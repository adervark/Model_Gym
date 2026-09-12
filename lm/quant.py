"""Post-training quantization (module 12): the pieces you can actually run.

- W8A8 dynamic int8 (linear weights quantized per-tensor)
- NF4 4-bit (QLoRA storage format, dequantize-on-the-fly)
- GPTQ-lite: layerwise 4-bit with inverse-Hessian error correction
  (the algorithm behind GPTQ/AWQ/exllama; Frantar et al. 2022)

For serving, use vLLM (PagedAttention + fused kernels beat anything here);
this file teaches the mechanics. See module 12.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- int8 W8A8 ---------------------------------------------------------------

def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mx = w.abs().max()
    scale = (mx / 127.0).clamp(min=1e-8)
    w_q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return w_q, scale


class Int8Linear(nn.Module):
    def __init__(self, w_q: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("w_q", w_q)
        self.register_buffer("scale", scale)

    def forward(self, x):
        return F.linear(x, self.w_q.to(x.dtype) * self.scale)


def quantize_linear_int8(model: nn.Module, skip: tuple[str, ...] = ("head",)):
    """Replace all Linear weights with int8. Skipping `head` (the logit projection)
    is standard: its errors land directly in the output distribution."""
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and not any(k in name for k in skip):
            w_q, scale = quantize_int8(m.weight.data)
            parent = model
            path = name.split(".")
            for part in path[:-1]:
                parent = getattr(parent, part)
            new = Int8Linear(w_q, scale)
            setattr(parent, path[-1], new)


# --- NF4 4-bit ---------------------------------------------------------------

class NF4:
    """NormalFloat4: 16 quantiles of N(0,1) — near-optimal for Gaussian weights."""
    OFFSETS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0])

    @classmethod
    def quantize(cls, w: torch.Tensor, block: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-block absmax into NF4 codes. Returns (codes uint8, absmax fp16)."""
        cls._block = block
        flat = w.flatten()
        pad = (block - flat.numel() % block) % block
        if pad:
            flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block)
        mx = blocks.abs().max(-1, keepdim=True).values
        norm = (blocks / (mx + 1e-8)).clamp(-1, 1)
        grid = cls.OFFSETS.to(w.device)
        idx = (norm.unsqueeze(-1) - grid.unsqueeze(0)).abs().argmin(-1)
        return idx.to(torch.uint8), mx.to(torch.float16)

    @classmethod
    def dequantize(cls, idx: torch.Tensor, mx: torch.Tensor,
                   orig_shape: torch.Size, block: int = 64) -> torch.Tensor:
        grid = cls.OFFSETS.to(idx.device)
        out = grid[idx.to(torch.long)] * mx.repeat_interleave(block, -1)
        return out.reshape(-1)[: orig_shape.numel()].reshape(orig_shape)


def nf4_size_reduction(w: torch.Tensor) -> float:
    idx, mx = NF4.quantize(w)
    raw = w.numel() * w.element_size()
    comp = idx.numel() * 1 + mx.numel() * 2
    return raw / comp


# --- GPTQ-lite ---------------------------------------------------------------

def gptq_layer(w: torch.Tensor, x: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Quantize W given calibration activations X (minimize ||WX - Wq X||_F).

    Iterates columns; the inverse-Hessian trick (OBD, LeCun) propagates each
    column's quantization error into the remaining columns analytically —
    no gradient descent needed.
    """
    n_in = w.shape[1]
    H = x.T @ x / max(x.shape[0], 1) + 1e-4 * torch.eye(n_in, device=w.device)
    Hinv = torch.inverse(H).float()
    Q = w.clone().float()
    err = torch.zeros_like(Q)
    per_col_scale = w.abs().max(dim=0).values / 7.5 + 1e-8  # 4-bit symmetric grid
    for i in range(n_in):
        col = Q[:, i] + err[:, i]
        q = (col / per_col_scale[i]).round().clamp(-7, 7) * per_col_scale[i]
        e = col - q
        if i < n_in - 1:
            err[:, i + 1:] -= e.unsqueeze(1) * (Hinv[i, i + 1:] / Hinv[i, i]).unsqueeze(0)
        Q[:, i] = q
    return Q


def gptq_model(model, calib_loader, layers_to_skip=("head",)):
    """Full-model GPTQ: calibration forward pass captures each Linear's input,
    then layers are quantized bottom-up with the inverse-Hessian correction."""
    model.eval()
    device = next(model.parameters()).device
    inputs, hooks = {}, []

    def hook_fn(name):
        def h(_, _in, _out):
            inputs[name] = _in[0].detach()
        return h

    linears = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    for name, m in linears.items():
        hooks.append(m.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        for i, x in enumerate(calib_loader):
            model(x[:, :-1].to(device))
            if i >= 7:  # ~8 calibration batches is plenty for stable Hessians
                break
    for h in hooks:
        h.remove()

    for name, m in linears.items():
        if any(k in name for k in layers_to_skip) or name not in inputs:
            continue
        x_in = inputs[name].flatten(0, -2).float()
        m.weight.data = gptq_layer(m.weight.data, x_in[: min(x_in.shape[0], 512)])
