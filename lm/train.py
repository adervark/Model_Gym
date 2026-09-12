"""The training loop. Everything in module 06 lives here.

Design: one function `train(cfg, model_cfg)` runnable standalone (single GPU)
or under torchrun (DDP/FSDP). bf16 autocast; AdamW with optional muP
parameter groups; cosine / warmup-stable-decay schedules; checkpointing.
"""
import os
import math
import time
import json
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from .config import TrainConfig, ModelConfig, save_config
from .model import Transformer
from .data import build_dataloaders
from .tokenizer import _VOCAB_SIZE


# Approximate peak tensor-core throughput for the MFU estimate.
#
# CONVENTION: dense (no structured sparsity) and FP32 accumulate — the mode
# training actually runs in. Two traps this table exists to avoid:
#   * vendor headline numbers are usually the WITH-SPARSITY figure (2x too high
#     for us): H100 BF16 is 1,979 TFLOPS* with sparsity, 989 dense.
#   * GeForce (Ada / Ampere consumer) runs tensor ops at HALF rate when
#     accumulating in FP32, so the widely-quoted "330 TFLOPS" for a 4090 is
#     165 for real training. Datacenter parts (A100/H100) have no such penalty.
# Getting either wrong silently halves or doubles every MFU number you report.
_PEAK_BF16 = {"h100": 989e12, "h200": 989e12, "a100": 312e12, "a6000": 155e12,
              "rtx 4090": 165e12, "rtx 4080": 97e12, "rtx 4070": 58e12,
              "rtx 4060": 29e12, "rtx 3090": 71e12, "rtx 3080": 60e12,
              "rtx 3070": 41e12, "rtx 3060": 26e12,
              # Volta/Turing: FP16 only (no bf16); these are already FP32-accumulate.
              "v100": 125e12, "t4": 65e12}


def estimate_peak(device: str) -> float:
    if not torch.cuda.is_available():
        return 1.0  # CPU: MFU meaningless, avoid div-by-zero weirdness
    name = torch.cuda.get_device_name(device).lower()
    return next((v for k, v in _PEAK_BF16.items() if k in name), 100e12)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# muP reference width: the size you TUNE the lr on. Everything else is
# expressed relative to it, so the same --lr transfers to any width.
MUP_BASE_WIDTH = 256


def mup_param_class(name: str, ndim: int) -> str:
    """Classify a parameter for muP (Yang et al., Tensor Programs V, Table 8).

    Three classes, because they scale differently with width w:
      "vector"   1-D params (RMSNorm gains). Init and lr both O(1).
      "input"    token embeddings. Init O(1), lr O(1) — the input dimension is
                 vocab, which does NOT grow with width, so nothing to correct.
      "output"   the readout (head, MTP heads). Init std ~ 1/w, lr ~ 1/w.
      "hidden"   every other matrix (q,k,v,o,gate,up,down,router,MLA projections):
                 fan_in grows with w, so init std ~ 1/sqrt(w) and — for Adam,
                 whose update is scale-invariant per parameter — lr ~ 1/w.
    """
    if ndim < 2:
        return "vector"
    if "tok_emb" in name:
        return "input"
    if "head" in name:            # head.weight, mtp_heads.N.weight
        return "output"
    return "hidden"


def mup_scale(model: Transformer, base_width: int = MUP_BASE_WIDTH):
    """muP init correction, applied on top of the model's standard init.

    The model inits every matrix at a width-independent std (see
    Transformer._init), so activations grow with width. muP fixes that by
    rescaling relative to the base width:
      hidden  x sqrt(base/w)   (emulates the 1/sqrt(fan_in) that muP requires)
      output  x (base/w)       (the readout is 1/w, not 1/sqrt(w))
      input, vector: untouched — they are already O(1) under muP.

    Note on attention logits: muP also replaces the 1/sqrt(head_dim) attention
    scale with 1/head_dim. This course holds head_dim fixed at 64 and grows
    width via n_heads (module 02 §1b), so that correction is a width-INDEPENDENT
    constant here and is deliberately omitted. If you ever scale head_dim with
    width, you must add it or transfer will break.

    LR groups are built in `make_optimizer`. See module 08."""
    ratio = base_width / model.cfg.hidden
    with torch.no_grad():
        for name, p in model.named_parameters():
            cls = mup_param_class(name, p.ndim)
            if cls == "hidden":
                p.mul_(math.sqrt(ratio))
            elif cls == "output":
                p.mul_(ratio)


def make_optimizer(model, cfg: TrainConfig, rank: int = 0):
    if cfg.optim == "muon":
        from .optim import Muon, split_muon_adam
        muon_p, adam_p = split_muon_adam(model)
        opt = Muon(muon_p, adam_p, lr=cfg.lr, weight_decay=cfg.weight_decay,
                   momentum=0.95, adam_lr=cfg.lr / 66)  # 0.02 / 66 ~ 3e-4
        if rank == 0:
            print(f"[muon] {len(muon_p)} 2D matrices + {len(adam_p)} 1D params "
                  f"(lr {cfg.lr}, adam lr {cfg.lr / 66:.2e})")
        return opt
    if cfg.use_mup:
        width = model.module.cfg.hidden if hasattr(model, "module") else model.cfg.hidden
        lr_scale = MUP_BASE_WIDTH / width
        # muP lr: ~1/width for hidden AND output matrices; O(1) for the input
        # embedding and the 1-D norm gains. Scaling the embedding lr (or leaving
        # q/k/v/gate/up unscaled) is the classic way to get a sweep that looks
        # like muP and does not transfer.
        scaled, flat = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (scaled if mup_param_class(n, p.ndim) in ("hidden", "output") else flat).append(p)
        groups = [
            {"params": scaled, "base_lr": cfg.lr * lr_scale},
            {"params": flat, "base_lr": cfg.lr},
        ]
        print(f"[muP] base {MUP_BASE_WIDTH} -> width {width}: lr x{lr_scale:.4f} on "
              f"{len(scaled)} hidden/output tensors, lr x1 on {len(flat)} input/vector")
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "base_lr": cfg.lr}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             weight_decay=cfg.weight_decay, fused=True)


def get_lr(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "cosine":
        t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))
    if cfg.lr_schedule == "ws":  # warmup-stable-decay (Chinchilla-style; LLM practice)
        if step < cfg.ws_stable_steps:
            return cfg.lr
        t = (step - cfg.ws_stable_steps) / max(1, cfg.max_steps - cfg.ws_stable_steps)
        return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))
    return cfg.lr


@torch.no_grad()
def evaluate(model, val_loader, cfg: TrainConfig, steps: int, device: str) -> float:
    """Distributed-aware: ALL ranks must run the eval forwards so DDP/FSDP
    collectives stay balanced (rank-0-only eval is a classic deadlock/crash).
    Losses are all-reduced; every rank returns the global mean."""
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(steps):
        x = next(it).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, _ = model(x[:, :-1], x[:, 1:])
        losses.append(loss.item())
    model.train()
    val = sum(losses) / len(losses)
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor(val, device=device)
        dist.all_reduce(t)
        val = t.item() / dist.get_world_size()
    return val


def save_checkpoint(model, optimizer, step: int, cfg: TrainConfig, model_cfg: ModelConfig,
                    best: bool = False):
    raw = model.module if hasattr(model, "module") else model
    tag = "best.pt" if best else f"step_{step}.pt"
    path = os.path.join(cfg.out_dir, tag)
    torch.save({"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "model_cfg": model_cfg.__dict__, "train_cfg": cfg.__dict__}, path)


def train(cfg: TrainConfig, model_cfg: ModelConfig):
    ddp, fsdp = cfg.ddp, cfg.fsdp
    rank, world = 0, 1
    if ddp or fsdp:
        # nccl on CUDA, gloo otherwise (gloor lets DDP/FSDP run on CPU — handy
        # for testing the mechanics without a GPU)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        if rank == 0:
            print(f"[dist] {world} ranks, backend={backend}, fsdp={fsdp}")
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed + rank)

    os.makedirs(cfg.out_dir, exist_ok=True)
    if rank == 0:
        save_config(cfg, os.path.join(cfg.out_dir, "train_config.json"))
        save_config(model_cfg, os.path.join(cfg.out_dir, "model_config.json"))

    model = Transformer(model_cfg, gradient_checkpointing=cfg.gradient_checkpointing)
    if cfg.use_mup:
        mup_scale(model)
    model.to(device)

    start_step = 0
    if cfg.resume:
        ckpt_path = cfg.resume if os.path.exists(cfg.resume) else \
            os.path.join(cfg.out_dir, cfg.resume)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        if rank == 0:
            print(f"[resume] from {ckpt_path} at step {start_step}")

    if fsdp:
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires a GPU (torch has no CPU FSDP). "
                               "Use --ddp to test distributed mechanics on CPU.")
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     mixed_precision=None, use_orig_params=True)
    elif ddp:
        device_ids = [rank] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=device_ids)
    if cfg.compile:
        model = torch.compile(model)

    optimizer = make_optimizer(model, cfg, rank)
    if cfg.resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(ckpt["optimizer"])
    train_loader, val_loader = build_dataloaders(cfg)
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if "cuda" in device \
        else nullcontext()

    if cfg.profile and rank == 0:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CUDA], schedule=torch.profiler.schedule(
            wait=2, warmup=2, active=3))

    tokens_per_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum * world
    peak = estimate_peak(device)

    use_wandb = cfg.use_wandb and rank == 0
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg.wandb_project or "lm-course", config=cfg.__dict__)
        except Exception as e:
            print(f"[wandb] disabled ({e.__class__.__name__}); run `wandb login` to enable")
            use_wandb = False
    model.train()
    t0, running_loss = time.time(), 0.0
    it = iter(train_loader)
    for step in range(start_step, cfg.max_steps):
        frac = get_lr(cfg, step) / cfg.lr  # 0..1 schedule fraction; muP groups keep ratio
        for g in optimizer.param_groups:
            g["lr"] = g.get("base_lr", cfg.lr) * frac
            if "adam_lr_base" in g:  # Muon's inner AdamW tracks the same schedule
                g["adam_lr"] = g["adam_lr_base"] * frac

        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(cfg.grad_accum):
            x = next(it).to(device)
            with ctx:
                _, loss, _ = model(x[:, :-1], x[:, 1:])
                loss = loss / cfg.grad_accum
            loss.backward()
            micro_loss += loss.item()
        # FSDP shards gradients, so the stock utility would compute each rank's
        # LOCAL norm and clip against that — a different (and weaker) clip on
        # every rank. FSDP's own method does the cross-rank reduction first.
        if fsdp:
            model.clip_grad_norm_(cfg.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if cfg.profile and rank == 0:
            prof.step()

        running_loss = 0.9 * running_loss + 0.1 * micro_loss if running_loss else micro_loss
        if rank == 0 and (step == 0 or (step + 1) % cfg.log_every == 0):
            dt = time.time() - t0
            tok_per_sec = tokens_per_step * cfg.log_every / dt
            if "cuda" in device:
                # MoE: FLOPs track ACTIVE params, not total — that's the efficiency
                mfu = tok_per_sec * 6 * model_cfg.active_params / peak
                mfu_s = f"| ~MFU {mfu*100:.0f}% (peak {peak/1e12:.0f} TF)"
            else:
                mfu_s = ""  # CPU: MFU is meaningless
            print(f"step {step+1:>6} | loss {running_loss:.4f} | lr {frac * cfg.lr:.2e} "
                  f"| {dt / cfg.log_every * 1000:.0f} ms/step | {tok_per_sec/1e3:.0f}k tok/s "
                  f"{mfu_s}")
            if use_wandb:
                wandb.log({"step": step + 1, "loss": running_loss,
                           "lr": frac * cfg.lr, "tok_per_sec": tok_per_sec})
            t0 = time.time()

        if (step + 1) % cfg.eval_every == 0:
            val_loss = evaluate(model, val_loader, cfg, cfg.eval_iters, device)
            if rank == 0:
                print(f"  val loss {val_loss:.4f}")
                if use_wandb:
                    wandb.log({"val_loss": val_loss, "step": step + 1})
        if (step + 1) % cfg.save_every == 0 and rank == 0:
            save_checkpoint(model, optimizer, step + 1, cfg, model_cfg)

    if rank == 0:
        save_checkpoint(model, optimizer, cfg.max_steps, cfg, model_cfg, best=True)
        print(f"done. artifacts in {cfg.out_dir}")
    if ddp or fsdp:
        dist.barrier()  # rank 0 may still be checkpointing; keep the group alive
        dist.destroy_process_group()
