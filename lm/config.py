from dataclasses import dataclass, field, asdict
import json


@dataclass
class ModelConfig:
    vocab_size: int = 50304          # gpt2 BPE vocab padded to 128-multiple
    hidden: int = 768                # d_model
    n_layers: int = 12
    n_heads: int = 12                # total query heads
    n_kv_heads: int = 4              # GQA: kv heads < q heads
    head_dim: int = 64               # RoPE applies per-head on this dim
    rope_theta: float = 10_000.0
    rope_method: str = "none"        # none | linear | yarn (context extension)
    rope_scale: float = 1.0          # target_len / trained_len for scaling
    ffn_mult: float = 8 / 3          # SwiGLU hidden = ffn_mult * d (Llama's 2/3 of 4d)
    ffn_multiple_of: int = 64        # round the hidden dim up to this (tensor cores)
    n_experts: int = 0               # 0 = dense; >0 = MoE MLP (module 15)
    n_experts_active: int = 2        # top-k routed experts per token
    moe_aux_loss_coef: float = 0.01  # load-balancing loss weight
    causal: bool = True              # False = bidirectional (diffusion LMs, module 18)
    use_mla: bool = False            # DeepSeek-V2 latent attention (module 17)
    kv_latent_dim: int = 128         # d_c: compressed K/V latent
    rope_latent_dim: int = 32        # d_r: decoupled RoPE slice
    mtp_depth: int = 0               # Multi-Token Prediction heads (V3, module 15/23)
    z_loss_coef: float = 0.0         # logit-z loss (PaLM; stabilization aux)
    max_seq_len: int = 1024
    tie_embeddings: bool = False     # Llama keeps untied; T5/GPT-style ties
    init_std: float = 0.02           # residual-stream init scale
    norm_eps: float = 1e-5

    @property
    def ffn_hidden(self) -> int:
        """SwiGLU hidden width. Llama's rule: 8d/3, rounded UP to a multiple of
        `ffn_multiple_of`. Three matrices at 8d/3 cost 8d^2 — the same as a
        two-matrix GELU MLP at 4d, which is what makes SwiGLU a free swap
        rather than a 1.5x parameter increase (module 03 §3)."""
        h = int(self.ffn_mult * self.hidden)
        m = self.ffn_multiple_of
        return ((h + m - 1) // m) * m

    @property
    def _per_attn_params(self) -> int:
        """One attention block's weights (all bias-free)."""
        d, H, D, n_kv = self.hidden, self.n_heads, self.head_dim, self.n_kv_heads
        if self.use_mla:
            d_c, d_r = self.kv_latent_dim, self.rope_latent_dim
            return (d * H * (D - d_r)          # wq   (compressed query part)
                    + d * H * d_r              # wq_r (roped query part)
                    + d * d_c                  # wkv  (the cached latent)
                    + d * d_r                  # wkr  (decoupled rope latent)
                    + d_c * n_kv * (D - d_r)   # wuk
                    + d_c * n_kv * D           # wuv
                    + H * D * d)               # wo
        # GQA: q and o are full width, k and v are n_kv-sized.
        return (d * H * D          # q
                + d * n_kv * D     # k
                + d * n_kv * D     # v
                + H * D * d)       # o

    @property
    def _per_mlp_params(self) -> int:
        """One MLP block: SwiGLU is THREE d x ffn_hidden matrices."""
        d, h = self.hidden, self.ffn_hidden
        dense = 3 * d * h
        if self.n_experts > 0:
            return self.n_experts * dense + d * self.n_experts  # experts + router
        return dense

    @property
    def n_params(self) -> int:
        d, V, L = self.hidden, self.vocab_size, self.n_layers
        emb = V * d
        head = 0 if self.tie_embeddings else V * d
        mtp = self.mtp_depth * d * V
        norms = 2 * d * L + d            # per-block norm1/norm2, plus final norm_f
        return emb + head + mtp + norms + (self._per_attn_params + self._per_mlp_params) * L

    @property
    def active_params(self) -> int:
        """Params actually used per token — the number FLOPs and MFU track.
        For MoE only n_experts_active of the experts run (the router always does)."""
        if self.n_experts == 0:
            return self.n_params
        d, V, L = self.hidden, self.vocab_size, self.n_layers
        emb = V * d
        head = 0 if self.tie_embeddings else V * d
        mtp = self.mtp_depth * d * V
        norms = 2 * d * L + d
        active_mlp = self.n_experts_active * 3 * d * self.ffn_hidden + d * self.n_experts
        return emb + head + mtp + norms + (self._per_attn_params + active_mlp) * L

    @property
    def kv_cache_tokens_per_position(self) -> int:
        """Floats stored per sequence position for the KV cache (all layers).

        GQA caches K and V per kv-head:      2 * n_kv * head_dim
        MLA caches ONE shared latent plus ONE shared roped key slice:
                                             d_c + d_r
        The MLA rope slice is shared across kv-heads by construction (it carries
        position, which is head-independent), so it is NOT multiplied by n_kv —
        that factor is the difference between MLA's real ~10-50x saving and a
        naive implementation's ~4x (module 17 §1)."""
        if self.use_mla:
            return self.n_layers * (self.kv_latent_dim + self.rope_latent_dim)
        return self.n_layers * 2 * self.n_kv_heads * self.head_dim


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints/run"
    dataset: str = "tiny_shakespeare"     # | fineweb_edu
    data_dir: str = "data"
    total_batch_tokens: int = 524_288     # tokens per optimizer step (0.5M)
    micro_batch_size: int = 8             # samples per device per step
    seq_len: int = 1024
    max_steps: int = 20_000
    warmup_steps: int = 500
    lr: float = 3e-4
    min_lr: float = 3e-5                  # cosine floor (set = lr to disable decay)
    lr_schedule: str = "cosine"           # | ws (warmup-stable-decay) | constant
    ws_stable_steps: int = 18_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    optim: str = "adamw"                 # adamw | muon (module 19)
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    use_mup: bool = False                 # muP with base width 256; lr is base_lr
    eval_every: int = 500
    eval_iters: int = 20
    log_every: int = 10
    save_every: int = 2000
    resume: str = ""                       # checkpoint path (or filename in out_dir)
    grad_accum_steps: int = 0             # 0 = auto from total_batch_tokens
    compile: bool = False
    profile: bool = False
    gradient_checkpointing: bool = False  # recompute activations in backward (VRAM saver)
    use_wandb: bool = False
    wandb_project: str = "lm-course"
    # data
    num_workers: int = 4
    # distributed
    ddp: bool = False
    fsdp: bool = False
    fsdp_shard: str = "full"              # full | grad | none
    # reproducibility
    seed: int = 1337

    @property
    def grad_accum(self) -> int:
        if self.grad_accum_steps > 0:
            return self.grad_accum_steps
        return max(1, self.total_batch_tokens // (self.micro_batch_size * self.seq_len))


def save_config(cfg, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_model_config(path: str) -> ModelConfig:
    with open(path) as f:
        return ModelConfig(**json.load(f))
