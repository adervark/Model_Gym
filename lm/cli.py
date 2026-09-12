"""One entry point: python -m lm.cli <subcommand>.

  pretrain   run the training loop (module 06/07), --resume to continue
  generate   sample from a checkpoint (module 03/12)
  eval       perplexity on a text file (module 11)
  sft        instruction-tune (module 09)
  lora       LoRA fine-tune: frozen base + rank-r adapters (module 14)
  dpo        preference training (module 10)
  grpo       RLVR training (module 10)
  quantize   int8/NF4/GPTQ on a checkpoint (module 12)
"""
import argparse
import json
import os

import torch

from .config import TrainConfig, ModelConfig, load_model_config
from .model import Transformer
from .train import train
from .generate import sample
from .tokenizer import get_tokenizer
from . import runners

PRESETS = {
    # n_heads * head_dim must equal hidden (the standard dense-attention shape,
    # required by llama.cpp and every real architecture)
    "tiny": dict(hidden=128, n_layers=4, n_heads=4, n_kv_heads=2,
                 head_dim=32, max_seq_len=256),
    "50m": dict(hidden=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    "100m": dict(hidden=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
    "300m": dict(hidden=1024, n_layers=16, n_heads=16, n_kv_heads=8, max_seq_len=1024),
}


def load_ckpt(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def cmd_pretrain(args):
    mc = ModelConfig(**PRESETS[args.size], n_experts=args.n_experts,
                     n_experts_active=args.n_experts_active,
                     rope_method=args.rope_method, rope_scale=args.rope_scale)
    tc = TrainConfig(out_dir=args.out, dataset=args.dataset, data_dir=args.data_dir,
                     max_steps=args.steps, seq_len=args.seq_len, micro_batch_size=args.batch_size,
                     lr=args.lr, use_mup=args.mup, ddp=args.ddp, fsdp=args.fsdp,
                     compile=args.compile, profile=args.profile, eval_every=args.eval_every,
                     save_every=args.save_every, log_every=args.log_every,
                     total_batch_tokens=args.total_batch_tokens,
                     grad_accum_steps=args.grad_accum_steps, warmup_steps=args.warmup,
                     gradient_checkpointing=args.gradient_checkpointing,
                     use_wandb=args.use_wandb, wandb_project=args.wandb_project,
                     num_workers=args.num_workers,
                     resume=args.resume)
    train(tc, mc)


def cmd_generate(args):
    model, _ = load_ckpt(args.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            break
        print(sample(model, prompt, device="cuda" if torch.cuda.is_available() else "cpu",
                     max_new=args.max_new, temperature=args.temperature, top_p=args.top_p))


def cmd_eval(args):
    from .eval import perplexity
    model, _ = load_ckpt(args.ckpt, "cuda" if torch.cuda.is_available() else "cpu")
    text = open(args.file).read()
    ppl = perplexity(model, text, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"perplexity: {ppl:.2f}")


def cmd_export_gguf(args):
    from .gguf_export import export_gguf
    model, _ = load_ckpt(args.ckpt)
    export_gguf(model, args.out)


def cmd_quantize(args):
    import lm.quant as q
    model, _ = load_ckpt(args.ckpt)
    if args.method == "int8":
        q.quantize_linear_int8(model)
        torch.save(model.state_dict(), args.out)
        print(f"saved int8 weights to {args.out}")
    elif args.method == "nf4":
        # NF4 is a storage format; report compression ratio on the big matrices
        w = model.blocks[0].mlp.gate.weight
        print(f"example gate weight: {w.numel()*w.element_size()/1e6:.1f} MB fp32 "
              f"-> ratio {q.nf4_size_reduction(w):.2f}x with NF4")
    elif args.method == "gptq":
        from .data import build_dataloaders
        tc = TrainConfig(data_dir=args.data_dir, dataset=args.dataset, seq_len=args.seq_len,
                         micro_batch_size=args.batch_size)
        _, loader = build_dataloaders(tc)
        q.gptq_model(model, loader)
        torch.save(model.state_dict(), args.out)
        print(f"saved 4-bit GPTQ weights to {args.out}")


def main():
    ap = argparse.ArgumentParser(prog="lm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--out", default="checkpoints/run")
    p.add_argument("--size", choices=PRESETS, default="100m")
    p.add_argument("--dataset", default="tiny_shakespeare")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--resume", default="")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--total-batch-tokens", type=int, default=524288)
    p.add_argument("--grad-accum", type=int, default=0, dest="grad_accum_steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--mup", action="store_true")
    p.add_argument("--ddp", action="store_true")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true", dest="gradient_checkpointing")
    p.add_argument("--wandb", action="store_true", dest="use_wandb")
    p.add_argument("--wandb-project", default="lm-course")
    p.add_argument("--moe-experts", type=int, default=0, dest="n_experts")
    p.add_argument("--moe-topk", type=int, default=2, dest="n_experts_active")
    p.add_argument("--rope-method", choices=["none", "linear", "yarn"], default="none")
    p.add_argument("--rope-scale", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=10)
    p.set_defaults(fn=cmd_pretrain)

    p = sub.add_parser("generate")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--file", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("sft")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.set_defaults(fn=lambda a: runners.run_sft(
        a.ckpt, a.data, a.out, a.steps, a.lr, a.batch_size, a.seq_len))

    p = sub.add_parser("dpo")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="checkpoints/dpo")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=128)
    p.set_defaults(fn=lambda a: runners.run_dpo(
        a.ckpt, a.out, a.steps, a.lr, a.beta, a.seq_len))

    p = sub.add_parser("grpo")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="checkpoints/grpo")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--max-new", type=int, default=48)
    p.set_defaults(fn=lambda a: runners.run_grpo(
        a.ckpt, a.out, a.steps, a.group, a.lr, a.beta, a.eps, a.max_new))

    p = sub.add_parser("lora")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/sft.jsonl")
    p.add_argument("--out", default="checkpoints/lora")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.set_defaults(fn=lambda a: runners.run_lora(
        a.ckpt, a.data, a.out, a.steps, a.lr, a.batch_size, a.seq_len, a.r, a.alpha))

    p = sub.add_parser("export-gguf")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_export_gguf)

    p = sub.add_parser("quantize")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["int8", "nf4", "gptq"], required=True)
    p.add_argument("--out", default="checkpoints/quantized.pt")
    p.add_argument("--dataset", default="tiny_shakespeare")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.set_defaults(fn=cmd_quantize)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
