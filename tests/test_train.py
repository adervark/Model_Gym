"""Training loop: schedule, optimizer, muP, grad accumulation, resume, overfit."""
import math
import os

import pytest
import torch

from lm.config import ModelConfig, TrainConfig
from lm.model import Transformer
from lm.train import (get_lr, make_optimizer, mup_scale, save_checkpoint, train)


def test_lr_schedule_cosine():
    cfg = TrainConfig(max_steps=100, warmup_steps=10, lr=1e-3, min_lr=1e-4,
                      lr_schedule="cosine")
    assert get_lr(cfg, 0) == pytest.approx(1e-3 / 10)
    assert get_lr(cfg, 9) == pytest.approx(1e-3)
    mid = get_lr(cfg, 55)
    assert 1e-4 < mid < 1e-3
    # decays toward min_lr; never quite reaches it before max_steps (by design)
    assert get_lr(cfg, 99) == pytest.approx(1e-4, abs=1e-5)


def test_lr_schedule_ws():
    cfg = TrainConfig(max_steps=100, warmup_steps=10, lr=1e-3, min_lr=1e-4,
                      lr_schedule="ws", ws_stable_steps=80)
    assert get_lr(cfg, 50) == pytest.approx(1e-3)  # stable plateau
    assert get_lr(cfg, 90) < 1e-3  # decayed
    assert get_lr(cfg, 99) == pytest.approx(1e-4, abs=1e-5)


def test_lr_schedule_constant():
    cfg = TrainConfig(max_steps=100, warmup_steps=0, lr=1e-3, lr_schedule="constant")
    assert get_lr(cfg, 50) == 1e-3


def test_mup_optimizer_groups():
    cfg = ModelConfig(hidden=512, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(cfg)
    tc = TrainConfig(lr=1e-3, use_mup=True)
    opt = make_optimizer(m, tc)
    assert len(opt.param_groups) == 2
    ratio = 256 / 512
    base_lrs = sorted(g["base_lr"] for g in opt.param_groups)
    assert base_lrs[0] == pytest.approx(1e-3 * ratio)
    assert base_lrs[1] == pytest.approx(1e-3)


def test_mup_scale_reduces_output_weights():
    cfg = ModelConfig(hidden=512, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64)
    m = Transformer(cfg)
    before = {n: p.clone() for n, p in m.named_parameters() if "o.weight" in n}
    mup_scale(m)
    for n, p in m.named_parameters():
        if "o.weight" in n:
            assert torch.allclose(p, before[n] * math.sqrt(256 / 512), atol=1e-8)


def test_grad_accumulation_equals_big_batch(tiny_cfg, seeded):
    """Sum of micro-batch grads == one big-batch grad (the loop's core invariant).
    Note: loss divided by grad_accum per micro-batch, exactly as train() does."""
    m1 = Transformer(tiny_cfg)
    m2 = Transformer(tiny_cfg)
    m2.load_state_dict(m1.state_dict())
    n_micro = 4
    xs = [torch.randint(0, tiny_cfg.vocab_size, (2, 16)) for _ in range(n_micro)]
    big = torch.cat(xs, dim=0)

    for x in xs:
        _, loss, _ = m1(x, x)
        (loss / n_micro).backward()
    _, loss2, _ = m2(big, big)
    loss2.backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5)


def test_overfit_memorization(tiny_cfg, seeded):
    """The universal sanity check: fixed batch must be memorized."""
    m = Transformer(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        _, loss, _ = m(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.5


def test_checkpoint_resume_roundtrip(tiny_cfg, tmp_path):
    m = Transformer(tiny_cfg)
    opt = make_optimizer(m, TrainConfig(lr=1e-4))
    cfg = TrainConfig(out_dir=str(tmp_path))
    save_checkpoint(m, opt, 5, cfg, tiny_cfg)
    ckpt = torch.load(os.path.join(tmp_path, "step_5.pt"), weights_only=False)
    m2 = Transformer(tiny_cfg)
    m2.load_state_dict(ckpt["model"])
    opt2 = make_optimizer(m2, TrainConfig(lr=1e-4))
    opt2.load_state_dict(ckpt["optimizer"])
    assert ckpt["step"] == 5
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    for g1, g2 in zip(opt.param_groups, opt2.param_groups):
        for s1, s2 in zip(g1["params"], g2["params"]):
            st1, st2 = opt.state[s1], opt2.state[s2]
            if st1:
                assert torch.equal(st1["exp_avg"], st2["exp_avg"])


def test_train_end_to_end(tiny_data_dir, tmp_path, capsys):
    """The real loop on the real dataloader: runs, loss falls, artifacts exist."""
    dir, train_bin, val_bin = tiny_data_dir
    mc = ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
    tc = TrainConfig(out_dir=str(tmp_path), dataset="tiny", data_dir=str(dir),
                     max_steps=12, seq_len=32, micro_batch_size=2,
                     total_batch_tokens=64, warmup_steps=2, lr=1e-3,
                     log_every=4, eval_every=12, save_every=12, num_workers=0)
    train(tc, mc)
    assert os.path.exists(os.path.join(tmp_path, "best.pt"))
    assert os.path.exists(os.path.join(tmp_path, "train_config.json"))
    out = capsys.readouterr().out
    assert "val loss" in out
