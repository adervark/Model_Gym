"""Distributed training tests: DDP and FSDP on CPU via gloo.

Runs each mode under torchrun in a subprocess (2 ranks), on the self-contained
tiny dataset. Verifies: training completes, loss falls, checkpoints written,
and both ranks contribute (world_size=2 tokens/step in the log).
"""
import os
import subprocess
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": REPO}

TORCHRUN = os.path.join(os.path.dirname(PY), "torchrun")


def _make_data(tmp_path):
    from lm.data import tokenize_to_bin
    text = "the quick brown fox jumps over the lazy dog. " * 400
    tokenize_to_bin(text, str(tmp_path / "tiny_train.bin"))
    tokenize_to_bin(text[:500], str(tmp_path / "tiny_val.bin"))


def _run_dist(mode: str, tmp_path, steps: int = 10, eval_every: int | None = None):
    _make_data(tmp_path)
    out = tmp_path / "out"
    eval_every = eval_every if eval_every is not None else steps
    cmd = [TORCHRUN, "--nproc-per-node=2", "-m", "lm.cli", "pretrain",
           "--out", str(out), "--size", "tiny", "--steps", str(steps),
           "--seq-len", "32", "--batch-size", "2", "--grad-accum", "2",
           "--warmup", "2", "--log-every", "5", "--eval-every", str(eval_every),
           "--save-every", str(steps), "--data-dir", str(tmp_path),
           "--dataset", "tiny", "--num-workers", "0", mode]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=600)
    return r, out


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["--ddp", "--fsdp"])
def test_distributed_training(tmp_path, mode):
    if not os.path.exists(TORCHRUN):
        pytest.skip("torchrun not available")
    if mode == "--fsdp" and not torch.cuda.is_available():
        pytest.skip("FSDP requires a GPU (verified by torch itself); "
                    "DDP on gloo covers the distributed code path here")
    r, out = _run_dist(mode, tmp_path)
    assert r.returncode == 0, f"{mode} failed:\n{r.stdout}\n{r.stderr}"
    assert "[dist] 2 ranks" in r.stdout
    assert "val loss" in r.stdout
    assert (out / "best.pt").exists()
    # best.pt saved by rank 0 only; configs present
    assert (out / "train_config.json").exists()


@pytest.mark.slow
def test_ddp_loss_falls(tmp_path):
    """Both ranks training: running loss must fall across steps."""
    r, _ = _run_dist("--ddp", tmp_path, steps=14)
    assert r.returncode == 0, r.stderr
    import re
    losses = [float(m) for m in re.findall(r"loss (\d+\.\d+)", r.stdout)]
    assert len(losses) >= 3
    assert losses[-1] < losses[0]


@pytest.mark.slow
def test_ddp_mid_training_eval(tmp_path):
    """Eval at MID-training steps: the original bug crashed exactly here
    (rank-0-only eval desynced the DDP collectives). Multiple evals must work."""
    r, _ = _run_dist("--ddp", tmp_path, steps=15, eval_every=5)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert r.stdout.count("val loss") >= 2


@pytest.mark.slow
def test_fsdp_cpu_error_is_clear(tmp_path):
    """On CPU, FSDP must fail with our clear message, not torch's cryptic one."""
    if torch.cuda.is_available():
        pytest.skip("CUDA machine: FSDP runs for real")
    _make_data(tmp_path)
    cmd = [TORCHRUN, "--nproc-per-node=2", "-m", "lm.cli", "pretrain",
           "--out", str(tmp_path / "out"), "--size", "tiny", "--steps", "3",
           "--seq-len", "32", "--batch-size", "2", "--grad-accum", "1",
           "--data-dir", str(tmp_path), "--dataset", "tiny",
           "--num-workers", "0", "--fsdp"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode != 0
    assert "FSDP requires a GPU" in r.stdout + r.stderr
