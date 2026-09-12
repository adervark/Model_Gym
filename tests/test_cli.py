"""End-to-end CLI smoke tests: every subcommand must run to completion.

These spawn real processes against real files (slow; marked). The generated
smoke checkpoint is shared via module-scoped fixture.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": REPO}


def run(args, timeout=600):
    return subprocess.run([PY, "-m", "lm.cli", *args], capture_output=True,
                          text=True, env=ENV, timeout=timeout)


@pytest.fixture(scope="module")
def smoke_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    r = run(["pretrain", "--out", str(out), "--size", "tiny", "--steps", "10",
             "--seq-len", "64", "--batch-size", "2", "--grad-accum", "2",
             "--warmup", "2", "--log-every", "5", "--save-every", "10"])
    assert r.returncode == 0, r.stderr
    assert os.path.exists(out / "best.pt")
    return str(out / "best.pt")


@pytest.mark.slow
def test_pretrain_produces_artifacts(tmp_path):
    r = run(["pretrain", "--out", str(tmp_path), "--size", "tiny", "--steps", "6",
             "--seq-len", "64", "--batch-size", "2", "--grad-accum", "2",
             "--warmup", "2", "--log-every", "3", "--save-every", "6",
             "--eval-every", "6"])
    assert r.returncode == 0, r.stderr
    assert "val loss" in r.stdout
    assert os.path.exists(tmp_path / "train_config.json")
    assert os.path.exists(tmp_path / "model_config.json")
    assert os.path.exists(tmp_path / "best.pt")
    cfg = json.load(open(tmp_path / "model_config.json"))
    assert cfg["hidden"] == 128  # tiny preset (consistent: 4 heads x 32)


@pytest.mark.slow
def test_pretrain_resume(tmp_path):
    r1 = run(["pretrain", "--out", str(tmp_path), "--size", "tiny", "--steps", "5",
              "--seq-len", "64", "--batch-size", "2", "--grad-accum", "2",
              "--warmup", "2", "--log-every", "5", "--save-every", "5"])
    assert r1.returncode == 0, r1.stderr
    ckpt = tmp_path / "step_5.pt"
    assert ckpt.exists()
    r2 = run(["pretrain", "--out", str(tmp_path), "--size", "tiny", "--steps", "9",
              "--seq-len", "64", "--batch-size", "2", "--grad-accum", "2",
              "--warmup", "2", "--log-every", "5", "--save-every", "20",
              "--resume", str(ckpt)])
    assert r2.returncode == 0, r2.stderr
    assert "[resume]" in r2.stdout


@pytest.mark.slow
def test_generate_and_eval(smoke_ckpt):
    r = run(["generate", "--ckpt", smoke_ckpt, "--max-new", "20"])
    assert r.returncode == 0, r.stderr
    # eval needs a text file
    txt = os.path.join(REPO, "data", "tiny_shakespeare.txt")
    if os.path.exists(txt):
        r = run(["eval", "--ckpt", smoke_ckpt, "--file", txt])
        assert r.returncode == 0, r.stderr
        assert "perplexity" in r.stdout


@pytest.mark.slow
def test_quantize_subcommands(smoke_ckpt, tmp_path):
    out = tmp_path / "q.pt"
    r = run(["quantize", "--ckpt", smoke_ckpt, "--method", "int8", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert out.exists()
    r = run(["quantize", "--ckpt", smoke_ckpt, "--method", "nf4"])
    assert r.returncode == 0, r.stderr
    assert "ratio" in r.stdout


@pytest.mark.slow
def test_sft_dpo_grpo_lora(smoke_ckpt, tmp_path):
    sft_jsonl = os.path.join(REPO, "data", "sft.jsonl")
    if not os.path.exists(sft_jsonl):
        subprocess.run([PY, os.path.join(REPO, "scripts", "make_sft_data.py")],
                       capture_output=True, env=ENV)
    assert os.path.exists(sft_jsonl)

    for sub, extra in [
        ("sft", ["--data", sft_jsonl]),
        ("dpo", []),
        ("grpo", []),
        ("lora", ["--data", sft_jsonl]),
    ]:
        out = tmp_path / sub
        r = run([sub, "--ckpt", smoke_ckpt, "--out", str(out),
                 "--steps", "10", "--lr", "1e-4", *extra])
        assert r.returncode == 0, f"{sub}: {r.stderr}"
        assert (out / "best.pt").exists(), sub
    assert (tmp_path / "lora" / "adapters.pt").exists()


@pytest.mark.slow
def test_export_hf(smoke_ckpt, tmp_path):
    r = subprocess.run([PY, os.path.join(REPO, "scripts", "export_hf.py"),
                        "--ckpt", smoke_ckpt, "--out", str(tmp_path)],
                       capture_output=True, text=True, env=ENV, timeout=300)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(tmp_path / "pytorch_model.bin")
    assert os.path.exists(tmp_path / "config.json")
