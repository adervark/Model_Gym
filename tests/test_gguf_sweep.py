"""GGUF export (module 22) + scaling-law fitter."""
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from lm.config import ModelConfig
from lm.model import Transformer
from lm.gguf_export import export_gguf, gpt2_tokenizer_metadata, tensor_map

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def consistent_model():
    """llama.cpp requires hidden == n_heads * head_dim."""
    cfg = ModelConfig(hidden=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=32, max_seq_len=64)
    return Transformer(cfg), cfg


def test_export_roundtrip_via_gguf_library(consistent_model, tmp_path):
    m, _ = consistent_model
    path = str(tmp_path / "m.gguf")
    export_gguf(m, path)
    from gguf import GGUFReader
    r = GGUFReader(path)
    assert r.get_field("general.architecture").parts[-1].tobytes() == b"llama"
    assert len(r.tensors) == len(tensor_map(m))
    tok = r.get_field("tokenizer.ggml.tokens")
    # parts = [klen, kdata, vtype, itype, alen] + 2 per string
    n_strings = (len(tok.parts) - 5) // 2
    assert n_strings == 50257
    # tensor data must roundtrip exactly
    for name, t in tensor_map(m).items():
        rt = [x for x in r.tensors if x.name == name][0]
        want = t.numpy()
        got = rt.data.reshape(want.shape).astype(np.float32)
        assert np.array_equal(got, want.astype(np.float32)), name


def test_export_rejects_inconsistent_shapes(tiny_cfg, tmp_path):
    """n_heads*head_dim != hidden must raise a clear error (llama.cpp refuses)."""
    m = Transformer(tiny_cfg)  # 96 hidden vs 4*64 heads — inconsistent by design
    with pytest.raises(ValueError, match="n_heads"):
        export_gguf(m, str(tmp_path / "bad.gguf"))


def test_tokenizer_metadata_counts():
    tokens, types, merges, scores = gpt2_tokenizer_metadata()
    assert len(tokens) == len(types) == len(scores) == 50257
    assert len(merges) == 50000
    assert types[50256] == 3           # eot is CONTROL
    assert all(t in (1, 3) for t in types)


def test_vocab_trimmed_to_50257(consistent_model, tmp_path):
    m, _ = consistent_model
    path = str(tmp_path / "m.gguf")
    export_gguf(m, path)
    from gguf import GGUFReader
    r = GGUFReader(path)
    emb = [x for x in r.tensors if x.name == "token_embd.weight"][0]
    assert emb.data.shape[0] == 50257  # trimmed from 50304


def _find_llama_cli():
    """Locate a llama-cli binary. A single hard-coded path is how this test
    silently stopped running: it pointed at another tool's /tmp directory,
    which any reboot wipes, so the course's headline "verified end-to-end"
    claim skipped instead of failing. Check $LLAMA_CLI, then PATH, then the
    usual build locations."""
    import shutil
    env = os.environ.get("LLAMA_CLI")
    if env and os.path.exists(env):
        return env
    on_path = shutil.which("llama-cli")
    if on_path:
        return on_path
    for base in (os.path.expanduser("~/Code/llama.cpp"),
                 os.path.expanduser("~/llama.cpp"),
                 "/tmp/opencode/llama.cpp", "/opt/llama.cpp"):
        cand = os.path.join(base, "build", "bin", "llama-cli")
        if os.path.exists(cand):
            return cand
    return None


@pytest.mark.slow
def test_llamacpp_inference(tmp_path):
    """The real deal: build llama.cpp once, run our export through it.
    Skips if cmake/compiler or the llama.cpp tree is unavailable."""
    llama = _find_llama_cli()
    if llama is None:
        pytest.skip("llama.cpp not built (see module 22, or set LLAMA_CLI=/path/to/llama-cli)")
    cfg = ModelConfig(hidden=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=32, max_seq_len=64)
    m = Transformer(cfg)
    path = str(tmp_path / "m.gguf")
    export_gguf(m, path)
    r = subprocess.run([llama, "-m", path, "-p", "Hello", "-n", "12",
                        "--temp", "0.7", "--no-conversation"],
                       capture_output=True, text=True, timeout=120,
                       input="/exit\n")  # end the interactive prompt cleanly
    assert "failed to load model" not in r.stdout + r.stderr, r.stderr
    assert "Hello" in r.stdout  # prompt echoed = tokenizer works


class TestScalingLawFit:
    def test_fit_recovers_known_law(self):
        from scripts.sweep_scaling import fit_law
        rng = np.random.default_rng(0)
        E, A, B, a, b = 1.7, 400.0, 400.0, 0.34, 0.28
        N = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 3e7], dtype=float)
        D = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8], dtype=float)
        L = E + A / N ** a + B / D ** b + rng.normal(0, 0.002, len(N))
        fit = fit_law(N, D, L)
        # synthetic data from the exact model: fit should be close
        assert abs(fit["alpha"] - a) < 0.1
        assert abs(fit["beta"] - b) < 0.1
        assert abs(fit["E"] - E) < 0.05
