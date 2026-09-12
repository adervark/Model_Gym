"""Network-gated integration test for scripts/distill.py (module 16).

Downloads a tiny real HF model + real GSM8K questions; verifies the full
mechanics: tokenize -> generate -> slice response -> write SFT-format JSONL
that `lm.cli sft` can consume. Skips when HF is unreachable.
"""
import json
import os
import subprocess
import sys
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": REPO}


def _network_ok():
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


@pytest.mark.slow
def test_distill_end_to_end(tmp_path):
    if not _network_ok():
        pytest.skip("no network access to huggingface.co")
    out = tmp_path / "traces.jsonl"
    r = subprocess.run(
        [PY, os.path.join(REPO, "scripts", "distill.py"),
         "--model", "sshleifer/tiny-gpt2", "--n", "2",
         "--out", str(out), "--max-new", "16"],
        capture_output=True, text=True, env=ENV, timeout=900)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert out.exists()
    lines = [json.loads(l) for l in open(out)]
    assert len(lines) >= 1
    assert set(lines[0]) == {"prompt", "response"}
    assert len(lines[0]["prompt"]) > 0 and len(lines[0]["response"]) > 0
