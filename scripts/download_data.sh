#!/usr/bin/env bash
# Module 05: fetch data. Requires `datasets` (pip install -r requirements.txt).
set -euo pipefail
mkdir -p data

echo "[1/2] tiny_shakespeare (smoke tests)"
if [ ! -f data/tiny_shakespeare.txt ]; then
  curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/tiny_shakespeare.txt
fi

echo "[2/2] FineWeb-Edu sample (~100M tokens, the real pretraining corpus)"
PYBIN="${PYTHON:-python}"
[ -x ".venv/bin/python" ] && PYBIN=".venv/bin/python"
if [ ! -f data/fineweb_edu_train.bin ]; then
  $PYBIN - <<'EOF'
from datasets import load_dataset
from lm.data import tokenize_to_bin

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT",
                  split="train", streaming=True)
take = 200_000  # ~150-200M tokens; enough for a 100M model at Chinchilla-ish scale
texts = []
for i, ex in enumerate(ds):
    texts.append(ex["text"])
    if i >= take - 1:
        break
raw = "\n".join(texts)
n = len(raw)
tokenize_to_bin(raw[: int(n * 0.99)], "data/fineweb_edu_train.bin")
tokenize_to_bin(raw[int(n * 0.99):], "data/fineweb_edu_val.bin")
print(f"wrote {take} docs ({n/1e6:.0f} MB text)")
EOF
fi

echo "done. ls data/"
