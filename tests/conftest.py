import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.config import ModelConfig
from lm.model import Transformer

torch.set_num_threads(2)  # keep CPU tests fast


@pytest.fixture
def tiny_cfg():
    return ModelConfig(hidden=96, n_layers=2, n_heads=4, n_kv_heads=2,
                       max_seq_len=64, init_std=0.02)


@pytest.fixture
def model(tiny_cfg):
    return Transformer(tiny_cfg)


@pytest.fixture
def seeded():
    torch.manual_seed(1337)
    yield
    torch.manual_seed(torch.initial_seed())


@pytest.fixture
def tiny_data_dir(tmp_path):
    """A self-contained pretraining dataset (no network, no repo data needed)."""
    from lm.data import tokenize_to_bin
    text = ("the quick brown fox jumps over the lazy dog. " * 400)
    train = str(tmp_path / "tiny_train.bin")
    val = str(tmp_path / "tiny_val.bin")
    tokenize_to_bin(text, train)
    tokenize_to_bin(text[:500], val)
    return tmp_path, train, val
