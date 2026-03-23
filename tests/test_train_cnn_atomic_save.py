"""Tests for scripts/train_cnn.py checkpoint atomic save helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_cnn.py"


def _load_train_cnn():
    spec = importlib.util.spec_from_file_location("train_cnn", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["train_cnn"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_atomic_torch_save_roundtrip(tmp_path: Path) -> None:
    mod = _load_train_cnn()
    path = tmp_path / "ckpt.pt"
    mod._atomic_torch_save({"k": 42, "t": torch.zeros(2)}, path)
    assert path.is_file()
    assert not path.with_suffix(".pt.tmp").exists()

    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["k"] == 42
    assert list(data["t"].shape) == [2]
