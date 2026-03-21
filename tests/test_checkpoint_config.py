"""checkpoint_dir_for_arch() paths."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CHECKPOINT_DIR, checkpoint_dir_for_arch  # noqa: E402


def test_checkpoint_dir_defaults() -> None:
    assert checkpoint_dir_for_arch("cnn") == CHECKPOINT_DIR / "cnn_baseline"
    assert checkpoint_dir_for_arch("cnnrnn") == CHECKPOINT_DIR / "cnnrnn"


def test_checkpoint_dir_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ARTY_CHECKPOINT_CNN_DIR", "cnn/custom")
    assert checkpoint_dir_for_arch("cnn") == CHECKPOINT_DIR / "cnn" / "custom"
