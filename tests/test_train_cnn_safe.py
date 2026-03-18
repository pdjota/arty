from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import importlib.util
import sys


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "train_cnn_safe.py"
spec = importlib.util.spec_from_file_location("train_cnn_safe", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_cnn_safe"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_train_cnn_safe_retries_on_failure() -> None:
    run_train = mod.run_train

    # Patch subprocess.run so first call fails, second succeeds.
    with patch("subprocess.run") as run:
        run.side_effect = [
            type("R", (), {"returncode": 1})(),
            type("R", (), {"returncode": 0})(),
        ]

        # First run: non-zero
        rc1 = run_train(arch="cnnrnn", epochs=1, batch_size=32, cpu=False, extra_args=None)
        assert rc1 == 1

        # Second run: zero
        rc2 = run_train(arch="cnnrnn", epochs=1, batch_size=16, cpu=False, extra_args=None)
        assert rc2 == 0

