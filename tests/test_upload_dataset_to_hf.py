"""Tests for scripts/upload_dataset_to_hf.py"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "upload_dataset_to_hf.py"
spec = importlib.util.spec_from_file_location("upload_dataset_to_hf", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["upload_dataset_to_hf"] = mod
spec.loader.exec_module(mod)

get_hf_username = mod.get_hf_username


def test_get_hf_username_returns_username() -> None:
    with patch("subprocess.run") as run:
        run.return_value = type("R", (), {"returncode": 0, "stdout": "  myuser  \n", "stderr": ""})()
        assert mod.get_hf_username(None) == "myuser"


def test_get_hf_username_returns_none_on_failure() -> None:
    with patch("subprocess.run") as run:
        run.side_effect = FileNotFoundError()
        assert mod.get_hf_username(None) is None
