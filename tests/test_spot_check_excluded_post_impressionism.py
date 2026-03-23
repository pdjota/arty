"""Tests for scripts/spot_check_excluded_post_impressionism.py helpers."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "spot_check_excluded_post_impressionism.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("spot_check_excluded_post_impressionism", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["spot_check_excluded_post_impressionism"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_load_id2label_roundtrip(tmp_path: Path) -> None:
    mod = _load_module()
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"0": "foo", "1": "bar"}), encoding="utf-8")
    assert mod.load_id2label(p) == {0: "foo", 1: "bar"}


def test_default_rel_paths_count() -> None:
    mod = _load_module()
    assert len(mod.DEFAULT_REL_PATHS) == 5


def test_resolve_device_force_cpu() -> None:
    mod = _load_module()
    assert mod.resolve_device(force_cpu=True).type == "cpu"


def test_load_label_maps_reads_repo_json() -> None:
    mod = _load_module()
    if not (mod.LABEL_MAPS_DIR / "genre_id2label.json").exists():
        return
    g, s, a = mod.load_label_maps()
    assert "Post_Impressionism" in s.values()
    assert len(g) >= 1 and len(a) >= 1
