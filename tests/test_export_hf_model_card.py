"""Tests for scripts/export_hf_model_card.py (render only; no checkpoint needed)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_export_module():
    path = ROOT / "scripts" / "export_hf_model_card.py"
    spec = importlib.util.spec_from_file_location("export_hf_model_card", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def test_render_model_card_contains_metrics_yaml() -> None:
    mod = _load_export_module()
    m = {
        "arch": "cnn",
        "checkpoint_name": "best.pt",
        "checkpoint_path": "/x/best.pt",
        "epoch": 3,
        "test_n": 50,
        "genre_top1": 0.5,
        "style_top1": 0.25,
        "artist_top1": 0.125,
        "artist_top5": 0.875,
    }
    text = mod.render_model_card(
        m,  # type: ignore[arg-type]
        model_index_name="Test Model",
        dataset_id="pdjota/artyset",
        source_name="unit test",
        source_url="https://example.org",
        license_id="apache-2.0",
        extra_body="",
    )
    assert "value: 0.500000" in text
    assert "value: 0.250000" in text
    assert "value: 0.125000" in text
    assert "value: 0.875000" in text
    assert "pdjota/artyset" in text
    assert "model-index:" in text
