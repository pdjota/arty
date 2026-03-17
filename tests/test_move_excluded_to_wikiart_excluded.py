"""Tests for scripts/move_excluded_to_wikiart_excluded.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "move_excluded_to_wikiart_excluded.py"
spec = importlib.util.spec_from_file_location("move_excluded", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["move_excluded"] = mod
spec.loader.exec_module(mod)

run = mod.run
FAKE_JPEG = b"\xff\xd8\xff"


def test_run_dry_run_returns_count(tmp_path: Path) -> None:
    """Dry run returns number that would be moved."""
    wikiart = tmp_path / "wikiart"
    (wikiart / "S").mkdir(parents=True)
    (wikiart / "S" / "a.jpg").write_bytes(FAKE_JPEG)
    (wikiart / "S" / "b.jpg").write_bytes(FAKE_JPEG)
    index = tmp_path / "index.csv"
    pd.DataFrame([{"local_path": "S/a.jpg"}]).to_csv(index, index=False)
    excluded = tmp_path / "excluded"
    n = run(wikiart=wikiart, index_selected=index, excluded_dir=excluded, dry_run=True)
    assert n == 1
    assert (wikiart / "S" / "a.jpg").exists()
    assert (wikiart / "S" / "b.jpg").exists()


def test_run_moves_excluded(tmp_path: Path) -> None:
    """Excluded image is moved to excluded_dir preserving path."""
    wikiart = tmp_path / "wikiart"
    (wikiart / "StyleA").mkdir(parents=True)
    (wikiart / "StyleA" / "one.jpg").write_bytes(FAKE_JPEG)
    (wikiart / "StyleA" / "two.jpg").write_bytes(FAKE_JPEG)
    index = tmp_path / "index.csv"
    pd.DataFrame([{"local_path": "StyleA/one.jpg"}]).to_csv(index, index=False)
    excluded = tmp_path / "excluded"
    n = run(wikiart=wikiart, index_selected=index, excluded_dir=excluded, dry_run=False)
    assert n == 1
    assert not (wikiart / "StyleA" / "two.jpg").exists()
    assert (excluded / "StyleA" / "two.jpg").read_bytes() == FAKE_JPEG
    assert (wikiart / "StyleA" / "one.jpg").exists()


def test_run_none_excluded_returns_zero(tmp_path: Path) -> None:
    """When all images are in index, returns 0."""
    wikiart = tmp_path / "wikiart"
    (wikiart / "S").mkdir(parents=True)
    (wikiart / "S" / "a.jpg").write_bytes(FAKE_JPEG)
    index = tmp_path / "index.csv"
    pd.DataFrame([{"local_path": "S/a.jpg"}]).to_csv(index, index=False)
    n = run(wikiart=wikiart, index_selected=index, excluded_dir=tmp_path / "excluded", dry_run=False)
    assert n == 0


def test_run_missing_index_raises(tmp_path: Path) -> None:
    """FileNotFoundError if index does not exist."""
    (tmp_path / "wikiart").mkdir()
    with pytest.raises(FileNotFoundError, match="not found"):
        run(wikiart=tmp_path / "wikiart", index_selected=tmp_path / "nonexistent.csv", excluded_dir=tmp_path / "excluded")
