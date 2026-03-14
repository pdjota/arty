"""Tests for scripts/build_artgan_index.py"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

# Load the script as a module (scripts/ is not a package)
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_artgan_index.py"
spec = importlib.util.spec_from_file_location("build_artgan_index", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["build_artgan_index"] = mod
spec.loader.exec_module(mod)

load_images_from_folder_to_dataframe = mod.load_images_from_folder_to_dataframe
add_style_ids = mod.add_style_ids
add_artist_ids = mod.add_artist_ids
add_genre_ids = mod.add_genre_ids
selected_rows = mod.selected_rows

# Minimal fake JPEG bytes for testing
FAKE_JPEG = b"\xff\xd8\xff"


def test_load_images_from_folder_to_dataframe_empty(tmp_path: Path) -> None:
    """No jpg files returns empty list."""
    assert load_images_from_folder_to_dataframe(tmp_path) == []


def test_load_images_from_folder_to_dataframe_one_file(tmp_path: Path) -> None:
    """One image under style/artist_title-foo.jpg returns one row with local_path, style, artist, title."""
    (tmp_path / "Impressionism").mkdir()
    (tmp_path / "Impressionism" / "claude-monet_water-lilies-1906.jpg").write_bytes(FAKE_JPEG)
    rows = load_images_from_folder_to_dataframe(tmp_path)
    assert len(rows) == 1
    assert rows[0]["local_path"] == "Impressionism/claude-monet_water-lilies-1906.jpg"
    assert rows[0]["style"] == "Impressionism"
    assert rows[0]["artist"] == "claude-monet"
    assert rows[0]["title"] == "water-lilies"


def test_load_images_from_folder_to_dataframe_sorted(tmp_path: Path) -> None:
    """Images are returned in sorted order by path."""
    (tmp_path / "Baroque").mkdir()
    (tmp_path / "Baroque" / "b-artist_second.jpg").write_bytes(FAKE_JPEG)
    (tmp_path / "Baroque" / "a-artist_first.jpg").write_bytes(FAKE_JPEG)
    rows = load_images_from_folder_to_dataframe(tmp_path)
    assert len(rows) == 2
    assert rows[0]["local_path"] == "Baroque/a-artist_first.jpg"
    assert rows[1]["local_path"] == "Baroque/b-artist_second.jpg"


def test_add_style_ids_known_style(tmp_path: Path) -> None:
    """Rows with style in the map get the correct style_id."""
    (tmp_path / "style_class.txt").write_text("0 Baroque\n1 Impressionism\n")
    rows = [{"style": "Baroque"}, {"style": "Impressionism"}]
    add_style_ids(rows, tmp_path / "style_class.txt")
    assert rows[0]["style_id"] == 0
    assert rows[1]["style_id"] == 1


def test_add_style_ids_unknown_style(tmp_path: Path) -> None:
    """Rows with style not in the map get style_id -1."""
    (tmp_path / "style_class.txt").write_text("0 Baroque\n")
    rows = [{"style": "Unknown_Style"}]
    add_style_ids(rows, tmp_path / "style_class.txt")
    assert rows[0]["style_id"] == -1


def test_add_style_ids_empty_rows(tmp_path: Path) -> None:
    """Empty list is a no-op."""
    (tmp_path / "style_class.txt").write_text("0 Baroque\n")
    rows: list[dict] = []
    add_style_ids(rows, tmp_path / "style_class.txt")
    assert rows == []


def test_add_artist_ids_known_artist(tmp_path: Path) -> None:
    """Rows with artist matching map slug get the correct artist_id."""
    (tmp_path / "artist_class.txt").write_text("0 Claude_Monet\n1 Edgar_Degas\n")
    rows = [{"artist": "claude-monet"}, {"artist": "Edgar_Degas"}]
    add_artist_ids(rows, tmp_path / "artist_class.txt")
    assert rows[0]["artist_id"] == 0
    assert rows[1]["artist_id"] == 1


def test_add_artist_ids_unknown_artist(tmp_path: Path) -> None:
    """Rows with artist not in the map get artist_id -1."""
    (tmp_path / "artist_class.txt").write_text("0 Claude_Monet\n")
    rows = [{"artist": "unknown-artist"}]
    add_artist_ids(rows, tmp_path / "artist_class.txt")
    assert rows[0]["artist_id"] == -1

def test_add_genre_ids_known_path(tmp_path: Path) -> None:
    """Rows with local_path in genre CSV get genre_id and genre name from genre_class.txt."""
    (tmp_path / "genre_class.txt").write_text("0 abstract_painting\n2 genre_painting\n")
    (tmp_path / "genre_train.csv").write_text("style/artist_title.jpg,2\n")
    (tmp_path / "genre_val.csv").write_text("other/path.jpg,0\n")
    rows = [{"local_path": "style/artist_title.jpg"}]
    add_genre_ids(rows, tmp_path, tmp_path / "genre_class.txt")
    assert rows[0]["genre_id"] == 2
    assert rows[0]["genre"] == "genre_painting"


def test_add_genre_ids_unknown_path(tmp_path: Path) -> None:
    """Rows with local_path not in CSVs get genre_id -1 and genre ''."""
    (tmp_path / "genre_class.txt").write_text("0 abstract_painting\n1 cityscape\n")
    (tmp_path / "genre_train.csv").write_text("other/path.jpg,0\n")
    (tmp_path / "genre_val.csv").write_text("another/path.jpg,1\n")
    rows = [{"local_path": "unknown/path.jpg"}]
    add_genre_ids(rows, tmp_path, tmp_path / "genre_class.txt")
    assert rows[0]["genre_id"] == -1
    assert rows[0]["genre"] == ""


def test_add_genre_ids_files_missing(tmp_path: Path) -> None:
    """When genre CSVs do not exist, all rows get genre_id -1 and genre ''."""
    (tmp_path / "genre_class.txt").write_text("0 abstract_painting\n")
    rows = [{"local_path": "style/artist_title.jpg"}]
    add_genre_ids(rows, tmp_path, tmp_path / "genre_class.txt")
    assert rows[0]["genre_id"] == -1
    assert rows[0]["genre"] == ""


def test_selected_rows_all_valid() -> None:
    """Rows with style_id, artist_id, genre_id >= 0 are all kept."""
    df = pd.DataFrame([
        {"local_path": "a", "style_id": 0, "artist_id": 0, "genre_id": 0},
        {"local_path": "b", "style_id": 1, "artist_id": 1, "genre_id": 1},
    ])
    out = selected_rows(df)
    assert len(out) == 2


def test_selected_rows_excludes_any_missing() -> None:
    """Rows with any of style_id, artist_id, genre_id == -1 are excluded."""
    df = pd.DataFrame([
        {"local_path": "a", "style_id": 0, "artist_id": 0, "genre_id": 0},
        {"local_path": "b", "style_id": -1, "artist_id": 0, "genre_id": 0},
        {"local_path": "c", "style_id": 0, "artist_id": -1, "genre_id": 0},
        {"local_path": "d", "style_id": 0, "artist_id": 0, "genre_id": -1},
    ])
    out = selected_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["local_path"] == "a"


def test_selected_rows_empty() -> None:
    """Empty DataFrame returns empty."""
    df = pd.DataFrame(columns=["local_path", "style_id", "artist_id", "genre_id"])
    out = selected_rows(df)
    assert len(out) == 0
