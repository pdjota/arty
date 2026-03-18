"""Tests for scripts/upload_model_to_hf.py"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "upload_model_to_hf.py"
spec = importlib.util.spec_from_file_location("upload_model_to_hf", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["upload_model_to_hf"] = mod
spec.loader.exec_module(mod)


def test_build_id2label_from_selected_index(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {"genre_id": 0, "genre": "g0", "style_id": 2, "style": "s2", "artist_id": 5, "artist": "a5"},
            {"genre_id": 1, "genre": "g1", "style_id": 3, "style": "s3", "artist_id": 6, "artist": "a6"},
        ]
    ).to_csv(index, index=False)

    genre, style, artist = mod.build_id2label_from_selected_index(index)
    assert genre == {"0": "g0", "1": "g1"}
    assert style == {"2": "s2", "3": "s3"}
    assert artist == {"5": "a5", "6": "a6"}


def test_export_labels(tmp_path: Path) -> None:
    out = tmp_path / "labels"
    mod.export_labels(out, {"0": "g0"}, {"1": "s1"}, {"2": "a2"})
    assert (out / "genre_id2label.json").exists()
    assert (out / "style_id2label.json").exists()
    assert (out / "artist_id2label.json").exists()


def test_upload_checkpoint_and_labels_mocks_hub(tmp_path: Path) -> None:
    # checkpoint
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": {}, "n_genre": 2, "n_style": 2, "n_artist": 2}, ckpt)
    # index for labels
    index = tmp_path / "index.csv"
    pd.DataFrame(
        [{"genre_id": 0, "genre": "g0", "style_id": 0, "style": "s0", "artist_id": 0, "artist": "a0"}]
    ).to_csv(index, index=False)

    with patch("huggingface_hub.create_repo") as create_repo_mock:
        with patch("huggingface_hub.HfApi") as api_class:
            api_instance = MagicMock()
            api_class.return_value = api_instance
            url = mod.upload_checkpoint_and_labels(
                repo_id="u/r",
                checkpoint_path=ckpt,
                token="t",
                index_path=index,
                export_labels_dir=tmp_path / "labels",
            )
            assert url.endswith("/u/r")
            create_repo_mock.assert_called_once()
            upload_calls = [c for c in api_instance.method_calls if c[0] == "upload_file"]
            # best_model.pt + 3 jsons
            assert len(upload_calls) == 4

