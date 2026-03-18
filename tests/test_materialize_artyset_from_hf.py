from pathlib import Path

import importlib.util
import pandas as pd
import sys


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "materialize_artyset_from_hf.py"
spec = importlib.util.spec_from_file_location("materialize_artyset_from_hf", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["materialize_artyset_from_hf"] = mod
spec.loader.exec_module(mod)


def test_materialize_snapshot_to_training_layout_copies_index_and_images(tmp_path: Path) -> None:
    # Arrange: fake snapshot layout
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    df = pd.DataFrame(
        [
            {"local_path": "Impressionism/claude-monet_water-lilies-1906.jpg", "style_id": 1, "artist_id": 2, "genre_id": 0},
            {"local_path": "Post_Impressionism/vincent-van-gogh_starry-night.jpg", "style_id": 2, "artist_id": 3, "genre_id": 1},
        ]
    )
    (snapshot / "wikiart_index_selected.csv").write_text(df.to_csv(index=False))

    # Create only one of the two images to exercise missing handling.
    img1_src = snapshot / "Impressionism/claude-monet_water-lilies-1906.jpg"
    img1_src.parent.mkdir(parents=True, exist_ok=True)
    img1_src.write_bytes(b"\xff\xd8\xff")  # minimal fake jpg header bytes

    data_dir = tmp_path / "data"

    # Act
    mod.materialize_snapshot_to_training_layout(snapshot, data_dir)

    # Assert
    assert (data_dir / "wikiart_index_selected.csv").exists()
    assert (data_dir / "wikiart/Impressionism/claude-monet_water-lilies-1906.jpg").exists()
    # Second image missing in snapshot: should not be copied.
    assert not (data_dir / "wikiart/Post_Impressionism/vincent-van-gogh_starry-night.jpg").exists()

