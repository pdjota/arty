"""
Upload a trained checkpoint and id→label JSONs to a Hugging Face model repo (for Spaces / demos).

Usage (repo root):

  python scripts/upload_model_to_hf.py --repo-id USER/reponame --checkpoint PATH/TO/best.pt

`HF_TOKEN`: repo `.env` (python-dotenv) wins over an existing shell `HF_TOKEN` when the key appears in `.env`
(`load_dotenv(..., override=True)`). Use a token with **write** access to the model repo. Local labels: `data/label_maps/`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_from_repo() -> None:
    """Load repo `.env` into os.environ. Keys in `.env` override the same keys already in the environment
    (fixes stale HF_TOKEN from the shell or IDE masking a valid token in `.env`)."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=True)


DATA_DIR = ROOT / "data"
INDEX_SELECTED = DATA_DIR / "wikiart_index_selected.csv"
LABEL_EXPORT_DEFAULT = DATA_DIR / "label_maps"


def build_id2label_from_selected_index(index_path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (genre_id2label, style_id2label, artist_id2label) with string keys for JSON."""
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    import pandas as pd

    df = pd.read_csv(index_path)
    required = {"genre_id", "genre", "style_id", "style", "artist_id", "artist"}
    if not required.issubset(df.columns):
        raise ValueError(f"Index missing columns: {sorted(required - set(df.columns))}")

    def _map(id_col: str, name_col: str) -> dict[str, str]:
        m = df.drop_duplicates(id_col).set_index(id_col)[name_col].astype(str).to_dict()
        return {str(int(k)): v for k, v in m.items()}

    return _map("genre_id", "genre"), _map("style_id", "style"), _map("artist_id", "artist")


def export_labels(labels_dir: Path, genre: dict[str, str], style: dict[str, str], artist: dict[str, str]) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "genre_id2label.json").write_text(json.dumps(genre, ensure_ascii=False))
    (labels_dir / "style_id2label.json").write_text(json.dumps(style, ensure_ascii=False))
    (labels_dir / "artist_id2label.json").write_text(json.dumps(artist, ensure_ascii=False))


def upload_checkpoint_and_labels(
    repo_id: str,
    checkpoint_path: Path,
    token: str,
    index_path: Path = INDEX_SELECTED,
    export_labels_dir: Path | None = None,
) -> str:
    from huggingface_hub import HfApi, create_repo

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    genre, style, artist = build_id2label_from_selected_index(index_path)
    if export_labels_dir is not None:
        export_labels(export_labels_dir, genre, style, artist)

    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", exist_ok=True, token=token)

    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="best_model.pt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    # Upload JSONs (write temp files so upload_file can stream from disk)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for name, data in (
            ("genre_id2label.json", genre),
            ("style_id2label.json", style),
            ("artist_id2label.json", artist),
        ):
            p = tmpd / name
            p.write_text(json.dumps(data, ensure_ascii=False))
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    _load_dotenv_from_repo()

    p = argparse.ArgumentParser(
        description="Upload model checkpoint + id2label JSONs to Hugging Face Hub. "
        "Loads repo-root .env by default (HF_TOKEN) when python-dotenv is installed."
    )
    p.add_argument("--repo-id", required=True, help="Model repo id, e.g. username/arty-cnn-baseline")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file to upload (e.g. checkpoints/cnn_baseline/best.pt or checkpoints/cnnrnn/best.pt)",
    )
    p.add_argument("--index", type=Path, default=INDEX_SELECTED, help="Selected index CSV (default: data/wikiart_index_selected.csv)")
    p.add_argument(
        "--export-labels-dir",
        type=Path,
        default=LABEL_EXPORT_DEFAULT,
        help=f"Write *_id2label.json here (default: {LABEL_EXPORT_DEFAULT})",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Missing token: add HF_TOKEN to repo-root .env",
            file=sys.stderr,
        )
        sys.exit(1)

    # quick sanity load of checkpoint format
    _ = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    try:
        url = upload_checkpoint_and_labels(
            repo_id=args.repo_id,
            checkpoint_path=args.checkpoint,
            token=token,
            index_path=args.index,
            export_labels_dir=args.export_labels_dir,
        )
        print(f"Uploaded model repo: {url}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

