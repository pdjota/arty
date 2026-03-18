"""
Materialize the Hugging Face dataset repo `pdjota/artyset` into the local
folder structure expected by this project:

- `data/wikiart_index_selected.csv`
- `data/wikiart/<local_path>/*.jpg`

The dataset uploader stores images at repo root (paths like `Post_Impressionism/...`),
and stores the index at repo root as `wikiart_index_selected.csv`.
This script copies the required subset defined by `local_path` in the CSV.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd


DEFAULT_EXCLUDE_LOCAL_PATHS: Sequence[str] = (
    # Known corrupted image in the selected subset (kept in HF dataset for historical reasons).
    "Post_Impressionism/vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg",
)


def snapshot_download(
    repo_id: str,
    *,
    token: Optional[str],
    local_dir: Path,
    snapshot_download_fn: Optional[Callable[..., str]] = None,
) -> Path:
    """
    Download a dataset repo snapshot and return the local snapshot directory.

    `snapshot_download_fn` is injectable for unit tests.
    """
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download as snapshot_download_fn  # type: ignore

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download_fn(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(snapshot_path)


def materialize_snapshot_to_training_layout(
    snapshot_path: Path,
    data_dir: Path,
    *,
    exclude_local_paths: Optional[Sequence[str]] = None,
) -> None:
    """
    Copy dataset snapshot contents into `data/wikiart*` files used by training.
    """
    snapshot_path = Path(snapshot_path)
    data_dir = Path(data_dir)

    index_src = snapshot_path / "wikiart_index_selected.csv"
    if not index_src.exists():
        raise FileNotFoundError(f"Missing `{index_src}` in dataset snapshot")

    out_index = data_dir / "wikiart_index_selected.csv"
    out_wikiart_root = data_dir / "wikiart"
    out_wikiart_root.mkdir(parents=True, exist_ok=True)

    # Copy only images referenced by local_path in the index,
    # while optionally excluding known-broken rows.
    df = pd.read_csv(index_src)
    if "local_path" not in df.columns:
        raise ValueError("Index CSV must contain `local_path` column")

    excludes = list(exclude_local_paths) if exclude_local_paths is not None else list(DEFAULT_EXCLUDE_LOCAL_PATHS)
    if excludes:
        df = df[~df["local_path"].astype(str).isin(excludes)].reset_index(drop=True)

    # Write filtered index CSV back to training layout.
    df.to_csv(out_index, index=False)

    local_paths = df["local_path"].astype(str).unique().tolist()
    missing = 0
    for lp in local_paths:
        src = snapshot_path / lp
        dst = out_wikiart_root / lp
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            missing += 1
            continue
        shutil.copy2(src, dst)

    if missing:
        # Training dataset has a broken-image fallback, so missing images are not fatal.
        print(f"[materialize] Warning: {missing} images referenced by the index were missing in the snapshot.")


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize HF dataset repo into local training layout.")
    p.add_argument("--repo-id", default="pdjota/artyset", help="HF dataset repo id")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Where to write data/")
    p.add_argument("--cache-dir", type=Path, default=Path("data/.hf_cache"), help="HF snapshot cache directory")
    p.add_argument(
        "--exclude-local-path",
        action="append",
        default=None,
        help="Exclude one or more dataset rows by their `local_path` value. Can be provided multiple times.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Optional HF token (defaults to HF_TOKEN env var if present).",
    )
    args = p.parse_args()

    import os

    token = args.token or os.environ.get("HF_TOKEN")

    out_index = args.data_dir / "wikiart_index_selected.csv"
    out_wikiart_root = args.data_dir / "wikiart"

    excludes = args.exclude_local_path if args.exclude_local_path else None
    if out_index.exists() and out_wikiart_root.exists():
        # Ensure the exclude list is applied even if files were already materialized.
        df = pd.read_csv(out_index)
        if "local_path" in df.columns and (excludes or DEFAULT_EXCLUDE_LOCAL_PATHS):
            df = df[~df["local_path"].astype(str).isin(list(excludes) if excludes else list(DEFAULT_EXCLUDE_LOCAL_PATHS))].reset_index(drop=True)
            df.to_csv(out_index, index=False)
        return

    with tempfile.TemporaryDirectory() as tmp:
        # We stage snapshots under cache-dir to get reuse across restarts.
        snapshot = snapshot_download(args.repo_id, token=token, local_dir=args.cache_dir)
        materialize_snapshot_to_training_layout(snapshot, args.data_dir, exclude_local_paths=excludes)

    print(f"[materialize] Done. Wrote dataset to {args.data_dir}")


if __name__ == "__main__":
    main()

