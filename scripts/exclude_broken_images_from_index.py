"""
Check all images in wikiart_index.csv and optionally exclude broken ones from both indexes.
Usage:
  python scripts/exclude_broken_images_from_index.py           # check, exclude broken, overwrite indexes
  python scripts/exclude_broken_images_from_index.py --check-only   # only report broken (exit 1 if any)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_FULL = DATA_DIR / "wikiart_index.csv"
INDEX_SELECTED = DATA_DIR / "wikiart_index_selected.csv"
WIKIART = DATA_DIR / "wikiart"
COLS = ["image_id", "local_path", "style", "style_id", "artist", "artist_id", "genre", "genre_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check images in index and exclude broken from both indexes.")
    parser.add_argument("--check-only", action="store_true", help="Only report broken images; do not overwrite indexes.")
    args = parser.parse_args()

    if not INDEX_FULL.exists():
        print(f"ERROR: {INDEX_FULL} not found. Run scripts/build_artgan_index.py first.", file=sys.stderr)
        sys.exit(2)
    if not WIKIART.exists():
        print(f"ERROR: {WIKIART} not found.", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(INDEX_FULL)
    n_before = len(df)
    ok = []
    broken = []  # (local_path, error_message)
    for i, local_path in enumerate(df["local_path"]):
        if (i + 1) % 10000 == 0:
            print(f"Checked {i + 1:,}/{n_before:,} images...")
        try:
            Image.open(WIKIART / local_path).convert("RGB")
            ok.append(True)
        except Exception as e:
            ok.append(False)
            broken.append((local_path, str(e)))

    if broken:
        print("Broken/missing images:")
        for local_path, err in broken:
            print(f"  {local_path}: {err}")
        print(f"Total broken: {len(broken)}")
    else:
        print("All images OK.")

    if args.check_only:
        sys.exit(1 if broken else 0)

    if not broken:
        print("Nothing to exclude.")
        return

    ok_series = pd.Series(ok, index=df.index)
    df = df.loc[ok_series].reset_index(drop=True)
    df["image_id"] = df.index
    df = df[COLS]
    df.to_csv(INDEX_FULL, index=False)
    print(f"wikiart_index.csv: {n_before:,} -> {len(df):,} rows")

    sel = df[(df["style_id"] >= 0) & (df["artist_id"] >= 0) & (df["genre_id"] >= 0)].reset_index(drop=True)
    sel["image_id"] = sel.index
    sel[COLS].to_csv(INDEX_SELECTED, index=False)
    print(f"wikiart_index_selected.csv: {len(sel):,} rows")


if __name__ == "__main__":
    main()
