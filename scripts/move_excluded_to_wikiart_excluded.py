"""
Move images that are not in the selected index from data/wikiart to data/wikiart_excluded.
After running, data/wikiart contains only images listed in wikiart_index_selected.csv.
Preserves directory structure under wikiart_excluded (e.g. Style/artist_file.jpg).
Usage: python scripts/move_excluded_to_wikiart_excluded.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIKIART = ROOT / "data" / "wikiart"
WIKIART_EXCLUDED = ROOT / "data" / "wikiart_excluded"
INDEX_SELECTED = ROOT / "data" / "wikiart_index_selected.csv"


def _normalize(path_str: str) -> str:
    return path_str.strip().replace("\\", "/")


def run(
    wikiart: Path = WIKIART,
    index_selected: Path = INDEX_SELECTED,
    excluded_dir: Path = WIKIART_EXCLUDED,
    dry_run: bool = False,
) -> int:
    """Move non-selected images from wikiart to excluded_dir. Returns number moved."""
    if not wikiart.exists():
        raise FileNotFoundError(f"{wikiart} not found")
    if not index_selected.exists():
        raise FileNotFoundError(f"{index_selected} not found")
    df = pd.read_csv(index_selected)
    selected = set(df["local_path"].astype(str).apply(_normalize))
    all_image_paths = list(wikiart.rglob("*.jpg"))
    present = set()
    to_move = []
    for p in all_image_paths:
        rel = _normalize(str(p.relative_to(wikiart)))
        present.add(rel)
        if rel not in selected:
            to_move.append((p, rel))
    missing = selected - present
    if missing and len(missing) <= 10:
        for m in sorted(missing):
            print(f"  (in index but missing in wikiart: {m})", file=sys.stderr)
    elif missing:
        print(f"  ({len(missing)} paths in index are missing in wikiart)", file=sys.stderr)
    if not to_move:
        return 0
    if dry_run:
        return len(to_move)
    excluded_dir.mkdir(parents=True, exist_ok=True)
    for src, rel in to_move:
        dst = excluded_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    return len(to_move)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be moved without moving")
    args = parser.parse_args()

    if not WIKIART.exists():
        print(f"ERROR: {WIKIART} not found.", file=sys.stderr)
        sys.exit(1)
    if not INDEX_SELECTED.exists():
        print(f"ERROR: {INDEX_SELECTED} not found. Run scripts/build_artgan_index.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INDEX_SELECTED)
    selected = set(df["local_path"].astype(str).apply(_normalize))
    all_image_paths = list(WIKIART.rglob("*.jpg"))
    to_move = []
    present = set()
    for p in all_image_paths:
        rel = _normalize(str(p.relative_to(WIKIART)))
        present.add(rel)
        if rel not in selected:
            to_move.append((p, rel))
    missing = selected - present
    if missing:
        print(f"Warning: {len(missing)} paths in index are not in wikiart (missing or wrong extension).", file=sys.stderr)
        for m in sorted(missing)[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more", file=sys.stderr)

    if not to_move:
        print("No excluded images to move.")
        return

    print(f"Excluded images to move: {len(to_move)} (index has {len(selected)}, wikiart had {len(all_image_paths)} image files)")
    if args.dry_run:
        for _, rel in to_move[:5]:
            print(f"  would move: {rel}")
        if len(to_move) > 5:
            print(f"  ... and {len(to_move) - 5} more")
        return

    WIKIART_EXCLUDED.mkdir(parents=True, exist_ok=True)
    for src, rel in to_move:
        dst = WIKIART_EXCLUDED / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print(f"Moved {len(to_move)} images to {WIKIART_EXCLUDED}")


if __name__ == "__main__":
    main()
