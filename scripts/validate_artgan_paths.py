"""
scripts/validate_artgan_paths.py
=================================
Check that every path in the ArtGAN train/val CSVs exists under data/wikiart/.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKIART_ROOT = REPO_ROOT / "data" / "wikiart"
CSV_DIR = REPO_ROOT / "data" / "artgan_csv"

CSV_FILES = [
    "style_train.csv",
    "style_val.csv",
    "genre_train.csv",
    "genre_val.csv",
    "artist_train.csv",
    "artist_val.csv",
]

def main() -> None:
    if not WIKIART_ROOT.exists():
        print(f"ERROR: {WIKIART_ROOT} not found.")
        sys.exit(1)

    total_files = 0
    all_ok = True
    for csv_file_name in CSV_FILES:
        csv_file_path = CSV_DIR / csv_file_name
        label = csv_file_name.replace(".csv", "")
        if not csv_file_path.exists():
            print(f"  {label}: SKIP (file not found)")
            continue

        missing = []
        total = 0
        with open(csv_file_path) as csv_file:
            for row in csv.reader(csv_file):
                if not row:
                    continue
                total += 1
                local_path = row[0].strip()
                if not (WIKIART_ROOT / local_path).exists():
                    missing.append(local_path)
        if missing:
            all_ok = False
            print(f"  {label}: {total} rows, MISSING {len(missing)} files")
            for p in missing[:5]:
                print(f"    - {p}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        else:
            print(f"  {label}: {total} rows, OK")
        total_files += total
    if not all_ok:
        sys.exit(1)
    print(f"{total_files} total files, OK")

if __name__ == "__main__":
    main()
