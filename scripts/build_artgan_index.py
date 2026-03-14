"""
Build data/wikiart_index.csv and data/wikiart_index_selected.csv from
data/wikiart/ and data/artgan_csv/ label files.
"""
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIKIART = ROOT / "data" / "wikiart"
CSV_DIR = ROOT / "data" / "artgan_csv"
COLS = ["image_id", "local_path", "style", "style_id", "artist", "artist_id", "genre", "genre_id"]

GENRE_NAMES = {
    0: "abstract_painting", 1: "cityscape", 2: "genre_painting", 3: "illustration",
    4: "landscape", 5: "nude_painting", 6: "portrait", 7: "religious_painting",
    8: "sketch_and_study", 9: "still_life",
}

SELECTED_CSVS = [
    "style_train.csv", "style_val.csv", "genre_train.csv", "genre_val.csv",
    "genre_train_genre.csv", "genre_val_genre.csv", "artist_train.csv", "artist_val.csv",
]


def load_class_txt(path: Path) -> dict[int, str]:
    out = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if s:
            p = s.split(None, 1)
            if len(p) == 2:
                out[int(p[0])] = p[1].strip()
    return out


def paths_from_csv(path: Path) -> set[str]:
    return {row[0].strip() for row in csv.reader(open(path)) if row}


def main() -> None:
    if not WIKIART.exists():
        print(f"ERROR: {WIKIART} not found.")
        sys.exit(1)

    # Scan images
    rows = []
    for img in sorted(WIKIART.glob("**/*.jpg")):
        rel = img.relative_to(WIKIART)
        style = rel.parts[0]
        artist = img.stem.split("_")[0]
        rows.append({"local_path": str(rel), "style": style, "artist": artist})
    df = pd.DataFrame(rows)

    # Style id
    style_map = load_class_txt(CSV_DIR / "style_class.txt")
    df["style_id"] = df["style"].map({v: k for k, v in style_map.items()}).fillna(-1).astype(int)

    # Artist id (ArtGAN: 23 artists; name "Vincent_van_Gogh" → slug "vincent-van-gogh")
    artist_path = CSV_DIR / "artist_class.txt"
    if artist_path.exists():
        artist_map = load_class_txt(artist_path)
        slug2id = {v.strip().lower().replace("_", "-"): k for k, v in artist_map.items()}
        df["artist_id"] = df["artist"].map(slug2id).fillna(-1).astype(int)
    else:
        df["artist_id"] = -1

    # Genre (from train + val CSVs)
    g_train = CSV_DIR / "genre_train_genre.csv" if (CSV_DIR / "genre_train_genre.csv").exists() else CSV_DIR / "genre_train.csv"
    g_val = CSV_DIR / "genre_val_genre.csv" if (CSV_DIR / "genre_val_genre.csv").exists() else CSV_DIR / "genre_val.csv"
    if g_train.exists() and g_val.exists():
        g = pd.concat([
            pd.read_csv(g_train, header=None, names=["local_path", "genre_id"]),
            pd.read_csv(g_val, header=None, names=["local_path", "genre_id"]),
        ]).drop_duplicates("local_path")
        df = df.merge(g[["local_path", "genre_id"]], on="local_path", how="left")
        df["genre_id"] = df["genre_id"].fillna(-1).astype(int)
    else:
        df["genre_id"] = -1
    df["genre"] = df["genre_id"].map(GENRE_NAMES).fillna("")

    # Finalise
    df = df.reset_index(drop=True)
    df.insert(0, "image_id", df.index)
    df = df[COLS]

    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "wikiart_index.csv", index=False)
    print(f"wikiart_index.csv: {len(df):,} rows")

    # Selected = paths that appear in any ArtGAN train/val CSV
    selected = set()
    for f in SELECTED_CSVS:
        p = CSV_DIR / f
        if p.exists():
            selected |= paths_from_csv(p)
    if selected:
        sel = df[df["local_path"].isin(selected)].reset_index(drop=True)
        sel["image_id"] = sel.index
        sel[COLS].to_csv(out_dir / "wikiart_index_selected.csv", index=False)
        print(f"wikiart_index_selected.csv: {len(sel):,} rows")


if __name__ == "__main__":
    main()
