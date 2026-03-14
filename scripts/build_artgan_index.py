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

def load_images_from_folder_to_dataframe(root: Path) -> list[dict]:
    """Scan folder and subfolders for images **/*.jpg; return DataFrame with local_path, style, artist."""
    rows = []
    for img in sorted(root.glob("**/*.jpg")):
        image_path = img.relative_to(root)
        stem = image_path.stem
        artist = stem.split("_")[0]
        title_part = stem.split("_", 1)[1] if "_" in stem else ""
        parts = title_part.rsplit("-", 1)
        title = parts[0] if len(parts) == 2 and parts[1].isdigit() else title_part
        rows.append({
            "local_path": str(image_path),
            "style": image_path.parts[0],
            "artist": artist,
            "title": title,
        })
    return rows

def add_style_ids(rows: list[dict], style_class_path: Path) -> None:
    """Load style id→name map from path; for each row set style_id from style name (-1 if unknown). Mutates rows in place."""
    style_map = load_class_txt(style_class_path)
    name2id = {name: sid for sid, name in style_map.items()}
    for row in rows:
        row["style_id"] = name2id.get(row["style"], -1)

def add_artist_ids(rows: list[dict], artist_class_path: Path) -> None:
    """Load artist id→name map; for each row set artist_id from artist slug (name lower + _→-). -1 if unknown or file missing. Mutates rows in place."""
    if not artist_class_path.exists():
        for row in rows:
            row["artist_id"] = -1
        return
    artist_map = load_class_txt(artist_class_path)
    slug2id = {v.strip().lower().replace("_", "-"): k for k, v in artist_map.items()}
    for row in rows:
        slug = row["artist"].strip().lower().replace("_", "-")
        row["artist_id"] = slug2id.get(slug, -1)

def add_genre_ids(rows: list[dict], csv_dir: Path, genre_class_path: Path) -> None:
    """Load genre path→id from genre_train(_genre).csv and genre_val(_genre).csv; genre id→name from genre_class.txt. For each row set genre_id and genre. -1 and '' if unknown or files missing. Mutates rows in place."""
    genre_names = load_class_txt(genre_class_path) if genre_class_path.exists() else {}
    g_train = csv_dir / "genre_train_genre.csv" if (csv_dir / "genre_train_genre.csv").exists() else csv_dir / "genre_train.csv"
    g_val = csv_dir / "genre_val_genre.csv" if (csv_dir / "genre_val_genre.csv").exists() else csv_dir / "genre_val.csv"
    path2id: dict[str, int] = {}
    if g_train.exists() and g_val.exists():
        for path in (g_train, g_val):
            for r in csv.reader(open(path)):
                if len(r) >= 2 and r[0].strip() not in path2id:
                    path2id[r[0].strip()] = int(r[1].strip())
    for row in rows:
        gid = path2id.get(row["local_path"], -1)
        row["genre_id"] = gid
        row["genre"] = genre_names.get(gid, "")


def selected_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a valid label for all three tasks (style_id, artist_id, genre_id >= 0)."""
    return df[
        (df["style_id"] >= 0) & (df["artist_id"] >= 0) & (df["genre_id"] >= 0)
    ].reset_index(drop=True)


def main() -> None:
    if not WIKIART.exists():
        print(f"ERROR: {WIKIART} not found.")
        sys.exit(1)

    rows = load_images_from_folder_to_dataframe(WIKIART)
    add_style_ids(rows, CSV_DIR / "style_class.txt")
    add_artist_ids(rows, CSV_DIR / "artist_class.txt")
    add_genre_ids(rows, CSV_DIR, CSV_DIR / "genre_class.txt")
    data_frame = pd.DataFrame(rows)

    # Finalise
    data_frame = data_frame.reset_index(drop=True)
    data_frame.insert(0, "image_id", data_frame.index)
    data_frame = data_frame[COLS]

    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(out_dir / "wikiart_index.csv", index=False)
    print(f"wikiart_index.csv: {len(data_frame):,} rows")

    # Selected = rows with a valid label for all three tasks (style, artist, genre)
    sel = selected_rows(data_frame)
    sel["image_id"] = sel.index
    sel[COLS].to_csv(out_dir / "wikiart_index_selected.csv", index=False)
    print(f"wikiart_index_selected.csv: {len(sel):,} rows")


if __name__ == "__main__":
    main()
