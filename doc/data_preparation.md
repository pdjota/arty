# ArtGAN WikiArt

We use WikiArt dataset which has the source images and the indexes for each of the 3 categories already classified.
We need to download the images dataset and generate `data/wikiart_index.csv`.

## 1. Images

The refined WikiArt dataset (images only) is provided by the [ArtGAN project](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset).

- **URL:** [wikiart.zip on Google Drive](https://drive.google.com/file/d/1vTChp3nU5GQeLkPwotrybpUGUXj12BTK/view) (~25.4 GB).
- **License:** Non-commercial research only; see [ArtGAN README](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md).

We download the file into the repo, extract it under `data/wikiart`. The zip and the extracted files are excluded from git. `rembrandt_woman-standing-with-raised-hands.jpg` and `vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg` contained errors.

The root data dir `data/wikiart/` contains 27 style subdirectories, each containing images named `{artist-slug}_{painting-title}-{year}.jpg`.

## 2. ArtGAN label files for metadata

ArtGAN only uses a subset of WikiArt, ArtGAN’s artist, style and genre labels [wikiart_csv.zip] (https://drive.google.com/file/d/1uug57zp13wJDwb2nuHOQfR2Odr0hh1a8/view)
which we include in the repo for convenience `data/artgan_csv`. There's a slight variation from in the style `Art_Nouveau` to `Art_Nouveau_Modern`.
Using `scripts/validate_artgan_paths.py` to validate


## 3. Rebuild the index

From the repo root:

```bash
python scripts/build_artgan_index.py
```

The script scans `data/wikiart/`, uses `style_class.txt` for `style_id`, merges genre from the two genre CSVs, and writes **`data/wikiart_index.csv`** with exactly:

| Column       | Description                                |
| ------------ | ------------------------------------------ |
| `image_id`   | Sequential 0-based id                       |
| `local_path` | Path relative to `data/wikiart/`            |
| `style`      | Style name (from directory)                 |
| `style_id`   | ArtGAN style class index                     |
| `artist`     | Artist slug (from filename)                  |
| `artist_id`  | ArtGAN artist class index; `-1` when unknown|
| `genre`      | Genre name; empty when unknown               |
| `genre_id`   | ArtGAN genre index; `-1` when unknown        |

**Artist:** Only 23 artists are in ArtGAN’s list; all others have `artist_id = -1`. **Genre:** The 10 genre names (e.g. landscape, portrait) use ArtGAN IDs 0–9; images not in the genre CSVs have `genre_id = -1`. When training, ignore rows with `-1` for that task (masked loss): do not backprop on artist or genre for those rows.
