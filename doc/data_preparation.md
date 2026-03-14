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