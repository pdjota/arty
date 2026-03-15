"""WikiArt dataset from selected index; images under data/wikiart/."""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image



class WikiArtDataset(Dataset):
    """Load rows from selected index CSV; each sample has image and style_id, artist_id, genre_id."""

    def __init__(
        self,
        index_path: Path,
        image_root: Path,
        transform: torch.nn.Module | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.df = pd.read_csv(index_path)
        self.df = self.df.astype({"style_id": "int", "artist_id": "int", "genre_id": "int"})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        row = self.df.iloc[idx]
        local_path = row["local_path"]
        full_path = self.image_root / local_path
        try:
            image = Image.open(full_path).convert("RGB")
        except OSError as e:
            import warnings
            warnings.warn(f"Broken image {local_path}: {e}; using placeholder.", UserWarning)
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["style_id"]), int(row["artist_id"]), int(row["genre_id"])
