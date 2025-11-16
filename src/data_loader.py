from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests


@dataclass
class FakedditExample:
    id: str
    text: str
    image_path: str


class FakedditDataset:
    def __init__(self, df: pd.DataFrame):
        # df now already contains absolute paths and only needed columns
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def get_example(self, idx: int) -> FakedditExample:
        row = self.df.iloc[idx]

        return FakedditExample(
            id=str(idx),                     # index = ID
            text=str(row["text"]),
            image_path=row["image_path"],    # already absolute!
        )

    def iter_examples(self):
        for i in range(len(self)):
            yield self.get_example(i)


def download_images_from_original_urls(df: pd.DataFrame, save_dir: str) -> pd.DataFrame:
    """
    Downloads images using 'original_url' and stores local paths in image_path.
    Drops failed downloads and keeps only needed columns.
    """
    os.makedirs(save_dir, exist_ok=True)
    local_paths = []

    print(f"[INFO] Downloading images into: {save_dir}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row["original_url"]
        local_path = os.path.join(save_dir, f"{idx}.jpg")

        # Skip if already downloaded
        if os.path.exists(local_path):
            local_paths.append(local_path)
            continue

        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(local_path)
                local_paths.append(local_path)
            else:
                print(f"[WARN] HTTP {resp.status_code} - {url}")
                local_paths.append(None)
        except Exception as e:
            print(f"[ERROR] Could not download: {url} | {e}")
            local_paths.append(None)

    # Update image paths
    df["image_path"] = local_paths

    # Drop failed image downloads
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)

    # Keep only text + image
    df = df[["text", "image_path"]]

    return df


def load_fakeddit_9k(
    split: str = "train",
    image_root: str = "data/images",
    num_rows: Optional[int] = None,
    download_images: bool = False,
) -> FakedditDataset:
    cache_path = f"data/cache_{split}.csv"

    # If cache exists and we are not downloading images again, load cache
    if os.path.exists(cache_path) and not download_images:
        print(f"[INFO] Loading cached dataset from: {cache_path}")
        df = pd.read_csv(cache_path)
        return FakedditDataset(df)

    # Load from HuggingFace
    ds = load_dataset("ams-99/fakeddit_9k", split=split)
    if num_rows is not None:
        ds = ds.select(range(num_rows))

    df = ds.to_pandas()
    print(f"[INFO] Loaded {len(df)} raw rows from HuggingFace")

    if download_images:
        df = download_images_from_original_urls(df, save_dir=image_root)
        print(f"[INFO] After download, {len(df)} valid text+image pairs")

        df.to_csv(cache_path, index=False)
        print(f"[INFO] Saved cleaned dataset to cache: {cache_path}")

    else:
        # Try to map existing local paths
        df["image_path"] = df["image_path"].apply(
            lambda p: os.path.join(image_root, os.path.basename(p))
        )
        df = df[df["image_path"].apply(os.path.exists)]

        print(f"[INFO] Found {len(df)} existing text+image pairs")
        df = df[["text", "image_path"]]

    return FakedditDataset(df)

