from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd


@dataclass
class RecipeExample:
    id: str       # string ID
    title: str    # Recipe title (shorter, fits CLIP text encoder)
    text: str     # Merged recipe text (full text for SBERT)
    image_path: str  # Valid image path


class RecipeDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def get_example(self, idx: int) -> RecipeExample:
        row = self.df.iloc[idx]
        return RecipeExample(
            id=str(idx),
            title=row["title"],
            text=row["merged_text"],
            image_path=row["image_path"],
        )

    def iter_examples(self):
        for i in range(len(self)):
            yield self.get_example(i)


def load_recipe_dataset(
    csv_path: str,
    image_dir: str,
    num_rows: int = None,
) -> RecipeDataset:
    """
    Loads a recipe dataset from a CSV file and associated image folder.

    :param csv_path: Path to CSV file.
    :param image_dir: Directory where images are stored.
    :param num_rows: Optional limit to number of rows.
    """
    print(f"[INFO] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[INFO] Raw rows in CSV: {len(df)}")

    # Ensure basic required columns exist
    required_columns = ["Title", "Ingredients", "Instructions", "Image_Name"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Construct merged text for SBERT
    df["merged_text"] = (
        "Ingredients:\n" + df["Ingredients"].fillna("") + "\n\n"
        "Instructions:\n" + df["Instructions"].fillna("")
    )

    # Rename Title explicitly for CLIP use
    df = df.rename(columns={"Title": "title"})

    # Build full image path using file names (usually image_name.jpg)
    df["image_path"] = df["Image_Name"].apply(
        lambda name: os.path.join(image_dir, f"{name}.jpg")
    )

    # Keep only rows where image exists
    df["image_exists"] = df["image_path"].apply(os.path.exists)
    missing_images = df[~df["image_exists"]]

    if len(missing_images) > 0:
        print(f"[WARN] {len(missing_images)} rows dropped: image not found.")

    df = df[df["image_exists"]]

    # Optional trimming of dataset
    if num_rows is not None:
        df = df.iloc[:num_rows]

    # Final rows retained
    print(f"[OK] Final dataset size: {len(df)}")

    # Keep only needed columns
    df = df[["title", "merged_text", "image_path"]]

    return RecipeDataset(df)
