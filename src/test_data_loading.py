from src.data_loader import load_recipe_dataset
import os

os.makedirs("out", exist_ok=True)

def main():
    num_rows = None
    dataset = load_recipe_dataset(
        csv_path="data\Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        image_dir="data\Food Images\Food Images",
        num_rows=num_rows, 
    )

    with open("out/data_preview.txt", "w", encoding="utf-8") as f:
        f.write(f"[INFO] Dataset contains {len(dataset)} recipes.\n\n")
        for i in range(len(dataset)):
            ex = dataset.get_example(i)
            f.write(f"Example {i}:\n")
            f.write(f"  ID: {ex.id}\n")
            f.write(f"  Text Preview: {ex.text[:200]}...\n")
            f.write(f"  Image: {ex.image_path}\n\n")

if __name__ == "__main__":
    main()
