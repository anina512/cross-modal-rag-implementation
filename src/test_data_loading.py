from src.data_loader import load_fakeddit_9k

def main():
    dataset = load_fakeddit_9k(
        split="train",
        image_root="data\images",
        num_rows=100,          # only first 100
        download_images=True   # TRUE ONLY FOR FIRST RUN
    )

    print(f"\n[INFO] Final dataset size: {len(dataset)}\n")

    # Show a few examples
    for i in range(3):
        ex = dataset.get_example(i)
        print(f"Example {i}:")
        print(f"  ID: {ex.id}")
        print(f"  Text: {ex.text[:80]}...")
        print(f"  Image: {ex.image_path}")
        print()

if __name__ == "__main__":
    main()
