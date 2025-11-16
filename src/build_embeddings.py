import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.data_loader import load_fakeddit_9k
from src.embedder import Embedder


def main():
    # Load from cached CSV instead of fresh HuggingFace load
    dataset = load_fakeddit_9k(
        split="train",
        image_root="data/images",
        download_images=False,  # Use cache
    )

    print(f"[INFO] Dataset size for embedding: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Did you generate the cache?")

    embedder = Embedder()  # auto-selects cuda if available

    text_embeddings = []
    image_embeddings = []
    ids = []

    for ex in tqdm(dataset.iter_examples(), total=len(dataset)):
        try:
            txt_emb = embedder.encode_text(ex.text)

            img = Image.open(ex.image_path).convert("RGB")
            img_emb = embedder.encode_image(img)

            text_embeddings.append(txt_emb)
            image_embeddings.append(img_emb)
            ids.append(ex.id)

        except Exception as e:
            print(f"[WARN] Skipping ID={ex.id} due to error: {e}")

    if not text_embeddings or not image_embeddings:
        raise RuntimeError("No embeddings were generated. Check image paths or embedding code.")

    text_embeddings = np.stack(text_embeddings)
    image_embeddings = np.stack(image_embeddings)
    ids = np.array(ids)

    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/text_embs.npy", text_embeddings)
    np.save("embeddings/image_embs.npy", image_embeddings)
    np.save("embeddings/ids.npy", ids)

    print("[OK] Saved:")
    print(f"  embeddings/text_embs.npy shape={text_embeddings.shape}")
    print(f"  embeddings/image_embs.npy shape={image_embeddings.shape}")
    print(f"  embeddings/ids.npy        shape={ids.shape}")


if __name__ == "__main__":
    main()
