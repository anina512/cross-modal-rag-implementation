import faiss
import numpy as np
import os

def main():
    # Load embeddings
    text_embs = np.load("embeddings/text_embs.npy")
    image_embs = np.load("embeddings/image_embs.npy")

    # Infer dimensions
    text_dim = text_embs.shape[1]
    image_dim = image_embs.shape[1]

    print(f"[INFO] Text embedding dimension: {text_dim}")
    print(f"[INFO] Image embedding dimension: {image_dim}")

    # Use cosine similarity = dot product on normalized vectors
    text_index = faiss.IndexFlatIP(text_dim)     # inner-product
    image_index = faiss.IndexFlatIP(image_dim)

    # Add vectors to the indexes
    text_index.add(text_embs)
    image_index.add(image_embs)

    # Save indexes
    os.makedirs("indexes", exist_ok=True)
    faiss.write_index(text_index, "indexes/text.index")
    faiss.write_index(image_index, "indexes/image.index")

    print("[OK] FAISS indexes saved:")
    print(" - indexes/text.index")
    print(" - indexes/image.index")


if __name__ == "__main__":
    main()
