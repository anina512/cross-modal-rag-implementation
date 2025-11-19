import os
import faiss
import numpy as np

def main():
    print("[INFO] Loading embeddings...")

    # Load all embeddings
    sbert_text_embs = np.load("embeddings/sbert_text_embs.npy")
    image_embs = np.load("embeddings/image_embs.npy")

    # Get embedding dimensions
    text_dim = sbert_text_embs.shape[1]
    image_dim = image_embs.shape[1]

    print(f"[INFO] SBERT text embedding dimension: {text_dim}")
    print(f"[INFO] CLIP image embedding dimension: {image_dim}")

    # FAISS indexes for cosine similarity (inner product on normalized vectors)
    text_index = faiss.IndexFlatIP(text_dim)
    image_index = faiss.IndexFlatIP(image_dim)

    # Add vectors to indexes
    print("[INFO] Adding vectors to indexes...")
    text_index.add(sbert_text_embs)     # SBERT text embeddings
    image_index.add(image_embs)         # CLIP image embeddings

    # Save indexes
    os.makedirs("indexes", exist_ok=True)
    faiss.write_index(text_index, "indexes/text.index")
    faiss.write_index(image_index, "indexes/image.index")

    print("\n[OK] Indexes saved successfully:")
    print(" - indexes/text.index  (SBERT text)")
    print(" - indexes/image.index (CLIP image)")


if __name__ == "__main__":
    main()
