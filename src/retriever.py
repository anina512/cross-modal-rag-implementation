# src/retriever.py

import os
import faiss
import numpy as np
import clip
import torch
from sentence_transformers import SentenceTransformer


class CrossModalRetriever:
    """
    Cross-modal retriever:

    Supports:
      - text → text      (SBERT space)
      - image → image    (CLIP space)
      - text → image     (CLIP space)
      - image → text     (CLIP space)
    """

    def __init__(self, device=None):
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Output logs
        os.makedirs("out", exist_ok=True)
        self.log_file = open("out/retriever_log.txt", "w", encoding="utf-8")
        self.log_file.write(f"[INFO] Retriever initialized on device={self.device}\n")

        # Load embeddings
        self.sbert_text_embs = np.load("embeddings/sbert_text_embs.npy")
        self.clip_text_embs = np.load("embeddings/clip_text_embs.npy")
        self.image_embs = np.load("embeddings/image_embs.npy")
        self.ids = np.load("embeddings/ids.npy")

        # Build FAISS indexes
        self.text_index = faiss.IndexFlatIP(self.sbert_text_embs.shape[1])
        self.text_index.add(self.sbert_text_embs)

        self.image_index = faiss.IndexFlatIP(self.image_embs.shape[1])
        self.image_index.add(self.image_embs)

        self.clip_text_index = faiss.IndexFlatIP(self.clip_text_embs.shape[1])
        self.clip_text_index.add(self.clip_text_embs)

        # Load encoders
        self.sbert_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device,
        )
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    # ---------- SBERT: Text → Text ----------
    def similar_texts(self, query_text, k=5):
        q = self.sbert_encoder.encode(query_text, normalize_embeddings=True).astype(np.float32).reshape(1, -1)
        D, I = self.text_index.search(q, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    # ---------- CLIP: Prepare Image ----------
    def encode_image(self, image):
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)[0]

    # ---------- CLIP: Image → Image ----------
    def similar_images(self, query_image_emb, k=5):
        q = query_image_emb.astype(np.float32).reshape(1, -1)
        D, I = self.image_index.search(q, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    # ---------- CLIP: Text → Image ----------
    def text_to_images(self, query_text, k=5):
        tokens = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().astype(np.float32)
        D, I = self.image_index.search(emb, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    # ---------- CLIP: Image → Text (CLIP text index) ----------
    def image_to_texts(self, query_image_emb, k=5):
        q = query_image_emb.astype(np.float32).reshape(1, -1)
        D, I = self.clip_text_index.search(q, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    # Cleanup
    def close(self):
        self.log_file.write("[INFO] Retriever closed.\n")
        self.log_file.close()
