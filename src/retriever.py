import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
import torch


class CrossModalRetriever:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load embeddings
        self.text_embs = np.load("embeddings/text_embs.npy")
        self.image_embs = np.load("embeddings/image_embs.npy")
        self.ids = np.load("embeddings/ids.npy")

        # Load indexes
        self.text_index = faiss.read_index("indexes/text.index")
        self.image_index = faiss.read_index("indexes/image.index")

        # Encoders
        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device
        )
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def similar_texts(self, query_text, k=5):
        q = self.text_encoder.encode(query_text, normalize_embeddings=True)
        q = np.array(q, dtype=np.float32).reshape(1, -1)
        D, I = self.text_index.search(q, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    def encode_image(self, image):
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]

    def similar_images(self, query_image_emb, k=5):
        q = query_image_emb.astype(np.float32).reshape(1, -1)
        D, I = self.image_index.search(q, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]

    def text_to_images(self, query_text, k=5):
        tokens = clip.tokenize([query_text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy(
            
        ).astype(np.float32)
        D, I = self.image_index.search(emb, k)
        return [(self.ids[idx], float(score)) for idx, score in zip(I[0], D[0])]
