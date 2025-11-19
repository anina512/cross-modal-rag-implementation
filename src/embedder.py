import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image


class Embedder:
    """
    Handles text and image embeddings:
      - SBERT: for text to text and image to text
      - CLIP: for image to image and text to image
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # SBERT for text embeddings
        self.sbert_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device,
        )

        # CLIP for image and textual embeddings
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_text_sbert(self, text: str) -> np.ndarray:
        emb = self.sbert_encoder.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def encode_text_clip(self, text: str) -> np.ndarray:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)[0]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)[0]
