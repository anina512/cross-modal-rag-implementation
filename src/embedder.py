import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image


class Embedder:
    """
    Handles text and image embeddings:
      - Text: Sentence-BERT (all-mpnet-base-v2)
      - Image: CLIP ViT-B/32
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Text encoder
        self.text_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=self.device
        )

        # CLIP image encoder
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalized embedding.
        Returns a float32 numpy vector.
        """
        emb = self.text_encoder.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image into a normalized CLIP embedding.
        Returns a float32 numpy vector.
        """
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().astype(np.float32)[0]
