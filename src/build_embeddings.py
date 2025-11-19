import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.data_loader import load_recipe_dataset
from src.embedder import Embedder


def main():
    os.makedirs("out", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    # Load recipe dataset, subset for now
    dataset = load_recipe_dataset(
        csv_path="data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        image_dir="data/Food Images/Food Images",
    )

    log_path = "out/embeddings_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    log_file.write(f"[INFO] Loaded {len(dataset)} recipe examples with images.\n")

    embedder = Embedder()

    sbert_text_embs = []
    clip_text_embs = []
    image_embs = []
    ids = []

    log_file.write("[INFO] Generating embeddings...\n")

    for idx, ex in tqdm(enumerate(dataset.iter_examples()), total=len(dataset)):
        try:
            # SBERT text embedding from merged text
            sbert_emb = embedder.encode_text_sbert(ex.text)

            # CLIP text embedding from title
            clip_text_emb = embedder.encode_text_clip(ex.title)

            # CLIP image embedding
            img = Image.open(ex.image_path).convert("RGB")
            img_emb = embedder.encode_image(img)

            sbert_text_embs.append(sbert_emb)
            clip_text_embs.append(clip_text_emb)
            image_embs.append(img_emb)
            ids.append(ex.id)

        except Exception as e:
            log_file.write(f"[WARN] SKIPPED ID={ex.id} | Reason: {e}\n")

    if not sbert_text_embs or not clip_text_embs or not image_embs:
        log_file.write("[ERROR] No embeddings generated.")
        return

    np.save("embeddings/sbert_text_embs.npy", np.stack(sbert_text_embs))
    np.save("embeddings/clip_text_embs.npy", np.stack(clip_text_embs))
    np.save("embeddings/image_embs.npy", np.stack(image_embs))
    np.save("embeddings/ids.npy", np.array(ids))

    log_file.write("[OK] Saved embeddings.\n")
    log_file.close()
    print(f"[OK] Embeddings saved. Log at {log_path}")


if __name__ == "__main__":
    main()
