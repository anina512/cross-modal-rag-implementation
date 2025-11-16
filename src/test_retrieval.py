from PIL import Image
from src.retriever import CrossModalRetriever
from src.data_loader import load_fakeddit_9k


def main():
    # Load dataset (uses cache, no need to re-download)
    dataset = load_fakeddit_9k(
        split="train",
        image_root="data/images",
        download_images=False,  # IMPORTANT: use cached CSV
    )

    retriever = CrossModalRetriever()

    # ---------- TEXT → TEXT ----------
    print("\n=== TEXT → TEXT ===")
    query_text = dataset.get_example(0).text
    results = retriever.similar_texts(query_text, k=5)
    print(f"Query text: {query_text[:80]}...")
    print("\nTop-5 similar texts:")
    for rank, (id_, score) in enumerate(results, start=1):
        print(f"{rank}. ID={id_} | Score={score:.4f}")

    # ---------- IMAGE → IMAGE ----------
    print("\n=== IMAGE → IMAGE ===")
    ex = dataset.get_example(0)
    img = Image.open(ex.image_path).convert("RGB")
    img_emb = retriever.encode_image(img)

    results = retriever.similar_images(img_emb, k=5)
    print(f"Query image ID: {ex.id}")
    print("\nTop-5 similar images:")
    for rank, (id_, score) in enumerate(results, start=1):
        print(f"{rank}. ID={id_} | Score={score:.4f}")

    # ---------- TEXT → IMAGE ----------
    print("\n=== TEXT → IMAGE ===")
    query_text_2 = "This image shows a flooded street"
    results = retriever.text_to_images(query_text_2, k=5)
    print(f"Query: {query_text_2}")
    print("\nTop-5 related images:")
    for rank, (id_, score) in enumerate(results, start=1):
        print(f"{rank}. ID={id_} | Score={score:.4f}")


if __name__ == "__main__":
    main()
