# src/test_rag_prompt.py

from PIL import Image
from src.retriever import CrossModalRetriever
from src.data_loader import load_fakeddit_9k
from src.prompt_builder import PromptBuilder
from src.llm_inference import LLMWrapper


def main():
    # Load dataset
    dataset = load_fakeddit_9k(
        split="train",
        image_root="data/images",
        download_images=True,
    )

    retriever = CrossModalRetriever()
    prompt_builder = PromptBuilder()
    llm = LLMWrapper()  # Use local sample model

    # Use example 0
    ex = dataset.get_example(0)

    # Retrieve evidence
    similar_texts = retriever.similar_texts(ex.text, k=5)
    img = Image.open(ex.image_path).convert("RGB")
    img_emb = retriever.encode_image(img)
    similar_images = retriever.similar_images(img_emb, k=5)

    # Build prompt
    prompt = prompt_builder.build_prompt(
        claim_text=ex.text,
        image_id=ex.id,
        similar_texts=similar_texts,
        similar_images=similar_images,
        dataset=dataset,
    )

    print("\n=== Generated Prompt ===\n")
    print(prompt)

    print("\n=== LLM Output ===\n")
    response = llm.generate(prompt)
    print(response)


if __name__ == "__main__":
    main()
