import os
from PIL import Image
from src.retriever import CrossModalRetriever
from src.data_loader import load_recipe_dataset
from src.prompt_builder import PromptBuilder
from src.llm_inference import LLMWrapper


def main():
    # Ensure output directory
    os.makedirs("out", exist_ok=True)
    
    # Configure input
    text_query = "Popular Indian street food recipes"  # or None
    image_path = None  # e.g. "data/query_images/spicy_pasta.jpg"

    # Load recipe dataset
    dataset = load_recipe_dataset(
        csv_path="data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        image_dir="data/Food Images/Food Images",
    )

    # Initialize tools
    retriever = CrossModalRetriever()
    prompt_builder = PromptBuilder()
    llm = LLMWrapper()

    similar_texts = similar_images = image_based_texts = image_based_images = None
    image_id = None
    img_emb = None

    # Identify Query Mode
    mode = "none"
    if text_query and image_path:
        mode = "text+image"
    elif text_query:
        mode = "text"
    elif image_path:
        mode = "image"

    print(f"[INFO] Running RAG Pipeline in '{mode}' mode\n")
    
    # TEXT-ONLY MODE
    if mode == "text":
        # 1. TEXT → TEXT
        similar_texts = retriever.similar_texts(text_query, k=5)

        # 2. TEXT → IMAGE
        similar_images = retriever.text_to_images(text_query, k=5)
        
        # Optionally also retrieve from top image result
        img_id, _ = similar_images[0]
        ex_img = dataset.get_example(int(img_id))
        img = Image.open(ex_img.image_path).convert("RGB")
        img_emb = retriever.encode_image(img)

        # IMAGE → IMAGE and IMAGE → TEXT based on top image evidence
        image_based_images = retriever.similar_images(img_emb, k=5)
        image_based_texts = retriever.image_to_texts(img_emb, k=5)

        image_id = ex_img.id

    # IMAGE-ONLY MODE
    elif mode == "image":
        img = Image.open(image_path).convert("RGB")
        img_emb = retriever.encode_image(img)

        # IMAGE → IMAGE
        image_based_images = retriever.similar_images(img_emb, k=5)

        # IMAGE → TEXT
        image_based_texts = retriever.image_to_texts(img_emb, k=5)

        image_id = "User image"

    # TEXT + IMAGE MODE
    elif mode == "text+image":
        # Text → Text
        similar_texts = retriever.similar_texts(text_query, k=5)

        # Image encoding
        img = Image.open(image_path).convert("RGB")
        img_emb = retriever.encode_image(img)

        # Image → Image
        image_based_images = retriever.similar_images(img_emb, k=5)

        # Image → Text
        image_based_texts = retriever.image_to_texts(img_emb, k=5)

        image_id = "User image"

    # Build prompt depending on what evidence exists
    prompt = prompt_builder.build_prompt(
        query=text_query or "User image-based query",
        dataset=dataset,
        similar_texts=similar_texts,
        similar_images=similar_images,
        similar_texts_from_image=image_based_texts,
        similar_images_from_image=image_based_images,
        image_id=ex_img.id if mode!="text" else None,
    )

    # Save prompt
    with open("out/generated_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # Generate LLM output
    print("[INFO] Generating response from LLaMA...\n")
    response = llm.generate(prompt)

    with open("out/llm_output.txt", "w", encoding="utf-8") as f:
        f.write(response)

    print("[OK] Files saved in 'out/' folder.")
    retriever.close()


if __name__ == "__main__":
    main()
