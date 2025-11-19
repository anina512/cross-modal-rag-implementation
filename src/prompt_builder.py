from typing import List, Tuple

class PromptBuilder:
    """
    Builds multimodal prompts for recipe recommendation using:
      - Text-based evidence
      - Image-based evidence
      - Image → Text matches
      - Image → Image similarities
    """
    def __init__(self):
        pass

    def build_prompt(
        self,
        query: str,
        dataset,
        similar_texts: List[Tuple[str, float]] = None,
        similar_images: List[Tuple[str, float]] = None,
        similar_texts_from_image: List[Tuple[str, float]] = None,
        similar_images_from_image: List[Tuple[str, float]] = None,
        image_id: str = None,
    ) -> str:
        """
        Build recipe recommendation prompt using multimodal evidence.
        
        Args:
        - query: Text query (e.g., "vegan curry")
        - dataset: Fetched examples
        - similar_texts: List of (id, score) for text→text matches
        - similar_images: List of (id, score) for text→image matches
        - similar_texts_from_image: List of (id, score) for image→text matches
        - similar_images_from_image: List of (id, score) for image→image
        - image_id: Optional image ID (if a reference image was used)
        """

        # Build evidence sections dynamically
        recipe_evidence_txt = "\n".join(
            f"- (Score: {score:.3f}) {dataset.get_example(int(id_)).text[:200]}..."
            for id_, score in (similar_texts or [])
        ) or "*No text-based recipe matches found*"

        recipe_evidence_img = "\n".join(
            f"- (Score: {score:.3f}) Image ID: {id_} (user image match)"
            for id_, score in (similar_images or [])
        ) or "*No image-based matches found*"

        recipe_evidence_img_to_txt = "\n".join(
            f"- (Score: {score:.3f}) {dataset.get_example(int(id_)).text[:200]}..."
            for id_, score in (similar_texts_from_image or [])
        ) or "*No matches found from image to recipe text*"

        recipe_evidence_img_to_img = "\n".join(
            f"- (Score: {score:.3f}) Image ID: {id_} (similar dish)"
            for id_, score in (similar_images_from_image or [])
        ) or "*No visually similar dish found*"

        # Optional: include reference dish image ID if present
        image_part = f"\nDish Reference Image ID: {image_id}" if image_id else ""

        # Final prompt structure
        prompt = f"""
You are a helpful cooking assistant. A user is seeking recipe help or suggestions.

Query:
"{query}"{image_part}

Relevant recipe text evidence:
{recipe_evidence_txt}

Relevant recipe dish images:
{recipe_evidence_img}

Relevant recipes based on dish similarity (image → text):
{recipe_evidence_img_to_txt}

Visually similar dishes (image → image):
{recipe_evidence_img_to_img}

Based on this multimodal evidence, suggest:
- The top 2–3 recipes that best match the query (from the evidence)
- Their ingredients
- Why they are a good match
- Any variations or customization ideas (dietary preferences, ingredients availability)

Give your answer in a clear, friendly, and enthusiastic tone.
        """.strip()

        return prompt
