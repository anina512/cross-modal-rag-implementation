from typing import List, Tuple

class PromptBuilder:
    def __init__(self):
        pass

    def build_prompt(
        self,
        claim_text: str,
        image_id: str,
        similar_texts: List[Tuple[str, float]],
        similar_images: List[Tuple[str, float]],
        dataset,
    ) -> str:
        """
        Build a structured prompt that combines evidence from text & images.

        similar_texts: List of (id, score)
        similar_images: List of (id, score)
        dataset: used to fetch text and image info from IDs
        """

        # Fetch text evidence
        text_evidence = []
        for id_, score in similar_texts:
            ex = dataset.get_example(int(id_))
            text_evidence.append(f"- ({score:.3f}) {ex.text}")

        # Add image evidence (use captions or explainers)
        image_evidence = []
        for id_, score in similar_images:
            ex = dataset.get_example(int(id_))
            # Treat the image ID or a placeholder caption as evidence
            image_evidence.append(f"- ({score:.3f}) Image ID: {ex.id} (user can inspect separately)")

        # Construct prompt
        prompt = f"""
You are an expert fact-checking AI model. A user has submitted a claim along with an image.

Claim:
"{claim_text}"

Image ID: {image_id}

Relevant text evidence:
{chr(10).join(text_evidence)}

Relevant image evidence:
{chr(10).join(image_evidence)}

Based on this multimodal evidence, analyze whether the claim is:
- True (supported)
- False (contradicted)
- Uncertain (not enough evidence)

Provide a short explanation with your final decision.
"""
        return prompt.strip()
