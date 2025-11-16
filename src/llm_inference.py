import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMWrapper:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device=None):
        """
        Wrapper to load and generate text using a LLaMA model.
        :param model_name: HuggingFace model ID for LLaMA
        :param device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[INFO] Loading LLaMA model: {model_name} on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",         # Auto-distributes model across devices
            trust_remote_code=True,
        )

        # Fix for models that don't include pad_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=512):
        """
        Generate a text response given a prompt, using LLaMA.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
