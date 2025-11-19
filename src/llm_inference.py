import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct", device=None):
        """
        Wrapper to load and generate text using a LLaMA model.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[INFO] Loading LLaMA model: {model_name} on {self.device}")
        os.makedirs("out", exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True
        )

        # Load model with optimization for memory and speed
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=1024, temperature=0.2):
        """
        Generate a text response given a prompt.
        Automatically adjusts the prompt if too long.
        """
        try:
            # Tokenize input prompt with truncation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - max_new_tokens,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,     # Deterministic output
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Clean the response from repetition of the prompt
            if response.startswith(prompt.strip()):
                response = response[len(prompt):].lstrip()

            return response

        except Exception as e:
            error_msg = f"[ERROR] LLM generation failed: {e}"
            print(error_msg)
            return error_msg
