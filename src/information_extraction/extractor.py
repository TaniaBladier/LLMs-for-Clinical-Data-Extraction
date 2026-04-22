from transformers import pipeline
from .prompts import build_prompt_cause
from .parsing import parse_json

class MedicalExtractor:
    def __init__(self, model_name: str, **pipeline_kwargs):
        print(f"Loading model: {model_name}...")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            max_new_tokens=1024,
            **pipeline_kwargs
        )
        

    def extract(self, text: str):
        # This replaces the need for a global _pipe
        return self.pipe(f"Extract medical info from: {text}")
        

    def extract_medical_info(self, text: str) -> dict:
        prompt = build_prompt_cause(text)
        output = self.pipe(prompt)[0]["generated_text"]
        
        # Strip the prompt from the output — pipeline returns prompt + completion
        if isinstance(output, str):
            completion = output[len(prompt):]
        else:
            completion = output  # some pipeline versions return only the new tokens
        
        return parse_json(completion)