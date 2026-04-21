from transformers import pipeline
from .prompts import build_prompt_cause
from .parsing import parse_json

pipe = pipeline(...)

def extract_medical_info(text: str) -> dict:
    prompt = build_prompt_cause(text)
    output = pipe(prompt)[0]["generated_text"]
    return parse_json(output)