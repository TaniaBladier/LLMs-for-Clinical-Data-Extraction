from transformers import pipeline
from .prompts import build_prompt_cause
from .parsing import parse_json

_pipe = None  

def init_extractor(model_name: str, **pipeline_kwargs):
    """Call this once from your notebook before using extract_medical_info."""
    global _pipe
    _pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        max_new_tokens=1024,
        **pipeline_kwargs
    )

def extract_medical_info(text: str) -> dict:
    prompt = build_prompt_cause(text)
    output = _pipe(prompt)[0]["generated_text"]
    
    # Strip the prompt from the output — pipeline returns prompt + completion
    if isinstance(output, str):
        completion = output[len(prompt):]
    else:
        completion = output  # some pipeline versions return only the new tokens
    
    return parse_json(completion)