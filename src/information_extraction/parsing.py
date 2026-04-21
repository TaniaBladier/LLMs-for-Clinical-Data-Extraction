import json

def parse_json(text: str):
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return {"error": "Invalid JSON", "raw_output": text}