import pandas as pd
from pathlib import Path
from src.information_extraction.extractor import extract_medical_info

def extract_clinical_text(raw_text: str) -> str:
    markers = [
        "CHIEF COMPLAINT:",
        "HISTORY OF PRESENT ILLNESS:",
        "REASON FOR VISIT:",
    ]

    lines = raw_text.splitlines()

    start_idx = next(
        (i for i, line in enumerate(lines)
         if any(line.strip().startswith(m) for m in markers)),
        None
    )

    if start_idx is None:
        return ""

    end_idx = next(
        (i for i, line in enumerate(lines)
         if "About This Sample:" in line),
        len(lines)
    )

    return "\n".join(l for l in lines[start_idx:end_idx] if l.strip())


def process_er_folder(folder_path: str, limit: int = 5) -> pd.DataFrame:
    records = []
    files = sorted(Path(folder_path).glob("*.txt"))[:limit]

    for file in files:
        raw = file.read_text(encoding="utf-8")
        text = extract_clinical_text(raw)

        if not text:
            continue

        result = extract_medical_info(text)
        result["source_file"] = file.name

        records.append(result)

    return pd.DataFrame(records)