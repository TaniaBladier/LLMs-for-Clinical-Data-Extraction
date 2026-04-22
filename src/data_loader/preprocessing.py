import pandas as pd
from pathlib import Path
from src.information_extraction.extractor import MedicalExtractor

class ClinicalPreprocessor:
    def __init__(self, language="en", markers=None, end_marker=None):
        self.language = language
        
        # Default to English if no markers provided
        self.markers = markers or [
            "CHIEF COMPLAINT:", 
            "HISTORY OF PRESENT ILLNESS:", 
            "REASON FOR VISIT:"
        ]
        self.end_marker = end_marker or "About This Sample:"

    def clean_text(self, raw_text: str) -> str:
        """Extracts the relevant clinical section from raw notes."""
        lines = raw_text.splitlines()

        start_idx = next(
            (i for i, line in enumerate(lines)
             if any(line.strip().startswith(m) for m in self.markers)),
            None
        )

        if start_idx is None:
            return ""

        end_idx = next(
            (i for i, line in enumerate(lines)
             if self.end_marker in line),
            len(lines)
        )

        return "\n".join(l for l in lines[start_idx:end_idx] if l.strip())
        

    def process_folder(self, folder_path: str, extractor_instance, limit: int = 5, prompt_func=None) -> pd.DataFrame:
        """Loops through files using the specific markers for this instance."""
        records = []
        files = sorted(Path(folder_path).glob("*.txt"))[:limit]
    
        for file in files:
            raw = file.read_text(encoding="utf-8")
            text = self.clean_text(raw)
    
            if not text:
                continue
    
            # Pass the prompt_func down to the extractor
            # It will use the lambda you passed from the notebook
            result = extractor_instance.extract_medical_info(text, prompt_func=prompt_func)
            
            result["source_file"] = file.name
            result["lang"] = self.language
            records.append(result)

        return pd.DataFrame(records)