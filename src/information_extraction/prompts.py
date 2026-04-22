from typing import List, Dict, Union

SYSTEM_PROMPT = """You are a clinical NLP system that extracts structured information from medical texts.
Always return valid JSON only — no preamble, no explanation, no markdown formatting.
If a field cannot be determined from the text, use null (not "unknown" or empty string).
"""


TEMPLATES = {
    "extraction": """Extract the following fields from the medical text below.
Return ONLY a valid JSON object with exactly these keys. Use null for any field not found in the text.

Fields:
- "circumstances": string — the cause or triggering event of the ER visit (e.g. "fall from ladder", "road traffic accident")
- "symptoms": array of strings — symptoms reported by or observed in the patient
- "diagnosis": string — the clinical diagnosis if explicitly stated
- "medications": array of strings — all medications mentioned (prescribed, taken, or administered)
- "patient_age": integer or null — patient's age in years
- "patient_gender": string or null — patient's gender ("male", "female", or as stated)

Medical text:
{text}
""",

    "normalize": """Extract and normalize clinical information from the medical text below.
Return ONLY a valid JSON object with exactly these keys:

- "symptoms_normalized": array of strings — symptoms mapped to standard clinical terms (SNOMED-CT preferred)
- "diagnosis_normalized": string or null — diagnosis mapped to standard clinical terminology
- "medications_normalized": array of strings — medications converted to their INN generic names

Medical text:
{text}
""",

    "timeline": """Extract a chronological timeline of clinical events from the medical text below.
Return ONLY a valid JSON array. Each element must have exactly these keys:

- "event": string — a brief description of what happened
- "time_expression": string — the time reference as stated in the text (e.g., "2 days ago", "on admission", "yesterday morning")

Order events from earliest to most recent. If timing is ambiguous, preserve the narrative order.

Medical text:
{text}
""",

    "cause": """Identify the precipitating external event that led to the medical emergency described below.

A precipitating event is a concrete external incident such as: a fall, a road accident, 
domestic violence, a suicide attempt, a sports injury, an overdose, or a workplace accident.

IMPORTANT: Do NOT return a symptom or diagnosis as the event (e.g. "chest pain", "myocardial infarction" are NOT events).

Return ONLY a valid JSON object with exactly these keys:
- "event": string or null — the precipitating external incident in a few words, or null if not mentioned
- "confidence": string — one of: "high", "medium", "low", "unknown"
- "explanation": string — one sentence justifying your answer

Examples:
{{"event": "fall from ladder", "confidence": "high", "explanation": "The patient fell from a ladder at a construction site."}}
{{"event": null, "confidence": "unknown", "explanation": "No external precipitating event is described in the text."}}

Medical text:
{text}
""",

    "negation": """Extract all symptoms mentioned in the medical text and classify each as present, absent, or uncertain.

- "present": symptom is affirmed (e.g., "complains of chest pain")
- "absent": symptom is negated (e.g., "denies fever", "no shortness of breath")
- "uncertain": symptom is qualified or doubtful (e.g., "possible dyspnea", "rule out nausea")

Return ONLY a valid JSON array. Each element must have exactly these keys:
- "symptom": string — the symptom as described
- "status": string — one of: "present", "absent", "uncertain"

Medical text:
{text}
""",

    "summary": """Write a concise clinical summary of the medical text below, suitable for a physician handover.

Requirements:
- Use precise clinical language
- Cover: patient profile, presenting complaint, key findings, diagnosis (if stated), and treatment (if stated)
- Maximum 5 sentences
- Do not invent or infer information not present in the text

Medical text:
{text}
""",

    "urgency": """Classify the clinical urgency of the case described in the medical text below.

Return ONLY a valid JSON object with exactly these keys:
- "triage_level": string — one of: "low", "moderate", "high", "emergency"
- "justification": string — one or two sentences citing the specific findings that drove your classification

Triage guidance:
- emergency: life-threatening, requires immediate intervention
- high: serious condition, risk of deterioration
- moderate: significant but stable, needs timely evaluation
- low: non-urgent, can wait for routine care

Medical text:
{text}
""",

    "inconsistency": """Identify factual inconsistencies or clinical contradictions within the medical text below.

Return ONLY a valid JSON array. Each element must have exactly these keys:
- "statement_1": string — the first conflicting statement (quote or close paraphrase)
- "statement_2": string — the second conflicting statement (quote or close paraphrase)
- "explanation": string — one sentence explaining why these statements are contradictory

If no inconsistencies are found, return an empty array: []

Medical text:
{text}
""",

    "missing": """Identify clinically relevant information that is absent from the medical text but would normally be required for proper assessment or treatment.

Return ONLY a valid JSON array of strings, each naming a missing data element.

Consider: allergies, current medications, past medical history, vital signs, relevant lab values, 
vaccination status, social history, family history, mechanism of injury, time of onset, etc.

Only flag information that is genuinely missing AND clinically important given the context.

Medical text:
{text}
""",

    "relations": """Extract relationships between clinical entities mentioned in the medical text below.

Return ONLY a valid JSON array. Each element must have exactly these keys:
- "subject": string — the source entity (e.g., patient, medication, condition)
- "relation": string — the relationship type using snake_case (e.g., "has_symptom", "treated_with", "diagnosed_with", "contraindicated_with", "caused_by")
- "object": string — the target entity

Medical text:
{text}
""",

    "noisy": """The medical text below may contain typos, abbreviations, fragmented sentences, or OCR errors.
Extract the key clinical information as accurately as possible despite this noise.

Return ONLY a valid JSON object with exactly these keys:
- "symptoms": array of strings — symptoms identified (corrected if clearly misspelled)
- "diagnosis": string or null — the diagnosis if identifiable
- "medications": array of strings — medications mentioned (use best interpretation of garbled text)

If a field cannot be confidently extracted, use null or an empty array.

Medical text:
{text}
""",

    "counterfactual": """Based on the medical text below, reason about what would likely have happened 
if the patient had NOT received the treatment described.

Return ONLY a valid JSON object with exactly these keys:
- "treatment_received": string — the specific treatment mentioned in the text
- "likely_outcome_without_treatment": string — a brief, evidence-based clinical explanation of the probable outcome
- "severity_increase": string — one of: "yes", "no", "uncertain"

Base your reasoning strictly on the information in the text and standard clinical knowledge.

Medical text:
{text}
""",

    "sensitivity": """Extract the diagnosis and symptoms from the medical text below.
Be precise: capture the exact clinical terms used, preserving important distinctions 
(e.g., "suspected" vs. "confirmed", "acute" vs. "chronic").

Return ONLY a valid JSON object with exactly these keys:
- "symptoms": array of strings — each symptom as clinically described in the text
- "diagnosis": string or null — the diagnosis exactly as stated, or null if not mentioned
"""
}


def build_chat_prompt(text: str, task: str = "extraction", system_prompt: str = None) -> List[Dict[str, str]]:
    """
    Builds a prompt. 'task' can be a key in TEMPLATES or a custom template string.
    'system_prompt' overrides the default SYSTEM_PROMPT if provided.
    """
    # 1. Try to get a template from the dictionary
    # 2. If it's not in the dictionary, use the 'task' string itself as the template
    user_template = TEMPLATES.get(task, task)
    
    # 3. If 'task' was none or empty, fall back to default extraction
    if not user_template:
        user_template = TEMPLATES["extraction"]
    user_template = user_template.replace("{text}", text)

    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user_template.format(text=text)}
    ]