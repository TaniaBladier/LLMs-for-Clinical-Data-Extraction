from typing import List, Dict, Union

SYSTEM_PROMPT = """You are a medical information extraction assistant.
Given a medical text, extract structured information and return it as JSON only,
with no preamble or explanation.
"""


TEMPLATES = {
    "extraction": """Extract the following fields from the medical text below.
If a field is not mentioned, use null.

Fields to extract:
- circumstances: the cause of the incident which lead to the ER visit
- symptoms: list of symptoms reported by the patient
- diagnosis: the diagnosis if mentioned
- medications: list of medications mentioned
- patient_age: age of the patient if mentioned
- patient_gender: gender of the patient if mentioned

Return only a JSON object with these fields.

Medical text:
{text}
""",

    "normalize": """Extract clinical information and normalize it using standard medical terminology.

- Map symptoms and diagnoses to standardized terms (e.g., SNOMED-like labels if possible).
- Normalize medications to their generic names.
- Normalize age to integer if possible.

Return JSON with:
- symptoms_normalized
- diagnosis_normalized
- medications_normalized

Medical text:
{text}
""",

    "timeline": """Extract a timeline of events from the medical text.

Return a JSON list of events in chronological order:
- event
- time_expression (exact or relative, e.g., "2 days ago")

Medical text:
{text}
""",

    "cause": """Identify the precipitating event that led to the medical emergency described in the text.

A precipitating event is a concrete external incident such as: a fall, a car accident, 
domestic violence, a suicide attempt, a sports injury, an overdose, a workplace accident, etc.

Do NOT return a diagnosis or symptom as the event (e.g. "myocardial infarction", "chest pain" are NOT events).
If no such external incident is mentioned, return null.

Example of valid output when event is known:
{{"event": "fall from ladder", "confidence": "high", "explanation": "The patient fell from a ladder."}}

Example of valid output when event is unknown:
{{"event": null, "confidence": "unknown", "explanation": "No precipitating event is clearly stated in the text."}}

Return ONLY a valid JSON object with exactly these fields:
- "event": the precipitating external incident in a few words, or null if not mentioned
- "confidence": "high" / "medium" / "low" / "unknown"
- "explanation": one sentence justifying your answer

Medical text:
{text}
""",

     "negation": """Extract symptoms and indicate whether they are:
- present
- absent (negated)
- uncertain

Return JSON list:
- symptom
- status (present / absent / uncertain)

Medical text:
{text}
""", 


    "summary": """Summarize the medical text for a physician.

- Use precise clinical language
- Be concise
- Include key findings only

Medical text:
{text}
""",


    "urgency": """Classify the urgency level of this case.

Return JSON:
- triage_level: (low / moderate / high / emergency)
- justification: short explanation

Medical text:
{text}
""",


    "inconsistency": """Detect inconsistencies or contradictions in the medical text.

Return JSON list:
- statement_1
- statement_2
- explanation

Medical text:
{text}
""",


    "missing": """Identify missing but clinically relevant information.

Return JSON list of missing fields (e.g., allergies, medications, history).

Medical text:
{text}
""",


    "relations": """Extract relationships between entities.

Return JSON list:
- subject
- relation (e.g., "has_symptom", "treated_with")
- object

Medical text:
{text}
""",


    "noisy": """Extract key medical information despite possible noise, typos, or incomplete sentences.

Return JSON:
- symptoms
- diagnosis
- medications

Medical text:
{text}
""",

    
    "counterfactual": """Based on the medical text, answer what would likely change 
if the patient had NOT received the mentioned treatment.

Return ONLY a JSON object with exactly these fields:
- treatment_received: the treatment mentioned in the text
- likely_outcome_without_treatment: short clinical explanation
- severity_increase: yes / no / uncertain

Medical text:
{text}
""", 


    "sensitivity": """Extract diagnosis and symptoms.

        Be precise: small wording differences matter.
        
        Return JSON:
        - symptoms
        - diagnosis
        
        Medical text:
        {text}
        """ 
}




def build_chat_prompt(text: str, task: str = "extraction") -> List[Dict[str, str]]:
    """
    The master function to build chat-templated prompts.
    
    Args:
        text: The raw medical text.
        task: The key from the TEMPLATES dictionary.
    """
    user_instruction = TEMPLATES.get(task, TEMPLATES["extraction"])
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction.format(text=text)}
    ]