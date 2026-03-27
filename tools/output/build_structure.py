import deepl
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def deepl_translation(text, auth_key = "3e8dba21-6f6c-4f49-9d7f-:fx"):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="IT", target_lang="EN-GB")

    return result.text


def collect_patient_input(raw_text: str) -> dict:
    """
    Accepts raw Italian clinical note as a single string.
    Returns a dict preserving the original structure for prompt building.
    """
    return {"clinical_note": raw_text}

import re

def preprocess_diagnosis(text: str) -> str:
    """
    Normalize all-caps diagnosis field before translation.
    Converts 'ATRIAL FIBRILLATION\nRAPID VENTRICULAR' to 
    'Atrial fibrillation with rapid ventricular response'
    """
    # join broken lines
    text = " ".join(text.strip().splitlines())
    # convert to title case
    return text.title()

def extract_patient_fields(translated_note: str) -> dict:

    fields = {}

    # demographics (everything before comma is gender, the first digits after the comma are the age)
    demo_match = re.match(r'^(.+?),\s*(\d+)\s*(?:years?\s*old|AA)?', translated_note, re.IGNORECASE)
    if demo_match:
        fields["gender"] = demo_match.group(1).strip()
        fields["age"] = demo_match.group(2).strip()

    # medical history
    recent_match = re.search(
        r'Pathological proximate[:\s]+(.*?)(?=Remote pathology)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # remote medical history
    recent_match = re.search(
        r'Remote pathology[:\s]+(.*?)(?=Objective examination)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["remote_medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # objective examination
    exam_match = re.search(
        r'Objective examination[:\s]+(.*?)(?=Clinical Diary)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["exams"] = exam_match.group(1).strip() if exam_match else ""

    # clinical diary
    diary_match = re.search(
        r'(?:Clinical record|Clinical diary)[:\s]+(.*?)(?=Diagnosis)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["clinical_diary"] = diary_match.group(1).strip() if diary_match else ""

    # diagnosis
    diag_match = re.search(r'Diagnosis[:\s]+(.+?)$', translated_note, re.IGNORECASE | re.DOTALL)
    fields["diagnosis"] = diag_match.group(1).strip() if diag_match else ""

    return fields

# NER management
ENTITY_TYPE_ANCHORS = {
    "problem":   "diagnosis comorbidities risk factors management",
    "treatment": "therapy recommendations pharmacological management",
    "test":      "diagnostic evaluation assessment workup",
}

# Clinically high-impact terms that should ALWAYS be included in queries
MANDATORY_TERMS = {
    "pregnant", "pregnancy", "paediatric", "pediatric", "neonatal",
    "renal", "hepatic", "dialysis", "transplant",
    "immunosuppressed", "immunocompromised", "hiv",
    "palliative", "terminal", "cancer", "malignancy",
    "allergic", "anaphylaxis", "anaphylactic",
    "breastfeeding", "lactation",
}

def load_ner_pipeline(model_name: str = "samrawal/bert-base-uncased_clinical-ner", device: int = -1):

    ner_pipe = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)
    return ner_pipe
 
 
def extract_entities(text: str, ner_pipe) -> dict[str, list[str]]:
   
    results = ner_pipe(text)
 
    entities: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
 
    for ent in results:
        etype = ent["entity_group"].lower()
        # Clean up the entity text: strip whitespace, lowercase for dedup
        word = ent["word"].strip()
        word_key = word.lower()
 
        if etype not in entities:
            entities[etype] = []
            seen[etype] = set()
 
        if word_key not in seen[etype]:
            entities[etype].append(word)
            seen[etype].add(word_key)
 
    return entities
 
def find_mandatory_terms(text: str) -> list[str]:
    text_lower = text.lower()
    return [term for term in MANDATORY_TERMS if term in text_lower]


# TF-IDF to summarize NER further
def tfidf_keywords(
    text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 8,
) -> list[str]:
    """
    Extract the top-N TF-IDF keywords from a clinical note section.
    TF is computed from the section text; IDF comes from the guideline corpus (learned at fit time).
    """

    tfidf_vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_vector.toarray().flatten()
 
    # Get indices sorted by descending score
    top_indices = scores.argsort()[::-1]
 
    keywords = []
    for idx in top_indices:
        # if not enough words have a positive score, simply send it as it is
        if scores[idx] == 0:
            break
        keywords.append(feature_names[idx])
        if len(keywords) >= top_n:
            break

    if len(keywords) == 0:
        raise ValueError("TF-IDF returned no results")
 
    return keywords
 
 
def find_mandatory_terms(text: str) -> list[str]:
    """
    Scan text for clinically high-impact terms that must always
    appear in queries regardless of TF-IDF ranking.
    """
    text_lower = text.lower()
    return [term for term in MANDATORY_TERMS if term in text_lower]

 
# Guideline-oriented anchor terms per section.
SECTION_ANCHORS = {
    "remote_medical_history": "patient history comorbidities risk factors",
    "medical_history":        "presenting complaint acute symptoms assessment",
    "exams":                  "clinical findings examination assessment",
    "clinical_diary":         "treatment management therapy intervention",
}
def build_queries(translated_note: str, vectorizer: TfidfVectorizer, top_n_terms: int = 8) -> list[str]:

    fields = extract_patient_fields(translated_note)
    diagnosis = fields["diagnosis"]
 
    # Detect mandatory clinical modifiers from the full note
    mandatory = find_mandatory_terms(translated_note)
    mandatory_str = " ".join(mandatory)
 
    queries = []
 
    for section_key, anchor_terms in SECTION_ANCHORS.items():
        # separate clinical document into distinct sections (and get the sections)
        section_text = fields[section_key]
        if not section_text:
            continue
        
        # get the TF-IDF results for each section
        keywords = tfidf_keywords(section_text, vectorizer, top_n=top_n_terms)
        if not keywords:
            continue
        
        # create the queries by joining the diagnosis (always present), the TF-IDF results for each section, the anchor instructions and if present the mandatory terms
        parts = [diagnosis, " ".join(keywords), anchor_terms]
        if mandatory_str:
            parts.append(mandatory_str)
 
        query = ". ".join(parts)
        queries.append(query)
 
    # Always include a broad diagnosis-level query
    broad = f"{diagnosis} recommendations discharge management"
    if mandatory_str:
        broad += f" {mandatory_str}"
    queries.append(broad)
 
    return queries


def build_query(translated_note: str) -> str:

    fields = extract_patient_fields(translated_note=translated_note)

    return(
        f"Diagnosis: {fields['diagnosis']}. "
        f"Diagnosis: {fields['diagnosis']}. "
        f"Past medical history: {fields['remote_medical_history']}. "
        "therapy recommendations hospitalization discharge"
    )


def build_prompt(patient_data: str, retrieved_chunks: list[dict]) -> str:
    prompt = []

    prompt.append(
        "You are a medical doctor writing a discharge document. "
        "The document is addressed to the patient. "
        "Follow instructions strictly. Write specific, patient-tailored recommendations."
        "Avoid generic statements.\n"
    )

    prompt.append("CLINICAL NOTE:\n")
    prompt.append(patient_data)
    prompt.append("\n")

    prompt.append("GUIDELINE EXCERPTS:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        prompt.append(
            f"[E{i}] "
            f"({chunk['doc_id']}, p.{chunk['page_start']}-{chunk['page_end']})\n"
        )
        prompt.append(chunk["text"])
        prompt.append("\n")

    prompt.append(
        "INSTRUCTIONS:\n"
        "- Use ONLY the provided guideline excerpts. Do not rely on prior knowledge.\n"
        "- Do NOT invent or assume any information.\n"
        "- If a recommendation cannot be supported by the excerpts, state that no guidance is available.\n"
        "- Each recommendation MUST explicitly cite at least one excerpt.\n"
        "- Structure the output EXACTLY as follows:\n\n"
        "Medical history:\n"
        "...(Copy medical history from clinical note)\n\n"
        "Objective examination:\n"
        "...(Copy objective examination from clinical note)\n\n"
        "Clinical Diary:\n"
        "...(Copy clinical diary from clinical note)\n\n"
        "Prognosis:\n"
        "...(Decide based on patient clinical data and the excerpts)\n\n"
        "Recommendations and Prescriptions:\n"
        "...(Might contain pharmacological prescriptions, future exams, behavioral recommendations, hospitalization or all of them, base this part on the provided excerpts)\n"
    )

    return "\n".join(prompt)

if __name__ == "__main__":
    with open("AFexample.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    # translated_text = deepl_translation(raw)

    with open('translationAF.txt', 'r', encoding='utf-8') as file:
        translated_text = file.read()
    
    output = build_queries(translated_text)

    with open('query.txt', 'w', encoding='utf-8') as file:
        file.write(output)