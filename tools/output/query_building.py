import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from storage.storage import load_vectorizer

def collect_patient_input(raw_text: str) -> dict:
    return {"clinical_note": raw_text}


def extract_patient_fields(translated_note: str) -> dict:

    fields = {}

    # demographics (everything before comma is gender, the first digits after the comma are the age)
    demo_match = re.match(r'^(.+?),\s*(\d+)\s*(?:years?\s*old|AA)?', translated_note, re.IGNORECASE)
    if demo_match:
        fields["gender"] = demo_match.group(1).strip()
        fields["age"] = demo_match.group(2).strip()

    demo_match = re.match(r'^(.+?),\s*(\d+)\s*(?:months)?', translated_note, re.IGNORECASE)
    if demo_match:
        fields["neonatal"] = True

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


# NER categories
ENTITY_TYPE_ANCHORS = {
    "DISEASE_DISORDER":      "diagnosis comorbidities risk factors management",
    "MEDICATION":            "therapy recommendations pharmacological management",
    "SIGN_SYMPTOM":          "clinical presentation symptoms assessment",
    "DIAGNOSTIC_PROCEDURE":  "diagnostic evaluation assessment workup",
}

# Clinically high-impact terms that should ALWAYS be included in queries
MANDATORY_TERMS = {
    "pregnant", "pregnancy", "renal", "hepatic", "dialysis", "transplant",
    "immunosuppressed", "immunocompromised", "hiv", "palliative", "terminal", "cancer", "malignancy", "anaphylaxis", "anaphylactic",
    "breastfeeding", "lactation", "CHA2DS2-VASc"
}

def load_ner_model(model_name = "Clinical-AI-Apollo/Medical-NER", device="cuda"):

    ner_model = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)
    return ner_model
 
 
def extract_entities(text: str, ner_model) -> dict[str, list[str]]:
    """
    Run NER on text and return entities grouped by type.
    Only keeps entity types listed in ENTITY_TYPE_ANCHORS.
    """ 
    results = ner_model(text)
 
    entity_group = set([ent["entity_group"] for ent in results if ent["entity_group"] in ENTITY_TYPE_ANCHORS])
    entities = {}
    
    # eliminate duplicates
    for entity in list(entity_group):
        seen = set()
        deduped = []
        for ent in results:
            if ent["entity_group"] != entity:
                continue
            word = ent["word"].strip()
            word_key = word.lower()
            if word_key not in seen:
                deduped.append(word)
                seen.add(word_key)
        entities[entity] = deduped
    
    return entities


def find_mandatory_terms(text: str) -> list[str]:
    text_lower = text.lower()
    return [term for term in MANDATORY_TERMS if term in text_lower]


# TF-IDF to summarize NER further
def score_entity(
    entity: str,
    vectorizer: TfidfVectorizer
) -> list[str]:
    """
    Computes TF-IDF scores for each word inside an element of the entities
    Words not in the vocabulary are dropped.
    """

    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    tokens = re.findall(r'[a-zA-Z0-9µ²³]+(?:[-/][a-zA-Z0-9µ²³]+)*', entity.lower())

    max_score = 0.0
    for token in tokens:
        # Keep only the max score for elements with 2+ words.
        if token in vocab:
            idx = vocab[token]
            max_score = max(max_score, idf[idx])

    return max_score

def rank_entities(
    entities: list[str],
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> list[str]:
    """
    Rank NER-extracted entities by max-token IDF score and return top-N.
    """
    scored = [(ent, score_entity(ent, vectorizer)) for ent in entities]

    # Sort by score descending; drop entities with score 0
    scored = [(ent, s) for ent, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [ent for ent, _ in scored[:top_n]]


def build_queries_ner(
    translated_note: str,
    ner_model,
    vectorizer: TfidfVectorizer
) -> list[str]:
    """
    NER + TF-IDF query decomposition.

    Pipeline per entity type:
      NER extraction -> max-token IDF scoring -> top-N -> query

    Produces one query per entity type (disease_disorder, medication,
    sign_symptom, diagnostic_procedure) plus a broad diagnosis query.
    """

    fields = extract_patient_fields(translated_note)
    diagnosis = fields.get("diagnosis", "")
    gender = fields.get("gender", "").lower()
    age = int(fields.get("age", ""))

    # Divide patient into relevant age brackets
    age_bracket = None
    if age <= 1 or fields["neonatal"]:
        age_bracket = "neonatal"
    elif age <= 14:
        age_bracket = "pediatric"
    if age >= 65:
        age_bracket = "elderly"
 
    # Detect mandatory clinical modifiers
    mandatory = find_mandatory_terms(translated_note)
    mandatory_str = " ".join(mandatory)
 
    # Run NER on the full note
    entities = extract_entities(translated_note, ner_model)
 
    queries = []
 
    for etype, anchor_terms in ENTITY_TYPE_ANCHORS.items():
        ent_list = entities.get(etype, [])
        if not ent_list:
            continue
 
        ranked = rank_entities(ent_list, vectorizer)
        if not ranked:
            # Fallback: if no entity survived IDF filtering use the raw entities (they may still help via embedding)
            ranked = ent_list

        entity_str = " ".join(ranked)

        # final query should be top entities + diagnosis + anchor + mandatory words
        parts = [entity_str, diagnosis, anchor_terms]
        if mandatory_str:
            parts.append(mandatory_str)

        query = ". ".join(filter(None, parts))
        queries.append(query)
 
    # Always include a broad diagnosis-level query
    if age_bracket:
        broad = f"{diagnosis} recommendations discharge management {age_bracket} {gender}"
    else:
        broad = f"{diagnosis} recommendations discharge management {gender}"
    if mandatory_str:
        broad += f" {mandatory_str}"
    queries.append(broad)
 
    return queries


def build_query(translated_note: str) -> str:
    """
    Old function used to build the single query to be embedded. build_queries_ner is the better working version.
    """

    fields = extract_patient_fields(translated_note=translated_note)

    return(
        f"Diagnosis: {fields['diagnosis']}. "
        f"Diagnosis: {fields['diagnosis']}. "
        f"Past medical history: {fields['remote_medical_history']}. "
        "therapy recommendations hospitalization discharge"
    )

if __name__ == "__main__":
    import torch

    with open("translationAF.txt", "r", encoding="utf-8") as file:
        translated_text = file.read()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NER pipeline...")
    ner_model = load_ner_model(device=device)


    print("\nExtracting entities...")
    entities = extract_entities(translated_text, ner_model)
    for etype, ents in entities.items():
        print(f"  {etype}: {ents}")


    print("\nBuilding queries...")
    vectorizer = load_vectorizer()
    queries = build_queries_ner(translated_text, ner_model, vectorizer)


    with open("queries.txt", "w", encoding="utf-8") as file:
        for i, q in enumerate(queries, 1):
            file.write(f"--- Query {i} ---\n{q}\n\n")

    print(f"\nGenerated {len(queries)} queries")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q}")