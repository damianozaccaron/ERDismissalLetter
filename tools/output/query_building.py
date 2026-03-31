import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from storage.storage import load_vectorizer

def collect_patient_input() -> str:

    def prompt_required(prompt_text: str) -> str:
        value = ""
        while not value.strip():
            print(prompt_text)
            value = input()
            if not value.strip():
                print("Questo campo non può essere vuoto.")
        return value.strip()

    def prompt_optional(prompt_text: str) -> str:
        print(prompt_text)
        return input().strip()
    
    clinical_note = []

    print("Genere (M/F/A): ")
    gender = ""
    while gender.lower() not in ('m', 'f', 'a'):
        gender = input()
        if gender.lower() not in ('m', 'f', 'a'):
            print("Genere deve essere M (maschio), F (femmina) o A (altro): ")
    
    gender_exp = "Maschio"
    if gender.lower() == 'f':
        gender_exp = "Femmina"
    elif gender.lower() == 'a':
        gender_exp = "Altro"

    age = prompt_required("\n\nEtà: ")
    clinical_note.append(gender_exp+", "+age)

    clinical_note.append("\nAnamnesi: ")

    pat_prox = prompt_required("\n\nPatologica Prossima: ")
    clinical_note.append("- Patologia Prossima:\n" + pat_prox)

    pat_rem = prompt_optional("\n\nPatologica Remota (lasciare vuoto se non presente): ")
    if pat_rem:
        clinical_note.append("- Patologia Remota:\n" + pat_rem)

    exam = prompt_required("\n\nEsame Obiettivo: ")
    clinical_note.append("\nEsame Obiettivo:\n" + exam)
    
    diary = prompt_required("\n\nDiario Clinico: ")
    clinical_note.append("\nDiario Clinico: \n" + diary)

    diagnosis = prompt_required("\n\nDiagnosi: ")
    clinical_note.append("\nDiagnosi: "+diagnosis)

    prognosis = prompt_required("\n\nPrognosi: ")
    clinical_note.append("\nPrognosi: "+prognosis)


    clinical_note_str = "\n".join(clinical_note)

    return clinical_note_str


def extract_patient_fields(translated_note: str) -> dict:

    fields = {}

    # demographics (everything before comma is gender, the first digits after the comma are the age)
    import re

    demo_match = re.match(r'^[^,]+,\s*(?:(\d+)\s*(?:years?\s*old|aa))?\s*(?:(\d+)\s*months?)?', translated_note, re.IGNORECASE)


    gender = translated_note.split(",", 1)[0].strip()
    if gender.lower() == "other":
        gender = ""
    fields["gender"] = gender

    # age
    years = int(demo_match.group(1)) if demo_match.group(1) else None
    months = int(demo_match.group(2)) if demo_match.group(2) else None

    if years is not None:
        fields["age"] = years
    elif months is not None:
        fields["age"] = 0
    else:
        fields["age"] = None

    # neonatal
    if years is None and months is not None:
        fields["neonatal"] = True
    else: 
        fields["neonatal"] = False

    # medical history
    recent_match = re.search(
        r'Pathological proximate[:\s]+(.*?)(?=Remote pathology\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # remote medical history
    recent_match = re.search(
        r'Remote pathology[:\s]+(.*?)(?=Objective examination\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["remote_medical_history"] = recent_match.group(1).strip() if recent_match else ""

    # objective examination
    exam_match = re.search(
        r'Objective examination[:\s]+(.*?)(?=Clinical Diary\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["exams"] = exam_match.group(1).strip() if exam_match else ""

    # clinical diary
    diary_match = re.search(
        r'(?:Clinical record|Clinical diary)[:\s]+(.*?)(?=Diagnosis\s*:)',
        translated_note, re.IGNORECASE | re.DOTALL
    )
    fields["clinical_diary"] = diary_match.group(1).strip() if diary_match else ""

    # diagnosis
    diag_match = re.search(r'Diagnosis[:\s]+(.*?)(?=Prognosis\s*:)', translated_note, re.IGNORECASE | re.DOTALL)
    fields["diagnosis"] = diag_match.group(1).strip() if diag_match else ""

    # prognosis
    prog_match = re.search(r'Prognosis[:\s]+(.+?)$', translated_note, re.IGNORECASE | re.DOTALL)
    fields["prognosis"] = prog_match.group(1).strip() if prog_match else ""

    return fields


# NER categories
ENTITY_TYPE_ANCHORS = {
    "DISEASE_DISORDER":      "diagnosis comorbidities risk factors management",
    "MEDICATION":            "therapy recommendations pharmacological management clinical scores diagnostic evaluation",
   # "SIGN_SYMPTOM":          "clinical presentation symptoms assessment",
    "DIAGNOSTIC_PROCEDURE":  "diagnostic evaluation assessment workup",
}


def load_ner_model(model_name = "Clinical-AI-Apollo/Medical-NER", device="cuda"):

    ner_model = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)
    return ner_model


NEGATION_PATTERNS = re.compile(
    r'\b('
    r'no|not|none|neither|never|nor|'
    r'without|absent|absence of|'
    r'denies|denied|deny|'
    r'negative|'
    r'rule[sd]? out|exclude[sd]?'
    r')\b',
    re.IGNORECASE
)
def is_negated(text: str, entity_start: int, entity_end: int, window: int = 40) -> bool:
   
    raw_start = max(0, entity_start - window)
    preceding_raw = text[raw_start:entity_start]
 
    # Trim to the last clause boundary (period or comma)
    last_boundary = max(preceding_raw.rfind('.'), preceding_raw.rfind(','), preceding_raw.rfind(':'), preceding_raw.rfind(';'))
    if last_boundary != -1:
        preceding = preceding_raw[last_boundary + 1:]
    else:
        preceding = preceding_raw
 
    if NEGATION_PATTERNS.search(preceding):
        return True
 
    # Look-ahead: "negative(s)" after the entity
    post_start = entity_end
    post_end = min(len(text), entity_end + 15)
    following_raw = text[post_start:post_end]
 
    # Trim to the next clause boundary (period or comma)
    first_boundary = len(following_raw)
    for sep in ['.', ',', ':', ';']:
        pos = following_raw.find(sep)
        if pos != -1:
            first_boundary = min(first_boundary, pos)
    following = following_raw[:first_boundary]
 
    if re.search(r'\bnegatives?\b', following, re.IGNORECASE):
        # print(f"Negated {text[entity_start:entity_end]} because of: {following} (Following)")
        return True
 
    return False


def extract_entities(text: str, ner_model, filter_negated = True) -> dict[str, list[str]]:
    """
    Run NER on text and return entities grouped by type.
    Only keeps entity types listed in ENTITY_TYPE_ANCHORS.
    """
    results = ner_model(text)
 
    entity_group = set([ent["entity_group"] for ent in results if ent["entity_group"] in ENTITY_TYPE_ANCHORS])
    entities = {}
 
    for entity in list(entity_group):
        seen = set()
        deduped = []
        for ent in results:
            if ent["entity_group"] != entity:
                continue
 
            # Skip negated entities
            if filter_negated and is_negated(text, ent["start"], ent["end"]):
                continue
            
            # Eliminate duplicates
            word = ent["word"].strip()
            word_key = word.lower()
            if word_key not in seen:
                deduped.append(word)
                seen.add(word_key)
        entities[entity] = deduped

    return entities


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

    print(scored)

    return [ent for ent, _ in scored[:top_n]]

def normalize(text: str) -> str:
    SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹', '01234567890123456789')
    return text.translate(SUBSCRIPT_MAP)

def condense_diary(
    clinical_diary: str,
    diagnostic_entities: list[str],
    max_sentences: int = 2,
) -> str:
    """
    Extract sentences from the clinical diary that contain diagnostic
    procedure entities, preserving their clinical context (results, findings, values).
    """
    # Split diary into sentences on period, newline, or dash-prefixed lines
    sentences = re.split(r'(?<=[.])\s+|(?=\n\s*-)', clinical_diary)
    sentences = [s.strip().lstrip('- ') for s in sentences if s.strip()]

    entity_keys = [normalize(e).lower() for e in diagnostic_entities]

    matched = []
    seen = set()
    for sent in sentences:
        if sent in seen:
            continue
        sent_norm = normalize(sent).lower()
        for rank_idx, ent in enumerate(entity_keys):
            hit = bool(re.search(r'(?<![a-z0-9])' + re.escape(ent) + r'(?![a-z0-9])', sent_norm))
            if hit:
                matched.append((rank_idx, sent))
                seen.add(sent)
                break

    matched.sort(key=lambda x: x[0])
    matched = [sent for _, sent in matched[:max_sentences]]

    return " ".join(matched) if matched else ""

SALVAGE_ENTITY_TYPES = {
    "DIAGNOSTIC_PROCEDURE": 2
}
def build_queries_ner(
    translated_note: str,
    ner_model,
    vectorizer: TfidfVectorizer):
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
    exams = fields.get("exams", "")
    clinical_diary = fields.get("clinical_diary", "")

    # Divide patient into relevant age brackets
    age_bracket = None
    if age <= 1 or fields.get("neonatal"):
        age_bracket = "neonatal"
    elif age <= 14:
        age_bracket = "pediatric"
    if age >= 65:
        age_bracket = "elderly"
 
    # Run NER on the full note
    entities = extract_entities(translated_note, ner_model)

    salvaged_entities = []
    for etype, top_n in SALVAGE_ENTITY_TYPES.items():
        ent_list = entities.get(etype, [])
        if not ent_list:
            continue
        ranked = rank_entities(ent_list, vectorizer, top_n=top_n)
        salvaged_entities.extend(ranked)

    retrieval_queries = []
    reranking_queries = []

    for etype, anchor_terms in ENTITY_TYPE_ANCHORS.items():
        ent_list = entities.get(etype, [])

        if not ent_list and etype != "MEDICATION":
            continue
 
        ranked = rank_entities(ent_list, vectorizer) if ent_list else []

        # Append salvaged entities to MEDICATION query
        if etype == "MEDICATION" :
            ranked = ranked + salvaged_entities

        if etype == "DIAGNOSTIC_PROCEDURE":
            continue
            # use condensed diary instead of bare entity names            
            condensed = condense_diary(clinical_diary, ranked) if ent_list else ""
            if condensed:
                query = f"{condensed}. {diagnosis}. {anchor_terms}"
                # retrieval_queries.append(query)
                reranking_queries.append(query)
            continue

        if not ranked:
            # Fallback: if no entity survived IDF filtering use the raw entities
            ranked = ent_list

        entity_str = " ".join(ranked)

        # final query should be top entities + diagnosis + anchor
        parts = [entity_str, diagnosis, anchor_terms]

        query = ". ".join(filter(None, parts))
        retrieval_queries.append(query)
        reranking_queries.append(query)

    # Include raw Objective Examination query for Cross-Encoder only
    exam_query = f"{exams}. {diagnosis}. clinical presentation symptoms assessment"
    #reranking_queries.append(exam_query)
 
    # Always include a broad diagnosis-level query
    if age_bracket:
        broad = f"{diagnosis} recommendations discharge management {age_bracket} {gender}"
    else:
        broad = f"{diagnosis} recommendations discharge management {gender}"

    retrieval_queries.append(broad)
    reranking_queries.append(broad)
 
    return retrieval_queries, reranking_queries


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


    with open("results/queries.txt", "w", encoding="utf-8") as file:
        for i, q in enumerate(queries, 1):
            file.write(f"--- Query {i} ---\n{q}\n\n")

    print(f"\nGenerated {len(queries)} queries")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q}")