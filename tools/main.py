import torch
import numpy as np

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    FINAL_J,
    MMR_LAMBDA,
    TEMPERATURE,
    QUANT,
    REPO,
    MODEL_QUANT)

from retrieval.storage import load_index_and_metadata
from retrieval.embedding import embed_query, load_embedder
from retrieval.retrieval import mmr_select, retrieve_top_k

from output.output_prod import (
    load_model_quant,
    load_model_transformer,
    build_prompt,
    generate_letter_quant,
    generate_letter_transformer
)

def collect_patient_input():
    """
    For first test: hardcoded example.
    Later: CLI or structured input.
    """
    return {
        "Name": "Ugo Bianchetti",
        "Age": "57",
        "Gender": "Male",
        "Relevant Medical History": (
        "Hypertension (on ACE inhibitors), Type 2 Diabetes Mellitus, "
        "Hyperlipidemia, previous transient ischemic attack (2019), "
        "former smoker (quit 10 years ago)"
        ),
        "Exams Performed": (
        "ECG: irregularly irregular rhythm, no P waves, ventricular rate ~110 bpm; "
        "Blood Panel: elevated fasting glucose (145 mg/dL), HbA1c 7.5%, normal electrolytes; "
        "Cardiac enzymes: within normal limits; "
        "Echocardiogram: mild left atrial enlargement, preserved ejection fraction (55%); "
        "Blood pressure at admission: 150/95 mmHg"
        ),
        "Diagnosis": (
        "Symptomatic atrial fibrillation with rapid ventricular response, "
        "likely non-valvular, in a patient with multiple cardiovascular risk factors "
        "(hypertension, diabetes, prior TIA)"
        ),
    }


def main():

    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    print("Collecting patient input...")
    patient_data = collect_patient_input()

    query_text = " ".join([
        patient_data["Diagnosis"],
        patient_data["Relevant Medical History"]
    ])

    print("Loading Embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)

    print("Embedding query...")
    query_embedding = embed_query(query_text, embedder=emb_model)

    print("Retrieving relevant chunks...")
    top_k_candidates, faiss_ids = retrieve_top_k(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        k=RETRIEVAL_K
    )

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved")

    selected_chunks = mmr_select(
        query_embedding=query_embedding,
        candidates=top_k_candidates,
        faiss_ids=faiss_ids,
        index=index,
        top_j=FINAL_J,
        lambda_=MMR_LAMBDA
    )

    if len(selected_chunks) == 0:
        raise ValueError("No chunks retrieved")

    print("Building prompt...")
    prompt = build_prompt(patient_data, selected_chunks)

    print(prompt)

    if QUANT:
        print("Loading LLM...")
        model = load_model_quant(repo=REPO, model_name=MODEL_QUANT)

        print("Generating letter...")
        letter = generate_letter_quant(
            prompt=prompt,
            llm=model,
            temperature=TEMPERATURE)
        
    else:
        print("Loading LLM...")
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_model_transformer(MODEL_NAME, device)

        print("Generating dismissal letter...\n")
        letter = generate_letter_transformer(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            temperature=TEMPERATURE,
        )

    print("=== GENERATED LETTER ===\n")
    print(letter)

def check_retrieval():

    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    print("Collecting patient input...")
    patient_data = collect_patient_input()

    query_text = " ".join([
        patient_data["Diagnosis"],
        patient_data["Relevant Medical History"]
    ])

    print("Loading Embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)

    print("Embedding query...")
    query_embedding = embed_query(query_text, embedder=emb_model)

    print("Retrieving relevant chunks...")
    top_k_candidates, faiss_ids = retrieve_top_k(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        k=RETRIEVAL_K
    )

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved")

    selected_chunks = mmr_select(
        query_embedding=query_embedding,
        candidates=top_k_candidates,
        faiss_ids=faiss_ids,
        index=index,
        top_j=FINAL_J,
        lambda_=MMR_LAMBDA
    )

    if len(selected_chunks) == 0:
        raise ValueError("No chunks retrieved")

    print("Building prompt...")
    prompt = build_prompt(patient_data, selected_chunks)

    print(prompt)


if __name__ == "__main__":
    check_retrieval()