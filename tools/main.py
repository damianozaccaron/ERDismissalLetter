import torch
import numpy as np

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    FINAL_J,
    MMR_LAMBDA,
    TEMPERATURE)

from retrieval.storage import load_index_and_metadata
from retrieval.embedding import embed_query, load_embedder
from retrieval.retrieval import mmr_select, retrieve_top_k
from retrieval.storage import load_index_and_metadata

from output.output_prod import (
    load_model,
    build_prompt,
    generate_letter,
)

def collect_patient_input():
    """
    For first test: hardcoded example.
    Later: CLI or structured input.
    """
    return {
        "Age": 67,
        "Gender": "Male",
        "Relevant Medical History": "Hypertension, Type 2 Diabetes",
        "Exams Performed": "ECG, Blood Panel",
        "Diagnosis": "Atrial Fibrillation with migraine episodes",
    }


def main():

    print("Loading FAISS index and metadata...")
    start_time = torch.cuda.Event(enable_timing=True)
    index, metadata = load_index_and_metadata()
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    end_time.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded
    elapsed_time_ms = start_time.elapsed_time(end_time)
    print(f"Index and metadata loaded in {elapsed_time_ms:.2f} ms")

    print("Loading LLM...")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(MODEL_NAME, device)

    print("Collecting patient input...")
    patient_data = collect_patient_input()

    # Use diagnosis as query seed
    query_text = patient_data["Diagnosis"]

    print("Loading LLM...")
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

    # Take the indexes listed in faiss_ids and fetch the corresponding embeddings from the FAISS index using the reconstruct method, which retrieves the original embedding vector for a given index. We stack these embeddings into a single numpy array called candidate_embeddings, which will have the shape (k, dim) where k is the number of retrieved candidates and dim is the dimensionality of the embeddings.
    candidate_embeddings = np.stack([index.reconstruct(int(i)) for i in faiss_ids])

    selected_chunks = mmr_select(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        candidates=top_k_candidates,
        top_j=FINAL_J,
        lambda_=MMR_LAMBDA
    )

    print(selected_chunks)

    print("Building prompt...")
    prompt = build_prompt(patient_data, selected_chunks)

    print("Generating dismissal letter...\n")
    letter = generate_letter(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        temperature=TEMPERATURE,
    )

    print("=== GENERATED LETTER ===\n")
    print(letter)


if __name__ == "__main__":
    main()