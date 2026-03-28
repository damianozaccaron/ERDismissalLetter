import torch, time
import numpy as np

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    CROSS_ENCODER,
    RETRIEVAL_K,
    MMR_J,
    FINAL_N,
    MMR_LAMBDA,
    TEMPERATURE,
    QUANT,
    REPO,
    MODEL_QUANT)

from storage.storage import load_index_and_metadata, load_vectorizer
from storage.embedding import embed_query, load_embedder
from output.retrieval import mmr_select, retrieve_top_k, load_crossEncoder, reranking_multi_query

from output.output_prod import (
    load_model_quant,
    load_model_transformer,
    generate_letter_quant,
    generate_letter_transformer
)

from output.build_structure import (
    build_prompt,
    build_queries_ner,
    extract_entities,
    load_ner_model,
    collect_patient_input,
    deepl_translation,
)

"""
def main():

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
"""

def check_retrieval():
    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    # raw_data = collect_patient_input()
    with open("AFexample.txt", "r", encoding="utf-8") as f:
        raw_data = f.read()
    print("Translating clinical note...")

    """ translated_note = deepl_translation(raw_data)

    with open('translationAF.txt', 'w', encoding='utf-8') as file:
        file.write(translated_note)"""

    with open("translationAF.txt", "r", encoding="utf-8") as file:
        translated_note = file.read()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NER pipeline...")
    ner_model = load_ner_model(device=device)


    print("Extracting entities...")
    entities = extract_entities(translated_note, ner_model)

    print("Building queries...")
    vectorizer = load_vectorizer()
    queries = build_queries_ner(translated_note, ner_model, vectorizer)

    with open("queries.txt", "w", encoding="utf-8") as file:
        for i, q in enumerate(queries, 1):
            file.write(f"--- Query {i} ---\n{q}\n\n")

    print("Loading embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)

    print("Retrieving relevant chunks...")
    all_candidates = {}   # faiss_id -> chunk dict
    query_embeddings = []

    # top-k
    for i, q in enumerate(queries, 1):
        q_emb = embed_query(q, embedder=emb_model)
        query_embeddings.append(q_emb)

        candidates, faiss_ids = retrieve_top_k(
            query_embedding=q_emb,
            index=index,
            metadata=metadata,
            k=RETRIEVAL_K
        )

        new_count = 0
        for cand, fid in zip(candidates, faiss_ids):
            if fid not in all_candidates:
                all_candidates[fid] = cand
                new_count += 1

        print(f"Q{i}: retrieved {len(candidates)}, {new_count} new unique chunks")

    top_k_candidates = list(all_candidates.values())
    faiss_ids = list(all_candidates.keys())
    print(f"Total unique candidates after merge: {len(top_k_candidates)}")

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved from top_k retrieval")

    """
    avg_embedding = np.mean(query_embeddings, axis=0)
    selected_chunks = mmr_select(
        query_embedding=avg_embedding,
        candidates=top_k_candidates,
        faiss_ids=faiss_ids,
        index=index,
        top_j=MMR_J,
        lambda_=MMR_LAMBDA
    )
    if len(selected_chunks) == 0:
        raise ValueError("No candidates retrieved from MMR")"""

    print("Loading cross-encoder...")
    cross_encoder = load_crossEncoder(CROSS_ENCODER)

    print("Reranking...")
    start = time.time()

    final_chunks = reranking_multi_query(queries, top_k_candidates, cross_encoder, top_n=FINAL_N)

    end = time.time()

    print(f"Total time for re-reranking: {end-start:.3f} seconds")

    if len(final_chunks) == 0:
        raise ValueError("No candidates retrieved from Cross-Encoder")

    print("Building prompt...")
    prompt = build_prompt(translated_note, final_chunks)

    return prompt, queries

if __name__ == "__main__":
    output, queries = check_retrieval()

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(output)
        file.write("\n\n\nQUERIES:\n")
        for i, q in enumerate(queries, 1):
                file.write(f"Q{i}: {q}\n")