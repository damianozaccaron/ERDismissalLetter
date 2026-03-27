import torch
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

from storage.storage import load_index_and_metadata
from storage.embedding import embed_query, load_embedder
from output.retrieval import mmr_select, retrieve_top_k, load_crossEncoder, reranking

from output.build_structure import (
    build_prompt,
    build_query,
    build_queries,
    build_vectorizer,
    collect_patient_input,
    deepl_translation,
)

from output.output_prod import (
    load_model_quant,
    load_model_transformer,
    generate_letter_quant,
    generate_letter_transformer
)


def check_retrieval():
    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    with open("AFexample.txt", "r", encoding="utf-8") as f:
        raw_data = f.read()
    print("Translating clinical note...")

    with open("translationAF.txt", "r", encoding="utf-8") as f:
        translated_note = f.read()

    patient_data = {
        "clinical_note": raw_data,
        "clinical_note_translated": translated_note
    }

    # ── Fit TF-IDF vectorizer on guideline corpus ──
    print("Fitting TF-IDF vectorizer on guideline corpus...")
    vectorizer = build_vectorizer(metadata)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} terms")

    # ── Build decomposed queries ──
    queries = build_queries(translated_note, vectorizer, top_n_terms=8)
    print(f"Generated {len(queries)} sub-queries:")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q[:100]}{'...' if len(q) > 100 else ''}")

    # ── Load embedder ──
    print("Loading embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)

    # ── Multi-query retrieval with deduplication ──
    print("Running multi-query retrieval...")
    all_candidates = {}   # faiss_id -> chunk dict
    query_embeddings = []

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

        print(f"  Q{i}: retrieved {len(candidates)}, {new_count} new unique chunks")

    merged_candidates = list(all_candidates.values())
    merged_faiss_ids = list(all_candidates.keys())
    print(f"Total unique candidates after merge: {len(merged_candidates)}")

    if len(merged_candidates) == 0:
        raise ValueError("No candidates retrieved from any sub-query")

    # ── MMR on merged set (use mean query embedding) ──
    print("Running MMR on merged candidates...")
    avg_embedding = np.mean(query_embeddings, axis=0)

    selected_chunks = mmr_select(
        query_embedding=avg_embedding,
        candidates=merged_candidates,
        faiss_ids=merged_faiss_ids,
        index=index,
        top_j=MMR_J,
        lambda_=MMR_LAMBDA
    )

    if len(selected_chunks) == 0:
        raise ValueError("No candidates survived MMR selection")

    print(f"MMR selected {len(selected_chunks)} chunks")

    # ── Reranking (concatenate sub-queries for cross-encoder) ──
    print("Reranking with cross-encoder...")
    combined_query = " | ".join(queries)

    cross_encoder = load_crossEncoder(CROSS_ENCODER)
    final_chunks = reranking(
        query=combined_query,
        retrieved_chunks=selected_chunks,
        reranker=cross_encoder,
        top_n=FINAL_N
    )

    if len(final_chunks) == 0:
        raise ValueError("No candidates survived cross-encoder reranking")

    print(f"Final chunks after reranking: {len(final_chunks)}")

    # ── Build prompt ──
    print("Building prompt...")
    prompt = build_prompt(translated_note, final_chunks)

    return prompt, queries


def check_retrieval_single():
    """
    Original single-query retrieval (for A/B comparison).
    """
    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()

    with open("translationAF.txt", "r", encoding="utf-8") as f:
        translated_note = f.read()

    query = build_query(translated_note)

    print("Loading embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)

    print("Embedding query...")
    query_embedding = embed_query(query, embedder=emb_model)

    print("Retrieving relevant chunks...")
    top_k_candidates, faiss_ids = retrieve_top_k(
        query_embedding=query_embedding,
        index=index,
        metadata=metadata,
        k=RETRIEVAL_K
    )

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved from top_k retrieval")

    selected_chunks = mmr_select(
        query_embedding=query_embedding,
        candidates=top_k_candidates,
        faiss_ids=faiss_ids,
        index=index,
        top_j=MMR_J,
        lambda_=MMR_LAMBDA
    )

    if len(selected_chunks) == 0:
        raise ValueError("No candidates retrieved from MMR")

    cross_encoder = load_crossEncoder(CROSS_ENCODER)
    final_chunks = reranking(
        query=query,
        retrieved_chunks=selected_chunks,
        reranker=cross_encoder,
        top_n=FINAL_N
    )

    if len(final_chunks) == 0:
        raise ValueError("No candidates retrieved from Cross-Encoder")

    print("Building prompt...")
    prompt = build_prompt(translated_note, final_chunks)

    return prompt, query


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "multi"

    if mode == "single":
        print("=== SINGLE-QUERY MODE (baseline) ===\n")
        output, query = check_retrieval_single()
        with open("output_single.txt", "w", encoding="utf-8") as file:
            file.write(output)
            file.write("\n\n\nQUERY:\n")
            file.write(query)
        print("Output written to output_single.txt")

    else:
        print("=== MULTI-QUERY MODE (decomposed + TF-IDF) ===\n")
        output, queries = check_retrieval()
        with open("output_multi.txt", "w", encoding="utf-8") as file:
            file.write(output)
            file.write("\n\n\nQUERIES:\n")
            for i, q in enumerate(queries, 1):
                file.write(f"Q{i}: {q}\n")
        print("Output written to output_multi.txt")
