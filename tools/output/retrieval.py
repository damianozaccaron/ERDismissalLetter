import numpy as np
from sentence_transformers import CrossEncoder

def retrieve_top_k(query_embedding, index, metadata, k=30):
    scores, indices = index.search(query_embedding, k)

    results = []
    faiss_ids = []

    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue

        item = metadata[idx].copy()
        item["score"] = float(score)

        results.append(item)
        faiss_ids.append(idx)

    return results, faiss_ids

def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict],
    faiss_ids: list[int],
    index,
    top_j=15,
    lambda_=0.5):
    """
    Reranks candidates using Maximal Marginal Relevance (MMR) to balance relevance and diversity.
    Deprecated but could potentially be re-implemented.
    """

    candidate_embeddings = np.stack([index.reconstruct(int(i)) for i in faiss_ids])

    selected = []
    selected_indices = [] # we also keep the indices

    # similarity(query, candidate)
    sim_to_query = candidate_embeddings @ query_embedding.T  # matrix multiplication with result dimensions (k, dim)x(dim,1) -> (k, 1)
    sim_to_query = sim_to_query.flatten() # a list of similarity scores between the query and each candidate, with shape (k,)

    for _ in range(top_j):
        mmr_scores = []

        for i in range(len(candidates)):
            # if the candidate is already selected, we set its MMR score to -inf to exclude it from selection
            if i in selected_indices:
                mmr_scores.append(-np.inf)
                continue
  
            if not selected_indices:
                diversity_penalty = 0
            else:
                # max similarity to already selected chunks
                sim_to_selected = candidate_embeddings[i] @ candidate_embeddings[selected_indices].T
                diversity_penalty = np.max(sim_to_selected)

            mmr_score = (
                lambda_ * sim_to_query[i]
                - (1 - lambda_) * diversity_penalty
            )
            mmr_scores.append(mmr_score)

        best_idx = int(np.argmax(mmr_scores))
        selected_indices.append(best_idx)
        selected.append(candidates[best_idx])

    return selected


def load_crossEncoder(model_name: str = "BAAI/bge-reranker-large"):
    return CrossEncoder(model_name)

def reranking(query: str, retrieved_chunks: list[dict], reranker: CrossEncoder, top_n: int = 6) -> list[dict]:
    texts = [chunk["text"] for chunk in retrieved_chunks]
    ranks = reranker.rank(query, texts, return_documents=True)
    
    # map ranked results back to original chunk dicts to preserve metadata (corpus_id maintains original order)
    ranked_chunks = []
    for rank in ranks[:top_n]:
        original_chunk = retrieved_chunks[rank["corpus_id"]]
        original_chunk["rerank_score"] = rank["score"]
        # chunks already automatically sorted by highest score
        ranked_chunks.append(original_chunk)
    
    return ranked_chunks


def _text_overlap(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Check if two chunk texts are near-duplicates using token overlap.
    Uses set intersection over the smaller set to detect when one
    chunk is largely contained within another.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())

    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))

    return (intersection / smaller) >= threshold


def remove_duplicates(chunks: list[dict]) -> list[dict]:
    """
    Remove near-duplicate chunks, keeping the higher-scored one.
    Assumes chunks are already sorted by score descending.
    """
    kept = []
    for chunk in chunks:
        is_dup = False
        for existing in kept:
            if _text_overlap(chunk["text"], existing["text"]):
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
    return kept


def reranking_multi_query(
    queries: list[str],
    retrieved_chunks: list[dict],
    reranker: CrossEncoder,
    top_n: int = 8,
    category_boost: float = 0.1
) -> list[dict]:
    """
    Rerank chunks by scoring each chunk against each query independently,
    then aggregating with max. Applies category leader boost and text deduplication.

    Pipeline:
      1. Score every (query, chunk) pair with the cross-encoder.
      2. For each chunk, take the MAX score across all queries.
      3. Identify the top-scoring chunk per query (category leader) and apply a percentage boost to its max score.
      4. Sort by final score, deduplicate near-identical chunks.
      5. Return top-N.
    """
    texts = [chunk["text"] for chunk in retrieved_chunks]
    n_chunks = len(texts)
    n_queries = len(queries)

    # Step 1
    all_pairs = []
    for q in queries:
        for t in texts:
            all_pairs.append((q, t))

    all_scores = reranker.predict(all_pairs) # Why not rank?

    # Reshape into (n_queries, n_chunks)
    score_matrix = np.array(all_scores).reshape(n_queries, n_chunks)

    # Step 2
    max_scores = score_matrix.max(axis=0)            # shape (n_chunks,)

    # Step 3
    # For each query, find the chunk that scored highest on that query
    category_leaders = set()
    for q_idx in range(n_queries):
        best_chunk_idx = int(score_matrix[q_idx].argmax())
        category_leaders.add(best_chunk_idx)

    boosted_scores = max_scores.copy()
    for chunk_idx in category_leaders:
        boosted_scores[chunk_idx] *= (1.0 + category_boost)

    # Step 4
    for i, chunk in enumerate(retrieved_chunks):
        chunk["final_score"] = float(boosted_scores[i])

    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda c: c["final_score"],
        reverse=True,
    )
    final = remove_duplicates(sorted_chunks)

    return final[:top_n]
