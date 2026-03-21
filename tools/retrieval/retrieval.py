import numpy as np
from config import RETRIEVAL_K, MMR_LAMBDA, FINAL_J

def retrieve_top_k(query_embedding, index, metadata, k=RETRIEVAL_K):
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
    top_j=FINAL_J,
    lambda_=MMR_LAMBDA
):
    """
    Reranks candidates using Maximal Marginal Relevance (MMR) to balance relevance and diversity.

    candidate_embeddings: shape (k, dim), normalized
    candidates: list of metadata dicts (same order)
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
