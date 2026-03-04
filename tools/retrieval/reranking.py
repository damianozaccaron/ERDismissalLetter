from sentence_transformers import CrossEncoder

# Load a pretrained CrossEncoder model (aka the reranker)
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def reranking(query, retrieved_chunks, reranker, top_n=5):
    ranks = reranker.rank(query,
                         [chunk for chunk, sim in retrieved_chunks],
                         return_documents=True)
    return ranks[:top_n]