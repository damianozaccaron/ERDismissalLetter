# emb_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1") <- to use for final model

# emb_model = SentenceTransformer("lokeshch19/ModernPubMedBERT") <- to try in evaluation phase

#BioCLinicalBERT <- to try in evaluation phase
"""from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
"""
from sentence_transformers import SentenceTransformer
import torch
from config import EMBEDDING_MODEL

def load_embedder(model_name=EMBEDDING_MODEL):
    """Load the sentence transformer model for embedding, returns the model instance."""
    
    emb_model = SentenceTransformer(model_name,
        device="cuda" if torch.cuda.is_available() else "cpu")
    return emb_model

def embed_docs(chunks, embedder, batch_size=32):
    texts = [c["text"] for c in chunks]
    """Embed a list of text chunks using the provided embedder, takes the parameter text from the chunk (dict type) and returns 
    a list of dicts with the same metadata and the embedding vector added as a new key "embedding".
    """

    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks

def embed_query(query, embedder):
    """Embed a query string using the provided embedder, returns a normalized embedding vector."""
    embedding = embedder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.reshape(1, -1) # reshape to (1, dim) for FAISS search