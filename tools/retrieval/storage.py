import numpy as np
import faiss, pickle
from pathlib import Path

# ASK FOR CLARIFICATION OVER WHAT THIS IS DOING LINE BY LINE
def build_faiss_index(chunks):
    """ extract embeddings in the same order as chunks """

    embeddings = np.stack(
        [c["embedding"] for c in chunks]
    ).astype("float32") # extract embeddings from chunks (list of floats), turn them with np.stack to a 2D NumPy array and convert to float32 for faiss (which only accepts float32)

    dim = embeddings.shape[1] # get the dimensionality of the embeddings (number of columns in the 2D array)
    index = faiss.IndexFlatIP(dim) # create a FAISS index for inner product similarity search (which is equivalent to cosine similarity if the embeddings are normalized, as they are in our case)

    index.add(embeddings)

    return index

def build_metadata(chunks):
    metadata = [
        {
            "doc_id": c["doc_id"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "text": c["text"],
        }
        for c in chunks
    ]
    return metadata

def sanity_check(index, metadata):
    assert index.ntotal == len(metadata)

def save_index_and_metadata(index, metadata, index_path="faiss.index", metadata_path="metadata.pkl"):

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def load_index_and_metadata(index_path="faiss.index", metadata_path="metadata.pkl"):

    if not Path(index_path).exists() or not Path(metadata_path).exists():
        raise RuntimeError(
            "FAISS index or metadata not found."
        )
    
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    sanity_check(index, metadata)

    return index, metadata