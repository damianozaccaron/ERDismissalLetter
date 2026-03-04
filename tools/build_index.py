from ingestion.pdf_retriever import extract_folder
from ingestion.chunking import create_chunks
from retrieval.embedding import load_embedder, embed_docs
from retrieval.storage import (
    build_faiss_index,
    build_metadata,
    save_index_and_metadata,
)
from pathlib import Path

def main():

    print("Extracting PDFs...")
    data_dir = Path("/home/zazza/Documents/NLP/Project/Guidelines")
    pages = extract_folder(data_dir)
    print(f"Extracted {len(pages)} pages")

    print("Chunking...")
    chunks = create_chunks(pages)

    print("Loading embedder...")
    embedder = load_embedder()

    print("Embedding...")
    chunks = embed_docs(chunks, embedder)

    print("Building FAISS index...")
    index = build_faiss_index(chunks)

    metadata = build_metadata(chunks)

    print("Saving index and metadata...")
    save_index_and_metadata(index, metadata)

    print("Index successfully built.")


if __name__ == "__main__":
    main()
