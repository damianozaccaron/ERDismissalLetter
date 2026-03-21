from ingestion.pdf_retriever import extract_folder
from ingestion.chunking import create_chunks
from retrieval.embedding import load_embedder, embed_docs
from retrieval.storage import build_faiss_index, build_metadata, save_index_and_metadata, sanity_check
from pathlib import Path

def main(rel_path):

    print("Extracting PDFs...")
    data_dir = rel_path
    pages = extract_folder(data_dir)
    print(f"Extracted {len(pages)} pages")

    print("Chunking...")
    chunks = create_chunks(pages)
    print(f"Extracted {len(chunks)} chunks")

    print("Loading embedder...")
    embedder = load_embedder()

    print("Embedding...")
    chunks = embed_docs(chunks, embedder)

    print("Building FAISS index...")
    index = build_faiss_index(chunks)
    metadata = build_metadata(chunks)

    sanity_check(index, metadata)

    print("Saving index and metadata...")
    save_index_and_metadata(index, metadata)

    print("Index successfully built.")


def check_chunking(rel_path):
    print("Extracting PDFs...")
    data_dir = rel_path
    pages = extract_folder(data_dir)
    print(f"Extracted {len(pages)} pages")

    print("Chunking...")
    chunks = create_chunks(pages)

    print(f"Extracted {len(chunks)} chunks")

    output = []
    for chunk in chunks:
        output.append(chunk["text"])

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write("\n\n\n".join(output))

if __name__ == "__main__":
    guidelines = Path("./Guidelines")
    main(guidelines)
