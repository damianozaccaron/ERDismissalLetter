import torch, time, sys
from pathlib import Path

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    NER_MODEL,
    CROSS_ENCODER,
    RETRIEVAL_K,
    FINAL_N,
    TEMPERATURE,
    QUANT,
    REPO,
    MODEL_QUANT)

from storage.storage import load_index_and_metadata, load_vectorizer
from storage.embedding import embed_query, load_embedder
from retrieval.retrieval import retrieve_top_k, load_crossEncoder, reranking_multi_query

from generation.output_prod import load_model_quant, load_model_transformer, generate_letter

from retrieval.query_building import build_queries_ner,load_ner_model
from generation.prompting import build_prompt, deepl_translation
from pipeline.user_input import collect_patient_input


def load_models():
    
    start_models = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NER pipeline...")
    ner_model = load_ner_model(model_name=NER_MODEL, device=device)
    print("Loading Vectorizer...")
    vectorizer = load_vectorizer()    
    print("Loading embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)
    print("Loading cross-encoder...")
    cross_encoder = load_crossEncoder(CROSS_ENCODER)
    end_models = time.time()
    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()
    print(f"Models loaded in {end_models-start_models:.3f} seconds")

    llm_model = None
    tokenizer = None

    if QUANT:
        # quantized models
        print("Loading LLM...")
        llm_model = load_model_quant(repo=REPO, model_name=MODEL_QUANT)
    else:
        # non-quantized models (transformers library)
        print("Loading LLM...")
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, llm_model = load_model_transformer(MODEL_NAME, device)

    return ner_model, vectorizer, emb_model, cross_encoder, index, metadata, llm_model, tokenizer

def main(input_file: str, ner_model, vectorizer, emb_model, cross_encoder, index, metadata, llm_model, tokenizer, use_existing_transl = True):
    print(f"\n{'='*50}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*50}")

    # PATIENT DATA COLLECTION
    # raw_data = collect_patient_input()
  
    if use_existing_transl:
        with open(f"Examples/ENG/translation{input_file.name}", "r", encoding="utf-8") as file:
            translated_note = file.read()
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            italian_data = f.read()

        print("Translating clinical note...")
        translated_note = deepl_translation(italian_data)

        eng_path = f"Examples/ENG/translation{input_file.name}"
        with open(eng_path, "w", encoding="utf-8") as f:
            f.write(translated_note)
        print(f"Translation saved to {eng_path}")


    # RETRIEVAL
    start_retrieval = time.time()
    print("Building queries...")
    queries, rerank_queries = build_queries_ner(translated_note, ner_model, vectorizer)

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

    top_k_candidates = list(all_candidates.values())
    faiss_ids = list(all_candidates.keys())
    print(f"Total unique candidates after merge: {len(top_k_candidates)}")

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved from top_k retrieval")

    # re-ranking
    print("Reranking...")
    start = time.time()

    final_chunks = reranking_multi_query(rerank_queries, top_k_candidates, cross_encoder, top_n=FINAL_N)

    end = time.time()

    print(f"Total time for re-reranking: {end-start:.3f} seconds")

    if len(final_chunks) == 0:
        raise ValueError("No candidates retrieved from Cross-Encoder")
    
    end_retrieval = time.time()
    print(f"Executed retrieval in {end_retrieval-start_retrieval:.3f} seconds.")

    # LETTER GENERATION
    print("Building prompt...")
    prompt = build_prompt(translated_note, final_chunks)

    print("Generating letter...\n")
    letter = generate_letter(prompt=prompt, tokenizer=tokenizer, model=llm_model, temperature=TEMPERATURE)

    print("=== GENERATED LETTER ===\n")

    return letter, rerank_queries

if __name__ == "__main__":

    input_dir = Path("Examples/ITA")
    eng_dir = Path("Examples/ENG")
    results_dir = Path("Examples/results")

    if len(sys.argv) > 1:
        single_file = input_dir / sys.argv[1]
        if not single_file.exists():
            print(f"File not found: {single_file}")
            exit(1)
        txt_files = [single_file]
    else:
        txt_files = sorted(input_dir.glob("*.txt"))
        if not txt_files:
            print(f"No .txt files found in {input_dir}/")
            exit(1)

    print(f"Found {len(txt_files)} file(s)\n")

    ner_model, vectorizer, emb_model, cross_encoder, index, metadata, model, tokenizer = load_models()

    for input_file in txt_files:
        start = time.time()
        output, queries = main(
            input_file=input_file,
            ner_model=ner_model,
            vectorizer=vectorizer,
            emb_model=emb_model,
            cross_encoder=cross_encoder,
            index=index,
            metadata=metadata,
            use_existing_transl=True,
            llm_model=model,
            tokenizer=tokenizer
        )
        end = time.time()
        print(f"{input_file} processed in {end-start:.3f} seconds.")

        stem = input_file.stem
        output_path = results_dir / f"{stem}_output.txt"
        queries_path = results_dir / f"{stem}_queries.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        with open(queries_path, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries, 1):
                f.write(f"--- Query {i} ---\n{q}\n\n")

        print(f"Results saved to {output_path} and {queries_path}")