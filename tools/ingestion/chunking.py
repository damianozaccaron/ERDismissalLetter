from typing import List, Dict

import pickle, time, re
from pathlib import Path
from nltk.tokenize.punkt import PunktTrainer, PunktSentenceTokenizer

from ingestion.cleaning import clean_lines, clean_sentences


def create_chunks(
    pages: List[Dict],
    max_chars: int = 1200,
    overlap_units: int = 2,
) -> List[Dict]:

    chunks = []
    current_chunk = []
    current_len = 0
    current_metadata = None

    merged_sentences = {} 

    for page in pages:
        if page["doc_id"] == "AtrialFibrillation" and page["page"] == 6:
            for i, line in enumerate(page["text"].split("\n")):
                print(f"{i}: {repr(line)}")
            break

    for page in pages:

        raw_lines = page["text"].split("\n")
        clean = clean_lines(raw_lines)

        merged_sentences[(page["doc_id"], page["page"])] = merge_lines(clean)

    all_units_flat = [unit for units in merged_sentences.values() for unit in units]
    tokenizer = get_tokenizer(all_units_flat)

    for page in pages:
        units = merged_sentences[(page["doc_id"], page["page"])]

        # flatten punkt output into a list of sentences and apply Punkt sentence splitting
        sentences = []
        for unit in units:
            sentences.extend(tokenizer.tokenize(unit))

        # remove captions and summaries
        sentences = clean_sentences(sentences)

        # check if there are sentences that are too long, if that's the case split them based on punctuation
        sentences = split_oversized_sentences(sentences, max_chars)

        for sentence in sentences:

            if current_metadata is None:
                current_metadata = {
                    "doc_id": page["doc_id"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                }

            if current_len + len(sentence) > max_chars and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    **current_metadata,
                    "text": chunk_text
                })

                current_chunk = current_chunk[-overlap_units:]
                current_len = sum(len(s) for s in current_chunk)

                current_metadata = {
                    "doc_id": page["doc_id"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                }

            current_chunk.append(sentence)
            current_len += len(sentence)
            current_metadata["page_end"] = page["page"]

    # last chunk
    if current_chunk:
        chunks.append({
            **current_metadata,
            "text": " ".join(current_chunk)
        })

    return chunks


def merge_lines(lines: List[str]) -> List[str]:
    """
    Merge broken lines from PDF extraction.
    """
    merged = []
    buffer = ""

    for line in lines:
        broken = False
        line = line.strip()
        if not line:
            continue

        # if the buffer is empty, start to populate it
        if not buffer:
            buffer = line
            continue

        #if the line ends with -, assume it is a broken word and fix it
        if buffer.endswith("-") or buffer.endswith("–") or buffer.endswith("\u00ad"):
            buffer = buffer[:-1]
            broken = True

        # if the line ends with punctuation, then the buffer is complete
        if buffer.endswith((".", ":", ";", "?", "!", ",")):
            merged.append(buffer)
            buffer = line
        
        # if the line is the continuation of the previous line, don't add the space
        elif broken:
            buffer += line
        else:
            buffer += " " + line
    # last line
    if buffer:
        merged.append(buffer)

    return merged

def train_punkt(texts: list[str], save_path: str = "Weights/punkt_medical.pkl") -> PunktSentenceTokenizer:

    trainer = PunktTrainer()
    # trainer.ABBREV_BACKOFF = 0  # more aggressive abbreviation learning

    print("Training a new Punkt tokenizer")
    
    start_time = time.time()

    for text in texts:
        trainer.train(text, finalize=False)
    
    trainer.finalize_training()
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer, f)

    end_time = time.time()

    print(f"Tokenizer trained in {end_time - start_time:.3f} seconds")
    
    return tokenizer


def load_punkt(path: str = "punkt_weights.pkl") -> PunktSentenceTokenizer:

    with open(path, "rb") as f:
        return pickle.load(f)

# Remember to modify paths for models
def get_tokenizer(texts: list[str] = None, path: str = "Weights/punkt_medical.pkl") -> PunktSentenceTokenizer:

    # load tokenizer if it exists, otherwise train and save it.
    if Path(path).exists():
        print("Loading existing tokenizer...")
        return load_punkt(path)
    
    if not texts:
        raise ValueError("No saved tokenizer found and no texts provided for training.")
    
    print("Training new tokenizer...")
    return train_punkt(texts, save_path=path)

def split_oversized_sentences(sentences: List[str], max_chars: int) -> List[str]:

    i = 0
    while i < len(sentences):
        if len(sentences[i]) > max_chars:
            split_parts = re.split(r'(?<=[;:!?])\s+', sentences[i])
            sentences = sentences[:i] + split_parts + sentences[i+1:]
            i += len(split_parts)
        else:
            i += 1
    return sentences
