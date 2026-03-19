from typing import List, Dict

def merge_lines(lines: List[str]) -> List[str]:
    """
    Merge broken lines from PDF extraction.
    """
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # if buffer empty → start
        if not buffer:
            buffer = line
            continue

        # if previous line ends with punctuation → new unit
        if buffer.endswith((".", ":", ";")):
            merged.append(buffer)
            buffer = line
        else:
            # merge lines
            buffer += " " + line

    if buffer:
        merged.append(buffer)

    return merged


def create_chunks(
    pages: List[Dict],
    max_chars: int = 1200,
    overlap_units: int = 2,
) -> List[Dict]:

    chunks = []
    current_chunk = []
    current_len = 0
    current_metadata = None

    for page in pages:
        raw_lines = [l.strip() for l in page["text"].split("\n") if l.strip()]

        # merge broken PDF lines
        units = merge_lines(raw_lines)

        for unit in units:
            unit_len = len(unit)

            if current_metadata is None:
                current_metadata = {
                    "doc_id": page["doc_id"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                }

            # if exceeds → emit chunk
            if current_len + unit_len > max_chars and current_chunk:
                chunk_text = " ".join(current_chunk)

                chunks.append({
                    **current_metadata,
                    "text": chunk_text
                })

                # overlap (last N units)
                current_chunk = current_chunk[-overlap_units:]
                current_len = sum(len(u) for u in current_chunk)

                current_metadata = {
                    "doc_id": page["doc_id"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                }

            current_chunk.append(unit)
            current_len += unit_len
            current_metadata["page_end"] = page["page"]

    # last chunk
    if current_chunk:
        chunks.append({
            **current_metadata,
            "text": " ".join(current_chunk)
        })

    return chunks


def is_noisy_chunk(text: str) -> bool:
    """
    Function used to identify if a chunk is a summary table, 
    which can skew significantly the similarity scores due to density of key words but lack of actual significance.

    The idea is that summary table contain many less periods than plain text, but normally higher counts of digits and , due to citations.
    """

    period_count = text.count(".")
    comma_count = text.count(",")
    semicolon_count = text.count(";")
 
    digits = sum(c.isdigit() for c in text)
    digit_ratio = digits / max(len(text), 1)

    if period_count < 2 and (
        (comma_count + semicolon_count) > 8 or digit_ratio > 0.1):
        return True

    return False