from typing import List, Dict

def create_chunks(
    pages: List[Dict],
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[Dict]:
    """
    pages: output of PDF extraction
    [
        {"doc_id": "...", "page_start": 1, "page_end": 1, "text": "..."},
        ...
    ]
    """

    chunks = []
    current_chunk = ""
    current_metadata = None

    for page in pages:
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]

        for line in lines:
            if current_metadata is None:
                current_metadata = {
                    "doc_id": page["doc_id"],
                    "page_start": page["page"],
                    "page_end": page["page"],
                }

            # If adding the line exceeds max size → emit chunk
            if len(current_chunk) + len(line) >= max_chars:
                chunks.append({
                    **current_metadata,
                    "text": current_chunk.strip()
                })

                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] + " "
                current_metadata["page_start"] = page["page"]

            current_chunk += line + " "
            current_metadata["page_end"] = page["page"]

    # Emit last chunk
    if current_chunk.strip():
        chunks.append({
            **current_metadata,
            "text": current_chunk.strip()
        })

    return chunks