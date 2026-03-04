from pathlib import Path
import fitz 
import statistics
import time

def empty_page_warning(doc_id: str, empty_page_counter: int, total_pages: int, threshold: float = 0.8):
    """
    Simple function to print a warning if too many pages are empty in a document.
    """
    if empty_page_counter / total_pages > threshold:
        print(
            f"[WARNING] {doc_id}: "
            f"{empty_page_counter*100/total_pages:.1f}% pages are empty. "
            "Possible scanned PDF or extraction failure."
        )

def extract_pdf_text(
    pdf_path: Path,
    min_chars_per_page: int = 10
) -> list[dict]:
    """
    Extract text from a single PDF using PyMuPDF.
    Returns a list of dicts:
    [
        {
            "doc_id": "af_guideline",
            "page": 1,
            "text": "...",
        },
        ...
    ]
    """
    doc = fitz.open(pdf_path)
    results = []

    doc_id = pdf_path.stem # use filename without extension as identification
    empty_page_counter = 0

    for page_idx, page in enumerate(doc):
        text = page.get_text("text").strip()

        is_empty = len(text) < min_chars_per_page

        # Skip empty / nearly empty pages
        if is_empty:
            empty_page_counter += 1
            continue

        results.append({
            "doc_id": doc_id,
            "page": page_idx + 1,
            "text": text,
            "empty": is_empty
        })

    empty_page_warning(doc_id, empty_page_counter, len(doc))
        
    return results

def extract_pdf_text_layout_aware(
    pdf_path: Path,
    min_chars_per_block: int = 10,
    header_ratio: float = 0.05,
    footer_ratio: float = 0.05,
    side_margin_ratio: float = 0.05,
) -> list[dict]:
    """
    Extract text from a single PDF using PyMuPDF.
    Returns a list of dicts:
    [
        {
            "doc_id": "af_guideline",
            "page": 1,
            "text": "...",
        },
        ...
    ]
    """

    doc = fitz.open(pdf_path)
    results = []
    doc_id = pdf_path.stem

    empty_page_counter = 0

    for page_idx, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width

        page_text_parts = []

        empty_block_counter = 0
        for block in page.get_text("blocks"): 
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()

            if len(text) < min_chars_per_block:
                empty_block_counter += 1
                continue

            # if the block is in the header, footer, or side margins, skip it
            if y1 < page_height * header_ratio:
                continue
            if y0 > page_height * (1 - footer_ratio):
                continue
            if x1 < page_width * side_margin_ratio or x0 > page_width * (1 - side_margin_ratio):
                continue

            page_text_parts.append(text)

        if empty_block_counter == len(page.get_text("blocks")):
            empty_page_counter += 1

        if page_text_parts:
            results.append({
                "doc_id": doc_id,
                "page": page_idx + 1,
                "text": "\n".join(page_text_parts)
            })

    empty_page_warning(doc_id, empty_page_counter, len(doc))

    return results


def extract_pdf_blocks_with_headers(
    pdf_path: Path,
    min_chars_per_block: int = 10,
    header_font_ratio: float = 1.3,
    header_ratio: float = 0.05,
    footer_ratio: float = 0.05,
    side_margin_ratio: float = 0.05,
) -> list[dict]:

    doc = fitz.open(pdf_path)
    results = []
    doc_id = pdf_path.stem

    empty_pages_counter = 0

    for page_idx, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_height = page.rect.height
        page_width = page.rect.width

        font_sizes = []

        # collect font sizes for the page
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    font_sizes.append(span["size"])
        
        # if no font sizes found, skip the page
        if not font_sizes:
            empty_pages_counter += 1
            continue

        median_font_size = statistics.median(font_sizes)

        page_blocks_kept = 0

        for block_idx, block in enumerate(page_dict["blocks"]):

            if block["type"] != 0:
                continue

            x0, y0, x1, y1 = block["bbox"]

            # geometry filters
            if y1 < page_height * header_ratio:
                continue
            if y0 > page_height * (1 - footer_ratio):
                continue
            if x1 < page_width * side_margin_ratio or x0 > page_width * (1 - side_margin_ratio):
                continue

            block_text_parts = []
            block_font_sizes = []

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        block_text_parts.append(text)
                        block_font_sizes.append(span["size"])

            block_text = " ".join(block_text_parts)

            if len(block_text) < min_chars_per_block:
                continue

            avg_font_size = statistics.mean(block_font_sizes)

            is_header = (
                len(block_text) <= 150 and
                avg_font_size >= median_font_size * header_font_ratio
            )

            results.append({
                "doc_id": doc_id,
                "page": page_idx + 1,
                "block": block_idx,
                "text": block_text,
                "is_header": is_header
            })
            page_blocks_kept += 1

        # if no blocks were kept for this page, count it as empty
        if page_blocks_kept == 0:
            empty_pages_counter += 1

    # if there are too many empty pages it's suspect, send a warning
    empty_page_warning(doc_id, empty_pages_counter, len(doc))

    return results


def extract_folder(
    folder_path: Path
) -> list[dict]:
    """
    Extract text from all PDFs in a folder.
    """
    all_pages = []

    start_time = time.time()
    for pdf_file in folder_path.glob("*.pdf"):
        print(f"Parsing {pdf_file.name}")

        # Final choice is to use the whole page but with layout aware extraction, meaning foot notes and headers are removed.
        pages = extract_pdf_text_layout_aware(pdf_file)
        all_pages.extend(pages)


    end_time = time.time()
    print(f"Extraction took {end_time - start_time:.3f} seconds")
    return all_pages

if __name__ == "__main__":
    data_dir = Path("/home/zazza/Documents/NLP/Project/Guidelines")
    print(f"Extracting PDFs from {data_dir}")
    blocks = extract_folder(data_dir)

    print(f"Extracted {len(blocks)} blocks")
    if blocks:
        print(blocks[0].keys())
        print(blocks[5]["text"][0:100]) 
        #print(blocks[130]["is_header"])