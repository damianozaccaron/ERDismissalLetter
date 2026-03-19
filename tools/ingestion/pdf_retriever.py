from pathlib import Path
import fitz 
import re, time

def empty_page_warning(doc_id: str, empty_page_counter: int, total_pages: int, threshold: float = 0.3):
    """
    Simple function to print a warning if too many pages are empty in a document.
    """
    if empty_page_counter / total_pages > threshold:
        print(
            f"[WARNING] {doc_id}: "
            f"{empty_page_counter*100/total_pages:.1f}% pages are empty. "
            "Possible scanned PDF or extraction failure."
        )


def is_page_bad(text: str) -> bool:
    """
    Detects if a page only contains Bibliography references or is part of the Index or Abbreviation section.
    """
    text_lower = text.lower()

    # DOI / URLs (Bibliography)
    doi_count = text_lower.count("doi")
    url_count = text_lower.count("http")

    if doi_count + url_count >= 8:
        return True

    # Et al.
    etal_counter = text_lower.count("et al")
    if etal_counter >= 8:
        return True
    
    # Index
    sep_counter = text.count(".............................")
    if sep_counter >= 8:
        return True
       
    return False

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
    Extract text from a single PDF using PyMuPDF. Excludes text written in page margins (page numbers, download references, footnotes, ...)
    Returns a list of dicts:
    [
        {
            "doc_id": "NameOfFile",
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

        textpage = []

        empty_block_counter = 0
        for block in page.get_text("blocks"): 
            # PyMuPDF divides chunk of text inside a pages in blocks and gives their position relative to the page. If it's a useful block, we'll keep it in the page.
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()

            # if the block does not meet minimum characters, skip it
            if len(text) < min_chars_per_block:
                empty_block_counter += 1
                continue

            # if the block is in the header, footer, or side margins, skip it
            if y1 < page_height * header_ratio:
                empty_block_counter += 1
                continue
            if y0 > page_height * (1 - footer_ratio):
                empty_block_counter += 1
                continue
            if x1 < page_width * side_margin_ratio or x0 > page_width * (1 - side_margin_ratio):
                empty_block_counter += 1
                continue
            
            textpage.append(text)

        # At this point, we have a page in text form and with block-level checks completed
        if empty_block_counter == len(page.get_text("blocks")):
            empty_page_counter += 1

        if textpage:
            full_text = "\n".join(textpage)

            # If the page is identified as bibliography/index, ignore it
            if is_page_bad(full_text):
                continue
            
            results.append({
                "doc_id": doc_id,
                "page": page_idx + 1,
                "text": full_text
            })

    # At this point, we have all the (hopefully) useful pages for the file and can check how many we have. 
    # If it's too little, something probably went wrong (threshold for warning set at 30% arbitrarily).
    empty_page_warning(doc_id, empty_page_counter, len(doc))

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

        pages = extract_pdf_text_layout_aware(pdf_file)
        all_pages.extend(pages)

        print(f"Extracted {len(pages)} pages from {pdf_file.name}")


    end_time = time.time()
    print(f"Extraction took {end_time - start_time:.3f} seconds")
    return all_pages

if __name__ == "__main__":
    data_dir = Path("./Guidelines")
    print(f"Extracting PDFs from {data_dir}")
    blocks = extract_folder(data_dir)

    print(f"Extracted {len(blocks)} blocks")
    if blocks:
        print(blocks[5]["text"][0:100]) 