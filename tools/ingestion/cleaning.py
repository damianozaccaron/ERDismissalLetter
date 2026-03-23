import re
from typing import List

def detect_biblio(text: str, threshold = 8) -> bool:
    text_lower = text.lower()

    # DOI / URLs (Bibliography)
    doi_count = text_lower.count("doi")
    url_count = text_lower.count("http")

    if doi_count + url_count >= threshold:
        return True

    # Et al.
    etal_counter = text_lower.count("et al")
    if etal_counter >= threshold:
        return True
    
    return False

def detect_index(text:str, threshold = 5) -> bool:
    
    sep_counter = text.count(".............................")
    return sep_counter >= threshold


def detect_abbreviations(text:str, threshold = 12) -> bool:
    abbrev_counter = sum(
        1 for line in text.split("\n")
        if re.match(r'^[A-Z][A-Z0-9\-]{1,9}\s*$', line.strip())
    )

    return abbrev_counter >= threshold


def is_page_bad(text: str) -> bool:
    """
    Detects if a page only contains Bibliography references or is part of the Index or Abbreviation section.
    """

    return detect_biblio(text) or detect_abbreviations(text) or detect_index(text)


def remove_reference_numbers(lines: List[str]) -> List[str]:

    cleaned = []
    for line in lines:
        # remove sequences of numbers that appear right after a letter, a parenthesis or a period without spaces in between (most likely references)
        line = re.sub(r'(?<=[a-zA-Z\)\.])\d+(?:[,-]\d+)*', '', line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return cleaned


def remove_captions(sentences: List[str]) -> List[str]:
    return [s for s in sentences
        if not re.match(r'^\s*(Figure|Fig\.|Table|Supplementary)\s*+\d+', s, re.IGNORECASE)]

def remove_figure_nonsense(lines: List[str]) -> List[str]:
    """
    Remove lines that are mostly garbled/unrenderable characters
    from broken font mappings in PDF extraction.
    """
    cleaned = []
    for line in lines:

        bad_chars = sum(1 for c in line if ord(c) > 127 or not c.isprintable())
        ratio = bad_chars / max(len(line), 1)

        if ratio > 0.8 or not line.strip():
            continue

        cleaned.append(line)

    return cleaned


def clean_lines(lines: List[str]) -> List[str]:

    lines = remove_reference_numbers(lines)
    lines = remove_figure_nonsense(lines)

    return lines