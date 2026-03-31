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

def detect_index(text: str, threshold: float = 0.4) -> bool:

    if not text or len(text.strip()) < 20:
        return False
 
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
 
    score = 0.0
 
    # ── 1. Keyword detection (weight: 0.30) ──
    # TOC/index headers
    keyword_pattern = re.compile(
        r'\b(table of contents|contents|index|'
        r'|abbreviations|list of figures|list of tables)\b',
        re.IGNORECASE
    )
    # Check only the first lines for header keywords
    header_text = " ".join(lines[:10])
    if keyword_pattern.search(header_text):
        score += 0.30
 
    # ── 2. Lines ending with page numbers (weight: 0.40) ──
    # Matches lines that end with a number, optionally preceded by
    # dots, dashes, spaces, or tabs (the "leader" pattern)
    line_with_number = re.compile(
        r'[a-zA-Z]'              # at least one letter somewhere
        r'.*'                     # any content
        r'[\s.\-·…\t]{2,}'       # leader: dots, spaces, dashes, etc.
        r'\d{1,4}\s*$'           # ends with a page number
    )
    number_ending_count = sum(1 for line in lines if line_with_number.match(line))
    number_ratio = number_ending_count / len(lines)
    score += 0.40 * min(number_ratio / 0.5, 1.0)  # full score at 50%+ lines matching
 
    # ── 3. Dot/dash leaders (weight: 0.15) ──
    # The classic "Chapter 1 ............... 5" pattern
    leader_pattern = re.compile(r'[.\-·…]{5,}')
    leader_count = sum(1 for line in lines if leader_pattern.search(line))
    leader_ratio = leader_count / len(lines)
    score += 0.15 * min(leader_ratio / 0.3, 1.0)  # full score at 30%+ lines
 
    return score >= threshold


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
        line = re.sub(r'(?<=[a-zA-Z\)\.,])\d+(?:[,\-–]\d+)*', '', line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return cleaned


def remove_captions(sentences: List[str]) -> List[str]:
    return [s for s in sentences
        if not re.match(r'^\s*(Figure|Fig\.|Table|Supplementary|Recommendation Table)\s*+\d+', s, re.IGNORECASE)]

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

def remove_summaries(lines: list[str]) -> list[str]:

    cleaned = []
    for line in lines:
        line = line.strip()

        if "©" in line:
            continue
        if line.count(";") >= 3:
            continue
        # if len(line) >= 300 and line.count(",") / len(line) > 0.05::
            # continue

        cleaned.append(line)

    return cleaned


def clean_sentences(lines: list[str]) -> list[str]:
    
    lines = remove_summaries(lines)
    lines = remove_captions(lines)

    return lines


def clean_lines(lines: List[str]) -> List[str]:

    lines = remove_reference_numbers(lines)
    lines = remove_figure_nonsense(lines)
    
    return lines
