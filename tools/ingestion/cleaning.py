import re
from typing import List

def remove_reference_numbers(lines: List[str]) -> List[str]:

    cleaned = []
    for line in lines:
        # remove sequences of numbers that appear right after a letter, a parenthesis or a period without spaces in between (most likely references)
        line = re.sub(r'(?<=[a-zA-Z\)\.])\d+(?:[,-]\d+)*', '', line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return cleaned


def remove_table_lines(lines: List[str]) -> List[str]:
    """
    Remove lines that look like table rows:
    - Contain recommendation markers like 'I C', 'I B', 'III C' in isolation
    - Very high density of uppercase abbreviations
    - Short lines with mostly non-prose content
    """
    cleaned = []
    for line in lines:
        # standalone recommendation class markers
        if re.fullmatch(r'\s*(I{1,3}|IV|V)\s+[ABC]\s*', line):
            continue
        # lines that are almost entirely uppercase abbreviations and numbers
        words = line.split()
        if len(words) < 6:
            upper_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)
            if upper_ratio > 0.6:
                continue
        cleaned.append(line)
    return cleaned


def remove_captions(sentences: List[str]) -> List[str]:
    return [s for s in sentences
        if not re.match(r'^\s*(Figure|Fig\.|Table|Supplementary)\s*+\d+', s, re.IGNORECASE)]


def clean_lines(lines: List[str]) -> List[str]:

    lines = remove_reference_numbers(lines)
    lines = remove_table_lines(lines)
    return lines