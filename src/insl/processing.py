"""Parallel processing utilities."""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .correction import correct_spelling, init_spelling_corrector
from .parser import count_verbs, init_parser

# Global variables for worker processes
_nlp = None
_corrector = None


def init_worker():
    """Initialize models once per worker process."""
    global _nlp, _corrector
    _nlp = init_parser()
    _corrector = init_spelling_corrector()


def process_text(text: str) -> int:
    """
    Process single text: correct spelling, parse, count verbs.

    Args:
        text: Input text string

    Returns:
        Number of verbs found
    """
    corrected = correct_spelling(text, _corrector)
    doc = _nlp(corrected)
    count, _ = count_verbs(doc)
    return count


def process_single_text(text: str) -> dict[str, str | int]:
    """
    Process single text without multiprocessing (for interactive use).

    Args:
        text: Input text string

    Returns:
        Dictionary with original text, corrected text, and verb count
    """
    # Initialize models if not already loaded
    corrector = init_spelling_corrector()
    nlp = init_parser()

    # Process text
    corrected = correct_spelling(text, corrector)
    doc = nlp(corrected)
    count, _ = count_verbs(doc)

    return {
        "original_text": text,
        "corrected_text": corrected,
        "verb_count": count
    }


def process_file(filepath: Path) -> dict[str, str | int]:
    """
    Process single text file.

    Args:
        filepath: Path to text file

    Returns:
        Dictionary with filename, original text, corrected text, and verb count
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    corrected = correct_spelling(text, _corrector)
    doc = _nlp(corrected)
    count, _ = count_verbs(doc)

    return {
        "filename": filepath.name,
        "original_text": text,
        "corrected_text": corrected,
        "verb_count": count
    }


def process_multiple_files(filepaths: list[Path], max_workers: int = None) -> pl.DataFrame:
    """
    Process multiple text files in parallel.

    Args:
        filepaths: List of paths to text files
        max_workers: Number of worker processes (default: CPU count)

    Returns:
        Polars DataFrame with columns: filename, original_text, corrected_text, verb_count
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        results = list(tqdm(
            executor.map(process_file, filepaths, chunksize=1),
            total=len(filepaths),
            desc="Processing files"
        ))

    return pl.DataFrame(results)


def process_multiple_texts(texts: list[str], max_workers: int = None) -> list[int]:
    """
    Process multiple texts in parallel.

    Args:
        texts: List of text strings
        max_workers: Number of worker processes (default: CPU count)

    Returns:
        List of verb counts
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        return list(tqdm(
            executor.map(process_text, texts, chunksize=1),
            total=len(texts),
            desc="Processing texts"
        ))
