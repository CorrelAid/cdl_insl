"""Text parsing and POS tagging using Stanza."""

import stanza
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def init_parser():
    """Initialize and return Stanza pipeline for German (cached)."""
    # Use a consistent model directory to avoid re-downloading
    model_dir = Path.home() / ".stanza_models"
    model_dir.mkdir(exist_ok=True)

    return stanza.Pipeline(
        lang='de',
        processors='tokenize,mwt,pos,lemma,depparse',
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        model_dir=str(model_dir),
        verbose=True
    )


def count_verbs(doc, include_aux=False, exclude=None) -> tuple[int, list]:
    """
    Count verbs in parsed Stanza document.

    Args:
        doc: Stanza parsed document
        include_aux: if True, also count AUX (auxiliary verbs)
        exclude: set of strings (lowercase) to exclude by word.text or word.lemma

    Returns:
        Tuple of (count, list of verb words)
    """
    if exclude is None:
        exclude = set()

    count = 0
    verbs = []
    for sent in doc.sentences:
        for word in sent.words:
            upos_ok = (word.upos == "VERB") or (include_aux and word.upos == "AUX")
            if not upos_ok:
                continue
            # Check exclusion list
            if word.text.casefold() in exclude:
                continue
            if hasattr(word, "lemma") and (word.lemma or "").casefold() in exclude:
                continue
            count += 1
            verbs.append(word)
    return count, verbs


def highlight_verbs(doc, include_aux=True, exclude=None) -> str:
    """
    Highlight verbs in the original text using character offsets.
    Main verbs (VERB) are highlighted in blue, auxiliary verbs (AUX) in orange.

    Args:
        doc: Stanza parsed document
        include_aux: if True, also highlight AUX verbs (in different color)
        exclude: set of strings (lowercase) to exclude by word.text or word.lemma

    Returns:
        Text with verbs wrapped in colored HTML spans
    """
    if exclude is None:
        exclude = set()

    # Colors for different verb types
    VERB_COLOR = "#0066cc"  # Blue for main verbs
    AUX_COLOR = "#ff8c00"   # Orange for auxiliary verbs

    def is_excluded(word):
        if word.text.casefold() in exclude:
            return True
        if hasattr(word, "lemma") and (word.lemma or "").casefold() in exclude:
            return True
        return False

    if not hasattr(doc, 'text') or doc.text is None:
        # Fallback: reconstruct from tokens
        result = []
        for sent in doc.sentences:
            for word in sent.words:
                if is_excluded(word):
                    result.append(word.text)
                elif word.upos == "VERB":
                    result.append(f'<span style="color: {VERB_COLOR}; font-weight: bold;">{word.text}</span>')
                elif include_aux and word.upos == "AUX":
                    result.append(f'<span style="color: {AUX_COLOR}; font-weight: bold;">{word.text}</span>')
                else:
                    result.append(word.text)
        return " ".join(result)

    # Build a list of (start_char, end_char, verb_type) tuples
    verb_positions = []
    for sent in doc.sentences:
        for token in sent.tokens:
            # Check all words in the token (for multi-word tokens)
            for word in token.words:
                if is_excluded(word):
                    continue
                if word.upos == "VERB":
                    verb_positions.append((token.start_char, token.end_char, "VERB"))
                    break
                elif include_aux and word.upos == "AUX":
                    verb_positions.append((token.start_char, token.end_char, "AUX"))
                    break

    # Sort by start position
    verb_positions.sort()

    # Build highlighted text
    text = doc.text
    result = []
    last_pos = 0

    for start, end, verb_type in verb_positions:
        # Add text before verb
        result.append(text[last_pos:start])
        # Add verb with appropriate color
        color = VERB_COLOR if verb_type == "VERB" else AUX_COLOR
        result.append(f'<span style="color: {color}; font-weight: bold;">{text[start:end]}</span>')
        last_pos = end

    # Add remaining text
    result.append(text[last_pos:])

    return "".join(result)
