"""Spelling correction using DSPy."""

import os
import threading

import dspy
import stanza
from dotenv import load_dotenv
from sacremoses import MosesDetokenizer

load_dotenv()

# Initialize Stanza tokenizer once
_tokenizer = None
_detokenizer = None
_corrector = None
_corrector_lock = threading.Lock()


def get_detokenizer():
    """Get or initialize Moses detokenizer for German."""
    global _detokenizer
    if _detokenizer is None:
        _detokenizer = MosesDetokenizer(lang='de')
    return _detokenizer


def get_tokenizer():
    """Get or initialize Stanza tokenizer for German."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = stanza.Pipeline(
            lang='de',
            processors='tokenize',
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
            verbose=False
        )
    return _tokenizer


class SpellingCorrection(dspy.Signature):
    """Correct German tokenized text. Fix spelling, orthography, and capitalization using sentence context.
    Rules:
    - Correct misspelled words while preserving meaning
    - Capitalize all nouns (German grammar)
    - Fix grammatical case/number agreement
    - Preserve verb tense from inflection endings (e.g., '-ten' indicates Präteritum: 'fahrten' → 'fuhren', not 'fahren')
    - Fix incorrect direct speech punctuation using ASCII quotes (e.g., ',,' → '"') - will be converted to German quotes later
    - Never delete tokens, only modify or insert
    - Never add words (e.g. if a verb is missing), only correct existing words but may insert missing punctuation
    - May correct pronouns and merge sentence fragments into proper sentences
    - Preserve word order
    """
    tokens: list = dspy.InputField(
        desc="German text tokens with potential spelling/grammar errors. May include misspellings, wrong cases, missing punctuation."
    )
    corrected: list = dspy.OutputField(
        desc="Corrected tokens list. Same length or longer than input (insertions allowed, deletions forbidden)."
    )


def init_spelling_corrector(
    provider: str = "openrouter",
    model: str | None = None,
    token: str | None = None,
    api_base: str | None = None,
):
    """
    Initialize and return a DSPy spelling corrector.
    Thread-safe singleton - returns cached corrector if already initialized.

    Args:
        provider: One of "openrouter", "modal", or "custom"
        model: Model name (default: provider-specific)
        token: API key (default: from env vars based on provider)
        api_base: Base URL (only used for "custom" provider)

    Returns:
        Compiled DSPy corrector

    Raises:
        ValueError: If required credentials are missing
    """
    global _corrector

    # Return cached corrector if available
    if _corrector is not None:
        return _corrector

    # Thread-safe initialization
    with _corrector_lock:
        # Double-check after acquiring lock
        if _corrector is not None:
            return _corrector

        examples = [
        # Example: Multiple errors with comma insertion
        dspy.Example(
            tokens=['Herr', 'Jakob', 'so', 'heiß', 'die', 'Haand', 'Figure', '.', 'Herr', 'Jakob', 'ist', 'neben', 'drei', 'männer', 'und', 'die', 'kuken', 'auf', 'dass', 'wasser'],
            corrected=['Herr', 'Jakob', ',', 'so', 'heißt', 'die', 'Hauptfigur', '.', 'Herr', 'Jakob', 'ist', 'neben', 'drei', 'Männern', 'und', 'sie', 'gucken', 'auf', 'das', 'Wasser', '.']
        ).with_inputs("tokens"),
        # Example: Capitalization and comma insertion
        dspy.Example(
            tokens=['der', 'lehrer', 'sakte', 'dehn', 'schülern', 'sie', 'sollen', 'ihrre', 'Handys', 'während', 'der', 'unterricht', 'ausschalten', '.'],
            corrected=['Der', 'Lehrer', 'sagte', 'den', 'Schülern', ',', 'sie', 'sollen', 'ihre', 'Handys', 'während', 'des', 'Unterrichts', 'ausschalten', '.']
        ).with_inputs("tokens"),
        # Example: Verb conjugation error
        dspy.Example(
            tokens=['Die', 'Boote', 'fahrten', 'im', 'runden', 'Brunnen', '.', 'Die', 'Boote', 'segelten', 'ins', 'saubere', 'Wasser', '.'],
            corrected=['Die', 'Boote', 'fuhren', 'im', 'runden', 'Brunnen', '.', 'Die', 'Boote', 'segelten', 'ins', 'saubere', 'Wasser', '.']
        ).with_inputs("tokens"),
        # Example: Case agreement error
        dspy.Example(
            tokens=['Er', 'gab', 'das', 'Buch', 'der', 'Mann', '.'],
            corrected=['Er', 'gab', 'das', 'Buch', 'dem', 'Mann', '.']
        ).with_inputs("tokens"),
        # Example: Incorrect direct speech punctuation (use ASCII quotes)
        dspy.Example(
            tokens=['Er', 'sagte', ':', ',', ',', 'Ich', 'habe', 'zu', 'Hause', 'noch', 'eine', 'Eisenbahn', '.', '"'],
            corrected=['Er', 'sagte', ':', '"', 'Ich', 'habe', 'zu', 'Hause', 'noch', 'eine', 'Eisenbahn', '.', '"']
        ).with_inputs("tokens"),
        # Example: Spelling error and case correction (preserve missing verb)
        dspy.Example(
            tokens=['Herr', 'Jakob', 'vor', 'den', 'Brunnen', 'und', 'kuckt', 'die', 'Schiffe', 'an', '.', 'Die', 'im', 'Brunnen', 'segeln', '.', 'Er', 'ging', 'zurück', '.', 'Er', 'kam', 'wenig', 'später', 'mit', 'einem', 'Karton', 'zurück', '.', 'Als', 'er', 'am', 'Brunnen', 'ankamm', 'leerte', 'er', 'den', 'Karton', 'aus', '.', 'Darin', 'waren', 'Schienen', 'und', 'Eisenbahnen', '.', 'Er', 'machte', 'die', 'Schienien', 'um', 'den', 'Brunnen', '.'],
            corrected=['Herr', 'Jakob', 'vor', 'dem', 'Brunnen', 'und', 'guckt', 'die', 'Schiffe', 'an', ',', 'die', 'im', 'Brunnen', 'segeln', '.', 'Er', 'ging', 'zurück', '.', 'Er', 'kam', 'wenig', 'später', 'mit', 'einem', 'Karton', 'zurück', '.', 'Als', 'er', 'am', 'Brunnen', 'ankam', ',', 'leerte', 'er', 'den', 'Karton', 'aus', '.', 'Darin', 'waren', 'Schienen', 'und', 'Eisenbahnen', '.', 'Er', 'machte', 'die', 'Schienen', 'um', 'den', 'Brunnen', '.']
        ).with_inputs("tokens")

        ]

        # Set defaults based on explicit provider selection
        headers = {}

        if provider == "openrouter":
            api_base = "https://openrouter.ai/api/v1"
            if token is None:
                token = os.getenv("OR_KEY")
            if model is None:
                model = "openrouter/mistralai/mistral-small-3.2-24b-instruct"
            elif not model.startswith("openrouter/"):
                model = f"openrouter/{model}"

        elif provider == "modal":
            api_base = os.getenv("MODAL_BASE_URL")
            if not api_base:
                raise ValueError("MODAL_BASE_URL environment variable not set")
            if token is None:
                token = os.getenv("VLLM_API_KEY")
            if model is None:
                model = "openai/mistralai/Mistral-7B-Instruct-v0.3"
            # Modal needs special auth headers
            headers["Authorization"] = f"Bearer {token}"
            if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
                headers["Modal-Key"] = os.getenv("MODAL_TOKEN_ID")
                headers["Modal-Secret"] = os.getenv("MODAL_TOKEN_SECRET")

        elif provider == "custom":
            if not api_base:
                raise ValueError("api_base required for custom provider")
            if model is None:
                model = "openai/custom-model"
            # token may be None for custom endpoints that don't require auth
        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'openrouter', 'modal', or 'custom'")

        if not token and provider != "custom":
            env_var = "OR_KEY" if provider == "openrouter" else "VLLM_API_KEY"
            raise ValueError(f"API key not provided. Set token parameter or {env_var} env var")

        lm = dspy.LM(model, api_key=token, api_base=api_base, headers=headers,cache=False)
        dspy.configure(lm=lm)
        prog = dspy.Predict(SpellingCorrection)
        teleprompter = dspy.LabeledFewShot()
        _corrector = teleprompter.compile(prog, trainset=examples)
        return _corrector


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize text using Stanza, preserving punctuation as separate tokens.

    Args:
        text: Input text string

    Returns:
        List of word and punctuation tokens
    """
    tokenizer = get_tokenizer()
    doc = tokenizer(text)

    # Extract all tokens from all sentences
    tokens = []
    for sentence in doc.sentences:
        for token in sentence.tokens:
            tokens.append(token.text)

    return tokens


def convert_to_german_quotes(text: str) -> str:
    """
    Convert ASCII quotes to proper German typographic quotes.

    German uses „..." (low-high) for direct speech.
    """
    # Track quote state to alternate between opening and closing
    result = []
    in_quote = False

    for char in text:
        if char == '"':
            if not in_quote:
                result.append('„')  # Opening quote (low)
                in_quote = True
            else:
                result.append('"')  # Closing quote (high)
                in_quote = False
        else:
            result.append(char)

    return ''.join(result)


def correct_spelling(text: str | list, corrector) -> str:
    """
    Correct spelling using DSPy corrector.

    Args:
        text: Either a string (will be tokenized) or list of tokens
        corrector: DSPy corrector instance

    Returns:
        Corrected text as a string
    """
    # Convert string to tokens if needed
    if isinstance(text, str):
        tokens = tokenize_text(text)
    else:
        tokens = text

    corrected_tokens = corrector(tokens=tokens).corrected

    # Use Moses detokenizer for proper German detokenization
    detokenizer = get_detokenizer()
    result = detokenizer.detokenize(corrected_tokens)

    # Convert ASCII quotes to German typographic quotes
    return convert_to_german_quotes(result)
