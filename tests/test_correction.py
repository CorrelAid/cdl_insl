"""Tests for correction module with multiple LLM providers."""

import os
from unittest.mock import patch

import pytest

from insl.correction import correct_spelling, init_spelling_corrector, tokenize_text, convert_to_german_quotes


@pytest.fixture
def sample_text():
    """Sample German text with errors."""
    return "Der Hunt läufd im Parc."


@pytest.fixture
def sample_tokens():
    """Sample tokenized German text."""
    return ['Der', 'Hunt', 'läufd', 'im', 'Parc', '.']


def test_tokenize_text(sample_text):
    """Test text tokenization."""
    tokens = tokenize_text(sample_text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
    # Stanza separates punctuation as separate tokens
    assert any('.' in t for t in tokens)


@pytest.mark.parametrize("provider,env_vars", [
    ("openrouter", {"OR_KEY": "test-key"}),
    ("modal", {"MODAL_BASE_URL": "https://test.modal.run/v1", "VLLM_API_KEY": "test-key"}),
])
def test_init_spelling_corrector_providers(provider, env_vars):
    """Test corrector initialization with different providers."""
    with patch.dict(os.environ, env_vars, clear=False):
        corrector = init_spelling_corrector(provider=provider)

        assert corrector is not None
        # Check that the corrector is a DSPy compiled program
        assert hasattr(corrector, '__call__')


def test_init_spelling_corrector_missing_key():
    """Test that missing API key raises an error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key not provided"):
            init_spelling_corrector(provider="openrouter")


def test_init_spelling_corrector_defaults():
    """Test default provider (openrouter)."""
    # Test with OR_KEY set
    with patch.dict(os.environ, {"OR_KEY": "test-key"}, clear=True):
        corrector = init_spelling_corrector()  # defaults to openrouter
        assert corrector is not None


def test_init_spelling_corrector_custom_missing_api_base():
    """Test that custom provider requires api_base."""
    with pytest.raises(ValueError, match="api_base required"):
        init_spelling_corrector(provider="custom")


def test_init_spelling_corrector_invalid_provider():
    """Test that invalid provider raises an error."""
    with pytest.raises(ValueError, match="Unknown provider"):
        init_spelling_corrector(provider="invalid")


def test_convert_to_german_quotes():
    """Test ASCII to German quote conversion."""
    # Basic conversion
    assert convert_to_german_quotes('Er sagte: "Hallo"') == 'Er sagte: „Hallo"'

    # Multiple quotes
    assert convert_to_german_quotes('"Ja" und "Nein"') == '„Ja" und „Nein"'

    # No quotes
    assert convert_to_german_quotes('Kein Zitat') == 'Kein Zitat'


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OR_KEY"),
    reason="OR_KEY not set - skipping integration test"
)
def test_correct_spelling_openrouter_integration(sample_text):
    """Integration test with OpenRouter (requires OR_KEY in env)."""
    corrector = init_spelling_corrector(provider="openrouter")

    corrected = correct_spelling(sample_text, corrector)

    assert isinstance(corrected, str)
    assert len(corrected) > 0
    # Corrected text should be different from original
    assert corrected != sample_text


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("MODAL_BASE_URL") and os.getenv("VLLM_API_KEY")),
    reason="MODAL_BASE_URL or VLLM_API_KEY not set - skipping integration test"
)
def test_correct_spelling_modal_integration(sample_text):
    """Integration test with Modal (requires MODAL_BASE_URL and VLLM_API_KEY in env)."""
    corrector = init_spelling_corrector(provider="modal")

    corrected = correct_spelling(sample_text, corrector)

    assert isinstance(corrected, str)
    assert len(corrected) > 0


def test_correct_spelling_with_tokens(sample_tokens):
    """Test correction with token list input."""
    # Mock the corrector to avoid actual API call
    class MockCorrector:
        def __call__(self, tokens):
            class Result:
                corrected = tokens  # Just return same tokens
            return Result()

    corrector = MockCorrector()
    corrected = correct_spelling(sample_tokens, corrector)

    assert isinstance(corrected, str)
    assert len(corrected) > 0


def test_correct_spelling_with_string(sample_text):
    """Test correction with string input (auto-tokenizes)."""
    class MockCorrector:
        def __call__(self, tokens):
            class Result:
                corrected = tokens
            return Result()

    corrector = MockCorrector()
    corrected = correct_spelling(sample_text, corrector)

    assert isinstance(corrected, str)
    assert len(corrected) > 0
