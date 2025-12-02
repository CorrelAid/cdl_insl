"""Tests for correction module with multiple LLM providers."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_init_spelling_corrector_thread_safety():
    """Test that init_spelling_corrector is thread-safe and returns singleton."""
    with patch.dict(os.environ, {"OR_KEY": "test-key"}, clear=False):
        # Reset the global corrector to test fresh initialization
        import insl.correction
        insl.correction._corrector = None

        results = []
        errors = []

        def init_corrector():
            try:
                corrector = init_spelling_corrector(provider="openrouter")
                results.append(id(corrector))  # Store object ID
            except Exception as e:
                errors.append(e)

        # Run 10 threads trying to initialize simultaneously
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=init_corrector)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All threads should get the same corrector instance (same ID)
        assert len(results) == 10
        assert len(set(results)) == 1, "Multiple corrector instances created"


def test_init_spelling_corrector_concurrent_calls():
    """Test concurrent corrector initialization using ThreadPoolExecutor."""
    with patch.dict(os.environ, {"OR_KEY": "test-key"}, clear=False):
        # Reset the global corrector
        import insl.correction
        insl.correction._corrector = None

        def get_corrector(_):
            return init_spelling_corrector(provider="openrouter")

        # Use ThreadPoolExecutor to run 20 concurrent initializations
        with ThreadPoolExecutor(max_workers=20) as executor:
            correctors = list(executor.map(get_corrector, range(20)))

        # All should be the same instance
        corrector_ids = [id(c) for c in correctors]
        assert len(set(corrector_ids)) == 1, "Multiple corrector instances created concurrently"

        # All should be callable
        assert all(hasattr(c, '__call__') for c in correctors)
