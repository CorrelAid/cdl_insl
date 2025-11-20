"""Tests for parser module."""

import pytest

from insl.parser import count_verbs, init_parser


@pytest.fixture(scope="module")
def nlp():
    """Initialize parser once for all tests."""
    return init_parser()


def test_count_verbs_simple(nlp):
    """Test verb counting in simple correct sentence."""
    text = "Er baute es auf. Er hat es aufgebaut."
    doc = nlp(text)
    count, verbs = count_verbs(doc)

    assert count == 2
    assert len(verbs) == 2
    assert verbs[0].lemma == "bauen"
    assert verbs[1].lemma == "aufbauen"


def test_count_verbs_multiple(nlp):
    """Test verb counting in sentence with multiple verbs."""
    text = "Der Hund läuft im Park. Er spielt mit einem Ball und bringt ihn zurück."
    doc = nlp(text)
    count, verbs = count_verbs(doc)

    assert count >= 3  # läuft, spielt, bringt (at least)
    lemmas = [v.lemma for v in verbs]
    assert "laufen" in lemmas
    assert "spielen" in lemmas


def test_count_verbs_none(nlp):
    """Test text with no verbs."""
    text = "Der große Hund."
    doc = nlp(text)
    count, verbs = count_verbs(doc)

    assert count == 0
    assert len(verbs) == 0
