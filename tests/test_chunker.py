"""Unit tests – Ollama is mocked, no running server required."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_chunker import (
    LLMChunker,
    fixed_size_chunk,
    sentence_chunk,
    paragraph_chunk,
    recursive_chunk,
    semantic_chunk,
)
from llm_chunker.chunker import _split_sentences, _cosine_similarity


# --- _split_sentences ---

def test_split_sentences_basic():
    sentences = _split_sentences("Hello world. This is a test. Another one!")
    assert sentences == ["Hello world.", "This is a test.", "Another one!"]


def test_split_sentences_empty():
    assert _split_sentences("") == []


# --- _cosine_similarity ---

def test_cosine_similarity_identical():
    v = [1.0, 0.0, 1.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- LLMChunker (mocked) ---

def make_chunker(**kwargs) -> LLMChunker:
    chunker = LLMChunker(filter_low_info=False, merge_similar=False, enrich_metadata=False, **kwargs)
    return chunker


def test_chunk_returns_list():
    chunker = make_chunker()
    chunker.client = MagicMock()
    chunker.client.generate.return_value = {"split_after": 1, "reason": "topic change"}

    text = "Machine learning needs data. Paris is in France. Water boils at 100 degrees."
    result = chunker.chunk(text)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(c.text for c in result)


def test_chunk_indices_sequential():
    chunker = make_chunker()
    chunker.client = MagicMock()
    chunker.client.generate.return_value = {"split_after": 0, "reason": "x"}

    text = ". ".join([f"Sentence {i}" for i in range(10)]) + "."
    result = chunker.chunk(text)
    for i, c in enumerate(result):
        assert c.index == i


def test_chunk_empty_text():
    chunker = make_chunker()
    assert chunker.chunk("") == []


def test_filter_removes_low_score():
    chunker = LLMChunker(filter_low_info=True, merge_similar=False, enrich_metadata=False)
    chunker.client = MagicMock()
    # boundary detection
    chunker.client.generate.side_effect = [
        {"split_after": 0, "reason": "x"},  # boundary
        {"score": 1, "keep": False},         # filter: drop
        {"score": 8, "keep": True},          # filter: keep
    ]
    text = "Filler text here. This is very informative content about machine learning."
    result = chunker.chunk(text)
    assert all(c.info_score is None or c.info_score >= chunker.filter_threshold for c in result)


def test_enrich_sets_title_and_keywords():
    # Use a single short text – window_size=6 takes it all as one chunk (no boundary call),
    # then enrich calls generate once for metadata.
    chunker = LLMChunker(filter_low_info=False, merge_similar=False, enrich_metadata=True)
    chunker.client = MagicMock()
    chunker.client.generate.return_value = {"title": "ML Basics", "keywords": ["ml", "data", "training"]}
    text = "ML models need data. They are trained with gradient descent."
    result = chunker.chunk(text)
    assert len(result) == 1
    assert result[0].title == "ML Basics"
    assert result[0].keywords == ["ml", "data", "training"]


# --- Baselines ---

def test_fixed_size_chunk():
    text = "a" * 1200
    chunks = fixed_size_chunk(text, size=500, overlap=50)
    assert len(chunks) == 3
    assert all(len(c.text) <= 500 for c in chunks)


def test_fixed_size_chunk_indices():
    chunks = fixed_size_chunk("x" * 600, size=500, overlap=0)
    assert [c.index for c in chunks] == [0, 1]


def test_recursive_chunk_respects_max_size():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = recursive_chunk(text, max_size=30)
    assert all(len(c.text) <= 30 for c in chunks)


def test_sentence_chunk():
    text = "Hello world. This is a test. Another sentence!"
    chunks = sentence_chunk(text)
    assert len(chunks) == 3
    assert chunks[0].text == "Hello world."
    assert chunks[2].text == "Another sentence!"


def test_paragraph_chunk():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird."
    chunks = paragraph_chunk(text)
    assert len(chunks) == 3
    assert chunks[0].text == "First paragraph here."


def test_semantic_chunk_percentile():
    """Semantic chunking with percentile threshold – splits at the largest distance gap."""
    call_count = 0

    def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        # groups 1-2 are similar (ML topic), group 3 is very different (Paris)
        if call_count <= 2:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    text = "ML needs data. Gradient descent optimizes models. Paris is in France."
    # With percentile=50, the split happens at the largest distance
    chunks = semantic_chunk(text, embed_fn=fake_embed,
                             threshold_type="percentile", threshold_value=50)
    assert len(chunks) >= 1


def test_semantic_chunk_std():
    """standard_deviation threshold type runs without error."""
    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0]

    text = "Sentence A. Sentence B. Sentence C."
    chunks = semantic_chunk(text, embed_fn=fake_embed,
                             threshold_type="standard_deviation", threshold_value=1.0)
    assert len(chunks) >= 1


def test_semantic_chunk_interquartile():
    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0]

    text = "Sentence A. Sentence B. Sentence C."
    chunks = semantic_chunk(text, embed_fn=fake_embed,
                             threshold_type="interquartile", threshold_value=1.5)
    assert len(chunks) >= 1


def test_semantic_chunk_invalid_threshold_type():
    import pytest
    with pytest.raises(ValueError, match="Unknown threshold_type"):
        semantic_chunk("A. B.", embed_fn=lambda t: [1.0],
                       threshold_type="invalid", threshold_value=1.0)
