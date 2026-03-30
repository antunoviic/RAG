"""Unit tests – Ollama is mocked, no running server required."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_chunker import LLMChunker
from llm_chunker.chunker import _split_sentences


# --- _split_sentences ---

def test_split_sentences_basic():
    sentences = _split_sentences("Hello world. This is a test. Another one!")
    assert sentences == ["Hello world.", "This is a test.", "Another one!"]


def test_split_sentences_empty():
    assert _split_sentences("") == []


# --- LLMChunker (mocked) ---

def make_chunker(**kwargs) -> LLMChunker:
    return LLMChunker(filter_low_info=False, **kwargs)


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
    chunker = LLMChunker(filter_low_info=True)
    chunker.client = MagicMock()
    chunker.client.generate.side_effect = [
        {"split_after": 0, "reason": "x"},
        {"score": 1, "keep": False},
        {"score": 8, "keep": True},
    ]
    text = "Filler text here. This is very informative content about machine learning."
    result = chunker.chunk(text)
    assert all(c.info_score is None or c.info_score >= chunker.filter_threshold for c in result)
