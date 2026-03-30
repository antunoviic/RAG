"""
Baseline chunking methods for comparison against LLMChunker.

Strategies (matching the taxonomy from the literature):
- fixed_size_chunk:   Split by character count with overlap (Lewis et al. 2020 use 100-word chunks)
- sentence_chunk:     One sentence per chunk
- paragraph_chunk:    One paragraph per chunk
- recursive_chunk:    Split by separator hierarchy (LangChain RecursiveCharacterTextSplitter)
- semantic_chunk:     Embedding-based boundary detection with configurable threshold types
                      (mirrors LangChain SemanticChunker: percentile / standard_deviation / interquartile)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Chunk:
    text: str
    index: int


# ---------------------------------------------------------------------------
# Reference configurations from the literature
# ---------------------------------------------------------------------------

# LlamaIndex defaults (developers.llamaindex.ai/python/framework/optimizing/basic_strategies)
LLAMAINDEX_CHUNK_SIZE = 1024    # tokens  (~4 chars/token → ~4096 chars)
LLAMAINDEX_CHUNK_OVERLAP = 20   # tokens
LLAMAINDEX_DEFAULT_TOP_K = 2    # similarity_top_k; double it when halving chunk size

# Lewis et al. (2020) RAG paper: Wikipedia split into disjoint 100-word chunks
RAG_PAPER_CHUNK_WORDS = 100     # ≈ 500 chars at ~5 chars/word


# ---------------------------------------------------------------------------
# Simple structural baselines
# ---------------------------------------------------------------------------

def fixed_size_chunk(text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split by character count with optional overlap.

    The original RAG paper (Lewis et al., 2020) uses disjoint 100-word chunks
    on Wikipedia – use size≈500 chars / overlap=0 to approximate that.
    """
    chunks: list[Chunk] = []
    start = 0
    i = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(text=text[start:end], index=i))
        i += 1
        if end == len(text):
            break
        start += size - overlap
    return chunks


def sentence_chunk(text: str) -> list[Chunk]:
    """One sentence per chunk – simplest possible semantic unit."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [Chunk(text=s, index=i) for i, s in enumerate(sentences) if s.strip()]


def paragraph_chunk(text: str) -> list[Chunk]:
    """One paragraph per chunk – preserves author's logical structure."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [Chunk(text=p.strip(), index=i) for i, p in enumerate(paragraphs) if p.strip()]


def recursive_chunk(
    text: str,
    max_size: int = 500,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """Hierarchical splitting by separator priority (like LangChain's
    RecursiveCharacterTextSplitter).  Tries \\n\\n first, then \\n, then
    ". ", then " ", until every chunk fits max_size.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    parts = _recursive_split(text, max_size, separators)
    return [Chunk(text=p, index=i) for i, p in enumerate(parts) if p.strip()]


# ---------------------------------------------------------------------------
# Semantic chunking (embedding-based) – mirrors LangChain SemanticChunker
# ---------------------------------------------------------------------------

def semantic_chunk(
    text: str,
    embed_fn: Callable,  # (str) -> list[float]  OR  (list[str]) -> list[list[float]]
    threshold_type: str = "percentile",
    threshold_value: float = 95.0,
    buffer_size: int = 1,
) -> list[Chunk]:
    """Embedding-based semantic chunking.

    Algorithm (inspired by LangChain SemanticChunker):
    1. Split into sentences.
    2. For each sentence, build a *group* = sentence ± buffer_size neighbours.
    3. Embed each group (gives richer context than single sentences).
    4. Compute cosine *distance* between consecutive groups.
    5. Find a threshold using threshold_type / threshold_value.
    6. Split wherever the distance exceeds the threshold.

    threshold_type options:
    - "percentile"        : split where distance > percentile(distances, threshold_value)
                            threshold_value = 95 → only the top-5% gaps cause splits
    - "standard_deviation": split where distance > mean + threshold_value × std_dev
                            threshold_value = 1.0 is a common starting point
    - "interquartile"     : split where distance > Q3 + threshold_value × IQR
                            threshold_value = 1.5 matches the "outlier fence" rule
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, index=0)]

    # Build buffered groups for richer embeddings
    groups = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        groups.append(" ".join(sentences[start:end]))

    # Try batch embed first (list[str] → list[list[float]]), fall back to single
    try:
        embeddings = embed_fn(groups)
        if not isinstance(embeddings[0], list):
            raise TypeError
    except (TypeError, KeyError):
        embeddings = [embed_fn(g) for g in groups]

    # Cosine *distance* (0 = identical, 1 = orthogonal)
    distances = [
        1.0 - _cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    threshold = _calc_threshold(distances, threshold_type, threshold_value)

    chunks: list[str] = []
    start = 0
    for i, dist in enumerate(distances):
        if dist > threshold:
            chunks.append(" ".join(sentences[start : i + 1]))
            start = i + 1
    chunks.append(" ".join(sentences[start:]))

    return [Chunk(text=c, index=i) for i, c in enumerate(chunks) if c.strip()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_threshold(distances: list[float], threshold_type: str, value: float) -> float:
    if not distances:
        return 0.0

    sorted_d = sorted(distances)
    n = len(sorted_d)

    if threshold_type == "percentile":
        idx = min(int(n * value / 100), n - 1)
        return sorted_d[idx]

    elif threshold_type == "standard_deviation":
        mean = sum(distances) / n
        std = math.sqrt(sum((d - mean) ** 2 for d in distances) / n)
        return mean + value * std

    elif threshold_type == "interquartile":
        q1 = sorted_d[n // 4]
        q3 = sorted_d[(3 * n) // 4]
        iqr = q3 - q1
        return q3 + value * iqr

    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type!r}. "
                         "Choose 'percentile', 'standard_deviation', or 'interquartile'.")


def _recursive_split(text: str, max_size: int, separators: list[str]) -> list[str]:
    if len(text) <= max_size or not separators:
        return [text]

    sep = separators[0]
    parts = text.split(sep) if sep else list(text)

    result: list[str] = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                result.extend(_recursive_split(current, max_size, separators[1:]))
            current = part
    if current:
        result.extend(_recursive_split(current, max_size, separators[1:]))
    return result


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
