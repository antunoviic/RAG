from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .llm_client import OllamaClient
from .prompts import BOUNDARY_PROMPT, FILTER_PROMPT, METADATA_PROMPT


@dataclass
class Chunk:
    text: str
    index: int
    title: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    info_score: Optional[int] = None


class LLMChunker:
    """
    Semantic text chunker using a local LLM via Ollama.

    Uses a sliding context window to detect natural semantic boundaries,
    then optionally filters low-information chunks, merges similar ones,
    and enriches each chunk with title and keywords.
    """

    def __init__(
        self,
        model: str = "qwen3.5:2b",
        base_url: str = "http://127.0.0.1:11434",
        window_size: int = 6,
        overlap: int = 1,
        filter_threshold: int = 3,
        merge_threshold: float = 0.88,
        filter_low_info: bool = True,
        merge_similar: bool = True,
        enrich_metadata: bool = True,
    ):
        self.client = OllamaClient(model=model, base_url=base_url)
        self.window_size = window_size
        self.overlap = overlap
        self.filter_threshold = filter_threshold
        self.merge_threshold = merge_threshold
        self.filter_low_info = filter_low_info
        self.merge_similar = merge_similar
        self.enrich_metadata = enrich_metadata

    def chunk(self, text: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        raw_texts = self._sliding_window(sentences)
        chunks = [Chunk(text=t, index=i) for i, t in enumerate(raw_texts)]

        if self.filter_low_info:
            chunks = self._filter(chunks)

        if self.merge_similar:
            chunks = self._merge(chunks)

        if self.enrich_metadata:
            chunks = self._enrich(chunks)

        for i, c in enumerate(chunks):
            c.index = i

        return chunks

    def _sliding_window(self, sentences: list[str]) -> list[str]:
        chunks = []
        pos = 0
        n = len(sentences)

        while pos < n:
            window = sentences[pos : pos + self.window_size]

            # Last window or too small – take everything remaining
            if len(window) <= 2 or pos + len(window) >= n:
                chunks.append(" ".join(sentences[pos:]))
                break

            formatted = "\n".join(f"[{i}] {s}" for i, s in enumerate(window))
            prompt = BOUNDARY_PROMPT.format(sentences=formatted)

            try:
                result = self.client.generate(prompt)
                split_at = int(result.get("split_after", len(window) - 2))
                split_at = max(0, min(split_at, len(window) - 2))
            except Exception:
                split_at = len(window) - 2

            chunks.append(" ".join(sentences[pos : pos + split_at + 1]))
            next_pos = pos + split_at + 1 - self.overlap
            pos = max(pos + 1, next_pos)

        return [c for c in chunks if c.strip()]

    def _filter(self, chunks: list[Chunk]) -> list[Chunk]:
        kept = []
        for chunk in chunks:
            prompt = FILTER_PROMPT.format(chunk=chunk.text[:500])
            try:
                result = self.client.generate(prompt)
                score = int(result.get("score", 5))
                chunk.info_score = score
                if score >= self.filter_threshold:
                    kept.append(chunk)
            except Exception:
                kept.append(chunk)
        return kept

    def _merge(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) <= 1:
            return chunks
        try:
            embeddings = [self.client.embed(c.text) for c in chunks]
        except Exception:
            return chunks

        merged = [chunks[0]]
        merged_embeddings = [embeddings[0]]

        for i in range(1, len(chunks)):
            sim = _cosine_similarity(merged_embeddings[-1], embeddings[i])
            if sim >= self.merge_threshold:
                prev = merged[-1]
                prev.text = prev.text + " " + chunks[i].text
                # Recompute embedding for merged chunk (averaged approximation)
                avg = [(a + b) / 2 for a, b in zip(merged_embeddings[-1], embeddings[i])]
                merged_embeddings[-1] = avg
            else:
                merged.append(chunks[i])
                merged_embeddings.append(embeddings[i])

        return merged

    def _enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            prompt = METADATA_PROMPT.format(chunk=chunk.text[:500])
            try:
                result = self.client.generate(prompt)
                chunk.title = result.get("title")
                chunk.keywords = result.get("keywords", [])
            except Exception:
                pass
        return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
