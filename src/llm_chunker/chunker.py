from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .llm_client import OllamaClient
from .prompts import BOUNDARY_PROMPT, FILTER_PROMPT


@dataclass
class Chunk:
    text: str
    index: int
    info_score: Optional[int] = None


class LLMChunker:
    """
    Semantic text chunker using a local LLM via Ollama.

    Pipeline:
    1. Split text into sentences
    2. Sliding window – LLM detects semantic boundaries
    3. Optional filtering – LLM scores information density, removes low-info chunks
    """

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://127.0.0.1:11434",
        window_size: int = 6,
        overlap: int = 1,
        filter_threshold: int = 3,
        filter_low_info: bool = True,
    ):
        self.client = OllamaClient(model=model, base_url=base_url)
        self.window_size = window_size
        self.overlap = overlap
        self.filter_threshold = filter_threshold
        self.filter_low_info = filter_low_info

    def chunk(self, text: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        raw_texts = self._sliding_window(sentences)
        chunks = [Chunk(text=t, index=i) for i, t in enumerate(raw_texts)]

        if self.filter_low_info:
            chunks = self._filter(chunks)

        for i, c in enumerate(chunks):
            c.index = i

        return chunks

    def chunk_pdf(self, path: str) -> list[Chunk]:
        """Read a PDF file and chunk its text content."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return self.chunk(text)

    def _sliding_window(self, sentences: list[str]) -> list[str]:
        chunks = []
        pos = 0
        n = len(sentences)

        while pos < n:
            window = sentences[pos : pos + self.window_size]

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


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]
