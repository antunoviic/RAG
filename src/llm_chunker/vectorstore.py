"""
ChromaDB vector store wrapper for storing and retrieving chunks.

Chunks from LLMChunker include title/keywords as metadata.
Baseline chunks are stored with basic metadata.

Usage:
    store = ChromaStore(model="qwen3.5:2b")
    store.add(chunks, collection="llm_chunker")
    results = store.query("What is gradient descent?", collection="llm_chunker", k=3)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass


class ChromaStore:
    def __init__(
        self,
        model: str = "qwen3.5:2b",
        base_url: str = "http://127.0.0.1:11434",
        persist_path: str | None = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install 'llm-chunker[eval]'")

        self.model = model
        self.base_url = base_url.rstrip("/")

        if persist_path:
            self._client = chromadb.PersistentClient(path=persist_path)
        else:
            self._client = chromadb.Client()

    # ------------------------------------------------------------------

    def add(self, chunks: list, collection: str) -> None:
        """Embed chunks and store in a named collection."""
        # Drop existing collection so reruns are clean
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass

        col = self._client.create_collection(collection)

        texts = [c.text for c in chunks]
        embeddings = self._embed_batch(texts)

        metadatas, ids = [], []
        for i, chunk in enumerate(chunks):
            ids.append(f"{collection}_{i}")
            meta: dict = {"index": chunk.index}
            if hasattr(chunk, "title") and chunk.title:
                meta["title"] = chunk.title
            if hasattr(chunk, "keywords") and chunk.keywords:
                meta["keywords"] = ", ".join(chunk.keywords)
            if hasattr(chunk, "info_score") and chunk.info_score is not None:
                meta["info_score"] = chunk.info_score
            metadatas.append(meta)

        col.add(documents=texts, embeddings=embeddings,
                metadatas=metadatas, ids=ids)

    def query(
        self, text: str, collection: str, k: int = 3
    ) -> list[dict]:
        """Return top-k results as list of {text, distance, metadata}."""
        col = self._client.get_collection(collection)
        q_emb = self._embed(text)
        results = col.query(query_embeddings=[q_emb], n_results=min(k, col.count()))

        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            output.append({"text": doc, "distance": dist, "meta": meta})
        return output

    def collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]

    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single Ollama request."""
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120.0,  # batch can take time on M1 CPU
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _embed(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
