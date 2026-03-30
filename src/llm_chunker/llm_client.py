from __future__ import annotations

import json
import re

import httpx


class OllamaClient:
    def __init__(self, model: str = "qwen3.5:2b", base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, retries: int = 2) -> dict:
        """Call Ollama generate and extract JSON from the response."""
        last_exc: Exception | None = None
        for _ in range(retries + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=60.0,
                )
                resp.raise_for_status()
                text = resp.json()["response"]
                match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except Exception as e:
                last_exc = e
        raise RuntimeError(f"Ollama generate failed after {retries + 1} attempts") from last_exc

    def embed(self, text: str) -> list[float]:
        """Get text embedding via Ollama."""
        resp = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
