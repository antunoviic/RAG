"""
RAGAS evaluation comparing LLMChunker vs baseline chunkers.

Prerequisites:
    pip install "llm-chunker[eval]" chromadb ragas sentence-transformers

Usage:
    ollama serve          # in separate terminal
    python examples/ragas_eval.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from llm_chunker import (
    LLMChunker,
    fixed_size_chunk,
    recursive_chunk,
    semantic_chunk,
    sentence_chunk,
    paragraph_chunk,
)

# ---------------------------------------------------------------------------
# Sample corpus + QA pairs (replace with your actual evaluation data)
# ---------------------------------------------------------------------------

CORPUS = """
Machine learning models require large amounts of training data.
The optimization process involves gradient descent to minimize the loss function.
Backpropagation computes gradients efficiently through the network layers.
Overfitting occurs when a model performs well on training data but poorly on unseen data.
Regularization techniques such as dropout and weight decay help prevent overfitting.

Retrieval-Augmented Generation (RAG) combines a retriever with a language model generator.
The retriever fetches relevant documents from a knowledge base given a user query.
The generator then conditions on both the query and the retrieved documents to produce an answer.
RAG reduces hallucinations because the model grounds its response in retrieved evidence.
Dense Passage Retrieval (DPR) is a popular retriever used in RAG systems.

Paris is the capital of France and one of the most visited cities in the world.
The Eiffel Tower was built in 1889 and stands 330 metres tall.
French cuisine is renowned globally for its sophistication and variety.
The Louvre museum houses more than 35,000 works of art including the Mona Lisa.
""".strip()

QA_PAIRS = [
    {
        "question": "What is gradient descent used for in machine learning?",
        "ground_truth": "Gradient descent is used to minimize the loss function during model optimization.",
    },
    {
        "question": "How does RAG reduce hallucinations?",
        "ground_truth": "RAG grounds the model's response in retrieved evidence from a knowledge base.",
    },
    {
        "question": "When was the Eiffel Tower built?",
        "ground_truth": "The Eiffel Tower was built in 1889.",
    },
]

OLLAMA_BASE = "http://127.0.0.1:11434"
MODEL = "qwen3.5:2b"


# ---------------------------------------------------------------------------
# Embedding helper (Ollama)
# ---------------------------------------------------------------------------

def ollama_embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": MODEL, "input": text},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


# ---------------------------------------------------------------------------
# Simple in-memory retriever (cosine similarity)
# ---------------------------------------------------------------------------

import math

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query: str, chunks: list, k: int = 3) -> list[str]:
    q_emb = ollama_embed(query)
    scored = [(cosine_sim(q_emb, ollama_embed(c.text)), c.text) for c in chunks]
    scored.sort(reverse=True)
    return [text for _, text in scored[:k]]


# ---------------------------------------------------------------------------
# Chunker registry
# ---------------------------------------------------------------------------

def get_chunkers(corpus: str) -> dict[str, list]:
    llm_chunker = LLMChunker(
        model=MODEL,
        filter_low_info=True,
        merge_similar=False,   # skip merge for faster eval
        enrich_metadata=False,
    )
    return {
        "llm_chunker":  llm_chunker.chunk(corpus),
        "fixed_100w":   fixed_size_chunk(corpus, size=500, overlap=0),
        "fixed_overlap": fixed_size_chunk(corpus, size=500, overlap=50),
        "sentence":     sentence_chunk(corpus),
        "paragraph":    paragraph_chunk(corpus),
        "recursive":    recursive_chunk(corpus, max_size=500),
        "semantic_pct": semantic_chunk(corpus, ollama_embed,
                                        threshold_type="percentile", threshold_value=90),
        "semantic_std": semantic_chunk(corpus, ollama_embed,
                                        threshold_type="standard_deviation", threshold_value=1.0),
    }


# ---------------------------------------------------------------------------
# Minimal evaluation (without full RAGAS to avoid API key requirement)
# Shows chunk count, avg length, and retrieved context for each QA pair.
# For full RAGAS metrics install ragas and set OPENAI_API_KEY.
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    chunker: str
    num_chunks: int
    avg_chunk_len: float
    retrieved_contexts: list[list[str]] = field(default_factory=list)


def evaluate_chunkers(corpus: str, qa_pairs: list[dict]) -> list[EvalResult]:
    print("Chunking corpus with all strategies...\n")
    chunkers = get_chunkers(corpus)

    results = []
    for name, chunks in chunkers.items():
        avg_len = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        print(f"  {name}: {len(chunks)} chunks, avg {avg_len:.0f} chars")

        contexts = []
        for qa in qa_pairs:
            ctx = retrieve(qa["question"], chunks, k=3)
            contexts.append(ctx)

        results.append(EvalResult(
            chunker=name,
            num_chunks=len(chunks),
            avg_chunk_len=avg_len,
            retrieved_contexts=contexts,
        ))

    return results


def print_report(results: list[EvalResult], qa_pairs: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for r in results:
        print(f"\n[{r.chunker}]  chunks={r.num_chunks}  avg_len={r.avg_chunk_len:.0f} chars")
        for i, qa in enumerate(qa_pairs):
            print(f"  Q: {qa['question'][:60]}...")
            for j, ctx in enumerate(r.retrieved_contexts[i][:1]):  # show top-1
                print(f"    Top-1 context: {ctx[:100]}...")


def ragas_evaluate(results: list[EvalResult], qa_pairs: list[dict]) -> None:
    """Full RAGAS evaluation – requires: pip install ragas openai chromadb."""
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_eval
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("\nInstall ragas for full metrics: pip install 'llm-chunker[eval]' ragas datasets")
        return

    # For answer generation you need a running LLM – using Ollama via OpenAI-compatible API
    # or just skip answer generation and evaluate context quality only.
    print("\n[RAGAS] Running context quality metrics (context_precision, context_recall)...")

    for r in results:
        rows = []
        for i, qa in enumerate(qa_pairs):
            rows.append({
                "question": qa["question"],
                "answer": "",  # fill in with generated answers for full eval
                "contexts": r.retrieved_contexts[i],
                "ground_truth": qa["ground_truth"],
            })
        ds = Dataset.from_list(rows)
        try:
            scores = ragas_eval(ds, metrics=[context_precision, context_recall])
            print(f"  {r.chunker}: context_precision={scores['context_precision']:.3f}  "
                  f"context_recall={scores['context_recall']:.3f}")
        except Exception as e:
            print(f"  {r.chunker}: RAGAS error – {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = evaluate_chunkers(CORPUS, QA_PAIRS)
    print_report(results, QA_PAIRS)
    ragas_evaluate(results, QA_PAIRS)

    print("\nDone. For full RAGAS metrics (faithfulness, answer_relevancy):")
    print("  1. pip install 'llm-chunker[eval]' ragas datasets")
    print("  2. Set OPENAI_API_KEY in your environment")
    print("  3. Fill in 'answer' field with LLM-generated answers")
