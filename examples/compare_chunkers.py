"""
Interactive comparison of all chunking strategies.

Chunks a sample text with every strategy, stores each in ChromaDB,
then lets you query all of them side-by-side.

Setup:
    pip install chromadb
    ollama serve          # separate terminal
    python examples/compare_chunkers.py
"""
from __future__ import annotations

import textwrap

from llm_chunker import (
    LLMChunker,
    fixed_size_chunk,
    sentence_chunk,
    paragraph_chunk,
    recursive_chunk,
    semantic_chunk,
)
from llm_chunker.vectorstore import ChromaStore

# ---------------------------------------------------------------------------
# Sample corpus – 3 clearly distinct topics to stress-test boundary detection
# ---------------------------------------------------------------------------

CORPUS = """
Machine learning models require large amounts of training data to generalise well.
The optimization process uses gradient descent to iteratively minimize the loss function.
Backpropagation efficiently computes gradients by applying the chain rule through each layer.
Overfitting occurs when a model memorizes training data but fails to generalize to new inputs.
Regularization techniques such as dropout, weight decay, and early stopping combat overfitting.
Transfer learning reuses weights from a pre-trained model, drastically reducing training time.

Retrieval-Augmented Generation (RAG) augments language models with an external knowledge base.
A retriever encodes documents into dense vectors and uses maximum inner-product search to find relevant passages.
The generator conditions on both the user query and the retrieved passages to produce a grounded answer.
RAG reduces hallucinations because responses are anchored in retrieved evidence rather than parametric memory.
Dense Passage Retrieval (DPR) is the bi-encoder retriever used in the original RAG paper by Lewis et al.

Paris is the capital of France and consistently ranks among the most visited cities in the world.
The Eiffel Tower was constructed in 1889 and stands 330 metres tall on the Champ de Mars.
French cuisine is renowned globally and was inscribed on the UNESCO Intangible Cultural Heritage list in 2010.
The Louvre museum houses more than 35,000 works of art, including the Mona Lisa and the Venus de Milo.
""".strip()

MODEL = "qwen3.5:2b"
BASE_URL = "http://127.0.0.1:11434"


# ---------------------------------------------------------------------------
# Build chunker registry
# ---------------------------------------------------------------------------

def build_chunkers(store: ChromaStore) -> dict[str, list]:
    embed = store._embed_batch  # batch embed: list[str] → list[list[float]]

    print("Running LLMChunker (this takes ~1–2 min)...")
    llm = LLMChunker(
        model=MODEL,
        base_url=BASE_URL,
        window_size=5,
        filter_low_info=True,
        merge_similar=True,
        enrich_metadata=True,
    )
    llm_chunks = llm.chunk(CORPUS)

    return {
        "llm_chunker":        llm_chunks,
        "fixed_100w":         fixed_size_chunk(CORPUS, size=500, overlap=0),
        "fixed_llamaindex":   fixed_size_chunk(CORPUS, size=512, overlap=50),
        "sentence":           sentence_chunk(CORPUS),
        "paragraph":          paragraph_chunk(CORPUS),
        "recursive_500":      recursive_chunk(CORPUS, max_size=500),
        "semantic_percentile": semantic_chunk(CORPUS, embed,
                                               threshold_type="percentile",
                                               threshold_value=90),
        "semantic_std":       semantic_chunk(CORPUS, embed,
                                              threshold_type="standard_deviation",
                                              threshold_value=1.0),
    }


# ---------------------------------------------------------------------------
# Store all chunkers in ChromaDB
# ---------------------------------------------------------------------------

def index_all(chunkers: dict[str, list], store: ChromaStore) -> None:
    print("\nIndexing into ChromaDB...")
    for name, chunks in chunkers.items():
        store.add(chunks, collection=name)
        avg = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
        print(f"  {name:<25}  {len(chunks):>3} chunks   avg {avg:>5.0f} chars")


# ---------------------------------------------------------------------------
# Print stats table
# ---------------------------------------------------------------------------

def print_stats(chunkers: dict[str, list]) -> None:
    print("\n" + "═" * 65)
    print(f"{'Strategy':<25} {'#Chunks':>7} {'Avg len':>8} {'Min':>6} {'Max':>6}")
    print("─" * 65)
    for name, chunks in chunkers.items():
        lengths = [len(c.text) for c in chunks]
        avg = sum(lengths) / max(len(lengths), 1)
        print(f"  {name:<23} {len(chunks):>7} {avg:>8.0f} {min(lengths):>6} {max(lengths):>6}")
    print("═" * 65)


# ---------------------------------------------------------------------------
# Show LLMChunker metadata (the unique feature vs baselines)
# ---------------------------------------------------------------------------

def print_llm_chunks(chunks: list) -> None:
    print("\n── LLMChunker chunks (with LLM-generated metadata) ──")
    for c in chunks:
        title = getattr(c, "title", None) or "(no title)"
        kw = ", ".join(getattr(c, "keywords", []) or []) or "—"
        score = getattr(c, "info_score", None)
        score_str = f"  score={score}" if score is not None else ""
        print(f"\n  [{c.index}] {title}{score_str}")
        print(f"       Keywords: {kw}")
        print(f"       {textwrap.shorten(c.text, 100)}")


# ---------------------------------------------------------------------------
# Interactive query loop
# ---------------------------------------------------------------------------

def query_loop(store: ChromaStore, chunkers: dict[str, list]) -> None:
    print("\n" + "═" * 65)
    print("INTERACTIVE QUERY  (type 'quit' to exit)")
    print("═" * 65)

    while True:
        query = input("\nYour query: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        print()
        for name in chunkers:
            results = store.query(query, collection=name, k=1)
            if results:
                top = results[0]
                dist = top["distance"]
                meta_str = ""
                if top["meta"].get("title"):
                    meta_str = f" [{top['meta']['title']}]"
                print(f"  {name:<25} dist={dist:.3f}{meta_str}")
                print(f"    {textwrap.shorten(top['text'], 110)}")
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_ollama() -> None:
    import httpx, sys
    try:
        httpx.get(f"{BASE_URL}/api/tags", timeout=3.0).raise_for_status()
    except Exception:
        print(f"ERROR: Ollama is not running on {BASE_URL}")
        print("Start it with:  ollama serve")
        sys.exit(1)


if __name__ == "__main__":
    _check_ollama()
    store = ChromaStore(model=MODEL, base_url=BASE_URL)

    print("=" * 65)
    print("LLM CHUNKER – Strategy Comparison")
    print("=" * 65)

    chunkers = build_chunkers(store)
    index_all(chunkers, store)
    print_stats(chunkers)
    print_llm_chunks(chunkers["llm_chunker"])
    query_loop(store, chunkers)
