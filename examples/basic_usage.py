"""
Basic usage example – requires Ollama running with qwen3.5:2b.

  ollama serve          # in one terminal
  ollama pull qwen3.5:2b
  python examples/basic_usage.py
"""

from llm_chunker import LLMChunker, fixed_size_chunk, recursive_chunk

SAMPLE_TEXT = """
Machine learning models require large amounts of training data.
The optimization process involves gradient descent to minimize the loss function.
Backpropagation computes gradients efficiently through the network layers.

Paris is the capital of France and one of the most visited cities in the world.
The Eiffel Tower was built in 1889 and stands 330 meters tall.
French cuisine is renowned globally for its sophistication and variety.

Water boils at 100 degrees Celsius at sea level.
The boiling point decreases at higher altitudes due to lower atmospheric pressure.
Steam engines exploited this property to power the industrial revolution.
""".strip()


def main():
    print("=== LLM Chunker ===")
    chunker = LLMChunker(
        model="qwen3.5:2b",
        window_size=5,
        filter_low_info=True,
        merge_similar=True,
        enrich_metadata=True,
    )
    chunks = chunker.chunk(SAMPLE_TEXT)
    for c in chunks:
        print(f"\n[Chunk {c.index}] {c.title or '(no title)'}")
        print(f"  Keywords: {c.keywords}")
        print(f"  Score:    {c.info_score}")
        print(f"  Text:     {c.text[:120]}...")

    print("\n\n=== Baselines ===")

    print("\n-- Fixed-size (500 chars, 50 overlap) --")
    for c in fixed_size_chunk(SAMPLE_TEXT, size=200, overlap=20):
        print(f"  [{c.index}] {c.text[:80]}...")

    print("\n-- Recursive (max 200 chars) --")
    for c in recursive_chunk(SAMPLE_TEXT, max_size=200):
        print(f"  [{c.index}] {c.text[:80]}...")


if __name__ == "__main__":
    main()
