"""
PDF chunking example – requires Ollama running with qwen3.5:2b.

  ollama serve
  ollama pull qwen3.5:2b
  python examples/pdf_usage.py path/to/file.pdf
"""

import sys
from pypdf import PdfReader
from llm_chunker import LLMChunker


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/pdf_usage.py path/to/file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Reading: {pdf_path}")

    text = extract_text(pdf_path)
    print(f"Extracted {len(text)} characters from PDF\n")

    chunker = LLMChunker(
        model="qwen3.5:2b",
        window_size=5,
        filter_low_info=True,
        merge_similar=True,
        enrich_metadata=True,
    )

    print("Chunking... (this may take a while)\n")
    chunks = chunker.chunk(text)

    print(f"=== {len(chunks)} Chunks ===\n")
    for c in chunks:
        print(f"[Chunk {c.index}] {c.title or '(no title)'}")
        print(f"  Keywords : {c.keywords}")
        print(f"  Score    : {c.info_score}")
        print(f"  Text     : {c.text[:200]}...")
        print()


if __name__ == "__main__":
    main()
