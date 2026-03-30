# LLM Chunker – Bachelorthesis

Supervisiert von Dr. Marian Lux, Universität Wien – Thema: AI 130261
Semantic Chunking mit LLMs für RAG – als Python-Library auf PyPI publiziert.

## Kernfeatures
1. Sliding-Window-Chunking via LLM (Hauptbeitrag)
2. Low-information chunk filtering via LLM
3. Similar chunk merging via Cosine-Similarity (Embeddings)
4. Metadata enrichment (title, keywords) via LLM

## Baseline-Vergleich (für Evaluation)
Implementiert in `baselines.py`:
- `fixed_size_chunk` – Lewis et al. (2020) nutzen 100-Wort-Chunks auf Wikipedia
- `sentence_chunk` – ein Satz pro Chunk
- `paragraph_chunk` – ein Absatz pro Chunk
- `recursive_chunk` – Separator-Hierarchie (wie LangChain RecursiveCharacterTextSplitter)
- `semantic_chunk` – Embedding-basiert mit 3 Threshold-Typen:
  - `percentile` (Standard): Split wo Abstand > X. Perzentile aller Abstände
  - `standard_deviation`: Split wo Abstand > mean + X × std
  - `interquartile`: Split wo Abstand > Q3 + X × IQR
  Implementiert mit Buffer-Sätzen (wie LangChain SemanticChunker)

Unser LLMChunker entspricht dem "Agentic Chunking" in der Literatur (Technik 15/15).

## Setup
- Ollama lokal auf `http://127.0.0.1:11434`
- Modell: `qwen3.5:2b` (getestet, liefert zuverlässig valides JSON)
- M1 MacBook Air 8GB RAM

## Projektstruktur
```
src/llm_chunker/
  chunker.py     # LLMChunker Klasse (Haupt-API)
  llm_client.py  # Ollama HTTP-Wrapper (httpx)
  prompts.py     # Prompt-Templates
  baselines.py   # Fixed-size, Recursive, Semantic-Chunking
tests/
examples/
```

## Evaluation
- Framework: RAGAS – Metriken: `context_precision`, `faithfulness`, `answer_relevancy`, `context_recall`
- Eval-Script: `examples/ragas_eval.py`
- Vektordatenbank: ChromaDB oder Qdrant (TBD)
- Hinweis aus Literatur: Semantic ≈ Naive Chunking bei RAGAS-Scores (Nayak 2024) → Mehrwert von LLM-Chunking muss gezielt gezeigt werden

## Schlüsselreferenzen
- Lewis et al. (2020) – RAG Paper (NeurIPS): Wikipedia → 100-Wort-Chunks, DPR + BART
- LlamaIndex Docs – Default: chunk_size=1024 Tokens, overlap=20; Regel: chunk_size/2 → top_k×2
- Nayak (2024) – Semantic Chunking for RAG: LangChain SemanticChunker, RAGAS-Eval
- Singh (2025) – 15 RAG Chunking Techniques: Taxonomie, "Agentic Chunking" = unser Ansatz
