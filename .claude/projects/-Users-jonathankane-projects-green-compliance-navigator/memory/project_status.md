---
name: Green Compliance Navigator — ingestion pipeline status
description: What has been built and what comes next in the RAG pipeline project
type: project
---

Ingestion pipeline (steps 1–7) is complete and all 59 tests pass.

Built modules:
- `src/config/settings.py` — env-var config with dotenv
- `src/loader/document_loader.py` — PDF (pdfplumber) + markdown loader, `Document` dataclass
- `src/chunker/chunker.py` — section-aware markdown chunker + paragraph-merge PDF chunker, with token-split overflow and overlap
- `src/embedder/embedding_client.py` — Voyage AI wrapper (`embed_texts` / `embed_query`), batching, protocol-based DI
- `src/vector_store/vector_store_client.py` — ChromaDB wrapper, `add_chunks` / `query` / `count`, protocol-based DI
- `src/pipeline/ingestion_pipeline.py` — composes all above, returns `IngestionResult`

**Why:** User said to stop before retrieval/generation and check in first.

**How to apply:** Next session should ask user to confirm scope before building retrieval (`src/retriever/`) and generation (`src/generator/`) layers. API keys (VOYAGE_API_KEY, ANTHROPIC_API_KEY) still needed for any live run.
