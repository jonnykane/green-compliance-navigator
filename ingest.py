"""Ingest the full corpus into the configured vector store (Pinecone or ChromaDB)."""
import os

from src.config import settings
from src.wiring import build_ingestion_pipeline


def main():
    store_name = "Pinecone" if os.getenv("PINECONE_API_KEY") else "ChromaDB"
    print(f"Ingesting to {store_name}...")
    pipeline = build_ingestion_pipeline()
    result = pipeline.run(
        directories=[
            (settings.CORPUS_REAL_DIR, "pdf"),
            (settings.CORPUS_MOCK_DIR, "md"),
        ]
    )
    print(f"Ingestion complete:")
    print(f"  Store            : {store_name}")
    print(f"  Documents loaded : {result.docs_loaded}")
    print(f"  Chunks created   : {result.chunks_created}")
    print(f"  Vectors stored   : {result.vectors_stored}")


if __name__ == "__main__":
    main()
