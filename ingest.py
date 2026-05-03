"""Run once to ingest the full corpus into ChromaDB."""
from src.config import settings
from src.wiring import build_ingestion_pipeline


def main():
    pipeline = build_ingestion_pipeline()
    result = pipeline.run(
        directories=[
            (settings.CORPUS_REAL_DIR, "pdf"),
            (settings.CORPUS_MOCK_DIR, "md"),
        ]
    )
    print(f"Ingestion complete:")
    print(f"  Documents loaded : {result.docs_loaded}")
    print(f"  Chunks created   : {result.chunks_created}")
    print(f"  Vectors stored   : {result.vectors_stored}")


if __name__ == "__main__":
    main()
