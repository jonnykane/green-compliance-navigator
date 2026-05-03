"""CLI: ask a question against the ingested corpus."""
import sys
from src.wiring import build_query_engine


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"Your question here\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    engine = build_query_engine()
    response = engine.ask(question)
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources ({response.chunks_used} chunks used):")
    for s in response.sources:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
