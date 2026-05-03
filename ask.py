"""CLI: ask a question against the ingested corpus."""
import sys
from src.classifier.classifier import CLARIFICATION_OPTIONS, CONTEXT_FROM_OPTION
from src.wiring import build_query_engine


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    engine = build_query_engine()
    result = engine.ask(question)

    if result.kind == "out_of_scope":
        print(f"\n{result.answer}")
        return

    if result.kind == "clarification_needed":
        print(f"\n{result.clarification_question}\n")
        for i, option in enumerate(CLARIFICATION_OPTIONS, 1):
            print(f"  {i}. {option}")

        raw = input("\nEnter number (1-6): ").strip()
        try:
            choice = int(raw) - 1
            if choice not in CONTEXT_FROM_OPTION:
                raise ValueError
        except ValueError:
            print("Invalid choice. Please run again and enter a number between 1 and 6.")
            sys.exit(1)

        company_context = CONTEXT_FROM_OPTION[choice]
        result = engine.ask(question, company_context=company_context)

    # kind == "answer" (reached directly or after clarification)
    print(f"\nAnswer:\n{result.answer}")
    if result.sources:
        print(f"\nSources ({len(result.sources)} cited):")
        for s in result.sources:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
