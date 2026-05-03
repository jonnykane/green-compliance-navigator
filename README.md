# UK Green Compliance Navigator

A RAG-powered assistant that helps UK businesses understand what green regulations apply to them, what those regulations require, and when deadlines fall — with every answer cited back to the specific source document.

---

## The problem

UK green regulation is fragmented across multiple frameworks — SECR, ESOS, TCFD, CSRD, PPN 06/21, and UK SDS incoming. Each has different thresholds, different deadlines, and different issuing bodies. The question "what do I actually have to do, and by when?" currently requires reading across 15+ government documents and understanding how the thresholds interact.

A company with 300 employees and £45m turnover sits differently under each framework. SECR uses a two-of-three criteria test. ESOS uses a different OR logic. FCA TCFD applies to listed companies regardless of size. CSRD only triggers via EU presence. Figuring this out is the problem this tool solves.

---

## What it does

Ask a plain-English question. Get a cited answer.

```
$ python ask.py "We have 300 employees and £45m turnover. What green reporting applies to us?"

To give you an accurate answer, which description is closest to your organisation?

  1. Large UK company — 250+ employees, or high turnover/balance sheet
  2. Listed or FCA-regulated — listed company, asset manager, insurer, or pension provider
  3. UK SME — under 250 employees and below large-company thresholds
  4. UK supplier bidding for public sector contracts
  5. UK company with significant EU operations — EU subsidiary, branch, or major EU turnover
  6. Not sure

Enter number (1-6): 1

Answer:
Based on your company profile, the following reporting frameworks apply:

**SECR (Streamlined Energy and Carbon Reporting):** You meet two of three
large company criteria (250+ employees and £36m+ turnover), so SECR applies.
You must report UK energy use, Scope 1 and 2 GHG emissions, at least one
intensity metric, and energy efficiency actions in your Directors' Report.
(secr_environmental_reporting_guidelines_2019.pdf)

**ESOS (Energy Savings Opportunity Scheme):** Your employee count of 300
exceeds the 250-employee threshold. An ESOS energy audit is required every
four years. Phase 4 deadline is 5 December 2027.
(esos_overview.md)
```

Every answer cites the specific source document. The tool flags when regulations are incoming or when guidance may have changed.

---

## Architecture

This project implements a RAG (Retrieval-Augmented Generation) pipeline — a pattern where an LLM generates answers grounded in retrieved documents rather than its training data alone. This is the right architecture for compliance questions because:

- **The corpus is too large for a single context window.** TCFD guidance alone is 74 pages. SECR is ~80 pages. Combined they exceed a single prompt.
- **Answers often span multiple regulations.** "What do we have to do?" may require chunks from SECR, ESOS, and TCFD simultaneously. Retrieval enables multi-document synthesis.
- **Every answer must be traceable.** RAG provides citations back to specific source documents. Hallucinated compliance advice is harmful — citations let users verify.

### Pipeline

```
User question
      │
      ▼
┌─────────────────┐
│   Classifier    │  Determines: clear / needs_clarification /
│   (Claude)      │  partial_answer_needs_clarification / out_of_scope
└────────┬────────┘
         │
    ┌────┴─────────────────────────────────┐
    │                                      │
    ▼                                      ▼
Clarification                        Retrieval
question returned                    query built
(if needed)                          (question + company context)
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │  Voyage AI  │  Embed query → vector
                                   │ (voyage-3)  │
                                   └──────┬──────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │  ChromaDB   │  Find nearest chunks
                                   │             │
                                   └──────┬──────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │   Claude    │  Generate cited answer
                                   │  Generator  │  from retrieved chunks
                                   └──────┬──────┘
                                          │
                                          ▼
                                     QueryResult
                                  (answer + sources)
```

### Module structure

| Module | Responsibility |
|--------|---------------|
| `src/loader/` | PDF extraction (pdfplumber) and markdown reading |
| `src/chunker/` | Section-aware chunking that respects regulatory document structure |
| `src/embedder/` | Voyage AI client wrapper (voyage-3) |
| `src/vector_store/` | ChromaDB client wrapper |
| `src/classifier/` | Pre-retrieval classifier — determines routing state |
| `src/retriever/` | Embeds query, queries ChromaDB, returns ranked chunks |
| `src/generator/` | Builds cited prompt, calls Claude, returns structured answer |
| `src/query.py` | `QueryEngine.ask()` — single public interface |
| `src/wiring.py` | Factory functions wiring real clients via dependency injection |
| `src/models.py` | Shared dataclasses: `DetectedContext`, `ClassificationResult`, `QueryResult` |
| `ingest.py` | One-shot ingestion: 8 documents → 276 chunks → ChromaDB |
| `ask.py` | CLI entry point |
| `run_evals.py` | Eval runner against the golden set |

---

## Corpus

Eight documents covering the main UK green compliance frameworks:

| Document | Type | Coverage |
|----------|------|----------|
| TCFD 2017 Recommendations | Real PDF (74 pages) | Climate-related financial disclosure framework |
| SECR Environmental Reporting Guidelines | Real PDF (~80 pages) | Streamlined Energy and Carbon Reporting |
| PPN 06/21 — Carbon Reduction Plans | Real PDF | Public procurement sustainability requirements |
| PPN 06/21 — Technical Standard | Real PDF | CRP completion requirements |
| CSRD UK Applicability Summary | Synthetic summary | EU Corporate Sustainability Reporting Directive — UK scope |
| FCA TCFD PS21/24 Summary | Synthetic summary | FCA mandatory climate disclosure rules |
| ESOS Overview | Synthetic summary | Energy Savings Opportunity Scheme |
| UK SDS Status and Timeline | Synthetic summary | UK Sustainability Disclosure Standards — incoming |

**On synthetic summaries:** Four documents are purpose-built summaries rather than primary legislation. They contain accurate thresholds and dates drawn from the real regulatory texts, are clearly labelled `source_type: synthetic_summary` in chunk metadata, and exist because the primary texts (CSRD runs to hundreds of pages of EU legislation; ESOS requires reading multiple statutory instruments) are impractical for a v0 corpus. The real PDF documents anchor the corpus with primary source material.

---

## Evaluation

The pipeline is evaluated against a golden set of 32 manually verified question/answer pairs covering:

- Threshold and applicability questions
- What must be reported under each framework
- Compliance deadlines and timelines
- Cross-framework synthesis questions
- Clarification-triggering (ambiguous) questions
- Out-of-scope boundary questions
- Hallucination traps (non-existent regulations, wrong premises, wrong dates)

### Eval metrics

| Metric | Method |
|--------|--------|
| State match | Classifier output matches expected state |
| Citation accuracy | All expected source documents appear in answer |
| Retrieval recall | % of expected documents surfaced |
| Fact presence | Expected facts appear in answer text |
| Hallucination rate | Forbidden phrases or fabricated content detected |

### Results (after classifier prompt revision)

| Metric | Score |
|--------|-------|
| Pass rate | **78.1%** |
| State match rate | **93.8%** |
| Citation accuracy | **81.2%** |
| Mean retrieval recall | **88.3%** |
| **Hallucination rate** | **0.0%** |

| Category | Result |
|----------|--------|
| Deadlines and timelines | 4/4 (100%) |
| What must be reported | 5/5 (100%) |
| Hallucination traps | 3/3 (100%) |
| Out of scope | 2/2 (100%) |
| Threshold / applicability | 5/6 (83%) |
| Cross-framework synthesis | 4/7 (57%) |
| Clarification triggering | 2/5 (40%) |

The 0% hallucination rate across both eval runs is the most important result. The system does not fabricate regulatory obligations. The remaining failures are citation precision gaps in multi-document cross-framework queries — the answers are substantively correct but not all expected source documents are retrieved simultaneously.

---

## Tech stack

| Component | Tool | Alternatives considered |
|-----------|------|------------------------|
| Answer generation | Claude (claude-sonnet-4-5) | GPT-4o, Gemini 1.5 Pro |
| Embeddings | Voyage AI (voyage-3) | OpenAI text-embedding-3-small, Cohere Embed |
| Vector store | ChromaDB (local) | Pinecone, Weaviate, pgvector |
| PDF extraction | pdfplumber | PyMuPDF, Unstructured.io |
| Testing | pytest + pytest-cov | unittest |
| Agentic coding | Claude Code | Cursor, GitHub Copilot |

---

## Getting started

### Prerequisites

- Python 3.11+
- A Voyage AI API key ([voyageai.com](https://www.voyageai.com))
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
git clone https://github.com/jonnykane/green-compliance-navigator.git
cd green-compliance-navigator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
VOYAGE_API_KEY=your_voyage_key_here
```

### Download the real corpus documents

```bash
python scripts/download_corpus.py
```

This downloads the four real regulatory PDFs into `corpus/real/`. The four synthetic summary documents are already present in `corpus/mock/`.

### Ingest the corpus

```bash
python ingest.py
```

This processes all 8 documents, creates 276 chunks with section-aware boundaries, embeds them with Voyage AI, and persists them to ChromaDB in `chroma_db/`.

### Ask a question

```bash
python ask.py "What are the SECR reporting thresholds for large UK companies?"
```

### Run the eval suite

```bash
# Dry run (no API calls)
python run_evals.py --dry-run

# Full live eval (uses real APIs)
python run_evals.py
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Design decisions

**Why section-aware chunking?** Regulatory documents have natural boundaries — articles, clauses, annexes. Chunking that respects these boundaries produces better retrieval than arbitrary token splits. A clause about SECR thresholds that spans a chunk boundary loses its meaning. This was validated by a real retrieval failure in Phase 2 where SECR threshold chunks were not being surfaced — the fix was in the chunker, not the retrieval parameters.

**Why a clarification agent?** UK green compliance obligations depend heavily on company size, listing status, EU presence, and procurement context. The same question — "what do we have to report?" — has completely different answers for an SME vs a listed company vs a company with EU operations. The pre-retrieval classifier routes ambiguous questions to a clarification step rather than attempting an incomplete answer.

**Why synthetic corpus documents?** The CSRD runs to hundreds of pages of EU legislation. The ESOS regulations require reading multiple statutory instruments. Purpose-built summaries with accurate thresholds and dates keep the corpus at a manageable size for a v0 build while covering the regulatory triggers that matter. They are clearly labelled in metadata so users know what they're reading.

**Why dependency injection throughout?** Every module accepts its dependencies via constructor injection rather than creating them directly. This makes every module independently testable with fake clients — the full 243-test suite runs without any network calls or API costs.

---

## Project status

- ✅ Phase 1 — Corpus assembly (8 documents)
- ✅ Phase 2 — RAG pipeline (ingestion, chunking, embeddings, retrieval, generation)
- ✅ Phase 3 — Clarification agent (four-state classifier, QueryResult, CLI)
- ✅ Phase 4 — Evals (32-question golden set, eval runner, 78.1% pass rate)
- 🔲 Phase 5 — UI (Lovable)

---

## Repository

[github.com/jonnykane/green-compliance-navigator](https://github.com/jonnykane/green-compliance-navigator)

---

## Disclaimer

This tool is a learning project and proof of concept. It is not a substitute for legal or compliance advice. Regulatory requirements change — always verify against current primary sources before making compliance decisions.
