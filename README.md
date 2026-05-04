# UK Green Compliance Navigator

A RAG-powered assistant that helps UK businesses understand what green regulations apply to them, what those regulations require, and when deadlines fall — with every answer cited back to the source document and section.

**Live UI:** `https://green-nav-guide.lovable.app`  
**Live API:** `https://green-compliance-navigator-production.up.railway.app`

---

## What it does

A user describes their company. The assistant retrieves relevant regulatory text from an 8-document corpus, classifies the query, asks one clarifying question if needed, and produces a plain-English cited answer. Every claim is traceable to a real regulation.

```
POST /ask
{ "question": "We have 300 employees and £45m turnover — what sustainability reporting applies to us?",
  "company_context": "Large UK company with 300+ employees and £45m turnover" }

→ { "kind": "answer",
    "answer": "Based on your size, SECR applies. You meet the 250+ employee and £36m+ turnover thresholds...",
    "clarification_question": null,
    "options": null,
    "sources": ["secr_environmental_reporting_guidelines_2019.pdf", "csrd_uk_applicability_summary.md"] }
```

---

## Architecture

```
User question
     │
     ▼
Lovable UI  (React · green-nav-guide.lovable.app)
     │
     ▼  POST /ask
FastAPI server  (Railway · green-compliance-navigator-production.up.railway.app)
     │
     ▼
QueryEngine  (src/query.py)
     │
     ├─ Classifier (Claude)
     │    ├─ out of scope → return directly
     │    ├─ needs clarification → return question + 6 options
     │    └─ clear / partial → continue
     │
     ▼  clear / partial answer
Retriever  (src/retriever/)
     ├─ enrich query with regulation vocabulary (_enrich_retrieval_query)
     ├─ embed enriched query → Voyage AI (voyage-3)
     └─ search → Pinecone (1,109 chunks)
     │
     ▼
Generator (Claude)
     │
     ▼
QueryResult JSON
{ kind, answer, clarification_question, options, sources }
```

---

## The corpus

Eight documents covering seven UK green regulatory frameworks:

| Document | Type | Frameworks covered |
|---|---|---|
| TCFD 2017 Recommendations | Real PDF (74p) | TCFD, UK SDS baseline |
| SECR Environmental Reporting Guidelines (2019) | Real PDF (~80p) | SECR |
| PPN 06/21 — Carbon Reduction Plans | Real PDF (~20p) | PPN 06/21 |
| PPN 06/21 — Technical Standard for CRP | Real PDF (~10p) | PPN 06/21 |
| CSRD UK Applicability Summary | Synthetic summary | CSRD |
| FCA TCFD PS21/24 Summary | Synthetic summary | FCA TCFD |
| ESOS Overview | Synthetic summary | ESOS |
| UK SDS Status and Timeline | Synthetic summary | UK SDS |

Synthetic summaries are labelled `source_type: synthetic_summary` in chunk metadata. Real PDFs are labelled `source_type: real_pdf`. The UI surfaces this distinction so users know which answers come from primary legislation.

---

## Eval results

32 golden Q&A pairs manually verified against source documents. Results after Phase 4 classifier and generator fixes:

| Metric | Result |
|---|---|
| Pass rate | 78.1% (25/32) |
| State match rate | 93.8% |
| Citation accuracy | 81.2% |
| Mean retrieval recall | 88.3% |
| **Hallucination rate** | **0.0%** |

Category breakdown:

| Category | Result |
|---|---|
| Threshold / applicability | 5/6 (83.3%) |
| What must be reported | 5/5 (100%) |
| Deadlines and timelines | 4/4 (100%) |
| Cross-framework synthesis | 4/7 (57.1%) |
| Clarification triggering | 2/5 (40.0%) |
| Out of scope | 2/2 (100%) |
| Hallucination traps | 3/3 (100%) |

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | Voyage AI (voyage-3, 1024-dim) | Anthropic-recommended partner; strong on technical regulatory text |
| Vector store | Pinecone (production) / ChromaDB (local) | Pinecone when `PINECONE_API_KEY` present; ChromaDB otherwise |
| Generation + classification | Anthropic claude-sonnet-4-6 | Strong system prompt adherence; 0% hallucination rate |
| API | FastAPI + uvicorn | Automatic validation, CORS, OpenAPI docs |
| Deployment | Railway (from GitHub) | Zero-config, deploys on push to master |
| UI | Lovable (React) | AI-generated React UI calling the Railway endpoint |
| PDF extraction | pdfplumber | Reliable on multi-column regulatory PDFs |
| Testing | pytest | 308 tests; all external APIs faked — runs without network access |

---

## Build standards

- **TDD throughout** — tests written before implementation at every module
- **Dependency injection** — every external client is injectable; wiring.py wires real clients
- **No hardcoded config** — all API keys and paths via environment variables
- **Fake discipline** — every external API call (Voyage AI, Pinecone, Anthropic) has a fake; tests never burn credits
- **Eval-gated CI** — `run_evals.py` exits 1 if pass rate drops below 70%

```bash
pytest              # 308 tests, runs in seconds, no network required
python run_evals.py # 32 live eval queries against real API
```

---

## Project structure

```
green-compliance-navigator/
├── api.py                  # FastAPI server (POST /ask, GET /health)
├── ask.py                  # CLI entry point
├── ingest.py               # Ingestion script: 8 docs → 1,109 chunks → vector store
├── run_evals.py            # Eval runner CLI
├── requirements.txt
├── TECHNICAL_DEBT.md       # Known gaps, deferred decisions, fix guidance
├── src/
│   ├── query.py            # QueryEngine.ask() — includes _enrich_retrieval_query()
│   ├── wiring.py           # Wires real clients (Voyage / vector store / Anthropic)
│   ├── models.py           # QueryResult, ClassificationResult, DetectedContext
│   ├── loader/             # PDF + markdown text extraction
│   ├── chunker/            # Section-aware chunking
│   ├── embedder/           # Voyage AI client wrapper
│   ├── vector_store/       # ChromaDB + Pinecone wrappers (VectorStoreProtocol)
│   ├── retriever/          # Embed query → search → rank chunks
│   ├── classifier/         # Pre-retrieval intent classifier
│   ├── generator/          # Build cited answer from retrieved chunks
│   └── evals/              # EvalRunner, reporter
├── evals/
│   └── golden_set.json     # 32 manually verified Q&A pairs
└── tests/                  # 308 tests across all modules
```

---

## Running locally

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=...
export VOYAGE_API_KEY=...
export PINECONE_API_KEY=...   # omit to use ChromaDB locally

python ingest.py              # one-time corpus ingestion
uvicorn api:app --reload      # start API server
python ask.py "We have 300 employees — what sustainability reporting applies to us?"
```

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API (classifier + generator) |
| `VOYAGE_API_KEY` | Yes | Voyage AI embeddings |
| `PINECONE_API_KEY` | No | Pinecone vector store (ChromaDB used if absent) |

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| 1–2 | RAG pipeline foundation | ✅ Complete |
| 3 | Clarification agent | ✅ Complete |
| 4 | Evals (78.1% pass rate, 0% hallucination) | ✅ Complete |
| 5 Step 1 | Pinecone migration (1,109 chunks) | ✅ Complete |
| 5 Step 2 | FastAPI server | ✅ Complete |
| 5 Step 3 | Railway deployment | ✅ Complete |
| 5 Step 4 | Lovable UI | ✅ Complete |
| 5 Post | SECR retrieval fix (_enrich_retrieval_query, 308 tests) | ✅ Complete |

---

## Known limitations

See [TECHNICAL_DEBT.md](./TECHNICAL_DEBT.md) for full detail. Key open items:

- **TD-005** — Classifier does not natively support `partial_answer_needs_clarification`. Partial-context questions are routed as `clear`; the generator handles the gap in prose.
- **TD-006** — Broad cross-framework questions (4+ frameworks) miss some documents. 57.1% on cross-framework synthesis eval category.
- **TD-007** — SECR threshold chunk does not consistently surface for threshold-specific queries despite existing in Pinecone.
- **TD-008** — Railway cold start latency (10–20s on first request after idle period).

---

## Design decisions worth noting

**Why RAG rather than long-context stuffing:** TCFD is 74 pages, SECR is ~80 pages. The combined corpus cannot fit in a single prompt. RAG also enables multi-document synthesis across frameworks.

**Why synthetic summaries for 4 of 8 documents:** The real CSRD text is hundreds of pages of EU legislation. Synthetic summaries with accurate thresholds and dates keep the corpus manageable and give full control over eval verifiability. They are clearly labelled.

**Why deterministic clarification questions:** Six constant options — predictable, testable, auditable. Essential for a compliance tool.

**Why query enrichment rather than re-chunking for the SECR retrieval gap:** The SECR threshold text exists correctly in Pinecone. The problem was vocabulary mismatch between natural-language queries and the flat similarity band of SECR chunks. `_enrich_retrieval_query()` appends regulation-specific vocabulary before embedding, fixing retrieval without requiring re-ingestion.
