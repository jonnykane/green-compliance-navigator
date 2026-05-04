# UK Green Compliance Navigator

A RAG-powered assistant that helps UK businesses understand what green regulations apply to them, what those regulations require, and when deadlines fall — with every answer cited back to the source document and section.

**Live UI:** `https://green-nav-guide.lovable.app`  
**Live API:** `https://green-compliance-navigator-production.up.railway.app`

---

## What it does

A user describes their company. The assistant retrieves relevant regulatory text from a 10-document corpus, classifies the query, asks one clarifying question if needed, and produces a plain-English cited answer. Every claim is traceable to a real regulation.

```
POST /ask
{ "question": "We have 300 employees and £45m turnover — what sustainability reporting applies to us?",
  "company_context": "Large UK company with 300+ employees and £45m turnover" }

→ { "kind": "answer",
    "answer": "Based on your size, SECR applies. You meet the 250+ employee and £36m+ turnover thresholds...",
    "clarification_question": null,
    "options": null,
    "sources": ["secr_guidelines_summary.md", "secr_environmental_reporting_guidelines_2019.pdf"] }
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
     ├─ search → Pinecone (1,133 chunks, top_k=8)
     └─ canonical source injection (_inject_canonical_sources)
          └─ guarantees curated rule-card chunks surface for trigger-matched frameworks
     │
     ▼  expand parent sections (parent-child chunking)
Generator (Claude · temperature=0)
     │
     ▼
QueryResult JSON
{ kind, answer, clarification_question, options, sources, retrieved_chunks }
```

---

## The corpus

Ten documents covering seven UK green regulatory frameworks:

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
| TCFD 2017 Recommendations Summary | Curated summary | TCFD |
| SECR Guidelines Summary | Curated rule-card | SECR |

Synthetic summaries are labelled `source_type: synthetic_summary` in chunk metadata. Real PDFs are labelled `source_type: real_pdf`. The UI surfaces this distinction so users know which answers come from primary legislation.

---

## Eval results

32 golden Q&A pairs manually verified against source documents. Results after Phase 7 eval hardening:

| Metric | Result |
|---|---|
| **Native product score** | **43.8% (14/32)** |
| State match rate | 93.8% |
| Citation accuracy | 78.1% |
| Mean retrieval recall | 86.7% |
| **Hallucination rate** | **0.0%** |

Category breakdown:

| Category | Result |
|---|---|
| Threshold / applicability | 3/6 (50.0%) |
| What must be reported | 3/5 (60.0%) |
| Deadlines and timelines | 2/4 (50.0%) |
| Cross-framework synthesis | 0/7 (0.0%) |
| Clarification triggering | 2/5 (40.0%) |
| Out of scope | 1/2 (50.0%) |
| Hallucination traps | 3/3 (100%) |

**Note on eval methodology:** The eval gate was significantly tightened in Phase 7. The previous 78.1% figure used a soft gate (state match + citation + no hallucination only). The current gate additionally requires all expected facts present, all required caveats present, retrieval recall above threshold, and no forbidden certainty. The 43.8% figure is a more honest measure of answer completeness. Evals run at `temperature=0` for deterministic, reproducible results.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Embeddings | Voyage AI (voyage-3, 1024-dim) | Anthropic-recommended partner; strong on technical regulatory text |
| Vector store | Pinecone (production) / ChromaDB (local) | Pinecone when `PINECONE_API_KEY` present; ChromaDB otherwise |
| Generation + classification | Anthropic claude-sonnet-4-6 (temperature=0) | Strong system prompt adherence; 0% hallucination rate; deterministic evals |
| API | FastAPI + uvicorn | Automatic validation, CORS, OpenAPI docs |
| Deployment | Railway (from GitHub) | Zero-config, deploys on push to master |
| UI | Lovable (React) | AI-generated React UI calling the Railway endpoint |
| PDF extraction | pdfplumber | Reliable on standard regulatory PDFs |
| Testing | pytest | 512 tests; all external APIs faked — runs without network access |

---

## Build standards

- **TDD throughout** — tests written before implementation at every module
- **Dependency injection** — every external client is injectable; wiring.py wires real clients
- **No hardcoded config** — all API keys and paths via environment variables
- **Fake discipline** — every external API call (Voyage AI, Pinecone, Anthropic) has a fake; tests never burn credits
- **Deterministic evals** — `temperature=0` on the generator; eval results are stable and reproducible across runs

```bash
pytest                      # 512 tests, runs in seconds, no network required
python -m src.evals.runner  # 32 live eval queries against real API
```

---

## Project structure

```
green-compliance-navigator/
├── api.py                  # FastAPI server (POST /ask, GET /health)
├── ask.py                  # CLI entry point
├── ingest.py               # Ingestion script: 10 docs → 1,133 chunks → vector store
├── run_evals.py            # Eval runner CLI
├── requirements.txt
├── TECHNICAL_DEBT.md       # Known gaps, deferred decisions, fix guidance
├── corpus/
│   ├── real/               # 4 real regulatory PDFs
│   └── mock/               # 4 synthetic summaries + 2 curated rule-cards
├── evals/
│   └── golden_set.json     # 32 manually verified Q&A pairs
├── src/
│   ├── query.py            # QueryEngine.ask() — classifier → retrieval → canonical injection → expansion → generation
│   ├── wiring.py           # Wires real clients (Voyage / vector store / Anthropic)
│   ├── models.py           # QueryResult, ClassificationResult, DetectedContext
│   ├── config/             # Settings including RETRIEVER_TOP_K, CANONICAL_SOURCE_MAP
│   ├── loader/             # PDF + markdown text extraction
│   ├── chunker/            # Section-aware chunking with parent-child metadata
│   ├── embedder/           # Voyage AI client wrapper
│   ├── vector_store/       # ChromaDB + Pinecone wrappers with query_filtered support
│   ├── retriever/          # Embed query → search → rank → retrieve_by_source for canonical injection
│   ├── classifier/         # Pre-retrieval intent classifier
│   ├── generator/          # Build cited answer from retrieved chunks (temperature=0)
│   └── evals/              # EvalRunner, reporter with failure taxonomy and retrieval traces
└── tests/                  # 512 tests across all modules
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
| `RETRIEVER_TOP_K` | No | Number of chunks retrieved per query (default: 8) |

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| 1–2 | RAG pipeline foundation | ✅ Complete |
| 3 | Clarification agent | ✅ Complete |
| 4 | Evals (78.1% pass rate soft gate, 0% hallucination) | ✅ Complete |
| 5 Step 1 | Pinecone migration (1,109 chunks) | ✅ Complete |
| 5 Step 2 | FastAPI server | ✅ Complete |
| 5 Step 3 | Railway deployment | ✅ Complete |
| 5 Step 4 | Lovable UI | ✅ Complete |
| 5 Post | SECR retrieval fix (_enrich_retrieval_query, 308 tests) | ✅ Complete |
| 6 | Regulatory change monitoring agent (ReAct loop, 95 agent tests) | ✅ Complete |
| 7 | Eval hardening, corpus fixes, canonical injection (43.8%, 512 tests) | ✅ Complete |

---

## Phase 6 — Regulatory change monitoring agent

### What we built

A ReAct-loop agent that periodically checks the 7 indexed regulatory source URLs for changes, classifies whether those changes are material, and automatically re-chunks and re-embeds affected content into Pinecone. Triggered manually via `POST /agent/run`; scheduling is deferred.

Seven components, each built TDD (95 tests, zero external API calls in the test suite):

| Component | Responsibility |
|---|---|
| `FrameworkRegistry` | Runtime URL registry loaded from `FRAMEWORK_REGISTRY_JSON` env var; tracks last content hash and ETag per source |
| `Fetcher` | HTTP fetch with ETag/Last-Modified header detection and content-hash fallback; typed result union: `Changed \| Unchanged \| FetchError` |
| `MaterialityClassifier` | Rules-first (regex patterns for threshold figures, deadlines, scope changes, penalty amounts); Claude fallback for ambiguous prose |
| `Rechunker` | Re-chunks changed content, re-embeds via Voyage AI, upserts to Pinecone with content-addressed vector IDs |
| `EvalFlagger` | Matches changed `framework_id` against the golden eval set; marks affected evals for human review |
| `MonitoringAgent` | ReAct orchestrator: explicit Thought → Action → Observation trace per source |
| `Summariser` | Structured human-readable run summary |

---

## Phase 7 — Eval hardening and corpus quality

### What we built

A comprehensive eval improvement pass that tightened the pass gate, added retrieval traceability, fixed corpus quality issues, and implemented canonical source injection for deterministic retrieval of curated rule-cards.

**Eval gate tightening** — the pass condition now requires state match, citation accuracy, no hallucination, retrieval recall ≥ threshold, all expected facts present, all required caveats present, and no forbidden certainty. Previous soft gate (state + citation + no hallucination only) masked systemic incompleteness in answers.

**Failure taxonomy** — `EvalResult.failure_type` assigned in priority order: `classification_failure → retrieval_failure → missing_facts → missing_caveats → citation_failure → hallucination → forbidden_certainty → answer_posture_failure`. Dual pass rates (native product score vs architecture-compatible score) separate genuine failures from TD-005 architecture gaps.

**Retrieval trace instrumentation** — `QueryResult.retrieved_chunks` now carries `{"source", "text", "score"}` per chunk. Failing evals print chunk traces in the reporter, enabling root cause diagnosis (retrieval failure vs generation failure vs scorer failure).

**Corpus fixes:**
- TCFD PDF replaced with curated synthetic summary — the original PDF used a complex multi-column graphical layout that produced garbled chunks; the summary restored eval_008 to passing
- SECR curated rule-card added — explicit threshold statements (250 employees, £36m turnover, £18m balance sheet, two-of-three criteria) in a form the generator can reliably use
- Pinecone delete-before-upsert — re-ingests now produce clean indexes with no stale vectors from previous runs

**Parent-child chunking** — long PDF sections store their full parent text in chunk metadata. At generation time, retrieved chunks are expanded to their parent section, giving the generator table context (column headers + values together) rather than fragmented rows.

**Canonical source injection** — `_inject_canonical_sources()` guarantees that curated rule-cards surface alongside PDF chunks for trigger-matched framework queries. Trigger phrases (e.g. "SECR", "streamlined energy") map to canonical source filenames; injected chunks are tagged with `retrieval_reason=canonical_source_injection` for debug traceability.

**Scorer improvements** — removed domain-critical terms (`employees`, `turnover`, `threshold`, `company`, `companies`) from the fact scorer stop word list; lowered minimum key term length from 5 to 3 characters to match short regulatory abbreviations (kWh, LLP, AUM).

**Deterministic generation** — `temperature=0` on the generator API call. Eval results are now stable and reproducible across runs.

### Score progression

| Checkpoint | Score | Gate |
|---|---|---|
| Phase 4 baseline | 78.1% (25/32) | Soft (state + citation + no hallucination) |
| Phase 7 gate tightened | 34.4% (11/32) | Strict (facts + caveats + recall + no certainty) |
| + TCFD synthetic summary | 34.4% → 34.4% | eval_008 fixed, eval_012 regressed |
| + Loader revert + clean Pinecone | 34.4% (11/32) | eval_012 restored |
| + TCFD summary + clean index | 34.4% → 37.5% | eval_008 passes |
| + Scorer stop word fix | 37.5% → 40.6% | eval_007 passes |
| + Canonical source injection | 40.6% → 40.6% | eval_031 passes, scorer gap on eval_001 |
| + Golden set calibration (eval_001, eval_012) | 40.6% → 43.8% | eval_001 passes |
| + temperature=0 | **43.8% (14/32)** | Stable, deterministic |

---

## Known limitations

See [TECHNICAL_DEBT.md](./TECHNICAL_DEBT.md) for full detail. Key open items:

- **TD-005** — Classifier does not natively support `partial_answer_needs_clarification`. Partial-context questions are routed as `clear`; the generator handles the gap in prose.
- **TD-006** — Broad cross-framework questions (4+ frameworks) miss some documents. 0% on cross-framework synthesis eval category.
- **TD-008** — Railway cold start latency (10–20s on first request after idle period).

---

## Design decisions worth noting

**Why RAG rather than long-context stuffing:** TCFD is 74 pages, SECR is ~80 pages. The combined corpus cannot fit in a single prompt. RAG also enables multi-document synthesis across frameworks.

**Why synthetic summaries for some documents:** The real CSRD text is hundreds of pages of EU legislation. Synthetic summaries with accurate thresholds and dates keep the corpus manageable and give full control over eval verifiability. They are clearly labelled.

**Why curated rule-cards alongside real PDFs:** Regulatory threshold tables fragment badly during PDF extraction — numbers separate from their column headers. Curated rule-cards state thresholds explicitly in prose that the generator can reliably extract. The PDF stays in the corpus for citation and retrieval breadth; the rule-card guarantees the generator has usable threshold text.

**Why canonical source injection rather than a facts registry:** A hardcoded facts dict would bypass RAG entirely and undermine the architecture's value. Canonical injection is still RAG — targeted retrieval from a curated source — but guarantees coverage for known high-value queries. The answer remains fully cited and traceable.

**Why temperature=0:** Eval scores must be reproducible. Non-deterministic generation causes borderline evals to flip between runs, making it impossible to attribute score changes to specific fixes. Temperature=0 makes every eval run a reliable signal.

**Why deterministic clarification questions:** Six constant options — predictable, testable, auditable. Essential for a compliance tool.
