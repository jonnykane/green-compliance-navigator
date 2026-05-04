# Technical Debt

Items deferred during active build phases. Each entry records what happened, why it was deferred, and what the fix would be.

---

## Open items

### TD-001 — Protocol placement inconsistency
**Phase introduced:** 2  
**Status:** Open  
**Impact:** Maintainability only — no runtime effect  
**Description:** Some Protocols live with their implementation module, some with their consumer module. No consistent convention was established before building. Different decisions were made at different Claude Code context windows.  
**Fix:** Decide on one convention (recommend: Protocol lives with its consumer), document in CONVENTIONS.md, move the misplaced Protocols in a single refactor commit.

---

### TD-002 — Generator system prompt not injectable via constructor
**Phase introduced:** 2 (fixed start of Phase 3)  
**Status:** Closed — fixed in Phase 3  
**Description:** Generator._SYSTEM_PROMPT was hardcoded as a module-level constant. Made prompt iteration require touching code structure. Fixed by making it injectable via constructor with the current prompt as default.

---

### TD-003 — Whitespace-only company_context bypasses clarification guard
**Phase introduced:** 3  
**Status:** Open  
**Impact:** Low — unreachable via CLI; only relevant if ask() is ever exposed beyond the CLI  
**Description:** A company_context value of whitespace-only (e.g. "   ") passes the non-empty check but generates no context prefix in the retrieval query. The clarification guard is bypassed but no enrichment occurs.  
**Fix:** Strip company_context before the guard check: `if company_context.strip()`.

---

### TD-004 — ClassifierClientProtocol defined in provider module (classifier.py) rather than consumer module (query.py)
**Phase introduced:** 3  
**Status:** Open  
**Impact:** Maintainability only  
**Description:** Same pre-existing convention inconsistency as TD-001, introduced during Phase 3 build.  
**Fix:** Move Protocol to query.py in the same refactor as TD-001.

---

### TD-005 — partial_answer_needs_clarification state mismatch between eval schema and classifier
**Phase introduced:** 4  
**Status:** Open  
**Impact:** Medium — product behaviour gap, not safety gap. Classifier routes some partial-context questions to 'clear'. Answers include caveats but the system doesn't natively express partial determination.  
**Description:** The eval golden set expects a partial_answer_needs_clarification state that the classifier doesn't natively support. Approximate scoring is used for these evals.  
**Fix:** Add partial_answer_needs_clarification as a native fourth state in the classifier prompt with 2–3 worked examples. Align eval scoring to use exact match once the state is live.

---

### TD-006 — Broad applicability retrieval gap (cross-framework synthesis)
**Phase introduced:** 4  
**Status:** Open  
**Impact:** High — directly affects core product promise for broad applicability questions. 0% on cross-framework synthesis eval category (7 evals).  
**Description:** Questions spanning four or more frameworks simultaneously (e.g. "what green regulations apply to us?") retrieve well for one or two frameworks but miss others. The canonical source injection mechanism (Phase 7) helps for single-framework queries but does not yet address multi-framework synthesis.  
**Fix:** For queries classified as broad applicability (no specific framework named), expand retrieval to query each indexed framework explicitly using per-framework enrichment vocabulary, then merge and deduplicate the top chunks before generation. Extend CANONICAL_SOURCE_MAP to trigger on broad applicability signals.

---

### TD-007 — SECR threshold chunks not consistently surfacing for threshold-specific queries
**Phase introduced:** 5 (discovered during live API testing)  
**Status:** Partially mitigated — canonical source injection added in Phase 7  
**Impact:** Low-medium — SECR threshold queries now reliably surface the curated SECR rule-card via canonical injection. Underlying PDF table fragmentation remains.  
**Description:** The SECR threshold table in the real PDF extracts as headerless number rows (`'UK 12,000 1,000 500 Not 1,500 6,000\n\nknown'`). The curated SECR rule-card (`secr_guidelines_summary.md`) was added in Phase 7 and contains explicit threshold statements. Canonical source injection guarantees it surfaces when "SECR" appears in the question. However, the injection mechanism currently marks all injected chunks as `reason=normal` rather than `reason=canonical_source_injection` — the trigger fires but the debug metadata isn't correctly set in all retrieval paths.  
**Fix:** Investigate why `retrieval_reason` is not being set on canonically injected chunks in the eval results JSON. Likely a path where the query engine bypasses `_inject_canonical_sources()` or where chunk metadata isn't propagated correctly. Verify by checking `retrieved_chunks` in a live eval result for eval_001.

---

### TD-008 — Railway cold start latency
**Phase introduced:** 5  
**Status:** Open  
**Impact:** Low — UX friction on first use  
**Description:** First request after an idle period incurs 10–20s cold start as Railway spins up the dyno. The QueryEngine singleton initialisation (Voyage AI client, Pinecone client) adds to this.  
**Fix:** Configure Railway keep-warm pings or upgrade to an always-on plan. Alternatively, lazy-load the Pinecone index connection on first query rather than at startup.

---

### TD-009 — Content hash computed on full HTML, not extracted body text
**Phase introduced:** 6  
**Status:** Open  
**Impact:** Medium — generates false-positive "changed" signals on every run for pages whose navigation, footers, or cookie banners update independently of regulatory content. Every false positive triggers materiality classification and potentially re-chunking.  
**Description:** `Fetcher` hashes the full HTTP response text before comparing to the stored hash. GOV.UK pages include navigation menus, footer links, and last-modified timestamps that change frequently without any change to the underlying regulatory content.  
**Fix:** Extract main body content (strip HTML to article/main element text) before computing the hash and before passing content to the materiality classifier. The same extraction step should apply to re-chunking to avoid embedding nav text as regulatory content.

---

### TD-010 — Registry state is in-memory only; lost between runs
**Phase introduced:** 6  
**Status:** Open  
**Impact:** High — every cold start resets `previous_hash` to `None` for all sources. Every run treats every framework as a first-time fetch, classifies all as material, and re-chunks all 7 frameworks regardless of whether anything changed. The agent is only useful for ongoing monitoring once it can persist state between invocations.  
**Description:** `FrameworkRegistry.update_check_result()` mutates the in-memory registry. The updated hashes and ETags are discarded when the Railway process restarts or the request completes.  
**Fix:** Persist `last_content_hash`, `last_etag`, and `last_checked_at` per framework to a durable store between runs. Options in order of simplicity: a JSON file written to Railway's persistent volume, a Redis instance, or a lightweight database. The registry interface already supports the mutation; only the persistence layer needs adding.

---

### TD-011 — Namespace mismatch between ingestion pipeline and rechunker
**Phase introduced:** 6  
**Status:** Open  
**Impact:** High — Phase 6 re-chunked content is not visible to query answers. The retriever queries `__default__` namespace only; Phase 6 vectors land in `{framework_id}` namespaces and are silently ignored.  
**Description:** The Phase 1–5 ingestion pipeline (`ingest.py`) wrote all vectors to Pinecone's `__default__` namespace. The Phase 6 `Rechunker` uses `framework_id` as the namespace (e.g. `ESOS`, `SECR`). These are separate namespaces in Pinecone. Without a migration, the two pipelines write to different locations and the query engine only sees the original ingestion.  
**Fix:** Either (a) update the rechunker to write to `__default__` to match the existing ingestion convention, or (b) migrate existing `__default__` vectors to framework-specific namespaces and update the retriever to query all relevant namespaces. Option (b) is architecturally cleaner and enables per-framework retrieval tuning. Whichever is chosen, document the namespace convention in CONVENTIONS.md to prevent recurrence.

---

### TD-012 — Fetcher has no retry or backoff for transient failures
**Phase introduced:** 6  
**Status:** Open  
**Impact:** Low — single transient failure produces a `FetchError` trace and skips that framework for the run. No data loss, but the agent silently misses a source until the next manual trigger.  
**Description:** `Fetcher.fetch()` makes a single HTTP GET with no retry on 429, 503, or network timeout. GOV.UK and FCA may rate-limit or experience transient unavailability.  
**Fix:** Add exponential backoff retry to `Fetcher` using the same pattern as `EmbeddingClient._embed_with_backoff()`. Configurable max retries and base delay, injectable for testing. Only retry on rate-limit (429) and server error (5xx) status codes; treat 404 as a permanent error.

---

### TD-013 — Materiality rules match on raw HTML, not extracted text
**Phase introduced:** 6  
**Status:** Open  
**Impact:** Low — could produce false-positive material classifications if threshold figures or date patterns appear in page navigation, sidebars, or footer content unrelated to the regulation being monitored.  
**Description:** The rules engine in `MaterialityClassifier` applies its regex patterns to the raw HTML string returned by the fetcher. Numeric patterns (`£\d[\d,]+`, `\d[\d,]*\s*kWh`) could match content in site-wide navigation or related-links sections that contain figures from other regulations.  
**Fix:** Apply the same HTML-to-body-text extraction recommended in TD-009 before passing content to the classifier. Once TD-009 is resolved, TD-013 is resolved as a side effect.

---

### TD-014 — missing_caveats is now the dominant eval failure type
**Phase introduced:** 7  
**Status:** Open  
**Impact:** Medium — 9/18 failures (50%) are missing_caveats after Phase 7 fixes. The system answers correctly but omits required qualifications such as "subject to consultation", "indicative timeline only", and "expected but not yet confirmed".  
**Description:** The generator prompt includes a REQUIRED CAVEATS RULE instructing it to reproduce hedging language. However, many caveats in the corpus appear in secondary sentences or footnotes rather than in the primary threshold/applicability chunk. When those secondary chunks don't surface in retrieval, the generator has no source material to draw the caveat from. Affected evals: 003, 009, 013, 014, 018, 020, 021, 022, 029.  
**Fix options:**  
- Add PPN and CSRD canonical rule-cards to CANONICAL_SOURCE_MAP to ensure caveat-bearing chunks surface for those frameworks  
- Extend parent-child expansion to include adjacent sibling sections (not just the parent), giving the generator broader context  
- Review affected evals to distinguish retrieval-missing-caveats from generation-skipping-caveats, then target the right layer  
**Recommended approach:** Run the retrieval diagnostic (check retrieved chunks for caveat text) on the top 3 missing_caveats evals before deciding which layer to fix.

---

### TD-015 — Cross-framework synthesis score is 0% (7/7 failing)
**Phase introduced:** 7 (revealed by tighter eval gate)  
**Status:** Open  
**Impact:** High — 7 evals in this category, all failing. Core product gap for broad applicability questions.  
**Description:** Questions like "what sustainability reporting do we have to do?" require the generator to synthesise across SECR, TCFD, ESOS, CSRD, UK SDS, and PPN simultaneously. The retriever currently returns the top-8 chunks across all documents, which tends to surface one or two frameworks well but miss others entirely. eval_027 is a persistent retrieval failure with irrelevant SECR biodiversity chunks scoring above the relevant TCFD/ESOS threshold chunks.  
**Fix:** Per-framework retrieval for broad applicability queries — detect "no specific framework named" intent in the classifier or query engine, then explicitly retrieve top-2 chunks per indexed framework before merging and deduplicating. This is the same approach recommended in TD-006 and should be implemented together with it.

---

## Closed items

| ID | Description | Fixed in phase |
|---|---|---|
| TD-002 | Generator system prompt not injectable | Phase 3 |
| — | Unexpected classifier state silently routed to answer path | Phase 3 coherence review |
| — | Dead QueryResponse dataclass | Phase 3 coherence review |
| — | embedding_client.py backoff retry path untested | Phase 2 coherence review |
| — | SECR retrieval gap (chunking — section boundary splitting) | Phase 2 fix |
| — | SECR retrieval query vocabulary mismatch | Phase 5 Step 3 (commit 71becd2) |
| — | Eval pass gate too soft (state + citation + no hallucination only) | Phase 7 — tightened to include facts, caveats, recall, certainty |
| — | No retrieval trace in eval results | Phase 7 — retrieved_chunks added to QueryResult and EvalResult |
| — | TCFD PDF garbled extraction (multi-column layout) | Phase 7 — replaced with curated synthetic summary |
| — | Pinecone accumulates stale vectors on re-ingest | Phase 7 — delete-before-upsert added to PineconeVectorStore |
| — | Generator non-deterministic (no temperature set) | Phase 7 — temperature=0 set on all generator API calls |
| — | Fact scorer stop words included domain-critical terms | Phase 7 — employees, turnover, threshold, company, companies removed |
| — | Fact scorer minimum word length 5 filtered short regulatory terms | Phase 7 — minimum lowered to 3 (catches kWh, LLP, AUM) |
