# Technical Debt

## Open items

### `ClassifierClientProtocol` defined in provider module, not consumer module

**File:** `src/classifier/classifier.py`  
**Convention broken:** every other Protocol in the codebase is defined in the
*consumer* module — `RetrieverProtocol`, `GeneratorProtocol`, `ClassifierProtocol`
live in `query.py`; `EmbedderProtocol`, `StoreProtocol` live in `retriever.py`;
`LoaderProtocol` etc. live in `ingestion_pipeline.py`.  
`ClassifierClientProtocol` is the only Protocol defined alongside the class it
constrains (the *provider*) rather than the class that depends on it.  
**Impact:** Low — no runtime effect. Inconsistent for a reader tracing dependency
direction through the codebase.  
**Fix:** Move `ClassifierClientProtocol` to `query.py` alongside the other
consumer-side Protocols, or establish an explicit `src/protocols.py` home for all
Protocols as part of a wider refactor.

---

### Whitespace-only `company_context` bypasses clarification guard in `ask()`

**File:** `src/query.py` — `QueryEngine.ask()`  
**Detail:** The clarification guard is `if classification.state == "needs_clarification"
and not company_context`. A whitespace-only string (e.g. `"   "`) is truthy in Python,
so the guard is bypassed and execution falls through to the retrieval path with an
effectively empty context. The generator handles it correctly (`.strip()` produces `""`),
but the clarification question is silently skipped.  
**Impact:** Low — unreachable via the CLI (all `CONTEXT_FROM_OPTION` values are
non-whitespace strings). Only relevant if `ask()` is called programmatically with
a whitespace string.  
**Fix:** Normalise at the top of `ask()`:
```python
company_context = company_context.strip()
```
This makes all three call sites (`clarification guard`, `retrieval_query`, `generate`)
consistent without changing observable behaviour for the CLI.
