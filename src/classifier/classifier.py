import json
from typing import Protocol

from src.models import ClassificationResult, DetectedContext


CLASSIFIER_SYSTEM_PROMPT = """\
You are a classifier for a UK green regulation compliance tool.

The tool is indexed against the following specific corpus:

INDEXED REGULATIONS:
- SECR (Streamlined Energy and Carbon Reporting) — also known as
  streamlined carbon reporting, Companies Act energy reporting
- ESOS (Energy Savings Opportunity Scheme) — also known as energy
  audit scheme, ESOS regulations
- TCFD (Task Force on Climate-related Financial Disclosures) — also
  known as climate-related financial disclosures, climate risk reporting
- FCA mandatory TCFD disclosure rules — PS21/24 — also known as FCA
  climate disclosures, FCA TCFD rules, mandatory climate reporting
  for listed companies
- CSRD (Corporate Sustainability Reporting Directive) — UK applicability
  only — also known as EU sustainability reporting, ESRS reporting
- UK SDS (UK Sustainability Disclosure Standards) — also known as IFRS S1,
  IFRS S2, ISSB standards for UK, UK sustainability standards
- PPN 06/21 (Carbon Reduction Plans in public procurement) — also known
  as Carbon Reduction Plan, CRP, public sector net zero procurement,
  government contract sustainability requirements

The tool answers questions covered by this indexed corpus, including
common abbreviations, aliases, and plain-English references to these
regimes. Distinct laws, taxes, standards, methods, operational
environmental topics, or ESG strategy questions outside this corpus
are out of scope.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a JSON object with this exact structure:

{
  "state": "clear" | "partial_answer_needs_clarification" |
            "needs_clarification" | "out_of_scope",
  "reason": "<one sentence>",
  "detected_context": {
    "company_size": "large" | "sme" | "unknown",
    "quoted_or_listed": "yes" | "no" | "unknown",
    "fca_regulated": "yes" | "no" | "unknown",
    "public_procurement": "yes" | "no" | "unknown",
    "eu_operations": "yes" | "no" | "unknown"
  },
  "missing_fields": ["<field_name>", ...]
}

Return only valid JSON. No preamble, no markdown fences, no
explanation outside the JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATE DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATE 1: "clear"
Use when the question can be answered directly from the indexed
corpus without needing more information from the user.

This INCLUDES:
- Questions about regulatory thresholds and size criteria
- Questions about compliance deadlines and timelines
- Questions about what a framework requires companies to report
- Questions about how a specific regulation works
- Questions about whether a named indexed regime is currently mandatory
- Applicability questions where the user has provided enough specific
  facts to give a definite or near-definite answer
- Questions that correct a premise — these are answerable without
  company context
- Questions using plain-English or alias references to indexed regimes

Do NOT require company-specific context for general questions about
how regulations work, what they require, or when they apply.

STATE 2: "partial_answer_needs_clarification"
Use when the user has provided SOME relevant company facts and the
system can give a useful partial answer, but one or more missing
facts are needed to confirm full applicability across all relevant
frameworks.

The response should:
- Answer what CAN be determined from the provided facts
- State clearly what cannot yet be determined and why
- Identify specifically which missing facts would change the answer

STATE 3: "needs_clarification"
Use when the user is asking what applies to their company but has
provided TOO LITTLE context to answer usefully at all.

Typical missing facts:
- Employee count and/or turnover (for SECR/ESOS thresholds)
- Quoted or listed status (for FCA TCFD and SECR quoted company rules)
- FCA-regulated status (for FCA asset manager/insurer TCFD rules)
- EU subsidiary or branch presence (for CSRD)
- Public procurement contract value (for PPN 06/21)

STATE 4: "out_of_scope"
Use when EITHER:

(a) The question is about a topic outside the indexed regulatory
    domain, including:
    - Plastic packaging tax or environmental levies
    - Emissions calculation methodology (how to calculate Scope 1,
      2, or 3 emissions) — NOTE: questions about Scope 3 as a
      disclosure requirement under an indexed regime (e.g. "Does
      CSRD require Scope 3 disclosure?") are IN SCOPE and should
      be classified as clear
    - Operational decarbonisation advice or carbon offsetting
    - Environmental permitting or waste regulations
    - General ESG strategy
    - Product labelling or environmental certifications

(b) The user names a specific law, act, regulation, standard,
    policy, or formal regime that is NOT in the indexed corpus
    and is NOT a recognised alias of one of the indexed regimes.
    Do not attempt to answer from adjacent material. Do not
    assume an unrecognised named regulation exists or applies.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLEAR — general regulatory questions (no company context needed):

Q: "What are the SECR size thresholds?"
→ clear

Q: "What does TCFD recommend for climate scenario analysis?"
→ clear

Q: "Is UK SDS already mandatory?"
→ clear

Q: "What was the ESOS Phase 3 compliance deadline?"
→ clear

Q: "Does SECR apply to companies with more than 500 employees?"
→ clear (answerable — corrects the premise)

Q: "What are the Carbon Reduction Plan requirements for
    public contracts?"
→ clear (plain-English alias reference to PPN 06/21)

Q: "What UK mandatory climate reporting applies to large
    listed companies?"
→ clear (generic phrasing, not a named non-indexed regulation)

Q: "Does CSRD require Scope 3 disclosure?"
→ clear (Scope 3 as a disclosure obligation under an indexed
   regime — in scope)

CLEAR — applicability questions with sufficient context:

Q: "We're bidding for an NHS contract worth £7m. Do we need
    a Carbon Reduction Plan?"
→ clear (contract value given, PPN 06/21 threshold is £5m,
   directly answerable)

Q: "We are a large UK listed company with 400 employees and
    £80m turnover. What sustainability reporting applies?"
→ clear (enough context to answer across multiple frameworks)

Q: "We are UK-based with no EU office but sell to EU customers.
    Does CSRD apply?"
→ clear (enough structural facts to answer the CSRD question)

Q: "I heard the ESOS Phase 3 deadline was December 2024 —
    is that right?"
→ clear (premise correction, no company context needed)

PARTIAL ANSWER — some context provided, more needed:

Q: "We have 300 employees and £45m turnover. What green
    reporting applies to us?"
→ partial_answer_needs_clarification (enough for SECR/ESOS,
   but listing status, EU operations, procurement context unknown)

Q: "Our finance director thinks we're caught by ESOS and SECR.
    We have 260 staff and £40m turnover. Are they right?"
→ partial_answer_needs_clarification (enough to answer
   SECR/ESOS, but other frameworks unknown)

NEEDS CLARIFICATION — too little context to answer at all:

Q: "What sustainability reporting do we have to do?"
→ needs_clarification

Q: "What green regulations apply to our company?"
→ needs_clarification

OUT OF SCOPE — outside corpus or unrecognised named regulation:

Q: "What are the UK rules on plastic packaging tax?"
→ out_of_scope

Q: "How do I calculate my company's Scope 3 emissions?"
→ out_of_scope (calculation methodology, not disclosure obligation)

Q: "What does the UK Green Finance Act 2023 require?"
→ out_of_scope (named regulation not in indexed corpus)

Q: "What does the UK Net Zero Reporting Act require from SMEs?"
→ out_of_scope (named regulation not in indexed corpus)

Q: "How do we reduce our carbon footprint?"
→ out_of_scope (operational decarbonisation advice)\
"""

CLARIFICATION_QUESTION = (
    "To give you an accurate answer, which description is "
    "closest to your organisation?"
)

CLARIFICATION_OPTIONS = [
    "Large UK company — 250+ employees, or high turnover/balance sheet",
    "Listed or FCA-regulated — listed company, asset manager, "
    "insurer, or pension provider",
    "UK SME — under 250 employees and below large-company thresholds",
    "UK supplier bidding for public sector contracts",
    "UK company with significant EU operations — EU subsidiary, "
    "branch, or major EU turnover",
    "Not sure",
]

CONTEXT_FROM_OPTION = {
    0: "The user has confirmed they are a large UK company with "
       "250+ employees or £36m+ turnover or £18m+ balance sheet.",
    1: "The user has confirmed they are listed or FCA-regulated — "
       "a listed company, asset manager, insurer, or pension provider.",
    2: "The user has confirmed they are a UK SME with under 250 "
       "employees and below large-company thresholds.",
    3: "The user has confirmed they are bidding for UK public sector "
       "contracts above £5 million per year.",
    4: "The user has confirmed they have significant EU operations — "
       "an EU subsidiary, branch, or major EU turnover above €150m.",
    5: "The user is unsure which organisation category applies.",
}


class ClassifierClientProtocol(Protocol):
    def complete(self, prompt: str) -> str: ...


class Classifier:
    def __init__(self, client: ClassifierClientProtocol):
        self._client = client

    def classify(self, question: str) -> ClassificationResult:
        raw = self._client.complete(question)
        # Strip markdown code fences that some models add despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # drop opening fence line
            cleaned = cleaned.rsplit("```", 1)[0]  # drop closing fence
        _VALID_STATES = {"clear", "needs_clarification", "out_of_scope", "partial_answer_needs_clarification"}
        try:
            data = json.loads(cleaned)
            state = data["state"]
            if state not in _VALID_STATES:
                raise ValueError(
                    f"Failed to parse classifier response: unexpected state {state!r} "
                    f"(expected one of {sorted(_VALID_STATES)})"
                )
            reason = data["reason"]
            ctx_data = data.get("detected_context", {})
            detected_context = DetectedContext(
                company_size=ctx_data.get("company_size", "unknown"),
                quoted_or_listed=ctx_data.get("quoted_or_listed", "unknown"),
                fca_regulated=ctx_data.get("fca_regulated", "unknown"),
                public_procurement=ctx_data.get("public_procurement", "unknown"),
                eu_operations=ctx_data.get("eu_operations", "unknown"),
            )
            missing_fields = data.get("missing_fields", [])
            return ClassificationResult(
                state=state,
                reason=reason,
                detected_context=detected_context,
                missing_fields=missing_fields,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Failed to parse classifier response: {exc!r}\nRaw response: {raw!r}"
            ) from exc
