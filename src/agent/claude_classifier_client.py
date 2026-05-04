from __future__ import annotations

import json


_SYSTEM = """\
You are a regulatory-change materiality classifier for UK corporate compliance.

Given a passage of changed regulatory text, decide whether the change is MATERIAL
to a company's compliance obligations.

Material changes include: new or revised thresholds (financial, energy, headcount),
new mandatory reporting requirements, changed deadlines, changed scope definitions,
new penalties or enforcement mechanisms.

Not material: typographical fixes, cosmetic restructuring, clarifying footnotes that
do not alter obligations.

Respond with valid JSON only — no prose, no markdown fences:
{"verdict": "material" | "not_material", "reasoning": "<one sentence>"}
"""


class AnthropicClassifierClient:
    def __init__(self, client) -> None:
        self._client = client

    def classify(self, change_text: str) -> dict:
        message = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": change_text}],
        )
        raw = message.content[0].text.strip()
        return json.loads(raw)
