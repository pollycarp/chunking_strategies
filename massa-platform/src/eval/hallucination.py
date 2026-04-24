"""
Hallucination detector — keyword-based claim grounding check.

Difference from faithfulness (answer_metrics.py):
    Faithfulness (LLM-as-judge): asks an LLM to holistically evaluate whether
        the answer is grounded. High accuracy, requires API call.
    Hallucination detector (this file): extracts specific numeric claims and
        checks whether those exact values appear in the retrieved context.
        No API call needed — runs instantly in CI.

Why keyword-based for hallucination?
Numbers are the most dangerous type of hallucination in finance. If an LLM
says "EBITDA margin was 27.3%" but the retrieved context says "25.0%", that's
a material error — the kind an analyst might rely on. Checking whether specific
numeric values in the answer appear in the context catches this class of error
without needing an LLM call.

This is not a replacement for faithfulness evaluation — it's a fast, cheap
first filter that runs on every response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Matches: integers, decimals, percentages, currency amounts
# Examples: 25.00%, 100,000, $45.2M, 2.3x, Q3 2024
_NUMERIC_CLAIM_PATTERN = re.compile(
    r"""
    (?:
        \$[\d,]+(?:\.\d+)?[MBK]?   |   # $45.2M  $100,000
        [\d,]+(?:\.\d+)?\s*%        |   # 25.00%  23.4 %
        [\d,]+(?:\.\d+)?x           |   # 2.3x  leverage
        [\d]{1,3}(?:,\d{3})*(?:\.\d+)?  # 100,000  45.2
    )
    """,
    re.VERBOSE,
)


@dataclass
class HallucinationResult:
    """
    Result of a hallucination check for one answer.

    grounded_claims  : numeric values from the answer that appear in the context
    ungrounded_claims: numeric values from the answer NOT found in the context
    has_hallucination: True if any numeric claim is ungrounded
    """
    grounded_claims: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)

    @property
    def has_hallucination(self) -> bool:
        return len(self.ungrounded_claims) > 0

    @property
    def grounding_rate(self) -> float:
        """Fraction of numeric claims that are grounded. 1.0 = fully grounded."""
        total = len(self.grounded_claims) + len(self.ungrounded_claims)
        if total == 0:
            return 1.0  # no numeric claims → no hallucination risk
        return len(self.grounded_claims) / total


def extract_numeric_claims(text: str) -> list[str]:
    """
    Extracts all numeric values and financial figures from text.

    Returns a deduplicated list, preserving order of first appearance.
    Normalises whitespace within matches.
    """
    matches = _NUMERIC_CLAIM_PATTERN.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for match in matches:
        normalised = re.sub(r"\s+", "", match)  # remove internal spaces
        if normalised not in seen:
            seen.add(normalised)
            result.append(normalised)
    return result


def is_claim_in_context(claim: str, context: str) -> bool:
    """
    Checks whether a numeric claim appears verbatim in the context.

    Normalises both claim and context (removes spaces, lowercases) before
    matching to handle minor formatting differences (e.g. "25.00 %" vs "25.00%").
    """
    normalised_claim = re.sub(r"\s+", "", claim).lower()
    normalised_context = re.sub(r"\s+", "", context).lower()
    return normalised_claim in normalised_context


def check_hallucination(answer: str, context: str) -> HallucinationResult:
    """
    Checks whether numeric claims in the answer are grounded in the context.

    Parameters:
        answer  : the agent's generated answer
        context : the retrieved chunks text used to generate the answer

    Returns a HallucinationResult listing grounded and ungrounded numeric claims.

    Limitations:
    - Only checks numeric/financial values, not named entities or conceptual claims
    - False positives: a number in the answer may be a different metric than the
      same number in the context (e.g. both "25%" but referring to different things)
    - False negatives: the LLM may paraphrase numbers (e.g. "one quarter" vs "25%")
    For comprehensive evaluation, combine with evaluate_faithfulness() (LLM-as-judge).
    """
    claims = extract_numeric_claims(answer)

    if not claims:
        return HallucinationResult()

    grounded: list[str] = []
    ungrounded: list[str] = []

    for claim in claims:
        if is_claim_in_context(claim, context):
            grounded.append(claim)
        else:
            ungrounded.append(claim)

    return HallucinationResult(
        grounded_claims=grounded,
        ungrounded_claims=ungrounded,
    )
