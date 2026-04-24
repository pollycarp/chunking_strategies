"""
Evaluation reporter — formats EvalReport as JSON or Markdown.

Why structured output?
An eval run that only prints to stdout is hard to track over time. JSON output
can be stored as an artefact in CI (GitHub Actions, etc.) and compared across
runs to detect regressions. Markdown output can be posted as a PR comment.

EvalReport captures all retrieval and answer quality metrics in one place
so stakeholders can see the full picture at a glance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class EvalReport:
    """
    Aggregated evaluation results for one benchmark run.

    All metric fields are floats in [0.0, 1.0] unless otherwise noted.
    """
    # Retrieval metrics
    recall_at_5: float = 0.0
    recall_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0

    # Answer quality metrics (from LLM-as-judge faithfulness)
    avg_faithfulness_score: float = 0.0
    faithfulness_pass_rate: float = 0.0   # fraction of answers with score >= threshold

    # Hallucination metrics (from keyword-based check)
    avg_grounding_rate: float = 0.0
    hallucination_rate: float = 0.0       # fraction of answers with ungrounded claims

    # Run metadata
    total_questions: int = 0
    k: int = 5

    # Thresholds used in regression gate
    recall_at_5_threshold: float = 0.6
    mrr_threshold: float = 0.4

    @property
    def recall_passes(self) -> bool:
        return self.recall_at_5 >= self.recall_at_5_threshold

    @property
    def mrr_passes(self) -> bool:
        return self.mrr >= self.mrr_threshold

    @property
    def all_pass(self) -> bool:
        return self.recall_passes and self.mrr_passes


def to_json(report: EvalReport, indent: int = 2) -> str:
    """Serialises the report to a JSON string."""
    return json.dumps(asdict(report), indent=indent)


def to_markdown(report: EvalReport) -> str:
    """
    Formats the report as a GitHub-flavoured Markdown table.
    Suitable for posting as a PR comment or saving as an artefact.
    """

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _pass(ok: bool) -> str:
        return "✅ PASS" if ok else "❌ FAIL"

    lines = [
        "## Evaluation Report",
        "",
        f"**Questions evaluated:** {report.total_questions}  |  **K:** {report.k}",
        "",
        "### Retrieval Metrics",
        "",
        "| Metric | Score | Threshold | Status |",
        "|--------|-------|-----------|--------|",
        f"| Recall@5 | {_pct(report.recall_at_5)} | {_pct(report.recall_at_5_threshold)} | {_pass(report.recall_passes)} |",
        f"| Recall@3 | {_pct(report.recall_at_3)} | — | — |",
        f"| Precision@5 | {_pct(report.precision_at_5)} | — | — |",
        f"| MRR | {report.mrr:.3f} | {report.mrr_threshold:.1f} | {_pass(report.mrr_passes)} |",
        "",
        "### Answer Quality Metrics",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Avg Faithfulness Score | {_pct(report.avg_faithfulness_score)} |",
        f"| Faithfulness Pass Rate | {_pct(report.faithfulness_pass_rate)} |",
        f"| Avg Grounding Rate | {_pct(report.avg_grounding_rate)} |",
        f"| Hallucination Rate | {_pct(report.hallucination_rate)} |",
        "",
        f"**Overall:** {'✅ All checks pass' if report.all_pass else '❌ One or more checks failed'}",
        "",
    ]
    return "\n".join(lines)
