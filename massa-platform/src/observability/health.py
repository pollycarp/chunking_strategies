"""
System health reporter — aggregates data quality and agent activity into one report.

This is the top-level observability surface for the MASSA platform.
Run it on a schedule (e.g., hourly) and store the JSON output as a CI artefact
or post the Markdown to a Slack channel / GitHub issue.

Design
------
HealthReporter pulls from two sources:
  1. DataQualityChecker  — structural health of the chunks/documents tables
  2. AgentLogger.stats() — runtime health from the last 24 hours of agent logs

These are independent concerns (data quality is about what's in the DB;
agent health is about how the system is performing at runtime) combined into
one report so an operator sees the full picture in one place.

Usage
-----
    reporter = HealthReporter(pool)
    report = await reporter.report()

    print(to_markdown(report))      # for a dashboard / PR comment
    print(to_json(report))          # for a CI artefact / time-series store
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import asyncpg

from src.observability.data_quality import DataQualityChecker, DataQualityReport
from src.observability.logger import AgentLogger


@dataclass
class SystemHealthReport:
    """
    Full system health snapshot combining data quality and agent runtime stats.

    Data quality fields
    -------------------
    total_chunks          : total number of chunks in the DB
    total_documents       : total number of ingested documents
    embedding_coverage    : fraction of chunks with embeddings (0.0–1.0)
    missing_embeddings    : count of chunks without an embedding
    empty_chunks          : count of chunks with blank content
    duplicate_chunks      : count of content hashes appearing more than once

    Agent activity fields (last 24 h)
    ----------------------------------
    total_calls_24h       : number of agent interactions logged
    avg_latency_ms        : mean wall-clock latency in milliseconds
    avg_input_tokens      : mean tokens in the prompt
    avg_output_tokens     : mean tokens in the generated answer
    avg_faithfulness_score: mean faithfulness score (None if not evaluated)
    hallucination_rate    : fraction of checked answers with ungrounded claims

    Quality gates
    -------------
    data_quality_pass     : True if DataQualityReport.all_pass
    """
    # Data layer
    total_chunks: int = 0
    total_documents: int = 0
    embedding_coverage: float = 0.0
    missing_embeddings: int = 0
    empty_chunks: int = 0
    duplicate_chunks: int = 0

    # Agent activity
    total_calls_24h: int = 0
    avg_latency_ms: float | None = None
    avg_input_tokens: float | None = None
    avg_output_tokens: float | None = None
    avg_faithfulness_score: float | None = None
    hallucination_rate: float | None = None

    # Gate
    data_quality_pass: bool = True
    data_quality_issues: list[str] = None   # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.data_quality_issues is None:
            self.data_quality_issues = []

    @property
    def all_pass(self) -> bool:
        return self.data_quality_pass


class HealthReporter:
    """
    Builds a SystemHealthReport by querying the DB.

    Parameters
    ----------
    pool           : asyncpg.Pool — shared connection pool
    lookback_hours : how many hours of agent logs to include in runtime stats
    """

    def __init__(self, pool: asyncpg.Pool, lookback_hours: int = 24) -> None:
        self._pool = pool
        self._lookback_hours = lookback_hours
        self._quality_checker = DataQualityChecker(pool)
        self._logger = AgentLogger(pool)

    async def report(self) -> SystemHealthReport:
        """
        Runs all checks and returns a SystemHealthReport.

        Data quality and agent stats are fetched sequentially so the report
        represents a consistent snapshot from roughly the same point in time.
        """
        dq: DataQualityReport = await self._quality_checker.run()
        stats: dict = await self._logger.stats(since_hours=self._lookback_hours)

        return SystemHealthReport(
            # Data layer
            total_chunks=dq.total_chunks,
            total_documents=dq.total_documents,
            embedding_coverage=dq.embedding_coverage,
            missing_embeddings=dq.missing_embedding_count,
            empty_chunks=dq.empty_chunk_count,
            duplicate_chunks=dq.duplicate_content_count,

            # Agent activity
            total_calls_24h=stats["total_calls"],
            avg_latency_ms=stats["avg_latency_ms"],
            avg_input_tokens=stats["avg_input_tokens"],
            avg_output_tokens=stats["avg_output_tokens"],
            avg_faithfulness_score=stats["avg_faithfulness_score"],
            hallucination_rate=stats["hallucination_rate"],

            # Gate
            data_quality_pass=dq.all_pass,
            data_quality_issues=dq.issues,
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def to_json(report: SystemHealthReport, indent: int = 2) -> str:
    """Serialises the report to JSON. Suitable for CI artefacts."""
    data = asdict(report)
    return json.dumps(data, indent=indent, default=str)


def to_markdown(report: SystemHealthReport) -> str:
    """
    Formats the report as GitHub-flavoured Markdown.
    Suitable for posting as a PR comment, Slack message, or dashboard widget.
    """

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _opt_pct(v: float | None) -> str:
        return _pct(v) if v is not None else "—"

    def _opt_ms(v: float | None) -> str:
        return f"{v:.0f} ms" if v is not None else "—"

    def _opt_tok(v: float | None) -> str:
        return f"{v:.0f}" if v is not None else "—"

    status_icon = "OK" if report.all_pass else "DEGRADED"

    lines = [
        "## System Health Report",
        "",
        f"**Status:** {status_icon}",
        "",
        "### Data Layer",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total documents | {report.total_documents} |",
        f"| Total chunks | {report.total_chunks} |",
        f"| Embedding coverage | {_pct(report.embedding_coverage)} |",
        f"| Missing embeddings | {report.missing_embeddings} |",
        f"| Empty chunks | {report.empty_chunks} |",
        f"| Duplicate content hashes | {report.duplicate_chunks} |",
        f"| Data quality | {'PASS' if report.data_quality_pass else 'FAIL'} |",
        "",
    ]

    if report.data_quality_issues:
        lines += ["**Issues detected:**", ""]
        for issue in report.data_quality_issues:
            lines.append(f"- {issue}")
        lines.append("")

    lines += [
        "### Agent Activity (last 24 h)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total interactions | {report.total_calls_24h} |",
        f"| Avg latency | {_opt_ms(report.avg_latency_ms)} |",
        f"| Avg input tokens | {_opt_tok(report.avg_input_tokens)} |",
        f"| Avg output tokens | {_opt_tok(report.avg_output_tokens)} |",
        f"| Avg faithfulness score | {_opt_pct(report.avg_faithfulness_score)} |",
        f"| Hallucination rate | {_opt_pct(report.hallucination_rate)} |",
        "",
    ]

    return "\n".join(lines)
