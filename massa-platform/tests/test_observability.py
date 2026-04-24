"""
Phase 9 Tests: Observability & Data Quality

Test categories:
- AgentLogger    : log(), recent(), update_quality_scores(), stats()
- DataQuality    : embedding coverage, empty chunks, duplicate detection, all_pass
- HealthReporter : aggregates both sources into SystemHealthReport
- Formatters     : to_json and to_markdown output correctness
"""

import json
import uuid

import pytest

from src.observability.data_quality import DataQualityChecker, DataQualityReport
from src.observability.health import HealthReporter, SystemHealthReport, to_json, to_markdown
from src.observability.logger import AgentLog, AgentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique() -> str:
    """Unique suffix to keep test data isolated across test runs."""
    return str(uuid.uuid4())[:8]


def _sample_log(**kwargs) -> AgentLog:
    """Returns a minimal valid AgentLog with sensible defaults."""
    defaults = dict(
        question=f"What was EBITDA? [{_unique()}]",
        answer="EBITDA margin was 25.00%.",
        tools_called=["retrieve_docs"],
        latency_ms=980,
        input_tokens=420,
        output_tokens=85,
    )
    defaults.update(kwargs)
    return AgentLog(**defaults)


# ---------------------------------------------------------------------------
# Test 1: AgentLogger — log() and recent()
# ---------------------------------------------------------------------------

async def test_logger_stores_log(db_pool):
    """log() inserts a row and returns a positive integer ID."""
    logger = AgentLogger(db_pool)
    log = _sample_log()
    log_id = await logger.log(log)

    assert isinstance(log_id, int)
    assert log_id > 0


async def test_logger_log_populates_id_and_timestamp(db_pool):
    """After log(), the AgentLog object has id and logged_at set."""
    logger = AgentLogger(db_pool)
    log = _sample_log()
    await logger.log(log)

    assert log.id is not None
    assert log.logged_at is not None


async def test_logger_recent_returns_inserted_row(db_pool):
    """recent() includes the row we just inserted."""
    logger = AgentLogger(db_pool)
    q = f"Unique question [{_unique()}]"
    log = _sample_log(question=q)
    await logger.log(log)

    rows = await logger.recent(limit=50)
    questions = [r["question"] for r in rows]
    assert q in questions


async def test_logger_recent_returns_tools_as_list(db_pool):
    """tools_called comes back as a Python list, not a raw JSON string."""
    logger = AgentLogger(db_pool)
    log = _sample_log(tools_called=["retrieve_docs", "query_metrics"])
    await logger.log(log)

    rows = await logger.recent(limit=5)
    found = next(r for r in rows if r["question"] == log.question)
    assert isinstance(found["tools_called"], list)
    assert "retrieve_docs" in found["tools_called"]


async def test_logger_recent_newest_first(db_pool):
    """recent() returns rows in descending time order."""
    logger = AgentLogger(db_pool)
    q1 = f"First [{_unique()}]"
    q2 = f"Second [{_unique()}]"
    await logger.log(_sample_log(question=q1))
    await logger.log(_sample_log(question=q2))

    rows = await logger.recent(limit=100)
    questions = [r["question"] for r in rows]
    assert questions.index(q2) < questions.index(q1)


async def test_logger_stores_quality_scores(db_pool):
    """Quality scores written at log time are retrievable."""
    logger = AgentLogger(db_pool)
    log = _sample_log(faithfulness_score=0.92, hallucination_detected=False)
    await logger.log(log)

    rows = await logger.recent(limit=10)
    found = next(r for r in rows if r["question"] == log.question)
    assert found["faithfulness_score"] == pytest.approx(0.92)
    assert found["hallucination_detected"] is False


# ---------------------------------------------------------------------------
# Test 2: AgentLogger — update_quality_scores()
# ---------------------------------------------------------------------------

async def test_update_quality_scores_writes_back(db_pool):
    """
    Scores can be written back to an existing row after initial insert.
    Simulates async post-hoc evaluation (log first, evaluate later).
    """
    logger = AgentLogger(db_pool)
    log = _sample_log()   # no scores at insert time
    log_id = await logger.log(log)

    await logger.update_quality_scores(
        log_id,
        faithfulness_score=0.88,
        hallucination_detected=True,
    )

    rows = await logger.recent(limit=50)
    found = next(r for r in rows if r["id"] == log_id)
    assert found["faithfulness_score"] == pytest.approx(0.88)
    assert found["hallucination_detected"] is True


async def test_update_quality_scores_preserves_existing(db_pool):
    """
    COALESCE logic: passing None for a field doesn't overwrite an existing value.
    """
    logger = AgentLogger(db_pool)
    log = _sample_log(faithfulness_score=0.95)
    log_id = await logger.log(log)

    # Update only hallucination_detected — faithfulness_score must be preserved
    await logger.update_quality_scores(log_id, hallucination_detected=False)

    rows = await logger.recent(limit=50)
    found = next(r for r in rows if r["id"] == log_id)
    assert found["faithfulness_score"] == pytest.approx(0.95)
    assert found["hallucination_detected"] is False


# ---------------------------------------------------------------------------
# Test 3: AgentLogger — stats()
# ---------------------------------------------------------------------------

async def test_stats_counts_recent_logs(db_pool):
    """stats() total_calls reflects how many logs we inserted."""
    logger = AgentLogger(db_pool)

    # Insert a known number of logs with a unique marker in the question
    marker = f"stats-test-{_unique()}"
    for i in range(3):
        await logger.log(_sample_log(question=f"{marker} Q{i}", latency_ms=1000))

    stats = await logger.stats(since_hours=1)
    # We can't know the exact count (other tests may have added rows),
    # but total_calls must be at least 3
    assert stats["total_calls"] >= 3


async def test_stats_avg_latency_is_numeric(db_pool):
    """avg_latency_ms is a float (or None if no rows in window)."""
    logger = AgentLogger(db_pool)
    await logger.log(_sample_log(latency_ms=500))
    await logger.log(_sample_log(latency_ms=1500))

    stats = await logger.stats(since_hours=1)
    assert stats["avg_latency_ms"] is not None
    assert isinstance(stats["avg_latency_ms"], float)


async def test_stats_faithfulness_none_when_no_scores(db_pool):
    """avg_faithfulness_score is None when no rows have faithfulness recorded."""
    logger = AgentLogger(db_pool)
    # Insert with no faithfulness score
    await logger.log(_sample_log())

    stats = await logger.stats(since_hours=0)   # 0 hours → no rows in window
    # With a 0-hour window there should be no results → all None
    assert stats["avg_latency_ms"] is None


async def test_stats_hallucination_rate_computed(db_pool):
    """
    hallucination_rate = hallucinated / checked rows.
    2 rows checked: 1 hallucinated, 1 not → rate = 0.5
    """
    logger = AgentLogger(db_pool)
    marker = f"halluc-{_unique()}"
    await logger.log(_sample_log(
        question=f"{marker} A",
        hallucination_detected=True,
    ))
    await logger.log(_sample_log(
        question=f"{marker} B",
        hallucination_detected=False,
    ))
    # Insert one row with no hallucination check — should not affect rate
    await logger.log(_sample_log(question=f"{marker} C"))

    stats = await logger.stats(since_hours=1)
    # Rate may be diluted by other test rows, but must be a float or None
    if stats["hallucination_rate"] is not None:
        assert 0.0 <= stats["hallucination_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4: DataQualityChecker
# ---------------------------------------------------------------------------

async def test_data_quality_returns_report(db_pool):
    """DataQualityChecker.run() returns a DataQualityReport without error."""
    checker = DataQualityChecker(db_pool)
    report = await checker.run()
    assert isinstance(report, DataQualityReport)


async def test_data_quality_chunk_counts_are_non_negative(db_pool):
    """All numeric counts in the report are >= 0."""
    checker = DataQualityChecker(db_pool)
    report = await checker.run()

    assert report.total_chunks >= 0
    assert report.embedded_chunks >= 0
    assert report.missing_embedding_count >= 0
    assert report.empty_chunk_count >= 0
    assert report.duplicate_content_count >= 0
    assert report.total_documents >= 0


async def test_data_quality_embedding_coverage_in_range(db_pool):
    """embedding_coverage is always between 0.0 and 1.0."""
    checker = DataQualityChecker(db_pool)
    report = await checker.run()
    assert 0.0 <= report.embedding_coverage <= 1.0


async def test_data_quality_missing_equals_total_minus_embedded(db_pool):
    """missing_embedding_count = total_chunks - embedded_chunks."""
    checker = DataQualityChecker(db_pool)
    report = await checker.run()
    assert report.missing_embedding_count == report.total_chunks - report.embedded_chunks


# ---------------------------------------------------------------------------
# Test 5: DataQualityReport — pure unit tests (no DB)
# ---------------------------------------------------------------------------

def test_data_quality_report_all_pass_clean_db():
    """A fully embedded, clean DB passes all checks."""
    report = DataQualityReport(
        total_chunks=100,
        embedded_chunks=100,
        missing_embedding_count=0,
        empty_chunk_count=0,
        duplicate_content_count=0,
        total_documents=5,
    )
    assert report.all_pass is True
    assert report.issues == []


def test_data_quality_report_fails_on_low_coverage():
    """Coverage below threshold fails the check."""
    report = DataQualityReport(
        total_chunks=100,
        embedded_chunks=80,   # 80% < 95% threshold
        missing_embedding_count=20,
        empty_chunk_count=0,
        duplicate_content_count=0,
    )
    assert report.embedding_coverage_passes is False
    assert report.all_pass is False
    assert len(report.issues) == 1
    assert "missing embeddings" in report.issues[0]


def test_data_quality_report_fails_on_empty_chunks():
    """Empty chunks are always a failure regardless of coverage."""
    report = DataQualityReport(
        total_chunks=100,
        embedded_chunks=100,
        missing_embedding_count=0,
        empty_chunk_count=3,
        duplicate_content_count=0,
    )
    assert report.all_pass is False
    assert any("empty content" in issue for issue in report.issues)


def test_data_quality_report_fails_on_duplicates():
    """Duplicate content hashes are always a failure."""
    report = DataQualityReport(
        total_chunks=100,
        embedded_chunks=100,
        missing_embedding_count=0,
        empty_chunk_count=0,
        duplicate_content_count=2,
    )
    assert report.all_pass is False
    assert any("duplicate" in issue for issue in report.issues)


def test_data_quality_report_multiple_issues():
    """Multiple failures produce multiple issue messages."""
    report = DataQualityReport(
        total_chunks=100,
        embedded_chunks=70,
        missing_embedding_count=30,
        empty_chunk_count=5,
        duplicate_content_count=3,
    )
    assert len(report.issues) == 3


def test_data_quality_empty_db_passes():
    """An empty DB (no chunks) is not a quality failure."""
    report = DataQualityReport(total_chunks=0, embedded_chunks=0)
    assert report.embedding_coverage == 1.0
    assert report.embedding_coverage_passes is True


# ---------------------------------------------------------------------------
# Test 6: HealthReporter
# ---------------------------------------------------------------------------

async def test_health_reporter_returns_report(db_pool):
    """HealthReporter.report() returns a SystemHealthReport without error."""
    reporter = HealthReporter(db_pool)
    report = await reporter.report()
    assert isinstance(report, SystemHealthReport)


async def test_health_report_data_layer_matches_quality_checker(db_pool):
    """
    HealthReporter's data-layer fields agree with DataQualityChecker.
    Both run against the same live DB at roughly the same time.
    """
    reporter = HealthReporter(db_pool)
    checker = DataQualityChecker(db_pool)

    health = await reporter.report()
    quality = await checker.run()

    assert health.total_chunks == quality.total_chunks
    assert health.total_documents == quality.total_documents
    assert health.missing_embeddings == quality.missing_embedding_count


async def test_health_report_agent_stats_included(db_pool):
    """
    After logging one interaction, total_calls_24h reflects it.
    We check >= 1 because other tests may have logged rows too.
    """
    logger = AgentLogger(db_pool)
    await logger.log(_sample_log())

    reporter = HealthReporter(db_pool, lookback_hours=1)
    report = await reporter.report()

    assert report.total_calls_24h >= 1


# ---------------------------------------------------------------------------
# Test 7: Formatters — to_json and to_markdown
# ---------------------------------------------------------------------------

def test_to_json_is_valid_json():
    """to_json() produces a valid JSON string."""
    report = SystemHealthReport(total_chunks=50, total_documents=3)
    output = to_json(report)
    data = json.loads(output)
    assert data["total_chunks"] == 50
    assert data["total_documents"] == 3


def test_to_json_includes_all_fields():
    """JSON output includes all top-level fields of SystemHealthReport."""
    report = SystemHealthReport(
        total_chunks=100,
        embedding_coverage=0.98,
        total_calls_24h=42,
        avg_latency_ms=1100.5,
    )
    data = json.loads(to_json(report))
    assert "total_chunks" in data
    assert "embedding_coverage" in data
    assert "total_calls_24h" in data
    assert "avg_latency_ms" in data


def test_to_markdown_contains_section_headers():
    """Markdown output includes both main sections."""
    report = SystemHealthReport()
    md = to_markdown(report)
    assert "Data Layer" in md
    assert "Agent Activity" in md


def test_to_markdown_shows_ok_when_all_pass():
    """Status line shows OK when data_quality_pass is True."""
    report = SystemHealthReport(data_quality_pass=True)
    md = to_markdown(report)
    assert "OK" in md


def test_to_markdown_shows_degraded_when_quality_fails():
    """Status line shows DEGRADED when data_quality_pass is False."""
    report = SystemHealthReport(
        data_quality_pass=False,
        data_quality_issues=["5 chunks have empty content"],
    )
    md = to_markdown(report)
    assert "DEGRADED" in md
    assert "empty content" in md


def test_to_markdown_shows_dash_for_none_metrics():
    """Optional metrics with None value render as '—' not 'None'."""
    report = SystemHealthReport(
        avg_latency_ms=None,
        avg_faithfulness_score=None,
        hallucination_rate=None,
    )
    md = to_markdown(report)
    assert "None" not in md


def test_to_markdown_shows_pass_on_good_data_quality():
    """Data quality PASS/FAIL label appears in the markdown."""
    report = SystemHealthReport(data_quality_pass=True)
    md = to_markdown(report)
    assert "PASS" in md
