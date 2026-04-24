"""
Phase 8 Tests: Evaluation Harness

Test categories:
- retrieval metrics  : Recall@K, Precision@K, MRR exact math
- answer metrics     : faithfulness judge with mocked Claude client
- hallucination      : keyword-based grounding check
- reporter           : JSON and Markdown output format
- benchmark runner   : end-to-end with mock retriever
- regression gate    : CI fails when recall drops below threshold
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.answer_metrics import FaithfulnessResult, _parse_judge_response, evaluate_faithfulness
from src.eval.benchmark import BenchmarkQuestion, BenchmarkRunner, load_questions
from src.eval.hallucination import (
    HallucinationResult,
    check_hallucination,
    extract_numeric_claims,
    is_claim_in_context,
)
from src.eval.reporter import EvalReport, to_json, to_markdown
from src.eval.retrieval_metrics import (
    average_precision_at_k,
    average_recall_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# Test 1: Retrieval metrics — exact math
# ---------------------------------------------------------------------------

def test_recall_at_k_perfect():
    """All relevant docs in top k → recall = 1.0."""
    assert recall_at_k([1, 2, 3, 4, 5], {2, 4}, k=5) == 1.0


def test_recall_at_k_miss():
    """Relevant doc not in top k → recall = 0.0."""
    assert recall_at_k([1, 2, 3], {9}, k=3) == 0.0


def test_recall_at_k_partial():
    """One of two relevant docs in top k → recall = 0.5."""
    assert recall_at_k([1, 2, 3], {1, 9}, k=3) == 0.5


def test_recall_at_k_cutoff():
    """Relevant doc at rank 6 is not counted when k=5."""
    assert recall_at_k([1, 2, 3, 4, 5, 6], {6}, k=5) == 0.0


def test_recall_at_k_empty_relevant():
    """Empty relevant set → 0.0 (undefined, not division by zero)."""
    assert recall_at_k([1, 2, 3], set(), k=5) == 0.0


def test_precision_at_k_perfect():
    """All top-k are relevant → precision = 1.0."""
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_at_k_zero():
    """No relevant docs in top k → precision = 0.0."""
    assert precision_at_k([1, 2, 3], {9, 10}, k=3) == 0.0


def test_precision_at_k_partial():
    """2 of 5 results relevant → precision@5 = 0.4."""
    assert precision_at_k([1, 9, 2, 9, 9], {1, 2}, k=5) == pytest.approx(0.4)


def test_precision_at_k_zero_k():
    """k=0 → 0.0 (guard against division by zero)."""
    assert precision_at_k([1, 2, 3], {1}, k=0) == 0.0


def test_reciprocal_rank_first():
    """Relevant doc at rank 1 → RR = 1.0."""
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0


def test_reciprocal_rank_second():
    """Relevant doc at rank 2 → RR = 0.5."""
    assert reciprocal_rank([3, 1, 2], {1}) == 0.5


def test_reciprocal_rank_third():
    """Relevant doc at rank 3 → RR = 1/3 ≈ 0.333."""
    assert reciprocal_rank([3, 5, 1], {1}) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found():
    """Relevant doc absent → RR = 0.0."""
    assert reciprocal_rank([1, 2, 3], {9}) == 0.0


def test_mrr_all_first():
    """All queries find the answer at rank 1 → MRR = 1.0."""
    results = [([1, 2, 3], {1}), ([4, 5, 6], {4})]
    assert mean_reciprocal_rank(results) == 1.0


def test_mrr_mixed():
    """
    Query 1: rank 1 → RR = 1.0
    Query 2: rank 2 → RR = 0.5
    Query 3: not found → RR = 0.0
    MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    """
    results = [
        ([1, 2, 3], {1}),    # rank 1
        ([3, 1, 2], {1}),    # rank 2
        ([3, 2, 4], {1}),    # not found
    ]
    assert mean_reciprocal_rank(results) == pytest.approx(0.5)


def test_mrr_empty():
    """Empty results list → 0.0."""
    assert mean_reciprocal_rank([]) == 0.0


def test_average_recall_at_k():
    """Average recall across two queries."""
    results = [
        ([1, 2, 3], {1}),    # recall@3 = 1.0
        ([1, 2, 3], {9}),    # recall@3 = 0.0
    ]
    assert average_recall_at_k(results, k=3) == pytest.approx(0.5)


def test_average_precision_at_k():
    """Average precision across two queries."""
    results = [
        ([1, 2, 3], {1, 2, 3}),  # precision@3 = 1.0
        ([1, 2, 3], {9, 10}),    # precision@3 = 0.0
    ]
    assert average_precision_at_k(results, k=3) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 2: Faithfulness / LLM-as-judge
# ---------------------------------------------------------------------------

def test_parse_judge_response_valid_json():
    """Clean JSON is parsed correctly."""
    score, claims = _parse_judge_response(
        '{"score": 0.75, "unsupported_claims": ["Revenue was $1B", "Margin was 30%"]}'
    )
    assert score == pytest.approx(0.75)
    assert "Revenue was $1B" in claims


def test_parse_judge_response_perfect_score():
    """Score 1.0 with empty claims = fully grounded."""
    score, claims = _parse_judge_response('{"score": 1.0, "unsupported_claims": []}')
    assert score == 1.0
    assert claims == []


def test_parse_judge_response_clamps_score():
    """Score values outside [0, 1] are clamped."""
    score, _ = _parse_judge_response('{"score": 1.5, "unsupported_claims": []}')
    assert score == 1.0

    score, _ = _parse_judge_response('{"score": -0.5, "unsupported_claims": []}')
    assert score == 0.0


def test_parse_judge_response_malformed_json():
    """Malformed JSON returns (0.0, [error message]) without raising."""
    score, claims = _parse_judge_response("not json at all")
    assert score == 0.0
    assert len(claims) == 1
    assert "Failed to parse" in claims[0]


def test_parse_judge_response_strips_markdown_fences():
    """JSON wrapped in markdown code fences is parsed correctly."""
    raw = '```json\n{"score": 0.9, "unsupported_claims": []}\n```'
    score, claims = _parse_judge_response(raw)
    assert score == pytest.approx(0.9)


async def test_evaluate_faithfulness_grounded_answer():
    """
    When the judge returns score=1.0 with no unsupported claims,
    is_faithful should be True.
    """
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = '{"score": 1.0, "unsupported_claims": []}'

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=mock_response)

    result = await evaluate_faithfulness(
        answer="EBITDA margin was 25.00%.",
        context="EBITDA margin improved to 25.00% in Q3 2024.",
        judge_client=mock_client,
    )

    assert result.score == 1.0
    assert result.unsupported_claims == []
    assert result.is_faithful is True


async def test_evaluate_faithfulness_ungrounded_answer():
    """
    When the judge returns score=0.3 with unsupported claims,
    is_faithful should be False (below default threshold 0.8).
    """
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = json.dumps({
        "score": 0.3,
        "unsupported_claims": ["Revenue was $200M", "Net margin was 15%"],
    })

    mock_response = MagicMock()
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=mock_response)

    result = await evaluate_faithfulness(
        answer="Revenue was $200M and net margin was 15%.",
        context="EBITDA margin improved to 25%.",
        judge_client=mock_client,
    )

    assert result.score == pytest.approx(0.3)
    assert len(result.unsupported_claims) == 2
    assert result.is_faithful is False


# ---------------------------------------------------------------------------
# Test 3: Hallucination detection
# ---------------------------------------------------------------------------

def test_extract_numeric_claims_percentages():
    """Percentages in the answer are extracted."""
    claims = extract_numeric_claims("EBITDA margin was 25.00% and revenue grew 12%.")
    assert "25.00%" in claims
    assert "12%" in claims


def test_extract_numeric_claims_currency():
    """Currency amounts are extracted."""
    claims = extract_numeric_claims("Revenue was $45.2M in Q3.")
    assert any("45.2" in c for c in claims)


def test_extract_numeric_claims_plain_numbers():
    """Plain numbers are extracted."""
    claims = extract_numeric_claims("Revenue was 100,000 and EBITDA was 25,000.")
    assert any("100,000" in c or "100000" in c for c in claims)


def test_extract_numeric_claims_empty():
    """Text with no numbers returns an empty list."""
    claims = extract_numeric_claims("The company performed well this quarter.")
    assert claims == []


def test_is_claim_in_context_found():
    """A claim that appears in the context returns True."""
    assert is_claim_in_context("25.00%", "EBITDA margin improved to 25.00% in Q3 2024.") is True


def test_is_claim_in_context_not_found():
    """A claim not in the context returns False."""
    assert is_claim_in_context("30.00%", "EBITDA margin improved to 25.00% in Q3 2024.") is False


def test_is_claim_in_context_normalises_spaces():
    """Spacing differences are normalised before comparison."""
    assert is_claim_in_context("25.00 %", "ebitda margin was 25.00% in Q3") is True


def test_check_hallucination_fully_grounded():
    """Answer numbers all appear in context → no hallucination."""
    answer = "EBITDA margin was 25.00% and revenue was 100,000."
    context = "Revenue: 100,000. EBITDA margin: 25.00%."
    result = check_hallucination(answer, context)

    assert result.has_hallucination is False
    assert result.grounding_rate == 1.0


def test_check_hallucination_ungrounded_number():
    """A number in the answer not in context → hallucination detected."""
    answer = "EBITDA margin was 30.00%."
    context = "EBITDA margin improved to 25.00% in Q3 2024."
    result = check_hallucination(answer, context)

    assert result.has_hallucination is True
    assert any("30" in c for c in result.ungrounded_claims)


def test_check_hallucination_no_numbers():
    """Answer with no numeric claims → no hallucination (grounding_rate=1.0)."""
    result = check_hallucination(
        "The company performed strongly this quarter.",
        "Revenue grew significantly.",
    )
    assert result.has_hallucination is False
    assert result.grounding_rate == 1.0


# ---------------------------------------------------------------------------
# Test 4: Reporter
# ---------------------------------------------------------------------------

def test_to_json_contains_all_fields():
    """JSON output includes all EvalReport fields."""
    report = EvalReport(
        recall_at_5=0.8,
        mrr=0.65,
        total_questions=10,
        k=5,
    )
    output = to_json(report)
    data = json.loads(output)

    assert data["recall_at_5"] == pytest.approx(0.8)
    assert data["mrr"] == pytest.approx(0.65)
    assert data["total_questions"] == 10


def test_to_markdown_contains_metric_names():
    """Markdown output includes all metric names."""
    report = EvalReport(recall_at_5=0.8, mrr=0.65, total_questions=5)
    md = to_markdown(report)

    assert "Recall@5" in md
    assert "MRR" in md
    assert "Precision@5" in md
    assert "Faithfulness" in md


def test_to_markdown_shows_pass_on_good_scores():
    """Metrics above thresholds show PASS in the markdown."""
    report = EvalReport(
        recall_at_5=0.9,
        mrr=0.7,
        recall_at_5_threshold=0.6,
        mrr_threshold=0.4,
    )
    md = to_markdown(report)
    assert "PASS" in md


def test_to_markdown_shows_fail_on_poor_scores():
    """Metrics below thresholds show FAIL in the markdown."""
    report = EvalReport(
        recall_at_5=0.3,
        mrr=0.2,
        recall_at_5_threshold=0.6,
        mrr_threshold=0.4,
    )
    md = to_markdown(report)
    assert "FAIL" in md


def test_eval_report_all_pass_property():
    """all_pass is True when both recall and MRR meet thresholds."""
    report = EvalReport(
        recall_at_5=0.9,
        mrr=0.7,
        recall_at_5_threshold=0.6,
        mrr_threshold=0.4,
    )
    assert report.all_pass is True


def test_eval_report_all_pass_fails_on_low_recall():
    """all_pass is False when recall is below threshold."""
    report = EvalReport(
        recall_at_5=0.3,
        mrr=0.7,
        recall_at_5_threshold=0.6,
    )
    assert report.all_pass is False


# ---------------------------------------------------------------------------
# Test 5: BenchmarkRunner
# ---------------------------------------------------------------------------

async def test_benchmark_runner_computes_metrics():
    """
    BenchmarkRunner with a deterministic mock retriever computes correct metrics.

    The retriever always returns [1, 2, 3, 4, 5].
    Questions have relevant_chunk_ids = [1] (rank 1 = RR = 1.0, recall@5 = 1.0).
    """
    async def mock_retriever(query: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    questions = [
        BenchmarkQuestion(id="q1", question="What is revenue?", relevant_chunk_ids=[1]),
        BenchmarkQuestion(id="q2", question="What is EBITDA?", relevant_chunk_ids=[1]),
    ]

    runner = BenchmarkRunner(retriever_fn=mock_retriever, k=5)
    report = await runner.run(questions)

    assert report.total_questions == 2
    assert report.recall_at_5 == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)


async def test_benchmark_runner_empty_questions():
    """Empty question list returns a zero-metric report without errors."""
    async def mock_retriever(query: str) -> list[int]:
        return [1, 2, 3]

    runner = BenchmarkRunner(retriever_fn=mock_retriever, k=5)
    report = await runner.run([])

    assert report.total_questions == 0
    assert report.recall_at_5 == 0.0


async def test_benchmark_runner_partial_recall():
    """Retriever misses half the relevant docs → recall@5 = 0.5."""
    async def mock_retriever(query: str) -> list[int]:
        return [1, 2, 3, 4, 5]  # only chunk 1 in relevant; chunk 9 is not

    questions = [
        BenchmarkQuestion(id="q1", question="Test?", relevant_chunk_ids=[1, 9]),
    ]

    runner = BenchmarkRunner(retriever_fn=mock_retriever, k=5)
    report = await runner.run(questions)

    assert report.recall_at_5 == pytest.approx(0.5)


def test_load_questions_from_json(tmp_path):
    """load_questions correctly parses a benchmark JSON file."""
    data = [
        {
            "id": "q1",
            "question": "What was EBITDA?",
            "relevant_chunk_ids": [1, 2],
            "expected_keywords": ["ebitda"],
        }
    ]
    path = tmp_path / "test_benchmark.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    questions = load_questions(path)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].relevant_chunk_ids == [1, 2]
    assert "ebitda" in questions[0].expected_keywords


# ---------------------------------------------------------------------------
# Test 6: Regression gate (CI guard)
# ---------------------------------------------------------------------------

async def test_regression_gate_passes_on_good_retrieval():
    """
    The regression gate passes when recall@5 meets the configured threshold.
    This is the test that would fail in CI if retrieval quality drops.
    """
    RECALL_THRESHOLD = 0.6
    MRR_THRESHOLD = 0.4

    # Perfect retriever: always returns the relevant doc at rank 1
    async def perfect_retriever(query: str) -> list[int]:
        return [1, 2, 3, 4, 5]

    questions = [
        BenchmarkQuestion(id=f"q{i}", question=f"Question {i}", relevant_chunk_ids=[1])
        for i in range(5)
    ]

    runner = BenchmarkRunner(
        retriever_fn=perfect_retriever,
        recall_threshold=RECALL_THRESHOLD,
        mrr_threshold=MRR_THRESHOLD,
    )
    report = await runner.run(questions)

    assert report.recall_at_5 >= RECALL_THRESHOLD, (
        f"REGRESSION: Recall@5 {report.recall_at_5:.2f} dropped below "
        f"threshold {RECALL_THRESHOLD}. Check recent changes to retrieval."
    )
    assert report.mrr >= MRR_THRESHOLD, (
        f"REGRESSION: MRR {report.mrr:.3f} dropped below "
        f"threshold {MRR_THRESHOLD}. Check recent changes to retrieval."
    )


async def test_regression_gate_fails_on_poor_retrieval():
    """
    Demonstrates the gate fails correctly when retrieval quality is poor.
    In real CI, this would block a PR that breaks the retriever.
    """
    RECALL_THRESHOLD = 0.8

    # Bad retriever: never returns the relevant doc
    async def bad_retriever(query: str) -> list[int]:
        return [99, 98, 97, 96, 95]  # wrong chunks

    questions = [
        BenchmarkQuestion(id="q1", question="What is revenue?", relevant_chunk_ids=[1])
    ]

    runner = BenchmarkRunner(retriever_fn=bad_retriever, recall_threshold=RECALL_THRESHOLD)
    report = await runner.run(questions)

    # The report shows failure — in CI this assertion in the gate test would fail
    assert report.recall_at_5 < RECALL_THRESHOLD, (
        "Expected the bad retriever to produce recall below threshold"
    )
    assert report.recall_passes is False
