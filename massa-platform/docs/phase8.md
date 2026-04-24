# Phase 8: Evaluation Harness

## What We Built

A complete evaluation framework that measures the quality of the MASSA platform across two dimensions: **retrieval quality** (did the system find the right documents?) and **answer quality** (is the generated answer faithful to what was retrieved?). The harness is designed to act as a regression gate in CI — if retrieval quality drops below a threshold after a code change, the build fails automatically.

```
Benchmark Dataset (JSON)
        │
        ▼
BenchmarkRunner.run(questions)
        │
        ├── for each question:
        │       retriever_fn(question) → [chunk_id, chunk_id, ...]
        │       compare to relevant_chunk_ids
        │
        ├── retrieval_metrics.py
        │       Recall@K, Precision@K, Reciprocal Rank
        │
        ├── reporter.py
        │       EvalReport → JSON artefact / Markdown PR comment
        │
        └── (optionally, per answer):
                hallucination.py     → fast numeric claim check (no API)
                answer_metrics.py    → faithfulness score (LLM-as-judge via Claude)
```

---

## Why Evaluation Matters in a RAG System

When you build a retrieval-augmented system you have two distinct failure modes:

| Failure mode | What goes wrong | Who suffers |
|---|---|---|
| **Retrieval failure** | The right document is never fetched | The LLM cannot possibly answer correctly |
| **Generation failure** | The right document was fetched, but the LLM ignored it or invented facts | The answer looks confident but is wrong |

Traditional software testing (unit tests, integration tests) can verify that code runs without errors. It cannot verify that the **quality of results** stays high as you change embedding models, chunking strategies, or reranking logic. That is what the evaluation harness does.

---

## Core Concepts

### Retrieval Metrics

All retrieval metrics work on the same inputs:

- `retrieved_ids` — the ordered list of chunk IDs your retriever returned (rank 1 first)
- `relevant_ids` — the set of chunk IDs that are considered correct for this query
- `k` — the cutoff (how many results do we consider?)

#### Recall@K

> "Of all the chunks that could answer this question, what fraction did we actually return?"

```
Recall@K = |{relevant} ∩ {top-K retrieved}| / |{relevant}|
```

**Example:** The query "What was EBITDA in Q3?" has two relevant chunks (IDs 12 and 15). Your retriever returns [12, 7, 3, 15, 9]. With k=5, both relevant chunks are in the top 5, so Recall@5 = 2/2 = **1.0**.

**Why it matters:** A low Recall@K means the LLM never even sees the right information. No amount of prompt engineering can fix a retriever that misses relevant chunks.

**The coverage vs. noise tradeoff:** A retriever that returns every chunk in the database scores Recall@K = 1.0 — but it's useless because the LLM gets buried in noise. That's why we also measure Precision.

#### Precision@K

> "Of the chunks we returned, what fraction were actually relevant?"

```
Precision@K = |{relevant} ∩ {top-K retrieved}| / K
```

**Example:** K=5, retrieved [12, 7, 3, 15, 9], relevant = {12, 15}. Two of the five results are relevant. Precision@5 = 2/5 = **0.4**.

**Why it matters:** Low precision means the LLM is given a lot of irrelevant context alongside the useful chunks. This increases the chance of the LLM getting confused or citing the wrong source.

#### Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)

> "How high up in the ranked list does the first correct answer appear?"

```
RR = 1 / rank_of_first_relevant_chunk
MRR = average RR across all queries
```

**Examples:**

| Scenario | Result |
|---|---|
| Relevant chunk at rank 1 | RR = 1/1 = **1.0** (perfect) |
| Relevant chunk at rank 2 | RR = 1/2 = **0.5** |
| Relevant chunk at rank 5 | RR = 1/5 = **0.2** |
| Relevant chunk not found | RR = **0.0** |

**Why it matters for finance:** Financial analysts using a chat interface expect the most relevant answer to surface immediately. A system that buries the right chunk at rank 10 is nearly useless in practice, even if Recall@10 = 1.0.

**MRR** averages RR over your entire benchmark dataset. A dataset-level MRR of 0.7+ is generally considered good for financial Q&A.

#### Why use all three metrics together?

| Using only… | The problem |
|---|---|
| Recall@K | A retriever returning every chunk scores 1.0 |
| Precision@K | A retriever returning one perfectly matched chunk scores 1.0, ignoring whether all relevant docs were found |
| MRR | Doesn't capture whether all relevant chunks were found, only where the first one appears |

Together, the three metrics capture the quality/noise/ranking tradeoff in one picture.

---

### Answer Quality Metrics

#### Hallucination Detection (`hallucination.py`)

The hallucination detector extracts all numeric values from the generated answer and checks whether each one appears verbatim in the retrieved context. No API call required — it runs instantly.

**Why numbers?** In financial analysis, numeric hallucinations are the most dangerous. If the LLM says "EBITDA margin was 27.3%" but the retrieved chunk says "25.0%", that's a material error an analyst might act on. Conceptual errors ("the company had a good quarter") are hard to verify automatically, but numeric errors are not.

**What it catches:**
- Percentages: `25.00%`, `12%`
- Currency amounts: `$45.2M`, `$100,000`
- Plain numbers with commas: `100,000`
- Leverage ratios: `2.3x`

**What it misses (known limitations):**
- Paraphrased numbers ("one quarter" vs "25%")
- Named entity errors ("Goldman Sachs" vs "Morgan Stanley")
- Conceptual/logical errors that don't involve numbers

The hallucination detector is a fast first filter. It runs on every response in CI. For deeper evaluation, combine it with the faithfulness judge.

#### Faithfulness Evaluation — LLM-as-Judge (`answer_metrics.py`)

The faithfulness judge asks Claude to holistically evaluate whether the generated answer is grounded in the retrieved context. This is sometimes called "contextual faithfulness" or "groundedness" in the RAG evaluation literature (RAGAS, TruLens, and similar frameworks use the same approach).

**How it works:**

1. The judge receives: the retrieved context + the generated answer
2. The judge identifies every factual claim in the answer
3. For each claim, it checks whether the context supports it
4. It returns a score (0.0–1.0) and a list of unsupported claims

**Score interpretation:**

| Score | Meaning |
|---|---|
| 1.0 | Every claim in the answer is supported by the context |
| 0.8+ | Mostly grounded — default pass threshold |
| 0.5 | Half the claims are unsupported — significant concern |
| 0.0 | The answer is essentially fabricated |

**Why use an LLM as a judge?** Answer quality cannot be measured with rule-based assertions. The same fact can be expressed in dozens of ways. An LLM can understand semantic equivalence ("EBITDA margin improved" ≈ "profitability increased") in a way that a keyword matcher cannot.

**The judge prompt is structured to return valid JSON**, making it machine-parseable for automated CI use. The `_parse_judge_response()` function strips markdown fences, clamps scores to [0,1], and falls back gracefully on malformed output rather than crashing the evaluation run.

---

### Benchmark Runner (`benchmark.py`)

The `BenchmarkRunner` is dependency-injected — it accepts any async callable that maps a query string to a list of chunk IDs. This means it can run against:

- A mock retriever (in unit tests — instant, no DB required)
- The real `HybridRetriever` (in integration tests — requires DB + embeddings)
- A future retriever (no code changes to the runner itself)

**Benchmark dataset format (JSON):**

```json
[
  {
    "id": "q001",
    "question": "What was the EBITDA margin for Q3 2024?",
    "relevant_chunk_ids": [12, 15],
    "expected_keywords": ["25", "ebitda", "margin"]
  }
]
```

`relevant_chunk_ids` must be populated from your actual database — these are the real chunk IDs that contain the answer for this question. The benchmark dataset is the investment that pays off over time: every time you change the retriever, you run the same questions and compare scores.

---

### Reporter (`reporter.py`)

The `EvalReport` dataclass captures all metrics in one object. Two output formats:

**JSON** — stored as a CI artefact and compared across runs to track score trends over time.

**Markdown** — suitable for posting as an automated PR comment, showing metrics with PASS/FAIL status against configured thresholds.

**Sample Markdown output:**

```
## Evaluation Report

**Questions evaluated:** 50  |  **K:** 5

### Retrieval Metrics

| Metric      | Score  | Threshold | Status   |
|-------------|--------|-----------|----------|
| Recall@5    | 82.0%  | 60.0%     | PASS     |
| Recall@3    | 74.0%  | —         | —        |
| Precision@5 | 41.0%  | —         | —        |
| MRR         | 0.751  | 0.4       | PASS     |

### Answer Quality Metrics

| Metric                  | Score  |
|-------------------------|--------|
| Avg Faithfulness Score  | 91.0%  |
| Faithfulness Pass Rate  | 88.0%  |
| Avg Grounding Rate      | 94.0%  |
| Hallucination Rate      | 6.0%   |

**Overall:** All checks pass
```

---

## File Structure

| File | Purpose |
|---|---|
| `src/eval/__init__.py` | Package marker |
| `src/eval/retrieval_metrics.py` | Pure functions: `recall_at_k`, `precision_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`, `average_recall_at_k`, `average_precision_at_k` |
| `src/eval/hallucination.py` | `extract_numeric_claims`, `is_claim_in_context`, `check_hallucination` — fast keyword-based numeric grounding check |
| `src/eval/answer_metrics.py` | `evaluate_faithfulness` — LLM-as-judge via Claude API, `_parse_judge_response` |
| `src/eval/benchmark.py` | `BenchmarkQuestion`, `BenchmarkRunner`, `load_questions` — dependency-injected runner |
| `src/eval/reporter.py` | `EvalReport` dataclass, `to_json`, `to_markdown` |
| `tests/test_eval.py` | 47 tests covering all six modules |

---

## Tests Written (47 total)

### Retrieval Metrics (18 tests)

| Test | What it verifies |
|---|---|
| `test_recall_at_k_perfect` | All relevant docs in top k → 1.0 |
| `test_recall_at_k_miss` | No relevant doc in top k → 0.0 |
| `test_recall_at_k_partial` | One of two relevant docs found → 0.5 |
| `test_recall_at_k_cutoff` | Relevant doc at rank 6 not counted when k=5 |
| `test_recall_at_k_empty_relevant` | Empty relevant set → 0.0, no division by zero |
| `test_precision_at_k_perfect` | All top-k are relevant → 1.0 |
| `test_precision_at_k_zero` | No relevant docs in top k → 0.0 |
| `test_precision_at_k_partial` | 2 of 5 results relevant → 0.4 |
| `test_precision_at_k_zero_k` | k=0 guard against division by zero |
| `test_reciprocal_rank_first` | Relevant at rank 1 → RR = 1.0 |
| `test_reciprocal_rank_second` | Relevant at rank 2 → RR = 0.5 |
| `test_reciprocal_rank_third` | Relevant at rank 3 → RR = 1/3 |
| `test_reciprocal_rank_not_found` | Absent → RR = 0.0 |
| `test_mrr_all_first` | All queries rank-1 hits → MRR = 1.0 |
| `test_mrr_mixed` | Mix of rank-1, rank-2, not-found → MRR = 0.5 |
| `test_mrr_empty` | Empty results list → 0.0 |
| `test_average_recall_at_k` | Average across two queries |
| `test_average_precision_at_k` | Average across two queries |

### Faithfulness Judge (7 tests)

| Test | What it verifies |
|---|---|
| `test_parse_judge_response_valid_json` | Clean JSON parsed correctly |
| `test_parse_judge_response_perfect_score` | Score 1.0, empty claims |
| `test_parse_judge_response_clamps_score` | Scores outside [0,1] are clamped |
| `test_parse_judge_response_malformed_json` | Malformed JSON returns (0.0, [error]) without raising |
| `test_parse_judge_response_strips_markdown_fences` | Code fences stripped before parsing |
| `test_evaluate_faithfulness_grounded_answer` | Score=1.0 → is_faithful=True (mocked client) |
| `test_evaluate_faithfulness_ungrounded_answer` | Score=0.3 → is_faithful=False, claims listed |

### Hallucination Detection (8 tests)

| Test | What it verifies |
|---|---|
| `test_extract_numeric_claims_percentages` | Percentages extracted from text |
| `test_extract_numeric_claims_currency` | Currency amounts extracted |
| `test_extract_numeric_claims_plain_numbers` | Numbers with commas extracted |
| `test_extract_numeric_claims_empty` | Text with no numbers → empty list |
| `test_is_claim_in_context_found` | Claim in context → True |
| `test_is_claim_in_context_not_found` | Claim not in context → False |
| `test_is_claim_in_context_normalises_spaces` | Spacing differences handled |
| `test_check_hallucination_fully_grounded` | All numbers in context → no hallucination |
| `test_check_hallucination_ungrounded_number` | Number not in context → hallucination detected |
| `test_check_hallucination_no_numbers` | No numeric claims → no hallucination, grounding_rate=1.0 |

### Reporter (6 tests)

| Test | What it verifies |
|---|---|
| `test_to_json_contains_all_fields` | JSON output includes all EvalReport fields |
| `test_to_markdown_contains_metric_names` | Markdown includes all metric names |
| `test_to_markdown_shows_pass_on_good_scores` | Above-threshold metrics show PASS |
| `test_to_markdown_shows_fail_on_poor_scores` | Below-threshold metrics show FAIL |
| `test_eval_report_all_pass_property` | `all_pass` True when both recall and MRR pass |
| `test_eval_report_all_pass_fails_on_low_recall` | `all_pass` False when recall below threshold |

### Benchmark Runner (4 tests)

| Test | What it verifies |
|---|---|
| `test_benchmark_runner_computes_metrics` | Mock retriever → correct recall and MRR |
| `test_benchmark_runner_empty_questions` | Empty question list → zero-metric report, no error |
| `test_benchmark_runner_partial_recall` | Retriever misses half the relevant docs → recall=0.5 |
| `test_load_questions_from_json` | Benchmark JSON file parsed correctly |

### Regression Gate (2 tests)

| Test | What it verifies |
|---|---|
| `test_regression_gate_passes_on_good_retrieval` | Perfect retriever exceeds all thresholds |
| `test_regression_gate_fails_on_poor_retrieval` | Bad retriever falls below threshold — gate correctly identifies regression |

---

## Design Decisions

### Why dependency injection for the retriever?

The `BenchmarkRunner` takes a `retriever_fn: Callable[[str], Awaitable[list[int]]]` instead of importing the `HybridRetriever` directly. This means:

1. **Unit tests run without a database.** All 47 Phase 8 tests pass in under 2 seconds with mock retrievers.
2. **Integration tests use the real retriever.** You can swap in `HybridRetriever` without changing the runner.
3. **Future retrievers slot in for free.** If you add a new retrieval strategy, you benchmark it by passing its callable — no runner changes needed.

This is the dependency inversion principle applied to evaluation infrastructure.

### Why two hallucination detection approaches?

| Approach | Speed | Cost | What it catches |
|---|---|---|---|
| Keyword-based (`hallucination.py`) | Instant | Free | Numeric/financial value mismatches |
| LLM-as-judge (`answer_metrics.py`) | ~1–2 seconds | API cost | Any factual claim — conceptual, numeric, or named entity |

The two complement each other. In CI, run the keyword check on every response (fast, free, catches the most dangerous class of financial errors). Run the LLM judge on a sample when you want deeper quality assurance.

### Why not just use RAGAS or TruLens?

Those are excellent frameworks, but they add significant dependencies and require specific integrations. Building our own evaluation harness:

1. **Teaches the concepts directly** — you see exactly how Recall@K is computed, not just a number from a black box.
2. **Stays lightweight** — no additional service dependencies, just pure Python and our existing Claude client.
3. **Is fully testable** — every metric function has exact mathematical tests. If Recall@K changes behavior due to a refactor, a test catches it.

In a production system you might migrate to RAGAS or a similar framework after validating your understanding — but the metric definitions are the same.

### Regression gate design

The regression gate is the test `test_regression_gate_passes_on_good_retrieval`. It is intentionally structured like a CI assertion:

```python
assert report.recall_at_5 >= RECALL_THRESHOLD, (
    f"REGRESSION: Recall@5 {report.recall_at_5:.2f} dropped below "
    f"threshold {RECALL_THRESHOLD}. Check recent changes to retrieval."
)
```

When you run the real benchmark (not the mock), this test becomes the quality gate. A PR that changes chunking, embedding model, or reranking strategy and causes a recall drop will fail CI with a clear, human-readable message pointing at the metric that regressed.

---

## How This Connects to Previous Phases

| Phase | What it produced | How Phase 8 uses it |
|---|---|---|
| Phase 3 (Ingestion) | Chunks stored in DB with chunk IDs | Benchmark `relevant_chunk_ids` map to real DB chunk IDs |
| Phase 4 (Retrieval) | `HybridRetriever` returning ranked chunks | `retriever_fn` in integration benchmarks wraps the hybrid retriever |
| Phase 7 (LLM) | `FinancialAgent` generating answers | `evaluate_faithfulness` and `check_hallucination` evaluate those answers |

---

## What Comes Next (Phase 9)

Phase 9 adds **Observability & Data Quality**:

- Structured logging of every agent call (latency, token usage, tool calls made)
- Data quality checks on ingested documents (missing fields, duplicate chunks, schema violations)
- A dashboard or summary report showing system health metrics over time
- Alerting hooks when data quality or response latency degrades

The evaluation harness built in Phase 8 becomes the quality signal that the observability layer monitors — closing the loop from raw data ingestion all the way through to answer quality measurement.
