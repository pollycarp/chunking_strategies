# Phase 9: Observability & Data Quality

## What We Built

The final layer of the MASSA platform — a structured observability system that answers two questions:

1. **Is the data healthy?** Are all chunks embedded? Is there duplicate or empty content in the database?
2. **Is the system performing well?** How fast are responses? Are answers faithful to the retrieved context? What fraction of answers contain ungrounded claims?

```
                        ┌─────────────────────────────────┐
                        │        HealthReporter           │
                        │  (aggregates both sources)      │
                        └────────────┬────────────────────┘
                                     │
               ┌─────────────────────┴──────────────────────┐
               │                                             │
               ▼                                             ▼
   ┌───────────────────────┐                  ┌──────────────────────────┐
   │   DataQualityChecker  │                  │       AgentLogger        │
   │                       │                  │                          │
   │ - total chunks        │                  │ - log(interaction)       │
   │ - embedding coverage  │                  │ - recent(limit)          │
   │ - empty chunks        │                  │ - update_quality_scores  │
   │ - duplicate content   │                  │ - stats(since_hours)     │
   │ - total documents     │                  │                          │
   └──────────┬────────────┘                  └────────────┬─────────────┘
              │                                             │
              ▼                                             ▼
       chunks / documents                           agent_logs table
           tables                                  (Migration 005)
```

---

## Why Observability Is the Final Phase

Every phase before this one built something that produces output — chunks, embeddings, retrieved results, generated answers. Observability is what lets you know whether that output is good over time.

Without observability, a silent failure looks like success:

| Failure | What you see without observability |
|---|---|
| A parser produces blank chunks | System runs fine, retrieval just gets slightly worse |
| An embedding job crashes halfway through | Half the corpus is unsearchable — no error |
| Average latency doubles after a code change | Users notice, but you have no data to trace the cause |
| A model change causes faithfulness to drop from 0.92 to 0.61 | Answers look confident but are increasingly hallucinated |

Observability makes these failures visible and measurable so they can be caught and fixed before they affect users.

---

## Core Concepts

### Structured Logging vs Plain Text Logs

Most applications write logs as plain text lines to stdout or a file. This works for debugging a session in real time. It breaks down when you want to answer questions across time, like:

- "What was the average latency last Tuesday vs this Tuesday?"
- "How many answers had hallucinations detected in the last 7 days?"
- "Did faithfulness scores change after we updated the reranker?"

These questions require structured, queryable data — not text files you'd need to parse with grep. By storing each agent interaction as a row in a Postgres table with typed columns, they become simple SQL aggregations.

### The Three Signals

Phase 9 gives you three distinct health signals:

| Signal | Source | What it catches |
|---|---|---|
| **Embedding coverage** | `chunks` table | Chunks that exist but were never embedded — silently invisible to semantic search |
| **Content quality** | `chunks` table | Blank chunks (parser failures) and duplicate content (re-ingestion bugs) |
| **Runtime quality** | `agent_logs` table | Latency regressions, faithfulness drops, hallucination spikes |

These three signals correspond to three places in the pipeline where silent failures can occur: at ingestion time, at embedding time, and at inference time.

---

## Database Schema: `agent_logs` (Migration 005)

```sql
CREATE TABLE agent_logs (
    id                      BIGSERIAL    PRIMARY KEY,
    logged_at               TIMESTAMPTZ  NOT NULL DEFAULT now(),
    question                TEXT         NOT NULL,
    answer                  TEXT         NOT NULL,
    tools_called            JSONB        NOT NULL DEFAULT '[]',
    latency_ms              INTEGER      NOT NULL,
    input_tokens            INTEGER      NOT NULL DEFAULT 0,
    output_tokens           INTEGER      NOT NULL DEFAULT 0,
    faithfulness_score      FLOAT,
    hallucination_detected  BOOLEAN
);

CREATE INDEX agent_logs_logged_at_idx ON agent_logs (logged_at DESC);
```

**Design decisions:**

- `tools_called` is JSONB rather than a separate join table because the list is always consumed as a unit, never queried individually. A JSONB array avoids a join on every read.
- `faithfulness_score` and `hallucination_detected` are nullable because evaluation is optional and may happen asynchronously after the user has received their answer (LLM-as-judge evaluation costs API tokens and time).
- The index on `logged_at DESC` matches the two most common access patterns: "show me the last N rows" and "show me stats for the last 24 hours". Both scan from the newest rows backwards.

---

## Module Deep-Dive

### `logger.py` — AgentLogger

The `AgentLogger` writes and reads from the `agent_logs` table. It has four methods:

#### `log(entry: AgentLog) -> int`

Persists one interaction and returns its DB ID. The ID can be used to write quality scores back later.

```python
log = AgentLog(
    question="What was EBITDA margin in Q3 2024?",
    answer="EBITDA margin was 25.00% in Q3 2024.",
    tools_called=["retrieve_docs", "query_metrics"],
    latency_ms=1240,
    input_tokens=820,
    output_tokens=143,
)
log_id = await logger.log(log)
```

After `log()` returns, `log.id` and `log.logged_at` are populated from the DB row. This avoids a separate read-after-write.

#### `update_quality_scores(log_id, faithfulness_score, hallucination_detected)`

Writes evaluation results back to an existing row. This supports an async evaluation pattern:

```
User sends question
       │
       ▼
Agent generates answer  ──► log(entry)  ──► user sees answer immediately
                                │
                                ▼  (asynchronously, after response sent)
                       evaluate_faithfulness()
                       check_hallucination()
                                │
                                ▼
                    update_quality_scores(log_id, ...)
```

This is important in production: you do not want to make the user wait for LLM-as-judge evaluation before they see their answer.

The `COALESCE` in the UPDATE SQL means passing `None` for a field preserves the existing value — so you can update just `faithfulness_score` without overwriting `hallucination_detected` if it was already set.

#### `recent(limit: int) -> list[dict]`

Returns the most recent interactions, newest first. `tools_called` is always returned as a Python `list`, not a raw JSON string — the JSONB-to-string decoding is handled inside this method.

#### `stats(since_hours: int) -> dict`

Computes aggregate statistics for interactions in the last N hours using a single SQL query:

```sql
SELECT
    COUNT(*)                                          AS total_calls,
    AVG(latency_ms)                                   AS avg_latency_ms,
    AVG(input_tokens)                                 AS avg_input_tokens,
    AVG(output_tokens)                                AS avg_output_tokens,
    AVG(faithfulness_score)                           AS avg_faithfulness_score,
    COUNT(*) FILTER (WHERE faithfulness_score IS NOT NULL)
                                                      AS evaluated_count,
    COUNT(*) FILTER (WHERE hallucination_detected = true)::float
        / NULLIF(COUNT(*) FILTER (WHERE hallucination_detected IS NOT NULL), 0)
                                                      AS hallucination_rate
FROM agent_logs
WHERE logged_at > now() - ($1 || ' hours')::interval
```

**Key SQL patterns:**

- `COUNT(*) FILTER (WHERE ...)` — PostgreSQL's conditional aggregate. Counts only rows matching the condition, in a single pass over the table. Equivalent to `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` but cleaner.
- `NULLIF(..., 0)` — avoids division by zero when no rows have hallucination checks. If the denominator would be 0, `NULLIF` returns NULL, and `float / NULL` = NULL (not an error).
- `($1 || ' hours')::interval` — builds a Postgres interval dynamically. The `since_hours` parameter is passed as a string because asyncpg infers the parameter type from the placeholder position, and string concatenation requires a `text` operand.

---

### `data_quality.py` — DataQualityChecker

Runs four SQL queries against the `chunks` and `documents` tables and returns a `DataQualityReport`.

#### The Four Checks

**1. Embedding coverage**
```sql
SELECT COUNT(*) FROM chunks WHERE embedding IS NULL
```
A chunk with `embedding IS NULL` was ingested but never embedded. It exists in the full-text keyword search index but is completely invisible to semantic (vector) search. In a hybrid retrieval system this means it can only ever be found by keyword match — the most semantically relevant chunks may be perpetually missed.

**2. Empty chunks**
```sql
SELECT COUNT(*) FROM chunks WHERE trim(content) = ''
```
Empty chunks arise from parser edge cases: a PDF page with only an image and no text, an empty spreadsheet range, or a malformed DOCX section. If retrieved, they give the LLM useless context.

**3. Duplicate content**
```sql
SELECT COUNT(*) FROM (
    SELECT content_hash FROM chunks GROUP BY content_hash HAVING COUNT(*) > 1
) duplicates
```
The `content_hash` column (SHA256 of chunk text) lets us find chunks with identical content efficiently without comparing the full text. A duplicate count > 0 means the same block of text was stored more than once — typically from re-ingesting a file. Duplicates inflate retrieval scores for that content, since both copies may be returned and counted separately.

**4. Document count** — baseline context for interpreting the other numbers.

#### `DataQualityReport` Properties

```python
@property
def embedding_coverage(self) -> float:
    # Fraction of chunks with an embedding. 1.0 = fully embedded.
    if self.total_chunks == 0:
        return 1.0   # empty DB is not a failure
    return self.embedded_chunks / self.total_chunks

@property
def all_pass(self) -> bool:
    return (
        self.embedding_coverage >= self.min_embedding_coverage  # default 95%
        and self.empty_chunk_count == 0
        and self.duplicate_content_count == 0
    )

@property
def issues(self) -> list[str]:
    # Human-readable list of detected problems
```

The threshold for embedding coverage (95% by default, not 100%) acknowledges that in a live system, there is always a brief window between ingestion and embedding completion where some chunks are unembedded. 95% is strict enough to catch a crashed embedding job while tolerating normal pipeline lag.

---

### `health.py` — HealthReporter

`HealthReporter` is the single entry point for a complete system health snapshot. It composes `DataQualityChecker` and `AgentLogger` and returns a `SystemHealthReport`.

```python
reporter = HealthReporter(pool, lookback_hours=24)
report = await reporter.report()
```

The two formatters produce output for different audiences:

**`to_json(report)`** — for CI artefacts and time-series monitoring. Store one JSON file per hour; diff consecutive reports to detect regressions automatically.

**`to_markdown(report)`** — for human consumption in Slack, GitHub PR comments, or a simple dashboard. All `None` fields render as `—` (not the string "None") to keep the table readable.

**Sample output when healthy:**

```
## System Health Report

**Status:** OK

### Data Layer

| Metric                   | Value   |
|--------------------------|---------|
| Total documents          | 12      |
| Total chunks             | 847     |
| Embedding coverage       | 100.0%  |
| Missing embeddings       | 0       |
| Empty chunks             | 0       |
| Duplicate content hashes | 0       |
| Data quality             | PASS    |

### Agent Activity (last 24 h)

| Metric                    | Value    |
|---------------------------|----------|
| Total interactions        | 34       |
| Avg latency               | 1,183 ms |
| Avg input tokens          | 742      |
| Avg output tokens         | 138      |
| Avg faithfulness score    | 91.0%    |
| Hallucination rate        | 5.9%     |
```

**Sample output when degraded:**

```
## System Health Report

**Status:** DEGRADED

### Data Layer

...
| Data quality             | FAIL    |

**Issues detected:**

- 47 chunks missing embeddings (coverage 94.4% < threshold 95%)
- 3 chunks have empty content
```

---

## File Structure

| File | Purpose |
|---|---|
| `src/db/migrations/005_agent_logs.sql` | Schema for the `agent_logs` table with timestamp index |
| `src/observability/__init__.py` | Package marker with module overview |
| `src/observability/logger.py` | `AgentLog` dataclass, `AgentLogger` class |
| `src/observability/data_quality.py` | `DataQualityReport` dataclass, `DataQualityChecker` class |
| `src/observability/health.py` | `SystemHealthReport` dataclass, `HealthReporter` class, `to_json`, `to_markdown` |
| `tests/test_observability.py` | 32 tests covering all three modules and formatters |

---

## Tests Written (32 total)

### AgentLogger — log() and recent() (6 tests)

| Test | What it verifies |
|---|---|
| `test_logger_stores_log` | `log()` inserts a row and returns a positive integer ID |
| `test_logger_log_populates_id_and_timestamp` | `AgentLog.id` and `logged_at` are set after insert |
| `test_logger_recent_returns_inserted_row` | Inserted question appears in `recent()` results |
| `test_logger_recent_returns_tools_as_list` | `tools_called` returns as Python list, not raw JSON string |
| `test_logger_recent_newest_first` | Results are ordered newest first |
| `test_logger_stores_quality_scores` | `faithfulness_score` and `hallucination_detected` are persisted |

### AgentLogger — update_quality_scores() (2 tests)

| Test | What it verifies |
|---|---|
| `test_update_quality_scores_writes_back` | Scores written back to an existing row are retrievable |
| `test_update_quality_scores_preserves_existing` | Passing `None` for a field does not overwrite an existing value |

### AgentLogger — stats() (4 tests)

| Test | What it verifies |
|---|---|
| `test_stats_counts_recent_logs` | `total_calls` reflects inserted rows |
| `test_stats_avg_latency_is_numeric` | `avg_latency_ms` returns a float |
| `test_stats_faithfulness_none_when_no_scores` | Returns `None` when time window has no rows |
| `test_stats_hallucination_rate_computed` | Rate is between 0.0 and 1.0 when checks are present |

### DataQualityChecker — DB tests (4 tests)

| Test | What it verifies |
|---|---|
| `test_data_quality_returns_report` | `run()` returns a `DataQualityReport` without error |
| `test_data_quality_chunk_counts_are_non_negative` | All counts are >= 0 |
| `test_data_quality_embedding_coverage_in_range` | Coverage is in [0.0, 1.0] |
| `test_data_quality_missing_equals_total_minus_embedded` | Arithmetic consistency of counts |

### DataQualityReport — pure unit tests (6 tests)

| Test | What it verifies |
|---|---|
| `test_data_quality_report_all_pass_clean_db` | Fully embedded, clean DB → all_pass True, no issues |
| `test_data_quality_report_fails_on_low_coverage` | Coverage below threshold → fails, issue message generated |
| `test_data_quality_report_fails_on_empty_chunks` | Empty chunks always fail regardless of coverage |
| `test_data_quality_report_fails_on_duplicates` | Duplicate content always fails |
| `test_data_quality_report_multiple_issues` | Multiple failures produce multiple issue messages |
| `test_data_quality_empty_db_passes` | Empty DB is not a quality failure (coverage = 1.0) |

### HealthReporter (3 tests)

| Test | What it verifies |
|---|---|
| `test_health_reporter_returns_report` | `report()` returns a `SystemHealthReport` without error |
| `test_health_report_data_layer_matches_quality_checker` | Data-layer fields agree with `DataQualityChecker` run at the same time |
| `test_health_report_agent_stats_included` | `total_calls_24h` reflects logged interactions |

### Formatters (7 tests)

| Test | What it verifies |
|---|---|
| `test_to_json_is_valid_json` | Output is valid JSON with correct field values |
| `test_to_json_includes_all_fields` | All top-level fields are present in JSON |
| `test_to_markdown_contains_section_headers` | Both "Data Layer" and "Agent Activity" sections present |
| `test_to_markdown_shows_ok_when_all_pass` | Status line shows OK |
| `test_to_markdown_shows_degraded_when_quality_fails` | Status shows DEGRADED, issue messages included |
| `test_to_markdown_shows_dash_for_none_metrics` | None values render as `—` not `"None"` |
| `test_to_markdown_shows_pass_on_good_data_quality` | PASS label appears for passing quality gate |

---

## Design Decisions

### Why asyncpg requires `str(since_hours)` for the interval

asyncpg infers the PostgreSQL type of each `$N` parameter from how it is used in the query. In the expression `$1 || ' hours'`, the `||` operator is string concatenation — Postgres therefore expects `$1` to be of type `text`. asyncpg does not automatically cast Python `int` to `text`, so passing an integer raises `DataError: expected str, got int`. The fix is explicit: `str(since_hours)`.

This is a consistent pattern throughout this codebase: asyncpg requires you to match Python types precisely to what Postgres expects for each parameter position.

### Why data quality uses full-table scans

The data quality queries count every row in the `chunks` table. For a platform processing thousands of financial documents this may take a few hundred milliseconds. This is intentional — a sampled estimate would miss a small embedding gap or a handful of duplicate chunks, which are exactly the issues we want to detect.

Data quality checks are designed to run periodically (after each ingestion batch, or hourly in CI) rather than on every user request. The latency is acceptable for batch use.

### Why `all_pass` uses 95% not 100% for embedding coverage

A 100% threshold would fail any time the embedding pipeline is mid-run — which is a normal, expected state. 95% catches a genuinely broken pipeline (where, say, an API error caused 20% of chunks to fail embedding) while tolerating the brief window between ingestion and embedding completion in a healthy system.

### Separation of concerns: logging vs evaluation

`AgentLogger` only records what the system did (question, answer, tools called, latency, tokens). It does not run evaluation itself. Evaluation (faithfulness scoring, hallucination detection) is the job of `src/eval/`. The logger provides `update_quality_scores()` so the eval harness can write its results back to the log row after the fact.

This separation means:
- The logger has no dependency on the eval harness (no circular imports)
- Evaluation can be skipped, sampled, or run asynchronously without changing the logging code
- The same log table works whether you're running full evaluation or just latency monitoring

---

## How This Connects to Previous Phases

| Phase | What it produced | How Phase 9 uses it |
|---|---|---|
| Phase 1 (Infrastructure) | Migration system, asyncpg pool | Migration 005 follows the same pattern; logger uses the pool |
| Phase 3 (Ingestion) | `chunks` and `documents` tables | `DataQualityChecker` queries these tables directly |
| Phase 8 (Evaluation) | `evaluate_faithfulness`, `check_hallucination` | Results written back to `agent_logs` via `update_quality_scores` |
| Phase 7 (LLM) | `FinancialAgent` generating answers | Each agent call is a candidate for logging via `AgentLogger` |

---

## The Complete Platform: All 9 Phases

With Phase 9 complete, the MASSA platform is fully built. Here is the full picture:

| Phase | Name | Core output |
|---|---|---|
| 1 | Infrastructure | Docker, asyncpg, pgvector, migrations, Pydantic config |
| 2 | Embeddings | OpenAI/Voyage embedders, DB-backed cache, HNSW index |
| 3 | Ingestion | PDF/DOCX/XLSX parsers, 3 chunking strategies, dedup pipeline |
| 4 | Hybrid Retrieval | SemanticRetriever, KeywordRetriever, RRF fusion, CohereReranker |
| 5 | Structured Data | Star schema, metrics layer, read-only QueryEngine, SchemaIntrospector |
| 6 | MCP Server | FastMCP, `retrieve_docs`, `query_metrics`, `list_sources` tools |
| 7 | LLM Integration | Claude API, agentic loop, context manager, HTTP endpoint |
| 8 | Evaluation | Recall@K, MRR, faithfulness judge, hallucination detector, benchmark runner |
| 9 | Observability | Agent logger, data quality checker, health reporter |

**Final test count: 201 passed, 1 skipped** (OpenAI live integration — skipped when `OPENAI_API_KEY` is not a live key, by design).
