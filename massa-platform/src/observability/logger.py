"""
Agent interaction logger — persists every request/response cycle to the DB.

Why structured DB logging instead of plain text logs?
-----------------------------------------------------
Plain text logs (stdout, log files) are great for debugging a single session.
They are poor for answering operational questions like:

  - "Has average response latency increased since we changed the retriever?"
  - "What fraction of answers had hallucinations detected last week?"
  - "Which questions most often trigger multiple tool calls?"

Storing each interaction as a DB row means these questions become SQL queries.
The quality scores from Phase 8 (faithfulness, hallucination) can be written
back into the log row after evaluation, connecting the eval harness to the
live system.

Usage
-----
    logger = AgentLogger(pool)

    log = AgentLog(
        question="What was EBITDA in Q3?",
        answer="EBITDA margin was 25.0% ...",
        tools_called=["retrieve_docs", "query_metrics"],
        latency_ms=1240,
        input_tokens=820,
        output_tokens=143,
        faithfulness_score=0.95,
        hallucination_detected=False,
    )
    log_id = await logger.log(log)

    stats = await logger.stats(since_hours=24)
    # {"total_calls": 47, "avg_latency_ms": 1180.3, ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import asyncpg


@dataclass
class AgentLog:
    """
    One captured agent interaction.

    All fields except the optional quality scores are required — they are
    produced by the agent without any extra evaluation step.

    faithfulness_score and hallucination_detected are optional because
    LLM-as-judge evaluation costs money and may not run on every call.
    Keyword-based hallucination detection is cheap enough to run always.
    """
    question: str
    answer: str
    tools_called: list[str]
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    faithfulness_score: float | None = None
    hallucination_detected: bool | None = None

    # Populated after DB insert
    id: int | None = field(default=None, compare=False)
    logged_at: datetime | None = field(default=None, compare=False)


class AgentLogger:
    """
    Reads and writes agent interaction logs from/to the agent_logs table.

    Parameters
    ----------
    pool : asyncpg.Pool
        Active DB connection pool — the same pool used throughout the system.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def log(self, entry: AgentLog) -> int:
        """
        Persists one agent interaction to the DB.

        Returns the auto-generated log ID so the caller can update the row
        later (e.g., to write back a faithfulness score after evaluation).
        """
        row = await self._pool.fetchrow(
            """
            INSERT INTO agent_logs
                (question, answer, tools_called, latency_ms,
                 input_tokens, output_tokens,
                 faithfulness_score, hallucination_detected)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, logged_at
            """,
            entry.question,
            entry.answer,
            json.dumps(entry.tools_called),
            entry.latency_ms,
            entry.input_tokens,
            entry.output_tokens,
            entry.faithfulness_score,
            entry.hallucination_detected,
        )
        entry.id = row["id"]
        entry.logged_at = row["logged_at"]
        return row["id"]

    async def update_quality_scores(
        self,
        log_id: int,
        faithfulness_score: float | None = None,
        hallucination_detected: bool | None = None,
    ) -> None:
        """
        Writes quality evaluation results back to an existing log row.

        Called after LLM-as-judge evaluation completes — which may happen
        asynchronously after the user receives their answer.
        """
        await self._pool.execute(
            """
            UPDATE agent_logs
            SET faithfulness_score     = COALESCE($2, faithfulness_score),
                hallucination_detected = COALESCE($3, hallucination_detected)
            WHERE id = $1
            """,
            log_id,
            faithfulness_score,
            hallucination_detected,
        )

    async def recent(self, limit: int = 20) -> list[dict]:
        """
        Returns the most recent interactions, newest first.

        Each row is a plain dict so it can be serialised to JSON directly.
        tools_called is returned as a Python list (decoded from JSONB).
        """
        rows = await self._pool.fetch(
            """
            SELECT id, logged_at, question, answer, tools_called,
                   latency_ms, input_tokens, output_tokens,
                   faithfulness_score, hallucination_detected
            FROM agent_logs
            ORDER BY logged_at DESC
            LIMIT $1
            """,
            limit,
        )
        result = []
        for row in rows:
            d = dict(row)
            # asyncpg returns JSONB as a string — parse it back to a list
            if isinstance(d["tools_called"], str):
                d["tools_called"] = json.loads(d["tools_called"])
            result.append(d)
        return result

    async def stats(self, since_hours: int = 24) -> dict:
        """
        Returns aggregate statistics for interactions in the last N hours.

        Keys:
            total_calls           : int
            avg_latency_ms        : float | None
            avg_input_tokens      : float | None
            avg_output_tokens     : float | None
            avg_faithfulness_score: float | None  (None if no scores recorded)
            hallucination_rate    : float | None  (None if no checks recorded)
            evaluated_count       : int           (rows with faithfulness_score)
        """
        row = await self._pool.fetchrow(
            """
            SELECT
                COUNT(*)                                          AS total_calls,
                AVG(latency_ms)                                   AS avg_latency_ms,
                AVG(input_tokens)                                 AS avg_input_tokens,
                AVG(output_tokens)                                AS avg_output_tokens,
                AVG(faithfulness_score)                           AS avg_faithfulness_score,
                COUNT(*) FILTER (WHERE faithfulness_score IS NOT NULL)
                                                                  AS evaluated_count,
                COUNT(*) FILTER (WHERE hallucination_detected = true)::float
                    / NULLIF(
                        COUNT(*) FILTER (WHERE hallucination_detected IS NOT NULL),
                        0
                      )                                           AS hallucination_rate
            FROM agent_logs
            WHERE logged_at > now() - ($1 || ' hours')::interval
            """,
            str(since_hours),
        )
        return {
            "total_calls": row["total_calls"],
            "avg_latency_ms": float(row["avg_latency_ms"]) if row["avg_latency_ms"] is not None else None,
            "avg_input_tokens": float(row["avg_input_tokens"]) if row["avg_input_tokens"] is not None else None,
            "avg_output_tokens": float(row["avg_output_tokens"]) if row["avg_output_tokens"] is not None else None,
            "avg_faithfulness_score": float(row["avg_faithfulness_score"]) if row["avg_faithfulness_score"] is not None else None,
            "evaluated_count": row["evaluated_count"],
            "hallucination_rate": float(row["hallucination_rate"]) if row["hallucination_rate"] is not None else None,
        }
