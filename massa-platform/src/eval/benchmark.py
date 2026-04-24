"""
Benchmark runner — loads Q&A pairs and computes eval metrics over them.

Design philosophy:
The benchmark runner is dependency-injected — it takes a retriever_fn
(any async callable that takes a query string and returns a list of chunk IDs)
and an optional answer_fn. This keeps it decoupled from the real retriever,
making it easily testable with mocks.

Benchmark dataset format (JSON):
[
  {
    "id": "q001",
    "question": "What was the EBITDA margin for Q3 2024?",
    "relevant_chunk_ids": [12, 15],
    "expected_keywords": ["25", "ebitda", "margin"]
  },
  ...
]

relevant_chunk_ids: DB chunk IDs that are considered correct answers for this query.
  These must be populated from the actual DB when the benchmark is configured.
  For CI, use a seeded DB with known chunk IDs.

expected_keywords: Optional keywords that should appear in a good answer.
  Used for lightweight answer correctness checking without an LLM judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

from src.eval.retrieval_metrics import (
    average_precision_at_k,
    average_recall_at_k,
    mean_reciprocal_rank,
)
from src.eval.reporter import EvalReport


@dataclass
class BenchmarkQuestion:
    """One question in the benchmark dataset."""
    id: str
    question: str
    relevant_chunk_ids: list[int]
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class QuestionResult:
    """Evaluation result for one question."""
    question_id: str
    question: str
    retrieved_ids: list[int]
    relevant_ids: set[int]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


# Type alias for the retriever function
RetrieverFn = Callable[[str], Awaitable[list[int]]]


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """
    Loads benchmark questions from a JSON file.

    Each entry must have: id, question, relevant_chunk_ids.
    expected_keywords is optional.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        BenchmarkQuestion(
            id=entry["id"],
            question=entry["question"],
            relevant_chunk_ids=entry["relevant_chunk_ids"],
            expected_keywords=entry.get("expected_keywords", []),
        )
        for entry in data
    ]


class BenchmarkRunner:
    """
    Runs a set of benchmark questions through a retriever and computes metrics.

    Usage:
        async def my_retriever(query: str) -> list[int]:
            chunks = await hybrid_retriever.search(query, top_k=5)
            return [c.chunk_id for c in chunks]

        runner = BenchmarkRunner(retriever_fn=my_retriever, k=5)
        report = await runner.run(questions)
    """

    def __init__(
        self,
        retriever_fn: RetrieverFn,
        k: int = 5,
        recall_threshold: float = 0.6,
        mrr_threshold: float = 0.4,
    ) -> None:
        self._retriever_fn = retriever_fn
        self._k = k
        self._recall_threshold = recall_threshold
        self._mrr_threshold = mrr_threshold

    async def run(
        self,
        questions: list[BenchmarkQuestion],
    ) -> EvalReport:
        """
        Runs all questions through the retriever and computes aggregate metrics.

        For each question:
        1. Calls retriever_fn(question) to get retrieved chunk IDs
        2. Compares to relevant_chunk_ids using Recall@K, Precision@K, RR

        Returns an EvalReport with aggregated scores.
        """
        if not questions:
            return EvalReport(total_questions=0, k=self._k)

        query_results: list[tuple[list[int], set[int]]] = []

        for q in questions:
            retrieved_ids = await self._retriever_fn(q.question)
            relevant_ids = set(q.relevant_chunk_ids)
            query_results.append((retrieved_ids, relevant_ids))

        return EvalReport(
            recall_at_5=average_recall_at_k(query_results, k=5),
            recall_at_3=average_recall_at_k(query_results, k=3),
            precision_at_5=average_precision_at_k(query_results, k=5),
            mrr=mean_reciprocal_rank(query_results),
            total_questions=len(questions),
            k=self._k,
            recall_at_5_threshold=self._recall_threshold,
            mrr_threshold=self._mrr_threshold,
        )
