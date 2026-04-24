"""
Data quality checker — inspects the chunks and documents tables for problems.

Why data quality matters in a RAG system
-----------------------------------------
The retrieval system is only as good as the data it retrieves. Silent data
quality issues are especially dangerous because the system keeps running and
returning answers — just worse ones. Common silent failures:

  Missing embeddings   The chunk exists in the DB but was never embedded.
                       Semantic search will never return it. The system won't
                       error, it will just silently miss relevant chunks.

  Empty chunks         A parser edge case left a chunk with blank content.
                       If retrieved, the LLM gets useless context.

  Duplicate chunks     The same content was ingested twice (e.g., file
                       re-uploaded). Duplicates inflate retrieval scores for
                       that content and waste embedding space.

This checker runs SQL queries against the live DB — no sample, no estimate.
It reports exact counts so issues can be investigated and fixed.

Usage
-----
    checker = DataQualityChecker(pool)
    report = await checker.run()

    print(f"Embedding coverage: {report.embedding_coverage:.1%}")
    if not report.all_pass:
        for issue in report.issues:
            print(f"  ISSUE: {issue}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncpg


@dataclass
class DataQualityReport:
    """
    Snapshot of data quality in the chunks and documents tables.

    All counts are exact (full-table scans) so this is intended to run
    periodically (e.g., after each ingestion job) rather than on every request.
    """
    # Chunk counts
    total_chunks: int = 0
    embedded_chunks: int = 0
    missing_embedding_count: int = 0
    empty_chunk_count: int = 0
    duplicate_content_count: int = 0   # distinct content_hashes with >1 chunk

    # Document counts
    total_documents: int = 0

    # Thresholds — what we consider acceptable
    min_embedding_coverage: float = 0.95   # 95% of chunks must be embedded

    @property
    def embedding_coverage(self) -> float:
        """Fraction of chunks that have an embedding. 1.0 = fully embedded."""
        if self.total_chunks == 0:
            return 1.0   # empty DB is not a quality failure
        return self.embedded_chunks / self.total_chunks

    @property
    def embedding_coverage_passes(self) -> bool:
        return self.embedding_coverage >= self.min_embedding_coverage

    @property
    def all_pass(self) -> bool:
        """True if all quality checks are within acceptable bounds."""
        return (
            self.embedding_coverage_passes
            and self.empty_chunk_count == 0
            and self.duplicate_content_count == 0
        )

    @property
    def issues(self) -> list[str]:
        """Human-readable list of detected issues. Empty list = all good."""
        found: list[str] = []
        if not self.embedding_coverage_passes:
            found.append(
                f"{self.missing_embedding_count} chunks missing embeddings "
                f"(coverage {self.embedding_coverage:.1%} < "
                f"threshold {self.min_embedding_coverage:.0%})"
            )
        if self.empty_chunk_count > 0:
            found.append(f"{self.empty_chunk_count} chunks have empty content")
        if self.duplicate_content_count > 0:
            found.append(
                f"{self.duplicate_content_count} content hashes appear in "
                f"more than one chunk (possible duplicate ingestion)"
            )
        return found


class DataQualityChecker:
    """
    Runs data quality checks against the live chunks and documents tables.

    Parameters
    ----------
    pool : asyncpg.Pool
        Active DB connection pool.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def run(self) -> DataQualityReport:
        """
        Executes all quality checks and returns a DataQualityReport.

        Runs four queries in sequence (not parallel — each is fast and we
        want a consistent snapshot at one point in time).
        """
        total_chunks = await self._count_total_chunks()
        embedded_chunks = await self._count_embedded_chunks()
        empty_chunks = await self._count_empty_chunks()
        duplicate_count = await self._count_duplicate_content()
        total_documents = await self._count_total_documents()

        missing = total_chunks - embedded_chunks

        return DataQualityReport(
            total_chunks=total_chunks,
            embedded_chunks=embedded_chunks,
            missing_embedding_count=missing,
            empty_chunk_count=empty_chunks,
            duplicate_content_count=duplicate_count,
            total_documents=total_documents,
        )

    async def _count_total_chunks(self) -> int:
        row = await self._pool.fetchrow("SELECT COUNT(*) AS n FROM chunks")
        return row["n"]

    async def _count_embedded_chunks(self) -> int:
        row = await self._pool.fetchrow(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        )
        return row["n"]

    async def _count_empty_chunks(self) -> int:
        """
        Counts chunks whose content is NULL or blank after trimming whitespace.

        A blank chunk can occur when a parser encounters an empty page or
        a spreadsheet cell range with no data.
        """
        row = await self._pool.fetchrow(
            "SELECT COUNT(*) AS n FROM chunks WHERE trim(content) = ''"
        )
        return row["n"]

    async def _count_duplicate_content(self) -> int:
        """
        Counts distinct content_hashes that appear in more than one chunk.

        A count > 0 means the same text block was stored multiple times —
        typically caused by ingesting the same file twice without the
        deduplication check catching it at the document level (e.g., if the
        file bytes changed slightly but the content stayed the same).
        """
        row = await self._pool.fetchrow(
            """
            SELECT COUNT(*) AS n
            FROM (
                SELECT content_hash
                FROM chunks
                GROUP BY content_hash
                HAVING COUNT(*) > 1
            ) duplicates
            """
        )
        return row["n"]

    async def _count_total_documents(self) -> int:
        row = await self._pool.fetchrow("SELECT COUNT(*) AS n FROM documents")
        return row["n"]
