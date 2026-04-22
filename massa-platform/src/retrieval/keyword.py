from asyncpg import Pool

from src.retrieval.filters import build_filter_clause
from src.retrieval.models import RetrievalFilter, RetrievedChunk


class KeywordRetriever:
    """
    Retrieves chunks by full-text keyword matching using Postgres tsvector.

    How it works:
    - The query is converted to a tsquery (tokenised, stop words removed)
    - We find chunks whose content_tsv matches the tsquery
    - ts_rank scores how well each chunk matches (term frequency, position)

    Strengths:
    - Exact term matching: "Q3 2024", "EBITDA", ticker symbols, client names
    - Handles typo-tolerant variants via ts_lexize normalization
    - No embedding model required — purely database-side

    Weaknesses:
    - No semantic understanding: "operating profitability" won't find "EBITDA"
    - Vocabulary mismatch: query and document must share root terms
    - Sensitive to language (we use 'english' stemming dictionary)

    Why this matters for financial data:
    Financial queries often contain precise identifiers that semantic search misses:
    - "Portfolio Company A Q3 2024 board report"
    - "Section 5.2 covenant compliance"
    - "EBITDA margin 23.4%"
    These require exact matching, not semantic proximity.
    """

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def search(
        self,
        query_text: str,
        top_k: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """
        Returns top_k chunks matching the query by full-text search.

        plainto_tsquery: converts plain text to tsquery (words joined by AND)
        ts_rank: scores the match quality based on term frequency and coverage
        """
        filter_clause, filter_params = build_filter_clause(filters, param_offset=2)

        query = f"""
            SELECT
                id,
                content,
                source_file,
                page_number,
                section_title,
                doc_type,
                chunk_strategy,
                parent_id,
                ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
            FROM chunks
            WHERE content_tsv @@ plainto_tsquery('english', $1)
            {filter_clause}
            ORDER BY score DESC
            LIMIT {top_k}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, query_text, *filter_params)

        return [
            RetrievedChunk(
                chunk_id=row["id"],
                content=row["content"],
                source_file=row["source_file"],
                page_number=row["page_number"],
                section_title=row["section_title"],
                doc_type=row["doc_type"],
                chunk_strategy=row["chunk_strategy"],
                parent_id=row["parent_id"],
                score=float(row["score"]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]
