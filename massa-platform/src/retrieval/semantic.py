from asyncpg import Pool

from src.retrieval.filters import build_filter_clause
from src.retrieval.models import RetrievalFilter, RetrievedChunk


class SemanticRetriever:
    """
    Retrieves chunks by vector cosine similarity using pgvector's HNSW index.

    How it works:
    - The query text is embedded into a vector (done by the caller)
    - We find the chunks whose embedding vectors are closest to the query vector
    - "Closest" = smallest cosine distance = highest cosine similarity

    Strengths:
    - Finds semantically related content even with different vocabulary
    - "operating profitability" matches "EBITDA", "net margin", "earnings"
    - Language-model aware — understands context, not just keywords

    Weaknesses:
    - Struggles with exact numeric values ("Q3 2024", "23.4%", ticker symbols)
    - All chunks look somewhat similar to any query — score threshold matters
    - Quality depends entirely on the embedding model used
    """

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """
        Returns the top_k most semantically similar chunks.

        query_vector: pre-computed embedding of the search query
        top_k: number of results to return (retrieve more than needed for reranking)
        filters: optional SQL WHERE conditions to narrow the search space
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
                1 - (embedding <=> $1::vector) AS score
            FROM chunks
            WHERE embedding IS NOT NULL
            {filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT {top_k}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, query_vector, *filter_params)

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
