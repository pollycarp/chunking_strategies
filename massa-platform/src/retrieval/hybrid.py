from asyncpg import Pool

from src.embeddings.base import EmbeddingModel
from src.retrieval.filters import build_filter_clause
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.models import RetrievalFilter, RetrievedChunk
from src.retrieval.semantic import SemanticRetriever


def rrf_fusion(
    semantic_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    top_k: int,
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion — merges two ranked lists into one.

    Formula: RRF_score(doc) = Σ  1 / (k + rank_i(doc))

    Why k=60?
    The constant k dampens the effect of very high ranks. Without it,
    a document ranked 1st in one list and 100th in the other would
    dominate over a document ranked 2nd in both lists. k=60 is the
    empirically validated default from the original RRF paper (Cormack 2009).

    Example with k=60:
        Doc A: rank 1 (semantic) + rank 10 (keyword)
               = 1/61 + 1/70 = 0.0164 + 0.0143 = 0.0307

        Doc B: rank 2 (semantic) + rank 2 (keyword)
               = 1/62 + 1/62 = 0.0161 + 0.0161 = 0.0323   ← wins

    Doc B is consistently good in both lists — RRF correctly ranks it higher
    than Doc A which is exceptional in one but weak in the other.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}

    for rank, chunk in enumerate(semantic_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        chunks[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(keyword_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id not in chunks:
            chunks[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for rank, chunk_id in enumerate(sorted_ids[:top_k], start=1):
        chunk = chunks[chunk_id]
        # Return a copy with updated score and rank from RRF
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            doc_type=chunk.doc_type,
            chunk_strategy=chunk.chunk_strategy,
            parent_id=chunk.parent_id,
            score=scores[chunk_id],
            rank=rank,
        ))

    return results


class HybridRetriever:
    """
    Combines semantic and keyword retrieval using Reciprocal Rank Fusion.

    This is the primary retriever used in production. It handles both:
    - Semantic queries: "what drove profitability improvement?"
    - Exact queries:    "EBITDA margin Q3 2024 portfolio company"

    Usage:
        retriever = HybridRetriever(pool=pool, embedder=embedder)
        results = await retriever.search("EBITDA margin improvement", top_k=5)
    """

    def __init__(
        self,
        pool: Pool,
        embedder: EmbeddingModel,
        candidate_k: int = 20,   # how many candidates each sub-retriever fetches
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._candidate_k = candidate_k
        self._semantic = SemanticRetriever(pool)
        self._keyword = KeywordRetriever(pool)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embeds the query, runs semantic + keyword search, fuses with RRF.

        The query is embedded once and reused for semantic search.
        Keyword search uses the raw query text.
        Both retrievers fetch candidate_k results, then RRF picks the top_k.
        """
        # Embed query once — reused for semantic search
        query_vector = await self._embedder.embed(query)

        # Run both retrievers concurrently
        import asyncio
        semantic_task = self._semantic.search(query_vector, self._candidate_k, filters)
        keyword_task = self._keyword.search(query, self._candidate_k, filters)

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        return rrf_fusion(semantic_results, keyword_results, top_k)

    async def fetch_parent(self, parent_id: int) -> RetrievedChunk | None:
        """
        Fetches a parent chunk by DB id — used in small-to-big retrieval.

        When a hierarchical child chunk is retrieved, call this to get the
        full parent section for richer LLM context.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, source_file, page_number, section_title,
                       doc_type, chunk_strategy, parent_id
                FROM   chunks
                WHERE  id = $1
                """,
                parent_id,
            )

        if row is None:
            return None

        return RetrievedChunk(
            chunk_id=row["id"],
            content=row["content"],
            source_file=row["source_file"],
            page_number=row["page_number"],
            section_title=row["section_title"],
            doc_type=row["doc_type"],
            chunk_strategy=row["chunk_strategy"],
            parent_id=row["parent_id"],
            score=1.0,
            rank=0,
        )
