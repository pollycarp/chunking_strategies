from abc import ABC, abstractmethod

import cohere

from src.retrieval.models import RetrievedChunk


class BaseReranker(ABC):
    """
    Abstract interface for re-ranking retrieved chunks.

    Re-ranking is a second-pass scoring step that takes the top-N candidates
    from hybrid retrieval and re-orders them using a more accurate but slower
    cross-encoder model.

    Cross-encoders differ from bi-encoders (embeddings):
    - Bi-encoder:    embed(query) + embed(doc) → cosine similarity
                     Fast — embeddings pre-computed; but query/doc never interact
    - Cross-encoder: score(query, doc) → relevance score
                     Slow — computed at query time; but query/doc interact directly
                     → much higher accuracy
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Re-scores chunks against the query and returns top_k in new order.
        The input list is not modified — a new sorted list is returned.
        """
        ...


class CohereReranker(BaseReranker):
    """
    Re-ranks chunks using Cohere's Rerank API.

    Model options:
    - rerank-english-v3.0     : English only, highest quality
    - rerank-multilingual-v3.0: Multiple languages
    - rerank-english-v2.0     : Older, slightly lower quality

    Why Cohere for reranking?
    Cohere's rerank models are cross-encoders trained specifically for
    relevance scoring. They understand financial terminology well and
    handle long documents gracefully (up to 4096 tokens per document).

    Cost: ~$1 per 1000 searches (assuming top-20 candidates reranked to top-5)
    Latency: ~200-400ms per rerank call
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ) -> None:
        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        top_k = min(top_k, len(chunks))

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=[c.content for c in chunks],
            top_n=top_k,
        )

        results = []
        for rank, result in enumerate(response.results, start=1):
            original = chunks[result.index]
            results.append(RetrievedChunk(
                chunk_id=original.chunk_id,
                content=original.content,
                source_file=original.source_file,
                page_number=original.page_number,
                section_title=original.section_title,
                doc_type=original.doc_type,
                chunk_strategy=original.chunk_strategy,
                parent_id=original.parent_id,
                score=result.relevance_score,
                rank=rank,
            ))

        return results


class PassThroughReranker(BaseReranker):
    """
    No-op reranker — returns the input unchanged.
    Used when no reranking API key is configured, or in tests.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]
