import voyageai

from src.embeddings.base import EmbeddingModel

_DIMENSIONS: dict[str, int] = {
    "voyage-finance-2": 1024,   # Finance-domain model — best for financial documents
    "voyage-3":         1024,   # General purpose, high accuracy
    "voyage-3-lite":    512,    # Fast and cheap, lower accuracy
}


class VoyageEmbedder(EmbeddingModel):
    """
    Embedding model backed by Voyage AI.

    Why Voyage for financial data?
    Voyage finance-2 is trained on financial corpora (SEC filings, earnings calls,
    analyst reports). For domain-specific retrieval over financial documents, it
    outperforms general-purpose models on precision because it understands terms
    like 'EBITDA', 'LBO', 'covenant', 'amortisation' in context rather than
    relying on surface-level word similarity.

    Trade-off: smaller context window and lower dimensions (1024 vs 1536) than
    OpenAI text-embedding-3-large, but better recall on financial queries.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-finance-2",
    ) -> None:
        if model not in _DIMENSIONS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_DIMENSIONS)}")
        self._model = model
        # voyageai uses a sync client internally but we wrap it for our async interface
        self._client = voyageai.AsyncClient(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS[self._model]

    async def embed(self, text: str) -> list[float]:
        result = await self._client.embed(
            texts=[text],
            model=self._model,
            input_type="document",   # "document" for storage; "query" for search queries
        )
        return result.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(
            texts=texts,
            model=self._model,
            input_type="document",
        )
        return result.embeddings
