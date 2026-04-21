from openai import AsyncOpenAI

from src.embeddings.base import EmbeddingModel

# Maps model name → output dimensions.
# text-embedding-3-small: fast, cheap, good quality (1536 dims)
# text-embedding-3-large: slower, costs more, higher accuracy (3072 dims)
_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedder(EmbeddingModel):
    """
    Embedding model backed by OpenAI's embeddings API.

    Default model: text-embedding-3-small
    - 1536 dimensions
    - ~$0.02 per million tokens
    - Good general-purpose quality; strong on financial text

    When to use text-embedding-3-large instead:
    - When retrieval recall matters more than cost/latency
    - For benchmark tuning on precision-sensitive queries
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        if model not in _DIMENSIONS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(_DIMENSIONS)}")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS[self._model]

    async def embed(self, text: str) -> list[float]:
        """Single text embedding. Prefer embed_batch() when you have multiple texts."""
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embedding — one API call for multiple texts.
        OpenAI returns results in the same order as the input list.
        """
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        # Sort by index to guarantee order (OpenAI spec says order is preserved,
        # but we sort defensively)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
