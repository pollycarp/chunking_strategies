from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """
    Abstract interface for all embedding models.

    Why an interface?
    The rest of the platform (chunker, retriever, cache) should not care whether
    embeddings come from OpenAI, Voyage, or an open-source model. They all call
    embed() and get back a list of floats. Swapping providers = swapping one object.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique identifier for this model — used as cache key."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the output vector (e.g. 1536 for text-embedding-3-small)."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.
        Returns a list of floats of length self.dimensions.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single API call.
        More efficient than calling embed() in a loop — most APIs batch at no extra cost.
        Returns a list of vectors in the same order as the input.
        """
        ...
