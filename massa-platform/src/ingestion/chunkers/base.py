import hashlib
from abc import ABC, abstractmethod

import tiktoken

from src.ingestion.models import Chunk, ParsedDocument

# cl100k_base is the encoding used by text-embedding-3-small and text-embedding-3-large.
# We use it for all token counting so chunk sizes are accurate for our embedding model.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> str:
    """Returns the number of tokens in text using OpenAI's cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def chunk_hash(content: str) -> str:
    """SHA256 of chunk content — used for chunk-level deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class BaseChunker(ABC):
    """
    Abstract interface for all chunking strategies.

    All chunkers take a ParsedDocument and return a flat list of Chunks.
    The chunk_strategy field on each Chunk identifies which strategy produced it
    — useful for evaluating strategies against each other in Phase 8.
    """

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Split a parsed document into retrieval-ready chunks.
        Returns chunks in document order (chunk_index 0, 1, 2, ...).
        """
        ...
