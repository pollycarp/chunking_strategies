import tiktoken

from src.ingestion.chunkers.base import BaseChunker, chunk_hash, count_tokens
from src.ingestion.models import Chunk, ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker(BaseChunker):
    """
    Splits documents into fixed-size token windows with overlap.

    Parameters:
        max_tokens  : maximum tokens per chunk (default 400)
                      OpenAI text-embedding-3-small limit is 8191 — we stay well below
                      so the LLM has room for the query + other context too.
        overlap     : tokens shared between consecutive chunks (default 50)
                      Overlap prevents a sentence that straddles a boundary from
                      being retrievable from neither chunk.

    When to use fixed-size:
    - When document structure is inconsistent (mixed formats, no headings)
    - As a baseline to benchmark against semantic/hierarchical strategies
    - When speed matters more than boundary quality

    When NOT to use fixed-size:
    - Financial tables — a fixed split will cut a table mid-row
    - Narrative reports — breaks mid-paragraph, losing argument flow
    """

    def __init__(self, max_tokens: int = 400, overlap: int = 50) -> None:
        if overlap >= max_tokens:
            raise ValueError(f"overlap ({overlap}) must be less than max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.content.strip():
                continue

            # Tokenise at the page level — we work in token IDs for precision
            token_ids = _ENCODING.encode(page.content)

            step = self.max_tokens - self.overlap
            start = 0

            while start < len(token_ids):
                end = min(start + self.max_tokens, len(token_ids))
                window_ids = token_ids[start:end]
                content = _ENCODING.decode(window_ids).strip()

                if content:
                    chunks.append(Chunk(
                        content=content,
                        source_file=document.filename,
                        page_number=page.page_number,
                        section_title=page.section_title,
                        doc_type=document.doc_type,
                        chunk_index=chunk_index,
                        chunk_strategy="fixed",
                        token_count=len(window_ids),
                        content_hash=chunk_hash(content),
                    ))
                    chunk_index += 1

                # Advance by step (not max_tokens) to create the overlap
                start += step

        return chunks
