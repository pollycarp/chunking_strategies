import re

from src.ingestion.chunkers.base import BaseChunker, chunk_hash, count_tokens
from src.ingestion.models import Chunk, ParsedDocument

# Sentence boundary: ends with . ! ? followed by whitespace or end of string
# We avoid splitting on decimals (3.14) and abbreviations (e.g. vs.)
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on clear sentence boundaries."""
    sentences = _SENTENCE_END.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (paragraph boundaries)."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


class SemanticChunker(BaseChunker):
    """
    Splits documents on natural language boundaries.

    Algorithm:
    1. Split page into paragraphs (double newline boundaries)
    2. If a paragraph is within max_tokens: add it to the current chunk buffer
    3. If adding it would exceed max_tokens: flush the buffer as a chunk, start new
    4. If a single paragraph exceeds max_tokens: split it at sentence boundaries

    Why semantic over fixed-size?
    - Preserves complete sentences and arguments
    - A retrieved chunk is a coherent unit of meaning, not a random window
    - Better for narrative documents (memos, reports, emails, meeting notes)

    When semantic chunking struggles:
    - Poorly formatted documents with no paragraph breaks
    - Tables and structured data (use fixed-size or a table-aware splitter)
    - Very long single paragraphs (falls back to sentence splitting)
    """

    def __init__(self, max_tokens: int = 400, min_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens   # don't create chunks smaller than this

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            if not page.content.strip():
                continue

            segments = self._get_segments(page.content)
            buffer: list[str] = []
            buffer_tokens = 0

            for segment in segments:
                seg_tokens = count_tokens(segment)

                if seg_tokens > self.max_tokens:
                    # Single segment too large — flush buffer first, then split segment
                    if buffer:
                        content = " ".join(buffer).strip()
                        chunks.append(self._make_chunk(
                            content, document, page, chunk_index
                        ))
                        chunk_index += 1
                        buffer, buffer_tokens = [], 0

                    # Split the oversized segment at sentence boundaries
                    for sent in _split_sentences(segment):
                        sent_tokens = count_tokens(sent)
                        if buffer_tokens + sent_tokens > self.max_tokens and buffer:
                            content = " ".join(buffer).strip()
                            chunks.append(self._make_chunk(
                                content, document, page, chunk_index
                            ))
                            chunk_index += 1
                            buffer, buffer_tokens = [], 0
                        buffer.append(sent)
                        buffer_tokens += sent_tokens

                elif buffer_tokens + seg_tokens > self.max_tokens:
                    # Adding this segment would overflow — flush buffer
                    content = " ".join(buffer).strip()
                    chunks.append(self._make_chunk(
                        content, document, page, chunk_index
                    ))
                    chunk_index += 1
                    buffer, buffer_tokens = [segment], seg_tokens

                else:
                    buffer.append(segment)
                    buffer_tokens += seg_tokens

            # Flush any remaining content in the buffer
            if buffer:
                content = " ".join(buffer).strip()
                if count_tokens(content) >= self.min_tokens:
                    chunks.append(self._make_chunk(
                        content, document, page, chunk_index
                    ))
                    chunk_index += 1

        return chunks

    def _get_segments(self, text: str) -> list[str]:
        """
        Returns paragraphs. Falls back to sentences if no paragraph breaks found.
        This handles documents that use single newlines instead of blank lines.
        """
        paragraphs = _split_paragraphs(text)
        if len(paragraphs) > 1:
            return paragraphs
        # No paragraph breaks — fall back to sentence splitting
        return _split_sentences(text) or [text]

    def _make_chunk(
        self, content: str, document: ParsedDocument, page, chunk_index: int
    ) -> Chunk:
        return Chunk(
            content=content,
            source_file=document.filename,
            page_number=page.page_number,
            section_title=page.section_title,
            doc_type=document.doc_type,
            chunk_index=chunk_index,
            chunk_strategy="semantic",
            token_count=count_tokens(content),
            content_hash=chunk_hash(content),
        )
