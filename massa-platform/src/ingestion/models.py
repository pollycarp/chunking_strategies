from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedPage:
    """
    One page (or sheet) extracted from a source document.

    Why per-page?
    Preserving page numbers is critical for citations. When the LLM answers
    "The EBITDA margin was 23.4%", the user needs to know it came from
    page 12 of the Q3 report — not just "somewhere in the document".
    """
    content: str                          # raw extracted text
    page_number: int                      # 1-based page number
    section_title: str | None = None      # nearest heading (if detectable)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """
    A fully parsed source document, ready to be chunked.

    file_hash is computed over the raw file bytes — used for deduplication.
    If the same file is ingested twice, the hash matches and we skip it.
    If the file changes (updated report), the hash changes and we re-ingest.
    """
    filename: str
    file_path: str
    doc_type: str                          # 'pdf' | 'docx' | 'xlsx'
    pages: list[ParsedPage]
    file_hash: str                         # SHA256 of raw file bytes
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenates all page content — used by chunkers."""
        return "\n\n".join(p.content for p in self.pages if p.content.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class Chunk:
    """
    A single retrieval unit — the atom of the RAG system.

    Every field has a purpose:
    - content       : the text the LLM will see
    - source_file   : citation source
    - page_number   : citation location
    - section_title : citation context
    - doc_type      : enables metadata filtering ("only search PDFs")
    - chunk_strategy: helps evaluate which strategy works best
    - token_count   : needed to stay within LLM context limits
    - parent_id     : for hierarchical chunks — reference to broader context
    """
    content: str
    source_file: str
    page_number: int | None
    section_title: str | None
    doc_type: str
    chunk_index: int                       # position in document (0-based)
    chunk_strategy: str                    # 'fixed' | 'semantic' | 'hierarchical'
    token_count: int
    content_hash: str                      # SHA256 of content
    parent_chunk_index: int | None = None  # index of parent chunk (hierarchical only)
    metadata: dict = field(default_factory=dict)
