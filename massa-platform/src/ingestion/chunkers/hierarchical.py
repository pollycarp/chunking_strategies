from src.ingestion.chunkers.base import BaseChunker, chunk_hash, count_tokens
from src.ingestion.chunkers.fixed_size import FixedSizeChunker
from src.ingestion.chunkers.semantic import SemanticChunker
from src.ingestion.models import Chunk, ParsedDocument, ParsedPage


class HierarchicalChunker(BaseChunker):
    """
    Two-level chunking: large parent sections + small child chunks.

    Structure:
        Parent (800 tokens) ──► child_0 (200 tokens)
                             ──► child_1 (200 tokens)
                             ──► child_2 (200 tokens)
                             ──► child_3 (200 tokens)

    Why hierarchical?
    A small chunk matches a precise query well, but it lacks context.
    The phrase "margin improved to 23.4%" is a great match for
    "what was the EBITDA margin?" — but the LLM also needs the
    surrounding section to know: which company? which quarter?
    what drove the improvement?

    The "small-to-big" retrieval pattern (Phase 4):
    1. Match the small child chunk (high precision)
    2. Return the parent chunk to the LLM (full context)

    This gives you the best of both worlds: precise matching with rich context.

    Parameters:
        parent_max_tokens : size of parent sections (default 800)
        child_max_tokens  : size of child chunks (default 200)
    """

    def __init__(
        self,
        parent_max_tokens: int = 800,
        child_max_tokens: int = 200,
    ) -> None:
        if child_max_tokens >= parent_max_tokens:
            raise ValueError("child_max_tokens must be smaller than parent_max_tokens")
        self.parent_max_tokens = parent_max_tokens
        self.child_max_tokens = child_max_tokens
        # Parents use semantic splitting to respect paragraph boundaries
        self._parent_chunker = SemanticChunker(max_tokens=parent_max_tokens)
        # Children use fixed-size for predictable size within a known text
        self._child_chunker = FixedSizeChunker(max_tokens=child_max_tokens, overlap=20)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        chunk_index = 0

        # Step 1: Create parent chunks from the full document
        parent_chunks = self._parent_chunker.chunk(document)

        for parent in parent_chunks:
            # Assign the parent chunk with its real index
            parent.chunk_index = chunk_index
            parent.chunk_strategy = "hierarchical_parent"
            all_chunks.append(parent)
            parent_index = chunk_index
            chunk_index += 1

            # Step 2: Split parent content into child chunks
            # We create a synthetic single-page document from the parent content
            # so the child chunker can process it normally
            parent_as_doc = ParsedDocument(
                filename=document.filename,
                file_path=document.file_path,
                doc_type=document.doc_type,
                pages=[ParsedPage(
                    content=parent.content,
                    page_number=parent.page_number or 1,
                    section_title=parent.section_title,
                )],
                file_hash=document.file_hash,
            )

            child_chunks = self._child_chunker.chunk(parent_as_doc)

            for child in child_chunks:
                child.chunk_index = chunk_index
                child.chunk_strategy = "hierarchical_child"
                child.parent_chunk_index = parent_index   # link to parent
                all_chunks.append(child)
                chunk_index += 1

        return all_chunks
