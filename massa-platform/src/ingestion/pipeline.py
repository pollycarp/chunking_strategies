from dataclasses import dataclass
from pathlib import Path

from asyncpg import Pool

from src.embeddings.base import EmbeddingModel
from src.ingestion.chunkers.base import BaseChunker
from src.ingestion.chunkers.fixed_size import FixedSizeChunker
from src.ingestion.deduplication import is_already_ingested
from src.ingestion.models import Chunk, ParsedDocument
from src.ingestion.parsers.base import DocumentParser
from src.ingestion.parsers.docx_parser import DocxParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.xlsx_parser import XLSXParser

# File extension → parser class
_PARSERS: dict[str, type[DocumentParser]] = {
    ".pdf":  PDFParser,
    ".docx": DocxParser,
    ".xlsx": XLSXParser,
}


@dataclass
class IngestionResult:
    """Returned by the pipeline after ingesting a document."""
    document_id: int | None    # None if skipped (already ingested)
    filename: str
    skipped: bool              # True if file hash already in DB
    chunk_count: int
    page_count: int


class IngestionPipeline:
    """
    Orchestrates: parse → chunk → embed → store.

    Design principles:
    - Each step is independent and testable on its own
    - Embedding happens in batch (one API call per document, not per chunk)
    - Deduplication is the first check — expensive steps are skipped early
    - The pipeline doesn't care which parser, chunker, or embedder is used

    Usage:
        pipeline = IngestionPipeline(pool=pool, embedder=embedder)
        result = await pipeline.ingest(Path("report.pdf"))
    """

    def __init__(
        self,
        pool: Pool,
        embedder: EmbeddingModel,
        chunker: BaseChunker | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        # Default to fixed-size chunking; caller can override
        self._chunker = chunker or FixedSizeChunker()

    async def ingest(self, file_path: Path) -> IngestionResult:
        """
        Full ingestion pipeline for a single file.

        Steps:
        1. Resolve parser by file extension
        2. Parse the file
        3. Check deduplication (skip if unchanged)
        4. Chunk the document
        5. Embed all chunks in one batch API call
        6. Store document + chunks in DB
        7. Record document in documents table
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in _PARSERS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {list(_PARSERS.keys())}"
            )

        # Step 1 & 2: Parse
        parser = _PARSERS[suffix]()
        document = parser.parse(file_path)

        # Step 3: Deduplication check
        if await is_already_ingested(document.file_hash, self._pool):
            return IngestionResult(
                document_id=None,
                filename=document.filename,
                skipped=True,
                chunk_count=0,
                page_count=document.page_count,
            )

        # Step 4: Chunk
        chunks = self._chunker.chunk(document)
        if not chunks:
            raise ValueError(f"No chunks produced from {file_path.name}")

        # Step 5: Embed all chunks in one batch call
        contents = [c.content for c in chunks]
        embeddings = await self._embedder.embed_batch(contents)

        # Step 6: Store document + chunks
        doc_id = await self._store(document, chunks, embeddings)

        return IngestionResult(
            document_id=doc_id,
            filename=document.filename,
            skipped=False,
            chunk_count=len(chunks),
            page_count=document.page_count,
        )

    async def _store(
        self,
        document: ParsedDocument,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Stores the document record first, then all chunks within a transaction.

        Why document first?
        The chunks table has a FK constraint on document_id. We must insert the
        document row to get the real ID before we can insert any chunks.

        Two-pass chunk insertion for hierarchical chunks:
        - Pass 1: insert parent chunks (parent_id = NULL) → collect their DB IDs
        - Pass 2: insert child chunks with resolved parent DB IDs
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Step 1: insert document record first to get the real ID
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documents
                        (file_hash, filename, doc_type, file_path, page_count, chunk_count)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    document.file_hash, document.filename, document.doc_type,
                    document.file_path, document.page_count, len(chunks),
                )

                # Step 2: insert parent chunks (those without a parent reference)
                # Map chunk_index → DB row id so children can reference their parent
                index_to_db_id: dict[int, int] = {}

                for chunk, embedding in zip(chunks, embeddings):
                    if chunk.parent_chunk_index is not None:
                        continue  # children handled in second pass

                    db_id = await conn.fetchval(
                        """
                        INSERT INTO chunks
                            (document_id, chunk_index, content, content_hash,
                             embedding, source_file, page_number, section_title,
                             doc_type, chunk_strategy, parent_id, token_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                        """,
                        doc_id,
                        chunk.chunk_index, chunk.content, chunk.content_hash,
                        embedding, chunk.source_file, chunk.page_number,
                        chunk.section_title, chunk.doc_type, chunk.chunk_strategy,
                        None, chunk.token_count,
                    )
                    index_to_db_id[chunk.chunk_index] = db_id

                # Step 3: insert child chunks with resolved parent DB IDs
                for chunk, embedding in zip(chunks, embeddings):
                    if chunk.parent_chunk_index is None:
                        continue  # already inserted

                    parent_db_id = index_to_db_id.get(chunk.parent_chunk_index)
                    db_id = await conn.fetchval(
                        """
                        INSERT INTO chunks
                            (document_id, chunk_index, content, content_hash,
                             embedding, source_file, page_number, section_title,
                             doc_type, chunk_strategy, parent_id, token_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                        """,
                        doc_id,
                        chunk.chunk_index, chunk.content, chunk.content_hash,
                        embedding, chunk.source_file, chunk.page_number,
                        chunk.section_title, chunk.doc_type, chunk.chunk_strategy,
                        parent_db_id, chunk.token_count,
                    )
                    index_to_db_id[chunk.chunk_index] = db_id

        return doc_id
