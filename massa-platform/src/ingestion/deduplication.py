from asyncpg import Pool


async def is_already_ingested(file_hash: str, pool: Pool) -> bool:
    """
    Returns True if a document with this file hash is already in the documents table.

    Why hash-based deduplication?
    File paths change (files get moved, renamed). File names change (v1 vs v2).
    The content hash is stable — it only changes when the content changes.

    If a client re-sends us "portfolio_report_v2.pdf" which is actually
    identical to "portfolio_report_v1.pdf" we skip it automatically.
    If they send a genuinely updated version, the hash changes and we re-ingest.
    """
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM documents WHERE file_hash = $1)",
            file_hash,
        )
    return bool(exists)


async def record_document(
    file_hash: str,
    filename: str,
    doc_type: str,
    file_path: str,
    page_count: int,
    chunk_count: int,
    pool: Pool,
) -> int:
    """
    Inserts a document record and returns its generated ID.
    Called after all chunks have been successfully stored.
    """
    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(
            """
            INSERT INTO documents
                (file_hash, filename, doc_type, file_path, page_count, chunk_count)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            file_hash, filename, doc_type, file_path, page_count, chunk_count,
        )
    return doc_id
