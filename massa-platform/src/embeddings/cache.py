import hashlib

from asyncpg import Pool

from src.embeddings.base import EmbeddingModel


class CachedEmbedder:
    """
    Wraps any EmbeddingModel with a Postgres-backed cache.

    Why cache embeddings?
    - API calls cost money (~$0.02 per million tokens for OpenAI small)
    - API calls take ~100ms; DB lookups take ~1ms
    - Documents are ingested once but may be re-processed (e.g. after chunking changes)
    - The same chunk text will appear across multiple ingestion runs

    How the cache works:
    1. Hash the input text + model name with SHA256 → cache key
    2. Look up the key in embeddings_cache table
    3. Hit → return stored embedding immediately (no API call)
    4. Miss → call the API, store result, return embedding

    Thread/concurrency safety:
    ON CONFLICT DO NOTHING handles the case where two coroutines try to insert
    the same embedding simultaneously — the second insert is silently ignored.
    """

    def __init__(self, embedder: EmbeddingModel, pool: Pool) -> None:
        self._embedder = embedder
        self._pool = pool

    def _make_hash(self, text: str) -> str:
        """
        SHA256 hash of (model_name + text).
        Including model_name means the same text embedded by two different models
        gets two separate cache entries — correct behaviour.
        """
        raw = f"{self._embedder.model_name}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    async def embed(self, text: str) -> list[float]:
        """Returns embedding from cache if available, otherwise calls the API."""
        content_hash = self._make_hash(text)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT embedding
                FROM   embeddings_cache
                WHERE  content_hash = $1
                AND    model_name   = $2
                """,
                content_hash,
                self._embedder.model_name,
            )

            if row is not None:
                # Cache hit — convert pgvector type to plain list[float]
                return list(row["embedding"])

            # Cache miss — call the API
            embedding = await self._embedder.embed(text)

            await conn.execute(
                """
                INSERT INTO embeddings_cache
                    (content_hash, model_name, content, embedding)
                VALUES
                    ($1, $2, $3, $4)
                ON CONFLICT (content_hash, model_name) DO NOTHING
                """,
                content_hash,
                self._embedder.model_name,
                text,
                embedding,
            )

            return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embed with cache awareness.
        Only calls the API for texts not already in cache.
        Returns results in the same order as input.
        """
        hashes = [self._make_hash(t) for t in texts]

        async with self._pool.acquire() as conn:
            # Fetch all cached embeddings in one query
            rows = await conn.fetch(
                """
                SELECT content_hash, embedding
                FROM   embeddings_cache
                WHERE  content_hash = ANY($1::text[])
                AND    model_name   = $2
                """,
                hashes,
                self._embedder.model_name,
            )

        cached = {row["content_hash"]: list(row["embedding"]) for row in rows}

        # Identify which texts need API calls
        missing_indices = [i for i, h in enumerate(hashes) if h not in cached]
        missing_texts = [texts[i] for i in missing_indices]

        if missing_texts:
            new_embeddings = await self._embedder.embed_batch(missing_texts)

            async with self._pool.acquire() as conn:
                for i, embedding in zip(missing_indices, new_embeddings):
                    await conn.execute(
                        """
                        INSERT INTO embeddings_cache
                            (content_hash, model_name, content, embedding)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (content_hash, model_name) DO NOTHING
                        """,
                        hashes[i],
                        self._embedder.model_name,
                        texts[i],
                        embedding,
                    )
                    cached[hashes[i]] = embedding

        # Return results in original input order
        return [cached[h] for h in hashes]
