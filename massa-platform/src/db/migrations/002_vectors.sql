-- Migration 002: Embeddings cache table + HNSW index
-- Stores embeddings so identical text is never sent to the API twice.

CREATE TABLE IF NOT EXISTS embeddings_cache (
    id            BIGSERIAL    PRIMARY KEY,
    content_hash  TEXT         NOT NULL,            -- SHA256 of (model_name + text)
    model_name    TEXT         NOT NULL,            -- e.g. "text-embedding-3-small"
    content       TEXT         NOT NULL,            -- original text (for debugging)
    embedding     vector(1536) NOT NULL,            -- the embedding vector
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),

    UNIQUE (content_hash, model_name)               -- prevents duplicate embeddings
);

-- HNSW index for fast cosine similarity search.
-- vector_cosine_ops = optimise for cosine distance (<=> operator).
-- m=16: number of connections per layer (higher = better recall, more memory).
-- ef_construction=64: search depth during index build (higher = better quality, slower build).
CREATE INDEX IF NOT EXISTS embeddings_cache_hnsw_idx
    ON embeddings_cache
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
