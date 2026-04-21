-- Migration 003: Documents and chunks tables
-- documents tracks ingested files (deduplication + audit trail)
-- chunks stores retrieval units with embeddings and metadata

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL    PRIMARY KEY,
    file_hash    TEXT         NOT NULL UNIQUE,  -- SHA256 of file bytes — dedup key
    filename     TEXT         NOT NULL,          -- original filename
    doc_type     TEXT         NOT NULL,          -- 'pdf' | 'docx' | 'xlsx'
    file_path    TEXT         NOT NULL,          -- path at time of ingestion
    page_count   INTEGER,                        -- number of pages/sheets
    chunk_count  INTEGER,                        -- number of chunks produced
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata     JSONB        NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL    PRIMARY KEY,
    document_id     BIGINT       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER      NOT NULL,        -- position within the document (0-based)
    content         TEXT         NOT NULL,        -- the actual text of this chunk
    content_hash    TEXT         NOT NULL,        -- SHA256 of content (for dedup at chunk level)
    embedding       vector(1536),                 -- embedding vector (NULL until embedded)
    source_file     TEXT         NOT NULL,        -- filename — for display in citations
    page_number     INTEGER,                      -- page this chunk came from
    section_title   TEXT,                         -- nearest heading above this chunk
    doc_type        TEXT         NOT NULL,        -- inherited from document
    chunk_strategy  TEXT         NOT NULL,        -- 'fixed' | 'semantic' | 'hierarchical'
    parent_id       BIGINT       REFERENCES chunks(id) ON DELETE CASCADE, -- hierarchical parent
    token_count     INTEGER      NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    metadata        JSONB        NOT NULL DEFAULT '{}',

    UNIQUE (document_id, chunk_index)             -- one chunk per position per document
);

-- Fast lookup of all chunks for a document
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- HNSW index for semantic similarity search
-- This is the index queried in Phase 4 retrieval
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index — used in Phase 4 hybrid retrieval
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING gin (content_tsv);
