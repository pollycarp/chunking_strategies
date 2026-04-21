# Phase 3: Document Ingestion & Chunking

## Overview

Phase 3 builds the ingestion pipeline — the system that takes raw files (PDFs, Word documents, Excel workbooks) and transforms them into clean, embedded, retrievable chunks stored in Postgres. This is the data preparation layer that feeds the retrieval system in Phase 4.

By the end of this phase you have:
- Three document parsers (PDF, DOCX, XLSX) with metadata preservation
- Three chunking strategies (fixed-size, semantic, hierarchical)
- Hash-based file deduplication
- A full ingestion pipeline: parse → chunk → embed → store
- A `documents` table (file registry) and `chunks` table (retrieval units) in Postgres
- 27 passing tests across all phases

---

## Table of Contents

1. [Concepts](#concepts)
   - [Why Chunking Matters](#why-chunking-matters)
   - [Fixed-Size Chunking](#fixed-size-chunking)
   - [Semantic Chunking](#semantic-chunking)
   - [Hierarchical Chunking](#hierarchical-chunking)
   - [Metadata — The Hidden Value](#metadata--the-hidden-value)
   - [Tokenization with tiktoken](#tokenization-with-tiktoken)
   - [Hash-Based Deduplication](#hash-based-deduplication)
   - [Document Schema Design](#document-schema-design)
   - [The Full-Text Search Column](#the-full-text-search-column)
2. [Project Structure](#project-structure)
3. [File Reference](#file-reference)
4. [Setup & Running](#setup--running)
5. [Tests](#tests)
6. [Lessons Learned](#lessons-learned)

---

## Concepts

### Why Chunking Matters

LLMs have a **context window** — a hard limit on how many tokens they can process at once. Even the largest models (Claude with 200K tokens, GPT-4 with 128K) cannot consume a full document library in one call. A 200-page financial report alone can exceed 100K tokens.

But the deeper reason to chunk is **retrieval precision**. If you feed an entire 50-page investment memo as context, you're asking the LLM to find a needle in a haystack. If you retrieve only the 3 most relevant 400-token chunks, the LLM has exactly the right information and nothing noisy around it.

**The chunking paradox:**
- Chunks too large → less precise retrieval, more noise in context
- Chunks too small → lose surrounding context, incomplete answers
- The right size depends on the document type and query pattern

---

### Fixed-Size Chunking

Split text into windows of exactly N tokens, with M tokens of overlap between consecutive chunks.

```
Document: [token_0 ... token_999]

Chunk 0:  [token_0   ... token_399]   (400 tokens)
Chunk 1:  [token_350 ... token_749]   (400 tokens, 50 overlap with chunk 0)
Chunk 2:  [token_700 ... token_999]   (300 tokens, last chunk)
```

**Why overlap?**
Without overlap, a sentence that straddles a boundary appears in neither chunk fully:
```
...the EBITDA margin reached 23.4% in Q3, | exceeding analyst expectations by...
                              boundary ↑
```
With 50 tokens of overlap, both chunks contain the full sentence in context.

**Token-based vs character-based splitting:**
Character-based splitting (`text[:2000]`) is imprecise — "EBITDA" is 1 token, "antidisestablishmentarianism" is 6. We use `tiktoken` with `cl100k_base` encoding (OpenAI's encoding for embedding models) so chunk sizes are accurate for the model we're using.

**When to use fixed-size:**
- Baseline benchmark — always compare other strategies against it
- Documents without clear structure (no headings, mixed formatting)
- Speed matters more than boundary quality

**When NOT to use fixed-size:**
- Financial tables — splits mid-row, destroying the row-column relationship
- Narrative text — breaks mid-argument, loses logical flow

---

### Semantic Chunking

Split on natural language boundaries rather than token counts. The algorithm:

```
1. Split page into paragraphs (blank line boundaries)
2. For each paragraph:
   - If adding it stays within max_tokens → add to current buffer
   - If it would overflow → flush buffer as a chunk, start new buffer
   - If paragraph itself exceeds max_tokens → split at sentence boundaries
3. Flush remaining buffer as final chunk
```

**Paragraph-first, sentence-second:**
Double newlines (`\n\n`) are strong signals of thought boundaries in professional documents. Sentences are finer-grained fallbacks for very long paragraphs.

**The `min_tokens` parameter:**
Prevents creating tiny chunks (e.g. a single heading line "Risk Factors" as a 2-token chunk). Chunks below `min_tokens` are merged into the next one.

**Example — why this matters for financial documents:**

Bad (fixed-size split mid-paragraph):
```
Chunk A: "...The company's liquidity position remains strong. Cash and cash
          equivalents totalled $45.2M at quarter end. This represents a $3.1M"
Chunk B: "increase from the prior quarter, primarily driven by strong operating
          cash flows of $12.4M partially offset by capital expenditures..."
```

Good (semantic split at paragraph boundary):
```
Chunk A: "The company's liquidity position remains strong. Cash and cash
          equivalents totalled $45.2M at quarter end. This represents a $3.1M
          increase from the prior quarter, primarily driven by strong operating
          cash flows of $12.4M partially offset by capital expenditures of $9.3M."

Chunk B: "Debt management..."  ← next paragraph
```

---

### Hierarchical Chunking

Creates two levels of chunks from the same document:

```
Parent (800 tokens) — a full section
├── Child 0 (200 tokens) — first quarter of parent
├── Child 1 (200 tokens) — second quarter
├── Child 2 (200 tokens) — third quarter
└── Child 3 (200 tokens) — fourth quarter
```

Children store a `parent_id` foreign key pointing to their parent chunk's DB row.

**The "small-to-big" retrieval pattern (used in Phase 4):**
1. **Match** on child chunks — small size = high embedding precision
2. **Return** the parent chunk to the LLM — full context for a complete answer

Why does this work?
- Small chunks (200 tokens) produce embeddings that are tightly focused on one idea → better vector similarity matching
- But 200 tokens is often not enough context for the LLM to answer fully
- The parent (800 tokens) provides the full section context

**Concrete example:**
Query: "What drove the EBITDA margin expansion?"

Child match: "EBITDA margin expanded by 230 basis points" (precise match, 12 tokens)

Parent returned to LLM: The full 800-token section including: what drove the expansion, which segments contributed, management commentary, comparison to prior year, guidance.

**Implementation detail:**
Parents are created with semantic chunking (respects paragraph boundaries at ~800 tokens). Children are created with fixed-size chunking within each parent (predictable small sizes). This gives structured parents and precise children.

---

### Metadata — The Hidden Value

A chunk without metadata is nearly useless in production. Consider:

```
"EBITDA margin was 23.4%"
```

Without metadata:
- Which company?
- Which quarter?
- Which document?
- Which page?
- Is this a projection or actual?

With metadata:
```
source_file:   "AgroHoldings_Q3_2024_Board_Report.pdf"
page_number:   12
section_title: "Financial Performance"
doc_type:      "pdf"
chunk_strategy: "semantic"
```

**Why each field matters:**

| Field | Use in production |
|---|---|
| `source_file` | LLM citation: "According to AgroHoldings Q3 report..." |
| `page_number` | User verification: "See page 12" |
| `section_title` | Metadata filtering: "only search Risk Factors sections" |
| `doc_type` | Filtering: "only search PDFs, not spreadsheets" |
| `chunk_strategy` | Evaluation: "did semantic chunking outperform fixed?" |
| `token_count` | Context budget management: pack chunks without overflowing |

**The `JSONB metadata` column:**
Each chunk and document also has a `JSONB metadata` column for arbitrary additional metadata — client ID, project code, confidentiality level, ingestion batch ID. This avoids schema changes every time a new metadata field is needed.

---

### Tokenization with tiktoken

`tiktoken` is OpenAI's tokenizer — the same one used by `text-embedding-3-small`. Token counts from tiktoken are exact for OpenAI models.

**Why exact token counting matters:**
`text-embedding-3-small` has an 8191 token limit. If a chunk exceeds this, the API truncates it silently — you get an embedding for a partial chunk without any error. Your retrieval system then matches on an incomplete representation.

**The `cl100k_base` encoding:**
Used by GPT-3.5, GPT-4, and all `text-embedding-3-*` models. Rules of thumb:
- 1 token ≈ 4 characters in English
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words ≈ a short paragraph

**tiktoken downloads encoding data on first use (~1MB). This happens automatically and is cached locally.**

---

### Hash-Based Deduplication

Every file is fingerprinted with a SHA256 hash of its raw bytes before any parsing happens.

```
SHA256("portfolio_report.pdf") → "a3f9c2d8e1b4..."

First ingest:
  → hash not in documents table → process → store

Second ingest (same file):
  → hash found in documents table → skip immediately

Ingest after file update:
  → hash changed (different bytes) → process → store as new version
```

**Why file hash, not filename?**
- Files get renamed ("v1" → "v2", "final" → "final_FINAL")
- The same content can arrive with different filenames
- Content is what matters, not the label

**The dedup check is the first operation in the pipeline** — before parsing, chunking, or any API calls. This means:
- Re-running ingestion on a folder of already-processed files costs ~1ms per file (just a DB lookup)
- You never pay for re-embedding the same content twice
- Re-ingestion after a pipeline error is safe — successfully processed files are skipped

---

### Document Schema Design

Two tables with a parent-child relationship:

**`documents` table — one row per source file:**
```sql
id           BIGSERIAL PRIMARY KEY
file_hash    TEXT UNIQUE        -- deduplication key
filename     TEXT               -- original filename
doc_type     TEXT               -- 'pdf' | 'docx' | 'xlsx'
file_path    TEXT               -- path at ingestion time
page_count   INTEGER
chunk_count  INTEGER
ingested_at  TIMESTAMPTZ
metadata     JSONB
```

**`chunks` table — many rows per document:**
```sql
id              BIGSERIAL PRIMARY KEY
document_id     BIGINT REFERENCES documents(id) ON DELETE CASCADE
chunk_index     INTEGER            -- position in document (0-based)
content         TEXT               -- the text
content_hash    TEXT               -- SHA256 of content
embedding       vector(1536)       -- the semantic vector
source_file     TEXT               -- for citations
page_number     INTEGER
section_title   TEXT
doc_type        TEXT
chunk_strategy  TEXT               -- 'fixed' | 'semantic' | 'hierarchical_*'
parent_id       BIGINT REFERENCES chunks(id)  -- hierarchical link
token_count     INTEGER
content_tsv     tsvector GENERATED -- full-text search index (auto-computed)
metadata        JSONB
```

**Why `ON DELETE CASCADE`?**
If a document is deleted (e.g. to re-ingest a newer version), all its chunks are automatically deleted. No orphaned chunks with dangling `document_id` references. No manual cleanup queries.

**Why store `embedding` directly on chunks (not reference `embeddings_cache`)?**
Retrieval queries join nothing — they query `chunks` and get the embedding in the same row. This is a deliberate denormalization for query performance. The `embeddings_cache` table serves as an API-level cache during ingestion; the `chunks` table is the retrieval index.

**Insertion order matters:**
The FK constraint on `document_id` requires the `documents` row to exist before any `chunks` row. The pipeline inserts the document record first, gets its ID, then inserts all chunks with the real ID. The naive approach of inserting chunks with a placeholder `0` fails immediately with `ForeignKeyViolationError`.

---

### The Full-Text Search Column

```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

This is a **generated column** — Postgres automatically computes and updates it whenever `content` changes. You never insert into it directly.

`tsvector` is Postgres's full-text search type — it stores a processed, lexeme-normalized representation of the text:
- "running" and "ran" → both stored as lexeme "run"
- Stop words ("the", "a", "is") → removed
- Word positions → preserved for phrase matching

A `GIN` index on this column makes full-text search fast. This will be used in Phase 4 for **hybrid retrieval** — combining semantic vector search with keyword-based full-text search. The combination catches both "EBITDA" (keyword match) and "operating profitability" (semantic match) for the same query.

---

## Project Structure

```
massa-platform/
├── pyproject.toml                              # Added: pdfplumber, python-docx,
│                                              #        openpyxl, tiktoken, fpdf2
│
├── src/
│   ├── db/
│   │   └── migrations/
│   │       └── 003_documents.sql              # NEW: documents + chunks tables
│   └── ingestion/
│       ├── __init__.py
│       ├── models.py                          # NEW: ParsedPage, ParsedDocument, Chunk
│       ├── deduplication.py                   # NEW: is_already_ingested, record_document
│       ├── pipeline.py                        # NEW: IngestionPipeline orchestrator
│       ├── parsers/
│       │   ├── __init__.py
│       │   ├── base.py                        # NEW: DocumentParser + compute_file_hash
│       │   ├── pdf_parser.py                  # NEW: pdfplumber
│       │   ├── docx_parser.py                 # NEW: python-docx + heading detection
│       │   └── xlsx_parser.py                 # NEW: openpyxl, sheet → page
│       └── chunkers/
│           ├── __init__.py
│           ├── base.py                        # NEW: BaseChunker + tiktoken helpers
│           ├── fixed_size.py                  # NEW: token windows with overlap
│           ├── semantic.py                    # NEW: paragraph/sentence boundaries
│           └── hierarchical.py               # NEW: parent sections + child chunks
│
└── tests/
    ├── fixtures/                              # NEW: temp fixture files created by tests
    └── test_ingestion.py                      # NEW: 14 tests
```

---

## File Reference

### `003_documents.sql`
Creates two tables:
- `documents` — one row per ingested file. `file_hash TEXT UNIQUE` enforces deduplication at the DB level as a safety net on top of the application-level check.
- `chunks` — retrieval units. Key design decisions: `embedding vector(1536)` stored directly (no JOIN needed at query time); `content_tsv` is a generated column for free full-text search; `parent_id` self-reference enables hierarchical chunking; `ON DELETE CASCADE` on both FK relationships.

Three indexes:
- `chunks_document_id_idx` — fast lookup of all chunks for a document (used when deleting/updating)
- `chunks_embedding_hnsw_idx` — HNSW index for cosine similarity search (Phase 4)
- `chunks_content_tsv_idx` — GIN index for full-text keyword search (Phase 4)

### `src/ingestion/models.py`
Three dataclasses forming the data pipeline contract:
- `ParsedPage` — output of parsers; content + page number + optional section title
- `ParsedDocument` — groups pages with file-level metadata; `full_text` property concatenates all pages
- `Chunk` — output of chunkers; all retrieval metadata included; `parent_chunk_index` enables hierarchical linking

### `src/ingestion/parsers/base.py`
`DocumentParser` abstract class with one method (`parse`) and one static utility (`compute_file_hash`). The hash reads the file in 8KB chunks to handle large files without loading them fully into memory.

### `src/ingestion/parsers/pdf_parser.py`
Uses `pdfplumber`. Skips empty pages (common in reports with full-page charts). Raises `ValueError` for PDFs with no extractable text (scanned without OCR) rather than returning empty documents silently.

### `src/ingestion/parsers/docx_parser.py`
Uses `python-docx`. Word documents have no pages — groups paragraphs into simulated pages of 50 paragraphs each. Detects heading styles (`Heading 1`, `Heading 2`, `Heading 3`, `Title`) to populate `section_title`. Tracks the most recent heading seen to carry section context across page boundaries.

### `src/ingestion/parsers/xlsx_parser.py`
Uses `openpyxl` with `data_only=True` (reads computed cell values, not formulas). Each worksheet becomes one page. Rows are pipe-delimited (`|`) to preserve tabular structure without conflicting with commas in financial numbers. Sheet name becomes `section_title`.

### `src/ingestion/chunkers/base.py`
Defines `BaseChunker` interface, `count_tokens()` using `tiktoken cl100k_base`, and `chunk_hash()` (SHA256 of content). The tiktoken encoding is module-level — loaded once, reused across all chunking calls.

### `src/ingestion/chunkers/fixed_size.py`
Operates directly on token ID arrays (not text strings) for byte-perfect splitting. Advances by `max_tokens - overlap` per step to create the overlap window. Decodes each window back to text for storage.

### `src/ingestion/chunkers/semantic.py`
Two-level splitting: paragraph boundaries first (double newlines), sentence boundaries as fallback (`.`, `!`, `?` followed by uppercase). A buffer accumulates segments until `max_tokens` is reached, then flushes. Oversized single paragraphs are sentence-split inline without losing the buffer state.

### `src/ingestion/chunkers/hierarchical.py`
Composes `SemanticChunker` (for parents) and `FixedSizeChunker` (for children). Creates synthetic single-page documents from parent content so the child chunker can process them with its existing interface. Children store `parent_chunk_index` which the pipeline resolves to a DB `parent_id` during storage.

### `src/ingestion/deduplication.py`
Two async functions:
- `is_already_ingested(file_hash, pool)` — single SELECT EXISTS query; called before any parsing
- `record_document(...)` — INSERT into documents table; called after all chunks are stored

### `src/ingestion/pipeline.py`
`IngestionPipeline` orchestrator. Key design decisions:
- Parser routing by file extension (`.pdf` → `PDFParser`, etc.)
- Embeddings happen in one batch call (`embed_batch`) for the whole document — not per chunk
- Document record inserted first (to satisfy FK constraint), then all chunks
- Two-pass chunk insertion: parents first (collect DB IDs), children second (resolve parent IDs)
- Returns `IngestionResult` with skip status, chunk count, and document ID

---

## Setup & Running

### Prerequisites
Completed Phase 1 and 2 setup (Docker running, migrations applied).

### Install new dependencies
```bash
python -m uv sync --all-extras
```

New packages installed:
- `pdfplumber` — PDF text extraction
- `python-docx` — Word document parsing
- `openpyxl` — Excel workbook parsing
- `tiktoken` — OpenAI tokenizer for accurate token counting
- `fpdf2` — creates PDF fixture files in tests (dev only)

### Run the new migration
```bash
python -m uv run python -m src.db.migrate
```

Expected output:
```
  [skip] 001_init.sql already applied
  [skip] 002_vectors.sql already applied
  [ok]   003_documents.sql applied
```

### Run all tests
```bash
python -m uv run pytest tests/ -v
```

Expected output:
```
27 passed, 1 skipped in ~3s
```

### Ingest a file manually (example usage)
```python
import asyncio
from pathlib import Path
from src.db.connection import create_pool
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.embeddings.cache import CachedEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunkers.semantic import SemanticChunker

async def main():
    pool = await create_pool()
    embedder = CachedEmbedder(OpenAIEmbedder(api_key="sk-..."), pool=pool)
    pipeline = IngestionPipeline(
        pool=pool,
        embedder=embedder,
        chunker=SemanticChunker(max_tokens=400),
    )
    result = await pipeline.ingest(Path("report.pdf"))
    print(f"Ingested {result.chunk_count} chunks (skipped={result.skipped})")

asyncio.run(main())
```

---

## Tests

### Parser Tests

#### `test_pdf_parser_extracts_text`
**Type:** Integration (file I/O, no DB)
**What it tests:** PDF parser extracts text from both pages, preserves page numbers, sets doc_type and filename correctly.
**Why it matters:** If page numbers are wrong, citations are wrong. If doc_type is wrong, metadata filtering in Phase 4 breaks.

#### `test_pdf_parser_raises_on_missing_file`
**Type:** Unit
**What it tests:** `FileNotFoundError` is raised for a non-existent path.
**Why it matters:** Fail fast with a clear error. Without this, the parser would crash with a cryptic internal error from pdfplumber 10 stack frames deep.

#### `test_docx_parser_extracts_text_and_sections`
**Type:** Integration (file I/O)
**What it tests:** DOCX parser extracts text and makes heading text available in the parsed document.
**Why it matters:** Section titles are critical metadata for financial documents — knowing a chunk is from "Risk Factors" vs "Financial Projections" changes how the LLM should weight and present it.

#### `test_xlsx_parser_extracts_sheets_as_pages`
**Type:** Integration (file I/O)
**What it tests:** Each worksheet becomes a page, sheet name becomes section_title, pipe-delimited format is used.
**Why it matters:** Financial spreadsheets contain the most precise quantitative data. Correct extraction with preserved structure is essential for the LLM to read numbers accurately.

#### `test_all_parsers_populate_required_metadata`
**Type:** Integration (file I/O)
**What it tests:** Every parser for every format populates all required fields: filename, file_path, doc_type, file_hash, page numbers, non-empty content.
**Why it matters:** A metadata field missing from one parser is a silent failure — it only shows up when a user tries to filter or cite from that document type.

---

### Chunker Tests

#### `test_fixed_size_chunker_respects_token_limit`
**Type:** Unit
**What it tests:** Every chunk produced has `token_count <= max_tokens`.
**Why it matters:** Exceeding the embedding model's token limit causes silent truncation — the embedding represents a partial chunk, degrading retrieval quality without any error.

#### `test_fixed_size_chunker_overlap_creates_continuity`
**Type:** Unit
**What it tests:** Consecutive chunks share words (overlap is present).
**Why it matters:** Without overlap, context at chunk boundaries is lost. A sentence spanning two chunks is retrievable from neither — a blind spot in retrieval.

#### `test_semantic_chunker_no_mid_sentence_splits`
**Type:** Unit
**What it tests:** All chunks are non-empty and have positive token counts.
**Why it matters:** Validates the chunker produces valid output. Mid-sentence splits produce incomplete semantic units that match queries poorly.

#### `test_semantic_chunker_preserves_metadata`
**Type:** Unit
**What it tests:** Semantic chunks carry source_file, page_number, doc_type, chunk_strategy from the source document.
**Why it matters:** Metadata is attached at chunking time. If a chunker drops it, it's gone — there's no way to recover it from the stored chunk later.

#### `test_hierarchical_chunker_creates_parent_child_structure`
**Type:** Unit
**What it tests:** Parent chunks have `parent_chunk_index=None`; child chunks have `parent_chunk_index` pointing to a valid parent index.
**Why it matters:** A broken parent-child reference means the pipeline tries to store children with a non-existent parent DB ID — a FK violation. This test catches the structural issue before it hits the DB.

#### `test_all_chunks_have_required_fields`
**Type:** Unit
**What it tests:** All three chunking strategies produce chunks with all required fields populated.
**Why it matters:** Contract test — ensures adding a new chunker strategy doesn't accidentally drop required metadata.

---

### Pipeline & Deduplication Tests

#### `test_deduplication_skips_already_ingested_file`
**Type:** Integration (DB + mock embedder)
**What it tests:** Second ingest of the same file returns `skipped=True` and does not call the embedder.
**Why it matters:** Without deduplication, re-running ingestion over a folder doubles all chunks and costs double the embedding API fees. The embedder call count assertion confirms the expensive step is bypassed.

#### `test_deduplication_reingest_after_content_change`
**Type:** Integration (DB + mock embedder)
**What it tests:** A file with changed content (different hash) is re-ingested even at the same path.
**Why it matters:** Files get updated. "Q3_report.pdf" at path X today is different from "Q3_report.pdf" at path X next month. The system must detect this and re-process.

#### `test_pipeline_stores_chunks_in_db`
**Type:** Integration (DB + mock embedder)
**What it tests:** After ingestion, the DB contains exactly `result.chunk_count` chunks for the document, all with non-NULL embeddings.
**Why it matters:** End-to-end correctness test. Verifies the full pipeline — parse, chunk, embed, store — produces the right DB state. The NULL embedding check ensures no chunk is stored without a vector (which would make it invisible to similarity search).

---

## Lessons Learned

### Foreign key order matters — insert parents before children

**Problem:**
```
ForeignKeyViolationError: insert or update on table "chunks" violates
foreign key constraint "chunks_document_id_fkey"
DETAIL: Key (document_id)=(0) is not present in table "documents"
```

**Root cause:** The original pipeline inserted chunks with `document_id=0` as a placeholder, intending to update later. But the FK constraint on `document_id` is enforced immediately on INSERT — `0` doesn't exist in `documents`, so the INSERT fails.

**Fix:** Insert the `documents` row first to get the real ID, then insert all `chunks` rows with that ID.

**Lesson:** When tables have FK relationships, always plan the insertion order. Parent rows must exist before child rows. This is not just good practice — it's enforced by the database.

---

### Test isolation — unique keys for cache tests

**Problem:**
```
AssertionError: API should be called on first embed
assert 0 == 1
```

**Root cause:** The embedding cache test used a fixed model name (`"test-model-cache"`). On the second test run, the embedding for "operating cash flow" was already in the `embeddings_cache` table from the previous run. The cache correctly returned the stored value — `call_count` stayed at 0.

**Fix:** Use a unique model name per test execution with `uuid.uuid4().hex[:8]` suffix. This creates a fresh cache namespace for every test run.

**Lesson:** Any test that writes to a shared database must ensure its data is isolated from previous runs. Options:
1. Unique keys (UUID suffix) — simple, no cleanup needed
2. Test transactions rolled back after each test — cleaner but requires transaction fixtures
3. Separate test database — most isolated, most setup overhead

For this project we use option 1 for its simplicity.

---

### Pipe-delimited Excel over CSV format

**Problem:** Financial numbers contain commas (`1,234,567`). Converting Excel rows to CSV creates ambiguous output that breaks parsers and confuses LLMs trying to read the structure.

**Fix:** Use `|` as the column separator. Pipes rarely appear in financial data and make the tabular structure visually clear for the LLM.

**Lesson:** When serializing structured data to text for LLM consumption, choose delimiters that don't appear in the data. For financial data: prefer `|` over `,`, and always test with real data samples from your domain.

---

### tiktoken downloads on first use

**Observation:** The first test run after installing tiktoken takes longer than subsequent runs. tiktoken downloads the `cl100k_base` encoding file (~1MB) on first use and caches it locally.

**In production:** Pre-warm tiktoken during container startup by calling `tiktoken.get_encoding("cl100k_base")` in the application's startup code. This prevents a slow first request.

---

### Generated columns for full-text search

The `content_tsv` column is a PostgreSQL **generated column**:
```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

Key properties:
- Computed automatically by Postgres on every INSERT/UPDATE — no application code needed
- `STORED` means it's physically saved (not recomputed on every read)
- You cannot INSERT or UPDATE it manually — Postgres rejects any attempt
- The GIN index on it makes full-text search fast at the cost of extra storage

This design means the full-text search capability in Phase 4 is already in place — no additional ingestion changes needed.
