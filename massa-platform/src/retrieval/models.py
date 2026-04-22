from dataclasses import dataclass, field


@dataclass
class RetrievalFilter:
    """
    Pre-search filters applied as SQL WHERE clauses before vector/keyword search.

    Filters run before the index is queried — they reduce the candidate set,
    improve precision, and in multi-tenant systems act as a security boundary
    (a user should never see another client's documents).

    All fields are optional — only set fields are added to the WHERE clause.
    """
    doc_type: str | None = None          # 'pdf' | 'docx' | 'xlsx'
    source_file: str | None = None       # exact filename match
    section_title: str | None = None     # exact section title match
    chunk_strategy: str | None = None    # 'fixed' | 'semantic' | 'hierarchical_child'
    document_id: int | None = None       # restrict to one document


@dataclass
class RetrievedChunk:
    """
    A chunk returned by the retrieval system, enriched with a relevance score and rank.

    score : relevance score (higher = more relevant)
            - Semantic:  cosine similarity (0–1)
            - Keyword:   ts_rank (0–1)
            - Hybrid:    RRF score (~0.01–0.03)
            - Reranked:  Cohere relevance score (0–1)

    rank  : position in the result list (1 = most relevant)
    parent_id : DB id of parent chunk — used in small-to-big retrieval
                (match child, return parent for full context)
    """
    chunk_id: int
    content: str
    source_file: str
    page_number: int | None
    section_title: str | None
    doc_type: str
    chunk_strategy: str
    score: float
    rank: int
    parent_id: int | None = None
    metadata: dict = field(default_factory=dict)
