from src.retrieval.models import RetrievalFilter


def build_filter_clause(
    filters: RetrievalFilter | None,
    param_offset: int = 1,
) -> tuple[str, list]:
    """
    Converts a RetrievalFilter into a parameterised SQL WHERE fragment.

    Returns:
        (clause, params) where clause is a string like "AND doc_type = $2 AND ..."
        and params is the list of values for each placeholder.

    param_offset: the $N number to start from (e.g. if $1 is already used
                  for the query vector, start filter params at $2).

    Security: all values are passed as parameters — never interpolated into the
    SQL string. This prevents SQL injection regardless of filter content.

    Example:
        clause, params = build_filter_clause(
            RetrievalFilter(doc_type="pdf", source_file="report.pdf"),
            param_offset=2
        )
        # clause → "AND doc_type = $2 AND source_file = $3"
        # params → ["pdf", "report.pdf"]
    """
    if filters is None:
        return "", []

    conditions: list[str] = []
    params: list = []
    i = param_offset

    if filters.doc_type is not None:
        conditions.append(f"doc_type = ${i}")
        params.append(filters.doc_type)
        i += 1

    if filters.source_file is not None:
        conditions.append(f"source_file = ${i}")
        params.append(filters.source_file)
        i += 1

    if filters.section_title is not None:
        conditions.append(f"section_title = ${i}")
        params.append(filters.section_title)
        i += 1

    if filters.chunk_strategy is not None:
        conditions.append(f"chunk_strategy = ${i}")
        params.append(filters.chunk_strategy)
        i += 1

    if filters.document_id is not None:
        conditions.append(f"document_id = ${i}")
        params.append(filters.document_id)
        i += 1

    if not conditions:
        return "", []

    return "AND " + " AND ".join(conditions), params
