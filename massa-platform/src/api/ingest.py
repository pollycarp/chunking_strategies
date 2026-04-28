"""
POST /ingest — accepts a document file upload and runs the ingestion pipeline.

The endpoint:
1. Reads the uploaded file bytes from the multipart form
2. Saves to a temporary file (the pipeline needs a real file path)
3. Runs IngestionPipeline.ingest() — parse → chunk → embed → store
4. Cleans up the temp file
5. Returns a JSON summary of the result
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse

from src.ingestion.pipeline import IngestionPipeline, _PARSERS as _SUPPORTED_PARSERS


async def ingest_endpoint(request: Request) -> JSONResponse:
    """
    Handles POST /ingest (multipart/form-data).

    Expected form field: "file" — the document to ingest (PDF, DOCX, or XLSX).

    Returns:
        200 {"filename": ..., "skipped": bool, "chunk_count": int, "page_count": int}
        400 {"error": ...}  on validation failure
        500 {"error": ...}  on pipeline failure
    """
    try:
        form = await request.form()
    except Exception:
        return JSONResponse({"error": "Expected multipart/form-data with a 'file' field"}, status_code=400)

    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "'file' field is required"}, status_code=400)

    filename: str = getattr(upload, "filename", None) or "upload"
    suffix = Path(filename).suffix.lower()

    supported = set(_SUPPORTED_PARSERS.keys())
    if suffix not in supported:
        return JSONResponse(
            {"error": f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(supported))}"},
            status_code=400,
        )

    contents: bytes = await upload.read()
    if not contents:
        return JSONResponse({"error": "Uploaded file is empty"}, status_code=400)

    # Save to a named temp file — the pipeline opens the file by path
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="massa_upload_"
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # Rename so the pipeline sees the real filename (used for source_file metadata)
        named_path = tmp_path.parent / filename
        tmp_path.rename(named_path)
        tmp_path = named_path

        pipeline: IngestionPipeline = request.app.state.pipeline
        result = await pipeline.ingest(tmp_path)

        return JSONResponse({
            "filename": result.filename,
            "skipped": result.skipped,
            "chunk_count": result.chunk_count,
            "page_count": result.page_count,
            "document_id": result.document_id,
            "message": (
                f"Already ingested — skipped." if result.skipped
                else f"Ingested {result.chunk_count} chunks from {result.page_count} pages."
            ),
        })

    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Ingestion failed: {exc}"}, status_code=500)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
