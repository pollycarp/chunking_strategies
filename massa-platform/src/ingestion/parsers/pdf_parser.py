from pathlib import Path

import pdfplumber

from src.ingestion.models import ParsedDocument, ParsedPage
from src.ingestion.parsers.base import DocumentParser


class PDFParser(DocumentParser):
    """
    Extracts text from PDF files using pdfplumber.

    Why pdfplumber over PyPDF2 or pymupdf?
    - pdfplumber preserves layout better (important for tables and financial data)
    - Handles multi-column layouts without merging columns
    - Provides bounding box data for tables (useful in Phase 5)
    - Good at scanned PDF text extraction when combined with OCR

    Limitations:
    - Scanned PDFs without text layers return empty pages
    - Complex tables may still merge columns — use camelot for serious table extraction
    """

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        file_hash = self.compute_file_hash(file_path)
        pages: list[ParsedPage] = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = text.strip()

                # Skip completely empty pages (common in financial reports
                # that have full-page charts with no text layer)
                if not text:
                    continue

                pages.append(ParsedPage(
                    content=text,
                    page_number=i + 1,   # 1-based for human-readable citations
                ))

        if not pages:
            raise ValueError(f"No extractable text found in {file_path.name}. "
                             "The PDF may be scanned — OCR required.")

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path),
            doc_type="pdf",
            pages=pages,
            file_hash=file_hash,
        )
