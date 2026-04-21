import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.models import ParsedDocument


class DocumentParser(ABC):
    """
    Abstract interface for all document parsers.

    Each parser handles one file format (PDF, DOCX, XLSX).
    All return the same ParsedDocument structure — the rest of the
    pipeline doesn't care about the source format.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a file and return a structured document.
        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if the file cannot be parsed.
        """
        ...

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """
        SHA256 hash of the raw file bytes.

        Why SHA256?
        - Deterministic: same file always produces same hash
        - Collision-resistant: two different files almost certainly produce different hashes
        - Fast enough: a 10MB PDF hashes in ~10ms

        Used for deduplication: if the hash is already in the documents table,
        the file has not changed and we skip re-ingestion.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
