from pathlib import Path

from pypdf import PdfReader

from app.config import UPLOAD_DIR


def save_upload(doc_id: str, content: bytes) -> Path:
    path = UPLOAD_DIR / f"{doc_id}.pdf"
    path.write_bytes(content)
    return path


def extract_pages(path: Path) -> list[str]:
    """Return the text of each page; index 0 is page 1."""
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def delete_file(path: str) -> None:
    Path(path).unlink(missing_ok=True)
