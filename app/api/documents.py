import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import MAX_FILE_SIZE_MB
from app.database.sqlite import DocumentRecord, get_db
from app.services import pdf_service, rag_service

router = APIRouter(prefix="/documents", tags=["documents"])


# Field names are camelCase to match the Angular UploadedDocument interface.
class DocumentOut(BaseModel):
    id: str
    name: str
    size: int
    pages: int
    uploadedAt: datetime


class Source(BaseModel):
    page: int
    snippet: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []


def to_out(doc: DocumentRecord) -> DocumentOut:
    return DocumentOut(id=doc.id, name=doc.name, size=doc.size, pages=doc.pages, uploadedAt=doc.uploaded_at.replace(tzinfo=timezone.utc))


@router.post("", response_model=DocumentOut)
def upload_document(file: UploadFile, db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File is too large. Maximum size is {MAX_FILE_SIZE_MB} MB.")

    doc_id = str(uuid.uuid4())
    path = pdf_service.save_upload(doc_id, content)
    try:
        pages = pdf_service.extract_pages(path)
        chunk_count = rag_service.index_document(doc_id, file.filename, pages)
    except Exception as exc:
        pdf_service.delete_file(str(path))
        raise HTTPException(400, f"Could not read PDF: {exc}")
    if chunk_count == 0:
        pdf_service.delete_file(str(path))
        raise HTTPException(400, "No text found in this PDF (scanned PDFs are not supported).")

    doc = DocumentRecord(id=doc_id, name=file.filename, size=len(content), pages=len(pages), file_path=str(path))
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return to_out(doc)


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(DocumentRecord).order_by(DocumentRecord.uploaded_at.desc())).all()
    return [to_out(d) for d in docs]


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.get(DocumentRecord, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found.")
    rag_service.delete_document(doc_id)
    pdf_service.delete_file(doc.file_path)
    db.delete(doc)
    db.commit()


@router.post("/{doc_id}/ask", response_model=AskResponse)
def ask_document(doc_id: str, body: AskRequest, db: Session = Depends(get_db)):
    if not db.get(DocumentRecord, doc_id):
        raise HTTPException(404, "Document not found.")
    context = rag_service.retrieve(body.question, doc_id=doc_id)
    try:
        answer = rag_service.generate_answer(body.question, context)
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}")
    sources = [Source(page=d.metadata["page"], snippet=d.page_content[:200]) for d in context]
    return AskResponse(answer=answer, sources=sources)
