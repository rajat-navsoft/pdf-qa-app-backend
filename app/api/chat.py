from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import rag_service

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_HISTORY_TURNS = 10


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


class ChatSource(BaseModel):
    documentName: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[ChatSource] = []


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest):
    # Searches across all uploaded documents.
    context = rag_service.retrieve(body.message)
    history = [{"role": t.role, "content": t.text} for t in body.history[-MAX_HISTORY_TURNS:]]
    try:
        reply = rag_service.generate_answer(body.message, context, history)
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}")
    sources = [
        ChatSource(
            documentName=d.metadata["document_name"],
            page=d.metadata["page"],
            snippet=d.page_content[:200],
        )
        for d in context
    ]
    return ChatResponse(reply=reply, sources=sources)
