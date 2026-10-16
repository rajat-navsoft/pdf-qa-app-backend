from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOP_K,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided document "
    "excerpts. If the answer is not in the excerpts, say you could not find it in the "
    "documents. Be concise."
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


@lru_cache
def get_vectorstore() -> Chroma:
    # Loading the embedding model is slow, so it happens once on first use.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


@lru_cache
def get_llm() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in backend/.env")
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def index_document(doc_id: str, doc_name: str, pages: list[str]) -> int:
    chunks: list[Document] = []
    for page_num, text in enumerate(pages, start=1):
        for chunk in splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={"document_id": doc_id, "document_name": doc_name, "page": page_num},
                )
            )
    if chunks:
        ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
        get_vectorstore().add_documents(chunks, ids=ids)
    return len(chunks)


def delete_document(doc_id: str) -> None:
    store = get_vectorstore()
    ids = store.get(where={"document_id": doc_id})["ids"]
    if ids:
        store.delete(ids=ids)


def retrieve(query: str, doc_id: str | None = None) -> list[Document]:
    filter_ = {"document_id": doc_id} if doc_id else None
    results = get_vectorstore().similarity_search_with_score(query, k=TOP_K, filter=filter_)
    return [doc for doc, _ in results]


def generate_answer(question: str, context: list[Document], history: list[dict] | None = None) -> str:
    excerpts = "\n\n".join(
        f"[{d.metadata['document_name']}, page {d.metadata['page']}]\n{d.page_content}"
        for d in context
    ) or "(no relevant excerpts found)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": f"Document excerpts:\n{excerpts}\n\nQuestion: {question}"})

    response = get_llm().chat.completions.create(model=LLM_MODEL, messages=messages)
    return response.choices[0].message.content or ""
