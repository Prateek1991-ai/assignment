"""
FastAPI route definitions — all endpoints in one flat file.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from config import settings
from pipeline import IngestionPipeline, IngestionSummary
from rag_chain import RAGChain, QueryRequest, QueryResponse
from vector_store import VectorStore
from llm import LLMClient

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Pydantic response models ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    indexed_documents: int
    total_chunks: int
    uptime_seconds: float


class DocumentListResponse(BaseModel):
    documents: list[str]
    total_documents: int


class DeleteResponse(BaseModel):
    message: str
    filename: str
    chunks_removed: int


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store

def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client

def get_start_time(request: Request) -> float:
    return request.app.state.start_time


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(
    vector_store: VectorStore = Depends(get_vector_store),
    llm_client: LLMClient = Depends(get_llm_client),
    start_time: float = Depends(get_start_time),
):
    """Returns model readiness, indexed document count, chunk count, and uptime."""
    return HealthResponse(
        status="ok",
        model=llm_client.llm_model,
        embedding_model=llm_client.embedding_model,
        indexed_documents=len(vector_store.indexed_documents),
        total_chunks=vector_store.total_chunks,
        uptime_seconds=round(time.time() - start_time, 1),
    )


# ── POST /ingest ──────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestionSummary, status_code=201, tags=["Ingestion"])
async def ingest(
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Upload a PDF. Extracts text, tables, and images; captions images with GPT-4o; embeds and indexes everything."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest_path = data_dir / file.filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pipeline = IngestionPipeline(llm_client=llm_client, vector_store=vector_store)
        return pipeline.run(dest_path)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


# ── POST /query ───────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["Retrieval"])
def query(
    request: QueryRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    llm_client: LLMClient = Depends(get_llm_client),
):
    """Ask a natural language question. Retrieves relevant chunks and generates a grounded GPT-4o answer."""
    if vector_store.total_chunks == 0:
        raise HTTPException(status_code=404, detail="No documents indexed. POST a PDF to /ingest first.")
    try:
        chain = RAGChain(llm_client=llm_client, vector_store=vector_store)
        return chain.query(request)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse, tags=["Management"])
def list_documents(vector_store: VectorStore = Depends(get_vector_store)):
    """List all indexed document filenames."""
    docs = vector_store.indexed_documents
    return DocumentListResponse(documents=docs, total_documents=len(docs))


# ── DELETE /documents/{filename} ──────────────────────────────────────────────

@router.delete("/documents/{filename}", response_model=DeleteResponse, tags=["Management"])
def delete_document(filename: str, vector_store: VectorStore = Depends(get_vector_store)):
    """Remove all chunks for the given PDF from the vector index."""
    if filename not in vector_store.indexed_documents:
        raise HTTPException(status_code=404, detail=f"'{filename}' not found in index.")
    removed = vector_store.delete_document(filename)
    return DeleteResponse(message=f"Removed {filename} from index.", filename=filename, chunks_removed=removed)
