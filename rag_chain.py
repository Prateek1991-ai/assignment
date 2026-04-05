"""
RAG chain: embed question → retrieve chunks → generate grounded answer.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel
from enum import Enum

from parser import ChunkType, DocumentChunk
from llm import LLMClient
from vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert financial research analyst assistant.
Answer questions using ONLY the context provided below.

Rules:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If context is insufficient, say "The provided documents do not contain enough information to answer this question."
3. Be precise with numbers, percentages, and dates.
4. Cite which document and page your answer is drawn from.
5. For table data, present it clearly. For image/figure data, describe what the figure shows.

Format: Clear answer (2-5 paragraphs), then a "Sources:" section.
"""


class SourceReference(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    chunk_type: ChunkType
    content_preview: str


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    question: str
    answer: str
    sources: list[SourceReference]
    chunks_retrieved: int
    model_used: str


class RAGChain:
    def __init__(self, llm_client: LLMClient, vector_store: VectorStore) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store

    def query(self, request: QueryRequest) -> QueryResponse:
        query_embedding = self.llm_client.embed_single(request.question)
        chunks = self.vector_store.search(query_embedding, top_k=request.top_k)

        if not chunks:
            return QueryResponse(
                question=request.question,
                answer="No documents indexed yet. Please ingest a PDF via POST /ingest.",
                sources=[], chunks_retrieved=0, model_used=self.llm_client.llm_model,
            )

        context = self._build_context(chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
        answer = self.llm_client.chat(SYSTEM_PROMPT, user_message)

        sources = [
            SourceReference(
                chunk_id=c.chunk_id, source_file=c.source_file,
                page_number=c.page_number, chunk_type=c.chunk_type,
                content_preview=c.content[:200],
            )
            for c in chunks
        ]
        return QueryResponse(
            question=request.question, answer=answer,
            sources=sources, chunks_retrieved=len(chunks),
            model_used=self.llm_client.llm_model,
        )

    @staticmethod
    def _build_context(chunks: list[DocumentChunk]) -> str:
        labels = {ChunkType.TEXT: "Text", ChunkType.TABLE: "Table", ChunkType.IMAGE: "Figure"}
        blocks = []
        for i, chunk in enumerate(chunks, 1):
            label = labels.get(chunk.chunk_type, "Content")
            blocks.append(f"[Source {i}] {label} from '{chunk.source_file}' (page {chunk.page_number})\n{chunk.content}")
        return "\n\n---\n\n".join(blocks)
