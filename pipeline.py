"""
Ingestion pipeline — parse → caption → embed → index.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from pydantic import BaseModel

from parser import PDFParser, ChunkType, DocumentChunk
from llm import LLMClient
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionSummary(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    processing_time_seconds: float


class IngestionPipeline:
    def __init__(self, llm_client: LLMClient, vector_store: VectorStore) -> None:
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.parser = PDFParser()

    def run(self, pdf_path: Path) -> IngestionSummary:
        start = time.perf_counter()
        raw_chunks = self.parser.parse(pdf_path)
        captioned = self._caption_images(raw_chunks)
        embedded = self._embed_chunks(captioned)
        self.vector_store.add_chunks(embedded)
        elapsed = time.perf_counter() - start

        return IngestionSummary(
            filename=pdf_path.name,
            text_chunks=sum(1 for c in captioned if c.chunk_type == ChunkType.TEXT),
            table_chunks=sum(1 for c in captioned if c.chunk_type == ChunkType.TABLE),
            image_chunks=sum(1 for c in captioned if c.chunk_type == ChunkType.IMAGE),
            total_chunks=len(captioned),
            processing_time_seconds=round(elapsed, 2),
        )

    def _caption_images(self, chunks):
        result = []
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.IMAGE:
                result.append(chunk)
                continue
            img_path = Path(chunk.metadata.get("image_path", ""))
            if not img_path.exists():
                continue
            try:
                caption = self.llm_client.caption_image(img_path)
                chunk = chunk.model_copy(update={"content": f"[Figure on page {chunk.page_number}] {caption}"})
            except Exception as exc:
                logger.error("Caption failed for %s: %s", img_path.name, exc)
                chunk = chunk.model_copy(update={"content": f"[Figure on page {chunk.page_number}] Could not caption: {img_path.name}"})
            result.append(chunk)
        return result

    def _embed_chunks(self, chunks):
        texts = [c.content for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 100):
            all_embeddings.extend(self.llm_client.embed(texts[i:i+100]))
        return list(zip(chunks, all_embeddings))
