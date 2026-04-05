"""
FAISS vector store with metadata sidecar.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import settings
from parser import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self.dim = settings.embedding_dim
        self.index_path = Path(settings.faiss_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._metadata: dict[int, dict] = {}
        self._next_id: int = 0
        self._load_or_create()

    def add_chunks(self, chunk_embeddings: list[tuple[DocumentChunk, list[float]]]) -> None:
        if not chunk_embeddings:
            return
        vectors = np.array([emb for _, emb in chunk_embeddings], dtype=np.float32)
        faiss.normalize_L2(vectors)
        ids = np.arange(self._next_id, self._next_id + len(chunk_embeddings), dtype=np.int64)
        self._index.add_with_ids(vectors, ids)
        for faiss_id, (chunk, _) in zip(ids, chunk_embeddings):
            self._metadata[int(faiss_id)] = chunk.model_dump()
        self._next_id += len(chunk_embeddings)
        self._persist()

    def delete_document(self, filename: str) -> int:
        remove_ids = [fid for fid, m in self._metadata.items() if m["source_file"] == filename]
        for fid in remove_ids:
            del self._metadata[fid]
        self._rebuild_index()
        self._persist()
        return len(remove_ids)

    def search(self, query_embedding: list[float], top_k: int | None = None) -> list[DocumentChunk]:
        k = top_k or settings.top_k
        if self._index.ntotal == 0:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        actual_k = min(k, self._index.ntotal)
        _, faiss_ids = self._index.search(vec, actual_k)
        results = []
        for fid in faiss_ids[0]:
            if fid == -1:
                continue
            meta = self._metadata.get(int(fid))
            if meta:
                results.append(DocumentChunk(**meta))
        return results

    @property
    def total_chunks(self) -> int:
        return self._index.ntotal

    @property
    def indexed_documents(self) -> list[str]:
        return sorted({m["source_file"] for m in self._metadata.values()})

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump((self._metadata, self._next_id), f)

    def _load_or_create(self) -> None:
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.pkl"
        if index_file.exists() and meta_file.exists():
            try:
                self._index = faiss.read_index(str(index_file))
                with open(meta_file, "rb") as f:
                    self._metadata, self._next_id = pickle.load(f)
                logger.info("Loaded index: %d chunks", self._index.ntotal)
                return
            except Exception as exc:
                logger.warning("Could not load index (%s) — creating fresh", exc)
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self._metadata = {}
        self._next_id = 0
        logger.info("Created new FAISS index (dim=%d)", self.dim)

    def _rebuild_index(self) -> None:
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        logger.info("Index rebuilt after deletion")
