"""
PDF parser — extracts text, tables (→ Markdown), and images from a PDF.
Uses PyMuPDF for text/images, pdfplumber for tables.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from enum import Enum
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)
MIN_IMAGE_AREA = 4000


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentChunk(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    chunk_type: ChunkType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PDFParser:
    def __init__(self, image_cache_dir: Path | None = None) -> None:
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.image_cache_dir = image_cache_dir or Path(".cache/images")
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, pdf_path: Path) -> list[DocumentChunk]:
        filename = pdf_path.name
        chunks: list[DocumentChunk] = []

        table_chunks = self._extract_tables(pdf_path, filename)
        chunks.extend(table_chunks)
        table_pages = {tc.page_number for tc in table_chunks}

        chunks.extend(self._extract_text(pdf_path, filename, skip_pages=table_pages))
        chunks.extend(self._extract_images(pdf_path, filename))

        logger.info("Parsed %s → %d chunks", filename, len(chunks))
        return chunks

    def _extract_text(self, pdf_path, filename, skip_pages):
        chunks = []
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            if page_num in skip_pages:
                continue
            text = self._clean_text(page.get_text("text"))
            if not text.strip():
                continue
            for chunk_text in self._split_text(text):
                if len(chunk_text.strip()) < 30:
                    continue
                chunks.append(DocumentChunk(
                    chunk_id=self._make_id(filename, page_num, chunk_text),
                    source_file=filename, page_number=page_num,
                    chunk_type=ChunkType.TEXT, content=chunk_text.strip(),
                ))
        doc.close()
        return chunks

    def _extract_tables(self, pdf_path, filename):
        chunks = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table in page.extract_tables() or []:
                    md = self._table_to_markdown(table)
                    if not md.strip():
                        continue
                    chunks.append(DocumentChunk(
                        chunk_id=self._make_id(filename, page_num, md),
                        source_file=filename, page_number=page_num,
                        chunk_type=ChunkType.TABLE, content=md,
                        metadata={"raw_rows": len(table)},
                    ))
        return chunks

    def _extract_images(self, pdf_path, filename):
        chunks = []
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, start=1):
            for img_index, img_ref in enumerate(page.get_images(full=True)):
                xref = img_ref[0]
                base_image = doc.extract_image(xref)
                w, h = base_image.get("width", 0), base_image.get("height", 0)
                if w * h < MIN_IMAGE_AREA:
                    continue
                ext = base_image["ext"]
                img_filename = f"{pdf_path.stem}_p{page_num}_i{img_index}.{ext}"
                img_path = self.image_cache_dir / img_filename
                img_path.write_bytes(base_image["image"])
                chunks.append(DocumentChunk(
                    chunk_id=self._make_id(filename, page_num, img_filename),
                    source_file=filename, page_number=page_num,
                    chunk_type=ChunkType.IMAGE,
                    content=f"[IMAGE PENDING CAPTION] {img_filename}",
                    metadata={"image_path": str(img_path), "width": w, "height": h, "ext": ext},
                ))
        doc.close()
        return chunks

    @staticmethod
    def _clean_text(text):
        text = re.sub(r"\x00", "", text)
        text = re.sub(r" {3,}", "  ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text):
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def _table_to_markdown(table):
        if not table:
            return ""
        def clean(c): return "" if c is None else str(c).replace("\n", " ").strip()
        rows = [[clean(c) for c in row] for row in table]
        header, body = rows[0], rows[1:]
        sep = ["---"] * len(header)
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
        for row in body:
            row = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    @staticmethod
    def _make_id(filename, page_num, content):
        digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{filename}_p{page_num}_{digest}"
