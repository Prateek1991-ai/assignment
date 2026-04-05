"""
LLM & Embedding client backed by OpenAI.
- Chat completions  → GPT-4o
- Vision captioning → GPT-4o with base64 image
- Embeddings        → text-embedding-3-small
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.llm_model = settings.openai_llm_model
        self.embedding_model = settings.openai_embedding_model
        logger.info("LLMClient ready — LLM: %s | Embed: %s", self.llm_model, self.embedding_model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def caption_image(self, image_path: Path) -> str:
        image_bytes = image_path.read_bytes()
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        suffix = image_path.suffix.lower()
        media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")

        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": (
                        "You are analysing a figure from a financial research PDF. "
                        "Describe: (1) type of visual, (2) key data/trends, (3) labels/axes/values visible, "
                        "(4) main insight an analyst would draw. Be specific and quantitative. "
                        "This description will be used for semantic search retrieval."
                    )},
                ],
            }],
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]
