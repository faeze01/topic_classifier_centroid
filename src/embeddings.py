"""Embeddings via OpenAI-compatible HTTP API (URL and key from .env)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from openai import OpenAI

from src.config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not EMBEDDING_BASE_URL or not EMBEDDING_API_KEY or not EMBEDDING_MODEL:
        raise RuntimeError(
            "Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL in .env."
        )
    return OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY, timeout=30.0)


def get_embedding(text: str) -> np.ndarray:
    """Return a sentence embedding vector for the given text."""
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Return embedding vectors for a list of texts, ordered like `texts`."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    return np.array([item.embedding for item in ordered], dtype=np.float32)
