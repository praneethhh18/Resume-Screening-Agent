"""Embedding utilities for resume ranking."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from .config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model(device: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""

    settings = get_settings()
    model_kwargs = {"device": device} if device else {}
    return HuggingFaceEmbeddings(
        model_name=settings.embeddings_model,
        model_kwargs=model_kwargs,
    )
