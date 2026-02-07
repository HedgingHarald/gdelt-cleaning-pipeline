"""Embedding pipeline for GDELT GKG records."""

from .embedder import EmbeddingResult, GKGEmbedder
from .batch import BatchEmbedder

__all__ = ["EmbeddingResult", "GKGEmbedder", "BatchEmbedder"]
