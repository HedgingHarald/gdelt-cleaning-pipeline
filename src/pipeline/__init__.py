"""End-to-end pipeline orchestration for GDELT ingestion, embedding, and search."""

from .run import main as run_pipeline

__all__ = ["run_pipeline"]
