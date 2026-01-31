# Pipeline Orchestration
"""End-to-end pipeline orchestration for GDELT ingestion and processing."""

from .orchestrator import GDELTPipeline, PipelineConfig

__all__ = ["GDELTPipeline", "PipelineConfig"]
