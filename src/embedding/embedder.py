"""
OpenAI embedding wrapper for GKG records.

Environment:
- OPENAI_API_KEY must be set

Usage:
    embedder = GKGEmbedder()
    vectors = embedder.embed_batch(records)
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding a GKG record."""
    gkg_record_id: str
    embedding: np.ndarray  # shape (1536,)
    text_hash: str         # MD5 of input text for dedup
    token_count: int


class GKGEmbedder:
    """Embeds GKG records using OpenAI text-embedding-3-small."""
    
    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    MAX_BATCH_SIZE = 500
    MAX_RETRIES = 5
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def _assemble_text(self, record) -> str:
        """
        Assemble embedding text from GKG record fields.
        
        Priority fields (in order):
        1. themes (semicolon-separated → space-separated)
        2. persons (top 5)
        3. organizations (top 5)
        4. locations (names only)
        5. tone summary (if available)
        
        Target: ~200-500 tokens per record for cost efficiency.
        """
        parts = []
        
        # Themes are the most semantic-rich
        if hasattr(record, 'themes') and record.themes:
            themes = record.themes[:20]  # Limit to top 20
            parts.append(f"Themes: {', '.join(themes)}")
        
        # Key entities - handle both string lists and object lists
        if hasattr(record, 'persons') and record.persons:
            persons = record.persons[:5]
            # Check if persons are strings or objects with .name
            if persons and hasattr(persons[0], 'name'):
                person_names = [p.name for p in persons]
            else:
                person_names = [str(p) for p in persons]
            parts.append(f"People: {', '.join(person_names)}")
        
        if hasattr(record, 'organizations') and record.organizations:
            orgs = record.organizations[:5]
            # Check if organizations are strings or objects with .name
            if orgs and hasattr(orgs[0], 'name'):
                org_names = [o.name for o in orgs]
            else:
                org_names = [str(o) for o in orgs]
            parts.append(f"Organizations: {', '.join(org_names)}")
        
        if hasattr(record, 'locations') and record.locations:
            locs = record.locations[:5]
            # Check if locations are objects with .full_name or strings
            if locs and hasattr(locs[0], 'full_name'):
                loc_names = [loc.full_name for loc in locs if loc.full_name]
            else:
                loc_names = [str(loc) for loc in locs]
            parts.append(f"Locations: {', '.join(loc_names)}")
        
        # Tone context
        if hasattr(record, 'tone') and record.tone:
            if hasattr(record.tone, 'tone'):
                tone_value = record.tone.tone
                if tone_value > 2:
                    tone_label = "positive"
                elif tone_value < -2:
                    tone_label = "negative"
                else:
                    tone_label = "neutral"
                parts.append(f"Tone: {tone_label}")
        
        # Source context
        if hasattr(record, 'source_common_name') and record.source_common_name:
            parts.append(f"Source: {record.source_common_name}")
        
        return " | ".join(parts) if parts else "Unknown event"
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a list of texts directly.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays with embeddings
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch_texts = texts[i:i + self.MAX_BATCH_SIZE]
            embeddings = self._embed_with_retry(batch_texts)
            all_embeddings.extend([np.array(emb, dtype=np.float32) for emb in embeddings])
        
        return all_embeddings
    
    def embed_batch(
        self,
        records: List,
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Embed a batch of GKG records.
        
        Handles:
        - Batching (500 records at a time)
        - Rate limiting with exponential backoff
        - Empty/invalid records (skipped)
        """
        results = []
        texts = []
        record_ids = []
        
        # Prepare texts
        for record in records:
            text = self._assemble_text(record)
            if text and len(text) > 10:  # Skip trivial texts
                texts.append(text)
                record_ids.append(record.gkg_record_id)
        
        if not texts:
            return results
        
        # Process in batches
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch_texts = texts[i:i + self.MAX_BATCH_SIZE]
            batch_ids = record_ids[i:i + self.MAX_BATCH_SIZE]
            
            embeddings = self._embed_with_retry(batch_texts)
            
            for j, emb in enumerate(embeddings):
                text_hash = hashlib.md5(batch_texts[j].encode()).hexdigest()
                results.append(EmbeddingResult(
                    gkg_record_id=batch_ids[j],
                    embedding=np.array(emb, dtype=np.float32),
                    text_hash=text_hash,
                    token_count=len(batch_texts[j].split())  # Approximate
                ))
            
            if show_progress:
                logger.info(f"Embedded {min(i + self.MAX_BATCH_SIZE, len(texts))}/{len(texts)} records")
        
        return results
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API with exponential backoff."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    model=self.MODEL,
                    input=texts,
                    dimensions=self.DIMENSIONS
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"Embedding failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        raise RuntimeError(f"Failed to embed after {self.MAX_RETRIES} attempts")
