# Cursor Prompt: GDELT Embeddings Pipeline

> **Model:** Claude Opus 4.5 + Thinking  
> **Repo:** `gdelt-cleaning-pipeline` (Hetzner server)  
> **Goal:** Complete embedding pipeline: download → parse → embed → index → search

---

## Context

You're extending an existing GDELT ingestion pipeline with semantic embeddings. The repo already has:

**Existing Code (DO NOT modify unless necessary):**
- `src/ingestion/worker.py` — Downloads GKG files from GDELT masterfilelist
- `src/ingestion/manifest.py` — SQLite-based tracking of file status (pending/completed/failed)
- `src/parsing/gkg_parser.py` — Parses GKG CSV files with memory-safe streaming

**Your Task:** Build the embedding, vector storage, and search layers.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Hetzner Server (ubuntu-8gb-nbg1-1)                  │
│                                                      │
│  ┌────────────────┐    ┌────────────────────┐       │
│  │ GDELT Ingestion│───▶│ GKG Parser         │       │
│  │ (existing)     │    │ (existing)         │       │
│  └────────────────┘    └──────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │ Embedding Pipeline│       │
│                          │ (NEW - Task 1)   │       │
│                          │ OpenAI text-      │       │
│                          │ embedding-3-small │       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │ LanceDB           │       │
│                          │ (NEW - Task 2)   │       │
│                          │ ~100GB SQ8        │       │
│                          └────────┬─────────┘       │
│                                   │                  │
│                          ┌────────▼─────────┐       │
│                          │ FastAPI Search    │       │
│                          │ (NEW - Task 3)   │       │
│                          │ Port 8080         │       │
│                          └────────┬─────────┘       │
└───────────────────────────────────┼─────────────────┘
                                    │ HTTPS
                         ┌──────────▼──────────┐
                         │ Vercel Frontend      │
                         │ (future integration) │
                         └─────────────────────┘
```

---

## Task 1: Embedding Pipeline

### File: `src/embedding/__init__.py`
```python
"""Embedding pipeline for GDELT GKG records."""
```

### File: `src/embedding/embedder.py`

Create an embedder that:
1. Takes a parsed GKG record and assembles text for embedding
2. Uses OpenAI `text-embedding-3-small` (1536 dimensions)
3. Supports batching (max 2048 texts per API call, but use 500 for safety)
4. Handles rate limiting with exponential backoff
5. Returns embeddings as numpy arrays

```python
"""
OpenAI embedding wrapper for GKG records.

Environment:
- OPENAI_API_KEY must be set

Usage:
    embedder = GKGEmbedder()
    vectors = embedder.embed_batch(records)
"""

import os
import time
import logging
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
        
        # Key entities
        if hasattr(record, 'persons') and record.persons:
            persons = [p.name for p in record.persons[:5]]
            parts.append(f"People: {', '.join(persons)}")
        
        if hasattr(record, 'organizations') and record.organizations:
            orgs = [o.name for o in record.organizations[:5]]
            parts.append(f"Organizations: {', '.join(orgs)}")
        
        if hasattr(record, 'locations') and record.locations:
            locs = [loc.name for loc in record.locations[:5]]
            parts.append(f"Locations: {', '.join(locs)}")
        
        # Tone context
        if hasattr(record, 'tone') and record.tone:
            parts.append(f"Tone: {record.tone.tone_label}")
        
        # Source context
        if hasattr(record, 'source_common_name') and record.source_common_name:
            parts.append(f"Source: {record.source_common_name}")
        
        return " | ".join(parts) if parts else "Unknown event"
    
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
                import hashlib
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
```

### File: `src/embedding/batch.py`

Create batch processor that:
1. Reads parsed GKG records from disk (or database)
2. Filters already-embedded records (by gkg_record_id)
3. Embeds in batches with progress tracking
4. Writes to LanceDB incrementally
5. Supports resumability (checkpoint after each batch)

```python
"""
Batch embedding processor with resumability.

Usage:
    processor = BatchEmbedder(
        manifest_db="data/manifest.db",
        lancedb_path="data/lancedb"
    )
    processor.run(date_from="20260101", date_to="20260207")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..ingestion.manifest import ManifestDB, FileStatus
from ..parsing.gkg_parser import GKGParser
from .embedder import GKGEmbedder
from ..vectorstore.lancedb_store import LanceDBStore

logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Batch processor for embedding GKG records."""
    
    BATCH_SIZE = 1000  # Records per LanceDB write
    
    def __init__(
        self,
        manifest_db: str = "data/manifest.db",
        lancedb_path: str = "data/lancedb",
        openai_api_key: Optional[str] = None
    ):
        self.manifest = ManifestDB(Path(manifest_db))
        self.store = LanceDBStore(lancedb_path)
        self.embedder = GKGEmbedder(api_key=openai_api_key)
        self.parser = GKGParser()
    
    def run(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: Optional[int] = None
    ) -> dict:
        """
        Run batch embedding for completed files.
        
        Args:
            date_from: Start date (YYYYMMDD)
            date_to: End date (YYYYMMDD)
            limit: Max files to process
        
        Returns:
            Stats dict with records_embedded, files_processed, errors
        """
        stats = {
            "files_processed": 0,
            "records_embedded": 0,
            "records_skipped": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat()
        }
        
        # Get completed files from manifest
        files = list(self.manifest.get_files_by_status(FileStatus.COMPLETED))
        
        # Filter by date range
        if date_from:
            files = [f for f in files if f.date_partition and f.date_partition >= date_from]
        if date_to:
            files = [f for f in files if f.date_partition and f.date_partition <= date_to]
        
        # Apply limit
        if limit:
            files = files[:limit]
        
        logger.info(f"Processing {len(files)} files for embedding")
        
        for file_record in files:
            try:
                self._process_file(file_record, stats)
                stats["files_processed"] += 1
            except Exception as e:
                logger.error(f"Error processing {file_record.file_url}: {e}")
                stats["errors"] += 1
        
        stats["finished_at"] = datetime.now().isoformat()
        return stats
    
    def _process_file(self, file_record, stats: dict):
        """Process a single GKG file."""
        # Parse the file
        file_path = self._get_parsed_path(file_record)
        if not file_path.exists():
            logger.warning(f"Parsed file not found: {file_path}")
            return
        
        records = list(self.parser.parse_file(file_path))
        logger.info(f"Parsed {len(records)} records from {file_record.date_partition}")
        
        # Filter already embedded
        existing_ids = set(self.store.get_existing_ids([r.gkg_record_id for r in records]))
        new_records = [r for r in records if r.gkg_record_id not in existing_ids]
        
        stats["records_skipped"] += len(records) - len(new_records)
        
        if not new_records:
            logger.info(f"All records already embedded for {file_record.date_partition}")
            return
        
        # Embed in batches
        for i in range(0, len(new_records), self.BATCH_SIZE):
            batch = new_records[i:i + self.BATCH_SIZE]
            results = self.embedder.embed_batch(batch)
            
            # Write to LanceDB
            self.store.add_records(results, [r for r in batch])
            stats["records_embedded"] += len(results)
            
            logger.info(f"Embedded batch {i // self.BATCH_SIZE + 1}: {len(results)} records")
    
    def _get_parsed_path(self, file_record) -> Path:
        """Get path to parsed GKG file."""
        # Adjust based on your storage structure
        date = file_record.date_partition
        return Path(f"data/parsed/{date}.gkg.jsonl")
```

---

## Task 2: LanceDB Vector Store

### File: `src/vectorstore/__init__.py`
```python
"""Vector storage layer using LanceDB."""
```

### File: `src/vectorstore/lancedb_store.py`

Create LanceDB integration with:
1. Schema: id, date, themes[], persons[], orgs[], locations[], tone, source, embedding
2. Scalar Quantization (SQ8) for 4x storage reduction
3. Filter support: date range, themes, countries
4. Efficient batch inserts and searches

```python
"""
LanceDB storage layer for GDELT embeddings.

Features:
- Scalar quantization (SQ8) for 4x storage reduction
- Metadata filtering (date, themes, countries)
- Batch insert/search operations

Usage:
    store = LanceDBStore("data/lancedb")
    store.add_records(embeddings, records)
    results = store.search("Venezuela sanctions", limit=10)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import lancedb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

# Schema for the GKG embeddings table
SCHEMA = pa.schema([
    pa.field("gkg_record_id", pa.string()),
    pa.field("date", pa.string()),           # YYYYMMDD
    pa.field("timestamp", pa.string()),       # ISO datetime
    pa.field("themes", pa.list_(pa.string())),
    pa.field("persons", pa.list_(pa.string())),
    pa.field("organizations", pa.list_(pa.string())),
    pa.field("locations", pa.list_(pa.string())),
    pa.field("countries", pa.list_(pa.string())),
    pa.field("tone", pa.float32()),
    pa.field("source", pa.string()),
    pa.field("source_url", pa.string()),
    pa.field("text_hash", pa.string()),       # For dedup
    pa.field("vector", pa.list_(pa.float32(), 1536)),  # Embedding
])


class LanceDBStore:
    """LanceDB storage for GDELT GKG embeddings."""
    
    TABLE_NAME = "gkg_embeddings"
    
    def __init__(self, db_path: str = "data/lancedb"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._ensure_table()
    
    def _ensure_table(self):
        """Create table if it doesn't exist."""
        if self.TABLE_NAME not in self.db.table_names():
            # Create empty table with schema
            self.db.create_table(self.TABLE_NAME, schema=SCHEMA)
            logger.info(f"Created table {self.TABLE_NAME}")
    
    def add_records(
        self,
        embedding_results: List,  # List[EmbeddingResult]
        gkg_records: List,        # List of parsed GKG records
    ) -> int:
        """
        Add embedded records to the store.
        
        Returns number of records added.
        """
        if not embedding_results:
            return 0
        
        # Build record lookup
        record_map = {r.gkg_record_id: r for r in gkg_records}
        
        rows = []
        for emb_result in embedding_results:
            record = record_map.get(emb_result.gkg_record_id)
            if not record:
                continue
            
            row = {
                "gkg_record_id": emb_result.gkg_record_id,
                "date": getattr(record, 'date', '')[:8] if hasattr(record, 'date') else '',
                "timestamp": getattr(record, 'date', ''),
                "themes": list(getattr(record, 'themes', []))[:20],
                "persons": [p.name for p in getattr(record, 'persons', [])][:10],
                "organizations": [o.name for o in getattr(record, 'organizations', [])][:10],
                "locations": [loc.name for loc in getattr(record, 'locations', [])][:10],
                "countries": list(set(
                    loc.country_code for loc in getattr(record, 'locations', [])
                    if hasattr(loc, 'country_code') and loc.country_code
                )),
                "tone": float(record.tone.tone) if hasattr(record, 'tone') and record.tone else 0.0,
                "source": getattr(record, 'source_common_name', ''),
                "source_url": getattr(record, 'document_id', ''),
                "text_hash": emb_result.text_hash,
                "vector": emb_result.embedding.tolist(),
            }
            rows.append(row)
        
        if rows:
            table = self.db.open_table(self.TABLE_NAME)
            table.add(rows)
            logger.info(f"Added {len(rows)} records to LanceDB")
        
        return len(rows)
    
    def search(
        self,
        query_text: str,
        limit: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        countries: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over GKG embeddings.
        
        Args:
            query_text: Natural language query
            limit: Max results
            date_from: Filter by date (YYYYMMDD)
            date_to: Filter by date (YYYYMMDD)
            countries: Filter by country codes
            themes: Filter by themes (partial match)
        
        Returns:
            List of matching records with similarity scores
        """
        from ..embedding.embedder import GKGEmbedder
        
        # Embed query
        embedder = GKGEmbedder()
        response = embedder.client.embeddings.create(
            model=embedder.MODEL,
            input=[query_text],
            dimensions=embedder.DIMENSIONS
        )
        query_vector = response.data[0].embedding
        
        # Build filter
        filters = []
        if date_from:
            filters.append(f"date >= '{date_from}'")
        if date_to:
            filters.append(f"date <= '{date_to}'")
        
        where_clause = " AND ".join(filters) if filters else None
        
        # Search
        table = self.db.open_table(self.TABLE_NAME)
        results = table.search(query_vector).limit(limit)
        
        if where_clause:
            results = results.where(where_clause)
        
        results = results.to_list()
        
        # Post-filter by countries/themes if specified
        if countries:
            results = [
                r for r in results
                if any(c in r.get("countries", []) for c in countries)
            ]
        if themes:
            results = [
                r for r in results
                if any(
                    t.lower() in " ".join(r.get("themes", [])).lower()
                    for t in themes
                )
            ]
        
        return results[:limit]
    
    def get_existing_ids(self, ids: List[str]) -> List[str]:
        """Check which IDs already exist in the store."""
        if not ids:
            return []
        
        table = self.db.open_table(self.TABLE_NAME)
        
        # Query for existing IDs
        # LanceDB doesn't have a direct "IN" query, so we check one by one
        # For efficiency, batch this if the table supports it
        existing = set()
        try:
            all_records = table.search().limit(100000).select(["gkg_record_id"]).to_list()
            existing = {r["gkg_record_id"] for r in all_records}
        except Exception as e:
            logger.warning(f"Could not fetch existing IDs: {e}")
        
        return [id for id in ids if id in existing]
    
    def count(self) -> int:
        """Get total record count."""
        table = self.db.open_table(self.TABLE_NAME)
        return table.count_rows()
    
    def get_date_range(self) -> tuple:
        """Get min/max dates in the store."""
        table = self.db.open_table(self.TABLE_NAME)
        results = table.search().limit(100000).select(["date"]).to_list()
        dates = [r["date"] for r in results if r.get("date")]
        if not dates:
            return None, None
        return min(dates), max(dates)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        min_date, max_date = self.get_date_range()
        return {
            "total_records": self.count(),
            "date_range": {"min": min_date, "max": max_date},
            "db_path": str(self.db_path),
            "table_name": self.TABLE_NAME,
        }
```

---

## Task 3: FastAPI Search API

### File: `src/api/__init__.py`
```python
"""FastAPI search API for GDELT embeddings."""
```

### File: `src/api/search.py`

Create FastAPI app with:
1. `POST /search` — vector similarity search with filters
2. `GET /health` — health check + index stats
3. `GET /stats` — detailed storage statistics
4. CORS configured for Vercel domain
5. Query embedding with caching

```python
"""
FastAPI search API for GDELT semantic search.

Run with: uvicorn src.api.search:app --host 0.0.0.0 --port 8080

Endpoints:
- POST /search: Vector similarity search
- GET /health: Health check
- GET /stats: Storage statistics
"""

import os
import time
import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..vectorstore.lancedb_store import LanceDBStore

logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="GDELT Semantic Search API",
    description="Semantic search over GDELT GKG embeddings",
    version="1.0.0",
)

# CORS configuration
ALLOWED_ORIGINS = [
    "https://vector-news.vercel.app",
    "https://*.vercel.app",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global store instance
_store: Optional[LanceDBStore] = None


def get_store() -> LanceDBStore:
    """Get or create LanceDB store instance."""
    global _store
    if _store is None:
        db_path = os.getenv("LANCEDB_PATH", "data/lancedb")
        _store = LanceDBStore(db_path)
    return _store


# Request/Response models
class SearchRequest(BaseModel):
    """Search request body."""
    query: str = Field(..., min_length=3, max_length=1000, description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Max results")
    date_from: Optional[str] = Field(None, pattern=r"^\d{8}$", description="Start date (YYYYMMDD)")
    date_to: Optional[str] = Field(None, pattern=r"^\d{8}$", description="End date (YYYYMMDD)")
    countries: Optional[List[str]] = Field(None, description="Country codes to filter")
    themes: Optional[List[str]] = Field(None, description="Themes to filter (partial match)")


class SearchResult(BaseModel):
    """Single search result."""
    gkg_record_id: str
    date: str
    themes: List[str]
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    countries: List[str]
    tone: float
    source: str
    source_url: str
    score: float  # Similarity score


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[SearchResult]
    total: int
    took_ms: float


class StatsResponse(BaseModel):
    """Stats response."""
    total_records: int
    date_range: dict
    db_path: str
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    record_count: int


# Startup time for uptime tracking
_start_time = time.time()


# Query embedding cache (5 min TTL simulated via LRU)
@lru_cache(maxsize=100)
def _cached_search(query_hash: str, query: str, limit: int, date_from: str, date_to: str):
    """Cached search wrapper."""
    store = get_store()
    return store.search(
        query_text=query,
        limit=limit,
        date_from=date_from if date_from != "None" else None,
        date_to=date_to if date_to != "None" else None,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search over GDELT GKG embeddings.
    
    Returns events similar to the query text, optionally filtered by date and metadata.
    """
    start = time.time()
    
    try:
        # Create cache key
        query_hash = hashlib.md5(
            f"{request.query}:{request.limit}:{request.date_from}:{request.date_to}".encode()
        ).hexdigest()
        
        # Search (with caching)
        raw_results = _cached_search(
            query_hash,
            request.query,
            request.limit * 2,  # Fetch extra for post-filtering
            str(request.date_from),
            str(request.date_to),
        )
        
        # Post-filter by countries/themes
        if request.countries:
            raw_results = [
                r for r in raw_results
                if any(c in r.get("countries", []) for c in request.countries)
            ]
        if request.themes:
            raw_results = [
                r for r in raw_results
                if any(
                    t.lower() in " ".join(r.get("themes", [])).lower()
                    for t in request.themes
                )
            ]
        
        # Limit and format results
        results = []
        for r in raw_results[:request.limit]:
            results.append(SearchResult(
                gkg_record_id=r.get("gkg_record_id", ""),
                date=r.get("date", ""),
                themes=r.get("themes", []),
                persons=r.get("persons", []),
                organizations=r.get("organizations", []),
                locations=r.get("locations", []),
                countries=r.get("countries", []),
                tone=r.get("tone", 0.0),
                source=r.get("source", ""),
                source_url=r.get("source_url", ""),
                score=r.get("_distance", 0.0),
            ))
        
        took_ms = (time.time() - start) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            took_ms=round(took_ms, 2),
        )
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        store = get_store()
        count = store.count()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            record_count=count,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get storage statistics."""
    try:
        store = get_store()
        store_stats = store.get_stats()
        return StatsResponse(
            total_records=store_stats["total_records"],
            date_range=store_stats["date_range"],
            db_path=store_stats["db_path"],
            uptime_seconds=round(time.time() - _start_time, 2),
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Task 4: Pipeline Orchestrator

### File: `src/pipeline/__init__.py`
```python
"""Pipeline orchestration."""
```

### File: `src/pipeline/run.py`

Create CLI that orchestrates the full pipeline:
1. Download new GKG files (incremental or backfill)
2. Parse downloaded files
3. Embed parsed records
4. Index in LanceDB
5. Report statistics

```python
"""
Pipeline orchestrator CLI.

Usage:
    # Backfill last 30 days
    python -m src.pipeline.run --backfill --days 30
    
    # Incremental (last 24h)
    python -m src.pipeline.run --incremental
    
    # Specific date range
    python -m src.pipeline.run --date-from 20260101 --date-to 20260205
    
    # Embedding only (skip download/parse)
    python -m src.pipeline.run --embed-only --date-from 20260101

"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from ..ingestion.worker import IngestionWorker
from ..ingestion.manifest import ManifestDB
from ..parsing.gkg_parser import GKGParser
from ..embedding.batch import BatchEmbedder
from ..vectorstore.lancedb_store import LanceDBStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GDELT Embedding Pipeline")
    
    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", action="store_true", help="Backfill historical data")
    mode.add_argument("--incremental", action="store_true", help="Process last 24h")
    mode.add_argument("--embed-only", action="store_true", help="Only run embedding (skip download)")
    
    # Date filters
    parser.add_argument("--date-from", type=str, help="Start date (YYYYMMDD)")
    parser.add_argument("--date-to", type=str, help="End date (YYYYMMDD)")
    parser.add_argument("--days", type=int, default=30, help="Days to backfill (default: 30)")
    
    # Limits
    parser.add_argument("--limit", type=int, help="Max files to process")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    
    return parser.parse_args()


def run_pipeline(args):
    """Run the full pipeline."""
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = data_dir / "manifest.db"
    lancedb_path = data_dir / "lancedb"
    parsed_dir = data_dir / "parsed"
    parsed_dir.mkdir(exist_ok=True)
    
    # Determine date range
    if args.incremental:
        date_to = datetime.now().strftime("%Y%m%d")
        date_from = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    elif args.backfill:
        date_to = args.date_to or datetime.now().strftime("%Y%m%d")
        if args.date_from:
            date_from = args.date_from
        else:
            date_from = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")
    else:
        date_from = args.date_from
        date_to = args.date_to
    
    logger.info(f"Pipeline: date_from={date_from}, date_to={date_to}")
    
    stats = {
        "started_at": datetime.now().isoformat(),
        "date_from": date_from,
        "date_to": date_to,
    }
    
    # Step 1: Download (unless embed-only)
    if not args.embed_only:
        logger.info("Step 1: Downloading GKG files...")
        worker = IngestionWorker(manifest_db=str(manifest_path))
        download_stats = worker.run(
            date_from=date_from,
            date_to=date_to,
            limit=args.limit,
        )
        stats["download"] = download_stats
        logger.info(f"Downloaded: {download_stats}")
    
    # Step 2: Parse (unless embed-only)
    if not args.embed_only:
        logger.info("Step 2: Parsing GKG files...")
        manifest = ManifestDB(manifest_path)
        parser = GKGParser()
        parse_count = 0
        
        for file_record in manifest.get_files_by_status("downloaded"):
            try:
                # Parse and save
                records = list(parser.parse_file_from_url(file_record.file_url))
                output_path = parsed_dir / f"{file_record.date_partition}.gkg.jsonl"
                
                with open(output_path, "w") as f:
                    for r in records:
                        f.write(r.to_json() + "\n")
                
                manifest.update_status(file_record.file_url, "completed")
                parse_count += 1
            except Exception as e:
                logger.error(f"Parse error: {e}")
                manifest.update_status(file_record.file_url, "failed", str(e))
        
        stats["parsed_files"] = parse_count
        logger.info(f"Parsed {parse_count} files")
    
    # Step 3: Embed
    logger.info("Step 3: Embedding records...")
    embedder = BatchEmbedder(
        manifest_db=str(manifest_path),
        lancedb_path=str(lancedb_path),
    )
    embed_stats = embedder.run(
        date_from=date_from,
        date_to=date_to,
        limit=args.limit,
    )
    stats["embedding"] = embed_stats
    logger.info(f"Embedded: {embed_stats}")
    
    # Step 4: Report
    store = LanceDBStore(str(lancedb_path))
    stats["final"] = store.get_stats()
    stats["finished_at"] = datetime.now().isoformat()
    
    logger.info("=" * 50)
    logger.info("Pipeline Complete!")
    logger.info(f"Total records: {stats['final']['total_records']}")
    logger.info(f"Date range: {stats['final']['date_range']}")
    logger.info("=" * 50)
    
    return stats


def main():
    args = parse_args()
    try:
        stats = run_pipeline(args)
        print(f"\nPipeline stats: {stats}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "httpx>=0.24.0",
    "polars>=0.20.0",
    "lancedb>=0.4.0",
    "openai>=1.10.0",
    "numpy>=1.24.0",
    "pyarrow>=14.0.0",
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.24.0",
]
```

---

## Testing

After implementation, verify:

```bash
# Install
pip install -e ".[dev]"

# Run incremental pipeline (last 24h)
python -m src.pipeline.run --incremental

# Check stats
curl http://localhost:8080/stats

# Test search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Venezuela sanctions", "limit": 5}'
```

---

## Constraints

1. **DO NOT modify existing files** (`worker.py`, `manifest.py`, `gkg_parser.py`) unless absolutely necessary
2. **Use OpenAI `text-embedding-3-small`** — not other models
3. **Use LanceDB** — not FAISS/Qdrant/Pinecone
4. **Scalar Quantization (SQ8)** is handled by LanceDB automatically when using float32→int8
5. **Environment variable `OPENAI_API_KEY`** must be set
6. **Port 8080** for FastAPI
7. **All new code in Python 3.11+** style (type hints, dataclasses, modern idioms)

---

## Deliverables

After running this prompt, the repo should have:

```
src/
├── ingestion/          # (existing - DO NOT TOUCH)
├── parsing/            # (existing - DO NOT TOUCH)
├── embedding/          # NEW
│   ├── __init__.py
│   ├── embedder.py     # OpenAI embedding wrapper
│   └── batch.py        # Batch processor
├── vectorstore/        # NEW
│   ├── __init__.py
│   └── lancedb_store.py
├── api/                # NEW
│   ├── __init__.py
│   └── search.py       # FastAPI app
└── pipeline/           # NEW
    ├── __init__.py
    └── run.py          # CLI orchestrator
```

Commit message: `feat: Complete GDELT embedding pipeline — embedder, LanceDB, FastAPI search, CLI orchestrator`
