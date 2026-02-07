"""
FastAPI search API for GDELT semantic search.

Run with: uvicorn src.api.search:app --host 0.0.0.0 --port 8080

Endpoints:
- POST /search: Vector similarity search
- GET /health: Health check
- GET /stats: Storage statistics
"""

import hashlib
import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
_store = None


def get_store():
    """Get or create LanceDB store instance."""
    global _store
    if _store is None:
        from ..vectorstore.lancedb_store import LanceDBStore
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


# Query embedding cache (LRU cache for efficiency)
@lru_cache(maxsize=100)
def _get_query_embedding(query_hash: str, query: str) -> List[float]:
    """Get or compute query embedding with caching."""
    from ..embedding.embedder import GKGEmbedder
    
    embedder = GKGEmbedder()
    response = embedder.client.embeddings.create(
        model=embedder.MODEL,
        input=[query],
        dimensions=embedder.DIMENSIONS
    )
    return response.data[0].embedding


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search over GDELT GKG embeddings.
    
    Returns events similar to the query text, optionally filtered by date and metadata.
    """
    start = time.time()
    
    try:
        store = get_store()
        
        # Create cache key
        query_hash = hashlib.md5(request.query.encode()).hexdigest()
        
        # Get query embedding (cached)
        query_vector = _get_query_embedding(query_hash, request.query)
        
        # Search with vector
        raw_results = store.search_by_vector(
            query_vector=query_vector,
            limit=request.limit * 2,  # Fetch extra for post-filtering
            date_from=request.date_from,
            date_to=request.date_to,
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
                score=1.0 - r.get("_distance", 0.0),  # Convert distance to similarity
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


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "GDELT Semantic Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "POST /search",
            "health": "GET /health",
            "stats": "GET /stats",
        },
    }
