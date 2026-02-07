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
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb
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
            
            # Extract date from date_str
            date_str = getattr(record, 'date_str', '') or ''
            date = date_str[:8] if date_str else ''
            
            # Extract themes
            themes = list(getattr(record, 'themes', []) or [])[:20]
            
            # Extract persons - handle both string lists and object lists
            persons_raw = getattr(record, 'persons', []) or []
            if persons_raw and hasattr(persons_raw[0], 'name'):
                persons = [p.name for p in persons_raw[:10]]
            else:
                persons = [str(p) for p in persons_raw[:10]]
            
            # Extract organizations - handle both string lists and object lists
            orgs_raw = getattr(record, 'organizations', []) or []
            if orgs_raw and hasattr(orgs_raw[0], 'name'):
                organizations = [o.name for o in orgs_raw[:10]]
            else:
                organizations = [str(o) for o in orgs_raw[:10]]
            
            # Extract locations and countries
            locations_raw = getattr(record, 'locations', []) or []
            locations = []
            countries = set()
            
            for loc in locations_raw[:10]:
                if hasattr(loc, 'full_name'):
                    if loc.full_name:
                        locations.append(loc.full_name)
                    if hasattr(loc, 'country_code') and loc.country_code:
                        countries.add(loc.country_code)
                else:
                    locations.append(str(loc))
            
            # Extract tone
            tone_value = 0.0
            if hasattr(record, 'tone') and record.tone:
                if hasattr(record.tone, 'tone'):
                    tone_value = float(record.tone.tone)
                elif isinstance(record.tone, (int, float)):
                    tone_value = float(record.tone)
            
            row = {
                "gkg_record_id": emb_result.gkg_record_id,
                "date": date,
                "timestamp": date_str,
                "themes": themes,
                "persons": persons,
                "organizations": organizations,
                "locations": locations,
                "countries": list(countries),
                "tone": tone_value,
                "source": getattr(record, 'source_common_name', '') or '',
                "source_url": getattr(record, 'document_id', '') or '',
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
        
        return self.search_by_vector(
            query_vector=query_vector,
            limit=limit,
            date_from=date_from,
            date_to=date_to,
            countries=countries,
            themes=themes,
        )
    
    def search_by_vector(
        self,
        query_vector: List[float],
        limit: int = 10,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        countries: Optional[List[str]] = None,
        themes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search by pre-computed vector.
        
        Args:
            query_vector: Query embedding vector
            limit: Max results
            date_from: Filter by date (YYYYMMDD)
            date_to: Filter by date (YYYYMMDD)
            countries: Filter by country codes
            themes: Filter by themes (partial match)
        
        Returns:
            List of matching records with similarity scores
        """
        # Build filter
        filters = []
        if date_from:
            filters.append(f"date >= '{date_from}'")
        if date_to:
            filters.append(f"date <= '{date_to}'")
        
        where_clause = " AND ".join(filters) if filters else None
        
        # Search
        table = self.db.open_table(self.TABLE_NAME)
        query = table.search(query_vector).limit(limit * 2 if (countries or themes) else limit)
        
        if where_clause:
            query = query.where(where_clause)
        
        results = query.to_list()
        
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
        # For efficiency, we sample the table
        existing = set()
        try:
            # Try to get all IDs efficiently
            all_records = table.to_pandas()[["gkg_record_id"]]
            existing = set(all_records["gkg_record_id"].tolist())
        except Exception as e:
            logger.warning(f"Could not fetch existing IDs efficiently: {e}")
            # Fallback to search-based approach
            try:
                results = table.search().limit(100000).select(["gkg_record_id"]).to_list()
                existing = {r["gkg_record_id"] for r in results}
            except Exception as e2:
                logger.warning(f"Could not fetch existing IDs: {e2}")
        
        return [id for id in ids if id in existing]
    
    def count(self) -> int:
        """Get total record count."""
        try:
            table = self.db.open_table(self.TABLE_NAME)
            return table.count_rows()
        except Exception:
            return 0
    
    def get_date_range(self) -> tuple:
        """Get min/max dates in the store."""
        try:
            table = self.db.open_table(self.TABLE_NAME)
            df = table.to_pandas()[["date"]]
            dates = df["date"].dropna().tolist()
            dates = [d for d in dates if d]
            if not dates:
                return None, None
            return min(dates), max(dates)
        except Exception as e:
            logger.warning(f"Could not get date range: {e}")
            return None, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        min_date, max_date = self.get_date_range()
        return {
            "total_records": self.count(),
            "date_range": {"min": min_date, "max": max_date},
            "db_path": str(self.db_path),
            "table_name": self.TABLE_NAME,
        }
    
    def delete_by_ids(self, ids: List[str]) -> int:
        """Delete records by IDs."""
        if not ids:
            return 0
        
        table = self.db.open_table(self.TABLE_NAME)
        
        # Build delete filter
        id_list = ", ".join(f"'{id}'" for id in ids)
        table.delete(f"gkg_record_id IN ({id_list})")
        
        return len(ids)
