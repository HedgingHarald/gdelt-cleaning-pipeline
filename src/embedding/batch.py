"""
Batch embedding processor with resumability.

Usage:
    processor = BatchEmbedder(
        manifest_db="data/manifest.db",
        lancedb_path="data/lancedb"
    )
    processor.run(date_from="20260101", date_to="20260207")
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..ingestion.manifest import FileStatus, ManifestDB
from ..parsing.gkg_parser import GKGStreamParser
from .embedder import GKGEmbedder

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
        self.manifest_path = Path(manifest_db)
        self.lancedb_path = lancedb_path
        self.openai_api_key = openai_api_key
        self._manifest = None
        self._store = None
        self._embedder = None
        self._parser = None
    
    @property
    def manifest(self) -> ManifestDB:
        """Lazy-load manifest database."""
        if self._manifest is None:
            self._manifest = ManifestDB(self.manifest_path)
        return self._manifest
    
    @property
    def store(self):
        """Lazy-load LanceDB store."""
        if self._store is None:
            from ..vectorstore.lancedb_store import LanceDBStore
            self._store = LanceDBStore(self.lancedb_path)
        return self._store
    
    @property
    def embedder(self) -> GKGEmbedder:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = GKGEmbedder(api_key=self.openai_api_key)
        return self._embedder
    
    @property
    def parser(self) -> GKGStreamParser:
        """Lazy-load parser."""
        if self._parser is None:
            self._parser = GKGStreamParser(
                fields_to_parse=["themes", "locations", "persons", "organizations", "tone"]
            )
        return self._parser
    
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
        # Try to find parsed file
        file_path = self._get_parsed_path(file_record)
        
        if file_path and file_path.exists():
            records = list(self._load_parsed_records(file_path))
        else:
            # Fall back to parsing from raw if available
            raw_path = self._get_raw_path(file_record)
            if raw_path and raw_path.exists():
                records = list(self.parser.parse_file(raw_path))
            else:
                logger.warning(f"No parsed or raw file found for {file_record.date_partition}")
                return
        
        logger.info(f"Processing {len(records)} records from {file_record.date_partition}")
        
        if not records:
            return
        
        # Filter already embedded
        record_ids = [r.gkg_record_id for r in records]
        existing_ids = set(self.store.get_existing_ids(record_ids))
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
            self.store.add_records(results, batch)
            stats["records_embedded"] += len(results)
            
            logger.info(f"Embedded batch {i // self.BATCH_SIZE + 1}: {len(results)} records")
    
    def _get_parsed_path(self, file_record) -> Optional[Path]:
        """Get path to parsed GKG file."""
        if not file_record.date_partition:
            return None
        
        date = file_record.date_partition
        
        # Check multiple possible locations
        possible_paths = [
            Path(f"data/parsed/{date}.gkg.jsonl"),
            Path(f"data/parsed/{date[:4]}/{date[4:6]}/{date}.gkg.jsonl"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return possible_paths[0]  # Return default path even if doesn't exist
    
    def _get_raw_path(self, file_record) -> Optional[Path]:
        """Get path to raw downloaded GKG file."""
        if not file_record.date_partition:
            return None
        
        date = file_record.date_partition
        year = date[:4]
        month = date[4:6]
        
        # Check possible raw file locations
        possible_paths = [
            Path(f"data/raw/{year}/{month}/{date}.gkg.csv"),
            Path(f"data/raw/{date}.gkg.csv"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_parsed_records(self, file_path: Path):
        """Load parsed records from JSONL file."""
        from ..parsing.gkg_parser import GKGRecord
        
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Create a simple record-like object
                        record = _ParsedRecord(data)
                        yield record
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {e}")
                        continue


class _ParsedRecord:
    """Simple wrapper for parsed JSON record data."""
    
    def __init__(self, data: dict):
        self._data = data
        self.gkg_record_id = data.get("gkg_record_id", "")
        self.date_str = data.get("date_str", "")
        self.source_common_name = data.get("source_common_name", "")
        self.document_id = data.get("document_id", "")
        self.themes = data.get("themes", [])
        self.persons = data.get("persons", [])
        self.organizations = data.get("organizations", [])
        
        # Handle locations
        locations_data = data.get("locations", [])
        self.locations = [_ParsedLocation(loc) for loc in locations_data]
        
        # Handle tone
        tone_data = data.get("tone")
        self.tone = _ParsedTone(tone_data) if tone_data else None


class _ParsedLocation:
    """Simple wrapper for parsed location data."""
    
    def __init__(self, data):
        if isinstance(data, dict):
            self.full_name = data.get("full_name", "")
            self.country_code = data.get("country_code", "")
        else:
            self.full_name = str(data)
            self.country_code = ""


class _ParsedTone:
    """Simple wrapper for parsed tone data."""
    
    def __init__(self, data):
        if isinstance(data, dict):
            self.tone = data.get("tone", 0.0)
        elif isinstance(data, (int, float)):
            self.tone = float(data)
        else:
            self.tone = 0.0
