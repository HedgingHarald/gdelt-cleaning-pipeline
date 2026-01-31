"""
Ingestion worker for GDELT masterfilelist.txt and file downloads.

Handles resumable, memory-safe streaming of GDELT file metadata.
"""

import hashlib
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse

import httpx

from .manifest import FileStatus, ManifestDB

logger = logging.getLogger(__name__)


@dataclass
class GDELTFileInfo:
    """Parsed file info from masterfilelist.txt."""
    file_url: str
    file_size_bytes: int
    checksum_md5: str
    date_partition: Optional[str] = None  # YYYYMMDD
    file_type: Optional[str] = None       # gkg, export, mentions

    @classmethod
    def from_line(cls, line: str) -> Optional["GDELTFileInfo"]:
        """
        Parse a line from masterfilelist.txt.
        
        Format: "size md5 url"
        Example: "123456 abc123def456 http://data.gdeltproject.org/gkg/20230101.gkg.csv.zip"
        """
        parts = line.strip().split()
        if len(parts) != 3:
            return None
        
        try:
            size = int(parts[0])
            md5 = parts[1]
            url = parts[2]
        except (ValueError, IndexError):
            return None
        
        # Extract date from filename
        # Patterns: YYYYMMDD.gkg.csv.zip, YYYYMMDDHHMMSS.gkg.csv.zip
        filename = urlparse(url).path.split("/")[-1]
        date_match = re.search(r"(\d{8})", filename)
        date_partition = date_match.group(1) if date_match else None
        
        # Determine file type
        file_type = None
        if ".gkg." in filename.lower():
            file_type = "gkg"
        elif ".export." in filename.lower():
            file_type = "export"
        elif ".mentions." in filename.lower():
            file_type = "mentions"
        
        return cls(
            file_url=url,
            file_size_bytes=size,
            checksum_md5=md5,
            date_partition=date_partition,
            file_type=file_type,
        )


class IngestionWorker:
    """
    Worker for ingesting GDELT file metadata and downloading files.
    
    Streams masterfilelist.txt to avoid memory issues with 10+ years of data.
    Supports resumable processing via ManifestDB.
    
    Example:
        db = ManifestDB("./data/manifest.db")
        worker = IngestionWorker(db, data_dir="./data/raw")
        
        # Ingest file list (streaming, memory-safe)
        worker.sync_masterfilelist(file_type="gkg")
        
        # Download pending files
        for result in worker.download_pending(batch_size=10):
            print(f"Downloaded: {result.file_url}")
    """
    
    # GDELT master file list URLs
    MASTERFILE_URLS = {
        "gkg": "http://data.gdeltproject.org/gkg/md5sums",
        "gkg_v2": "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt",
        "events": "http://data.gdeltproject.org/events/md5sums",
    }
    
    # Default timeouts
    CONNECT_TIMEOUT = 30.0
    READ_TIMEOUT = 300.0  # 5 min for large files
    
    def __init__(
        self,
        manifest_db: ManifestDB,
        data_dir: str | Path = "./data/raw",
        http_client: Optional[httpx.Client] = None,
        chunk_size: int = 8192,
    ):
        """
        Initialize ingestion worker.
        
        Args:
            manifest_db: ManifestDB instance for tracking
            data_dir: Directory to store downloaded files
            http_client: Optional custom httpx client
            chunk_size: Chunk size for streaming downloads
        """
        self.manifest = manifest_db
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(
                connect=self.CONNECT_TIMEOUT,
                read=self.READ_TIMEOUT,
                write=30.0,
                pool=5.0,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
            ),
        )
    
    def stream_masterfilelist(
        self,
        source: str = "gkg",
        file_type_filter: Optional[str] = None,
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
    ) -> Iterator[GDELTFileInfo]:
        """
        Stream file metadata from GDELT masterfilelist.
        
        Memory-safe: processes line by line, never loads full list.
        
        Args:
            source: Source list ("gkg", "gkg_v2", "events")
            file_type_filter: Filter by type ("gkg", "export", "mentions")
            date_min: Minimum date partition (YYYYMMDD)
            date_max: Maximum date partition (YYYYMMDD)
            
        Yields:
            GDELTFileInfo objects for matching files
        """
        url = self.MASTERFILE_URLS.get(source)
        if not url:
            raise ValueError(f"Unknown source: {source}. Valid: {list(self.MASTERFILE_URLS.keys())}")
        
        logger.info(f"Streaming masterfilelist from {url}")
        
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line.strip():
                    continue
                
                file_info = GDELTFileInfo.from_line(line)
                if file_info is None:
                    logger.debug(f"Skipping unparseable line: {line[:100]}")
                    continue
                
                # Apply filters
                if file_type_filter and file_info.file_type != file_type_filter:
                    continue
                
                if date_min and file_info.date_partition:
                    if file_info.date_partition < date_min:
                        continue
                
                if date_max and file_info.date_partition:
                    if file_info.date_partition > date_max:
                        continue
                
                yield file_info
    
    def sync_masterfilelist(
        self,
        source: str = "gkg",
        file_type_filter: Optional[str] = None,
        date_min: Optional[str] = None,
        date_max: Optional[str] = None,
        batch_size: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict:
        """
        Sync masterfilelist to manifest database.
        
        Streaming insert - processes in batches for efficiency.
        Only inserts new files, doesn't overwrite existing.
        
        Args:
            source: Source list ("gkg", "gkg_v2", "events")
            file_type_filter: Filter by type
            date_min: Minimum date partition
            date_max: Maximum date partition
            batch_size: Batch size for database inserts
            progress_callback: Optional callback(files_processed)
            
        Returns:
            Dict with sync statistics
        """
        stats = {
            "total_scanned": 0,
            "new_inserted": 0,
            "already_tracked": 0,
            "started_at": datetime.utcnow().isoformat(),
        }
        
        batch: list[tuple[str, Optional[str], Optional[int]]] = []
        
        for file_info in self.stream_masterfilelist(
            source=source,
            file_type_filter=file_type_filter,
            date_min=date_min,
            date_max=date_max,
        ):
            stats["total_scanned"] += 1
            
            batch.append((
                file_info.file_url,
                file_info.date_partition,
                file_info.file_size_bytes,
            ))
            
            if len(batch) >= batch_size:
                inserted = self.manifest.upsert_files_batch(batch)
                stats["new_inserted"] += inserted
                stats["already_tracked"] += len(batch) - inserted
                batch.clear()
                
                if progress_callback:
                    progress_callback(stats["total_scanned"])
        
        # Final batch
        if batch:
            inserted = self.manifest.upsert_files_batch(batch)
            stats["new_inserted"] += inserted
            stats["already_tracked"] += len(batch) - inserted
        
        stats["ended_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"Sync complete: {stats['new_inserted']} new files, "
            f"{stats['already_tracked']} already tracked"
        )
        
        return stats
    
    def download_file(
        self,
        file_url: str,
        verify_md5: Optional[str] = None,
    ) -> Path:
        """
        Download a single GDELT file.
        
        Streams to disk to avoid memory issues.
        Automatically extracts if zipped.
        
        Args:
            file_url: URL to download
            verify_md5: Expected MD5 checksum (optional)
            
        Returns:
            Path to downloaded (and extracted) file
            
        Raises:
            httpx.HTTPError: On download failure
            ValueError: On checksum mismatch
        """
        # Determine output path
        filename = urlparse(file_url).path.split("/")[-1]
        
        # Extract date partition for subdirectory
        date_match = re.search(r"(\d{8})", filename)
        if date_match:
            date_str = date_match.group(1)
            year = date_str[:4]
            month = date_str[4:6]
            subdir = self.data_dir / year / month
        else:
            subdir = self.data_dir / "unknown"
        
        subdir.mkdir(parents=True, exist_ok=True)
        output_path = subdir / filename
        
        # Stream download with checksum
        hasher = hashlib.md5() if verify_md5 else None
        
        logger.debug(f"Downloading {file_url} to {output_path}")
        
        with self._client.stream("GET", file_url) as response:
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
        
        # Verify checksum
        if verify_md5 and hasher:
            actual_md5 = hasher.hexdigest()
            if actual_md5 != verify_md5:
                output_path.unlink()  # Clean up bad download
                raise ValueError(
                    f"MD5 mismatch for {file_url}: "
                    f"expected {verify_md5}, got {actual_md5}"
                )
        
        # Extract if zipped
        if output_path.suffix == ".zip":
            extracted_path = self._extract_zip(output_path)
            output_path.unlink()  # Remove zip after extraction
            return extracted_path
        
        return output_path
    
    def _extract_zip(self, zip_path: Path) -> Path:
        """Extract a zip file and return path to extracted content."""
        extract_dir = zip_path.parent
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Get the first (usually only) file in the archive
            names = zf.namelist()
            if not names:
                raise ValueError(f"Empty zip file: {zip_path}")
            
            # Extract first file
            extracted_name = names[0]
            zf.extract(extracted_name, extract_dir)
            
            return extract_dir / extracted_name
    
    def download_pending(
        self,
        batch_size: int = 10,
        max_files: Optional[int] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Iterator[dict]:
        """
        Download pending files from manifest.
        
        Atomically claims files to prevent duplicate processing
        by concurrent workers.
        
        Args:
            batch_size: Files to claim per batch
            max_files: Maximum total files to download
            progress_callback: Optional callback(file_url, status)
            
        Yields:
            Dict with download result for each file
        """
        files_processed = 0
        
        while True:
            if max_files and files_processed >= max_files:
                break
            
            # Claim a batch of pending files
            remaining = batch_size
            if max_files:
                remaining = min(batch_size, max_files - files_processed)
            
            pending = self.manifest.get_next_pending(batch_size=remaining)
            if not pending:
                logger.info("No more pending files")
                break
            
            for file_record in pending:
                file_url = file_record.file_url
                result = {
                    "file_url": file_url,
                    "status": None,
                    "path": None,
                    "error": None,
                }
                
                try:
                    # Get expected checksum from manifest
                    expected_md5 = file_record.checksum_md5
                    
                    # Download file
                    path = self.download_file(file_url, verify_md5=expected_md5)
                    
                    # Update manifest
                    self.manifest.update_status(
                        file_url,
                        FileStatus.DOWNLOADED,
                        file_size_bytes=path.stat().st_size,
                    )
                    
                    result["status"] = "downloaded"
                    result["path"] = str(path)
                    
                    if progress_callback:
                        progress_callback(file_url, "downloaded")
                    
                except Exception as e:
                    logger.error(f"Failed to download {file_url}: {e}")
                    
                    self.manifest.update_status(
                        file_url,
                        FileStatus.FAILED,
                        error_message=str(e)[:500],
                    )
                    
                    result["status"] = "failed"
                    result["error"] = str(e)
                    
                    if progress_callback:
                        progress_callback(file_url, "failed")
                
                files_processed += 1
                yield result
    
    def get_file_content_stream(
        self,
        file_url: str,
    ) -> Iterator[bytes]:
        """
        Stream file content directly without saving to disk.
        
        Useful for parsing in memory-constrained environments.
        Handles zip extraction in-memory.
        
        Args:
            file_url: URL to stream
            
        Yields:
            Chunks of file content
        """
        with self._client.stream("GET", file_url) as response:
            response.raise_for_status()
            
            # Check if it's a zip file
            if file_url.endswith(".zip"):
                # Buffer the zip content
                buffer = io.BytesIO()
                for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    buffer.write(chunk)
                
                buffer.seek(0)
                
                with zipfile.ZipFile(buffer, "r") as zf:
                    names = zf.namelist()
                    if not names:
                        return
                    
                    with zf.open(names[0]) as f:
                        while True:
                            chunk = f.read(self.chunk_size)
                            if not chunk:
                                break
                            yield chunk
            else:
                # Stream directly
                for chunk in response.iter_bytes(chunk_size=self.chunk_size):
                    yield chunk
    
    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()
    
    def __enter__(self) -> "IngestionWorker":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
