"""
SQLite-based manifest database for tracking GDELT file ingestion status.

Provides resumable processing by persisting file states across runs.
"""

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional


class FileStatus(Enum):
    """Status of a file in the ingestion pipeline."""
    PENDING = "pending"           # Discovered, not yet processed
    DOWNLOADING = "downloading"   # Currently being downloaded
    DOWNLOADED = "downloaded"     # Downloaded, awaiting parsing
    PARSING = "parsing"           # Currently being parsed
    COMPLETED = "completed"       # Successfully processed
    FAILED = "failed"             # Failed (will retry)
    SKIPPED = "skipped"           # Intentionally skipped


@dataclass
class FileRecord:
    """Represents a tracked file in the manifest."""
    file_url: str
    status: FileStatus
    record_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    date_partition: Optional[str] = None  # YYYYMMDD from filename
    checksum_md5: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row: tuple) -> "FileRecord":
        """Create FileRecord from database row."""
        return cls(
            file_url=row[0],
            status=FileStatus(row[1]),
            record_count=row[2],
            file_size_bytes=row[3],
            date_partition=row[4],
            checksum_md5=row[5],
            error_message=row[6],
            retry_count=row[7],
            created_at=datetime.fromisoformat(row[8]) if row[8] else None,
            updated_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )


class ManifestDB:
    """
    SQLite-based manifest for tracking GDELT file ingestion.
    
    Thread-safe with connection pooling per thread.
    Supports resumable processing and status queries.
    
    Example:
        db = ManifestDB("./data/manifest.db")
        db.upsert_file("http://data.gdeltproject.org/gkg/20230101.gkg.csv.zip")
        db.update_status("http://...", FileStatus.DOWNLOADING)
        
        for file in db.get_files_by_status(FileStatus.PENDING, limit=100):
            process(file)
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str | Path, max_retries: int = 3):
        """
        Initialize manifest database.
        
        Args:
            db_path: Path to SQLite database file (created if not exists)
            max_retries: Maximum retry attempts for failed files
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self._local = threading.local()
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent reads
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn
    
    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database cursor with auto-commit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cur:
            # Main files table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_url TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    record_count INTEGER,
                    file_size_bytes INTEGER,
                    date_partition TEXT,
                    checksum_md5 TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # Indexes for common queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_status 
                ON files(status)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_date_partition 
                ON files(date_partition)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_status_retry 
                ON files(status, retry_count)
            """)
            
            # Schema version tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT DEFAULT (datetime('now'))
                )
            """)
            cur.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,)
            )
            
            # Processing stats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_started_at TEXT,
                    run_ended_at TEXT,
                    files_processed INTEGER DEFAULT 0,
                    records_processed INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0
                )
            """)
    
    def upsert_file(
        self,
        file_url: str,
        status: FileStatus = FileStatus.PENDING,
        date_partition: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
    ) -> None:
        """
        Insert or update a file record.
        
        Args:
            file_url: Full URL to the GDELT file
            status: Initial status (default: PENDING)
            date_partition: YYYYMMDD extracted from filename
            file_size_bytes: File size if known
        """
        with self._cursor() as cur:
            cur.execute("""
                INSERT INTO files (file_url, status, date_partition, file_size_bytes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_url) DO UPDATE SET
                    updated_at = datetime('now')
                WHERE status = 'pending'
            """, (file_url, status.value, date_partition, file_size_bytes))
    
    def upsert_files_batch(
        self,
        files: list[tuple[str, Optional[str], Optional[int]]]
    ) -> int:
        """
        Batch insert/update files for efficiency.
        
        Args:
            files: List of (file_url, date_partition, file_size_bytes) tuples
            
        Returns:
            Number of new files inserted
        """
        with self._cursor() as cur:
            cur.executemany("""
                INSERT INTO files (file_url, status, date_partition, file_size_bytes)
                VALUES (?, 'pending', ?, ?)
                ON CONFLICT(file_url) DO NOTHING
            """, files)
            return cur.rowcount
    
    def update_status(
        self,
        file_url: str,
        status: FileStatus,
        record_count: Optional[int] = None,
        error_message: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        checksum_md5: Optional[str] = None,
    ) -> None:
        """
        Update file status and metadata.
        
        Args:
            file_url: File URL to update
            status: New status
            record_count: Number of records parsed (for COMPLETED)
            error_message: Error details (for FAILED)
            file_size_bytes: Actual file size after download
            checksum_md5: MD5 checksum after download
        """
        with self._cursor() as cur:
            # Increment retry count on failure
            retry_increment = 1 if status == FileStatus.FAILED else 0
            
            cur.execute("""
                UPDATE files SET
                    status = ?,
                    record_count = COALESCE(?, record_count),
                    file_size_bytes = COALESCE(?, file_size_bytes),
                    checksum_md5 = COALESCE(?, checksum_md5),
                    error_message = ?,
                    retry_count = retry_count + ?,
                    updated_at = datetime('now')
                WHERE file_url = ?
            """, (
                status.value,
                record_count,
                file_size_bytes,
                checksum_md5,
                error_message,
                retry_increment,
                file_url,
            ))
    
    def get_file(self, file_url: str) -> Optional[FileRecord]:
        """Get a single file record by URL."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT file_url, status, record_count, file_size_bytes,
                       date_partition, checksum_md5, error_message, retry_count,
                       created_at, updated_at
                FROM files WHERE file_url = ?
            """, (file_url,))
            row = cur.fetchone()
            return FileRecord.from_row(tuple(row)) if row else None
    
    def get_files_by_status(
        self,
        status: FileStatus,
        limit: int = 100,
        include_retryable: bool = True,
    ) -> Iterator[FileRecord]:
        """
        Stream files by status.
        
        Args:
            status: Status to filter by
            limit: Maximum files to return
            include_retryable: For FAILED, include files under max_retries
            
        Yields:
            FileRecord objects matching criteria
        """
        with self._cursor() as cur:
            if status == FileStatus.FAILED and include_retryable:
                cur.execute("""
                    SELECT file_url, status, record_count, file_size_bytes,
                           date_partition, checksum_md5, error_message, retry_count,
                           created_at, updated_at
                    FROM files 
                    WHERE status = ? AND retry_count < ?
                    ORDER BY updated_at ASC
                    LIMIT ?
                """, (status.value, self.max_retries, limit))
            else:
                cur.execute("""
                    SELECT file_url, status, record_count, file_size_bytes,
                           date_partition, checksum_md5, error_message, retry_count,
                           created_at, updated_at
                    FROM files 
                    WHERE status = ?
                    ORDER BY date_partition ASC, created_at ASC
                    LIMIT ?
                """, (status.value, limit))
            
            for row in cur.fetchall():
                yield FileRecord.from_row(tuple(row))
    
    def get_next_pending(self, batch_size: int = 1) -> list[FileRecord]:
        """
        Atomically claim pending files for processing.
        
        Marks files as DOWNLOADING and returns them.
        Safe for concurrent workers.
        
        Args:
            batch_size: Number of files to claim
            
        Returns:
            List of claimed FileRecord objects
        """
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            # Begin exclusive transaction for atomic claim
            cur.execute("BEGIN IMMEDIATE")
            
            # Find pending files
            cur.execute("""
                SELECT file_url FROM files
                WHERE status = 'pending'
                ORDER BY date_partition ASC, created_at ASC
                LIMIT ?
            """, (batch_size,))
            urls = [row[0] for row in cur.fetchall()]
            
            if not urls:
                conn.commit()
                return []
            
            # Claim them atomically
            placeholders = ",".join("?" * len(urls))
            cur.execute(f"""
                UPDATE files SET
                    status = 'downloading',
                    updated_at = datetime('now')
                WHERE file_url IN ({placeholders})
            """, urls)
            
            # Return full records
            cur.execute(f"""
                SELECT file_url, status, record_count, file_size_bytes,
                       date_partition, checksum_md5, error_message, retry_count,
                       created_at, updated_at
                FROM files WHERE file_url IN ({placeholders})
            """, urls)
            
            records = [FileRecord.from_row(tuple(row)) for row in cur.fetchall()]
            conn.commit()
            return records
            
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    
    def reset_stale_downloads(self, stale_minutes: int = 30) -> int:
        """
        Reset files stuck in DOWNLOADING state.
        
        Useful for recovering from worker crashes.
        
        Args:
            stale_minutes: Consider downloads stale after this many minutes
            
        Returns:
            Number of files reset
        """
        with self._cursor() as cur:
            cur.execute("""
                UPDATE files SET
                    status = 'pending',
                    updated_at = datetime('now')
                WHERE status = 'downloading'
                AND updated_at < datetime('now', ?)
            """, (f"-{stale_minutes} minutes",))
            return cur.rowcount
    
    def get_stats(self) -> dict:
        """Get summary statistics of the manifest."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as count,
                    SUM(record_count) as total_records,
                    SUM(file_size_bytes) as total_bytes
                FROM files
                GROUP BY status
            """)
            
            stats = {
                "by_status": {},
                "total_files": 0,
                "total_records": 0,
                "total_bytes": 0,
            }
            
            for row in cur.fetchall():
                status = row[0]
                stats["by_status"][status] = {
                    "count": row[1],
                    "records": row[2] or 0,
                    "bytes": row[3] or 0,
                }
                stats["total_files"] += row[1]
                stats["total_records"] += row[2] or 0
                stats["total_bytes"] += row[3] or 0
            
            # Date range
            cur.execute("""
                SELECT MIN(date_partition), MAX(date_partition)
                FROM files WHERE date_partition IS NOT NULL
            """)
            row = cur.fetchone()
            stats["date_range"] = {
                "min": row[0],
                "max": row[1],
            }
            
            return stats
    
    def get_date_coverage(self) -> dict[str, int]:
        """Get record counts by date partition."""
        with self._cursor() as cur:
            cur.execute("""
                SELECT date_partition, SUM(record_count)
                FROM files
                WHERE date_partition IS NOT NULL AND record_count IS NOT NULL
                GROUP BY date_partition
                ORDER BY date_partition
            """)
            return {row[0]: row[1] for row in cur.fetchall()}
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __enter__(self) -> "ManifestDB":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
