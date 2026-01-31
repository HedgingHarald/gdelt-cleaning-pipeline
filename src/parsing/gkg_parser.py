"""
Memory-safe streaming parser for GDELT GKG (Global Knowledge Graph) CSV files.

Processes files line-by-line with configurable field extraction.
Handles malformed rows gracefully with detailed error reporting.
"""

import csv
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


class GKGVersion(Enum):
    """GDELT GKG format version."""
    V1 = "1.0"    # Original GKG (27 fields)
    V2 = "2.0"    # GKG 2.0 (27 fields, different schema)
    V2_1 = "2.1"  # GKG 2.1 (enhanced themes/locations)


# GKG 2.0 field definitions (tab-separated)
GKG_V2_FIELDS = [
    "gkg_record_id",           # 0: Unique identifier
    "date",                    # 1: YYYYMMDDHHMMSS
    "source_collection_id",    # 2: Source collection identifier
    "source_common_name",      # 3: Human-readable source name
    "document_id",             # 4: URL or document identifier
    "counts",                  # 5: CAMEO event counts
    "v2_counts",               # 6: Enhanced counts
    "themes",                  # 7: Themes/topics (semicolon-separated)
    "v2_enhanced_themes",      # 8: Enhanced themes with offsets
    "locations",               # 9: Locations mentioned
    "v2_enhanced_locations",   # 10: Enhanced locations with coordinates
    "persons",                 # 11: Person names
    "v2_enhanced_persons",     # 12: Enhanced persons with offsets
    "organizations",           # 13: Organization names
    "v2_enhanced_organizations", # 14: Enhanced orgs with offsets
    "v2_tone",                 # 15: Tone metrics (6 values)
    "v2_enhanced_dates",       # 16: Date mentions with offsets
    "v2_gcam",                 # 17: Global Content Analysis Measures
    "v2_sharing_image",        # 18: Social sharing image URL
    "v2_related_images",       # 19: Related images
    "v2_social_image_embeds",  # 20: Social image embeds
    "v2_social_video_embeds",  # 21: Social video embeds
    "v2_quotations",           # 22: Direct quotations
    "v2_all_names",            # 23: All names mentioned
    "v2_amounts",              # 24: Amounts/quantities mentioned
    "v2_translation_info",     # 25: Translation metadata
    "v2_extras_xml",           # 26: Extra XML data
]


@dataclass
class ToneMetrics:
    """Parsed tone/sentiment metrics from GKG."""
    tone: float = 0.0                  # Overall tone (-100 to +100)
    positive_score: float = 0.0        # Positive word percentage
    negative_score: float = 0.0        # Negative word percentage
    polarity: float = 0.0              # Polarity score
    activity_ref_density: float = 0.0  # Activity reference density
    group_ref_density: float = 0.0     # Group/self reference density
    word_count: int = 0                # Total word count

    @classmethod
    def from_string(cls, tone_str: str) -> "ToneMetrics":
        """Parse tone string (comma-separated values)."""
        if not tone_str:
            return cls()
        
        try:
            parts = tone_str.split(",")
            return cls(
                tone=float(parts[0]) if len(parts) > 0 and parts[0] else 0.0,
                positive_score=float(parts[1]) if len(parts) > 1 and parts[1] else 0.0,
                negative_score=float(parts[2]) if len(parts) > 2 and parts[2] else 0.0,
                polarity=float(parts[3]) if len(parts) > 3 and parts[3] else 0.0,
                activity_ref_density=float(parts[4]) if len(parts) > 4 and parts[4] else 0.0,
                group_ref_density=float(parts[5]) if len(parts) > 5 and parts[5] else 0.0,
                word_count=int(float(parts[6])) if len(parts) > 6 and parts[6] else 0,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse tone: {tone_str[:100]}, error: {e}")
            return cls()


@dataclass
class Location:
    """Parsed location from GKG."""
    location_type: int = 0       # 1=Country, 2=State, 3=City, 4=Feature
    full_name: str = ""
    country_code: str = ""
    adm1_code: str = ""          # State/province code
    adm2_code: str = ""          # County/district code
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    feature_id: str = ""
    char_offset: Optional[int] = None

    @classmethod
    def from_string(cls, loc_str: str) -> Optional["Location"]:
        """Parse location string (hash-separated fields)."""
        if not loc_str:
            return None
        
        try:
            parts = loc_str.split("#")
            return cls(
                location_type=int(parts[0]) if len(parts) > 0 and parts[0] else 0,
                full_name=parts[1] if len(parts) > 1 else "",
                country_code=parts[2] if len(parts) > 2 else "",
                adm1_code=parts[3] if len(parts) > 3 else "",
                adm2_code=parts[4] if len(parts) > 4 else "",
                latitude=float(parts[5]) if len(parts) > 5 and parts[5] else None,
                longitude=float(parts[6]) if len(parts) > 6 and parts[6] else None,
                feature_id=parts[7] if len(parts) > 7 else "",
                char_offset=int(parts[8]) if len(parts) > 8 and parts[8] else None,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse location: {loc_str[:100]}, error: {e}")
            return None


@dataclass
class GKGRecord:
    """
    Parsed GKG record with structured fields.
    
    Memory-efficient: only populates requested fields.
    """
    # Core identifiers
    gkg_record_id: str = ""
    date: Optional[datetime] = None
    date_str: str = ""  # Raw YYYYMMDDHHMMSS
    
    # Source info
    source_collection_id: int = 0
    source_common_name: str = ""
    document_id: str = ""  # Usually URL
    
    # Extracted entities (parsed on demand)
    themes: List[str] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    
    # Sentiment
    tone: Optional[ToneMetrics] = None
    
    # Raw fields for lazy parsing
    _raw_fields: Dict[str, str] = field(default_factory=dict, repr=False)
    
    # Parsing metadata
    line_number: int = 0
    parse_errors: List[str] = field(default_factory=list)

    def get_raw_field(self, field_name: str) -> str:
        """Get raw unparsed field value."""
        return self._raw_fields.get(field_name, "")
    
    @property
    def is_valid(self) -> bool:
        """Check if record has minimum required fields."""
        return bool(self.gkg_record_id and self.document_id)
    
    @property
    def year_month(self) -> Optional[str]:
        """Extract YYYYMM from date for partitioning."""
        if self.date_str and len(self.date_str) >= 6:
            return self.date_str[:6]
        return None


class ParseError(Exception):
    """Error during GKG parsing."""
    def __init__(self, message: str, line_number: int, line_content: str = ""):
        self.line_number = line_number
        self.line_content = line_content[:200] if line_content else ""
        super().__init__(f"Line {line_number}: {message}")


class GKGStreamParser:
    """
    Memory-safe streaming parser for GKG CSV files.
    
    Processes files line-by-line, never loading full file into memory.
    Supports selective field parsing for performance optimization.
    
    Example:
        parser = GKGStreamParser(
            fields_to_parse=["themes", "locations", "tone"],
            skip_malformed=True,
        )
        
        for record in parser.parse_file("./data/20230101.gkg.csv"):
            print(record.themes, record.tone.tone)
    
        # Or with raw bytes stream:
        for record in parser.parse_stream(byte_chunks_iterator):
            process(record)
    """
    
    FIELD_DELIMITER = "\t"
    LIST_DELIMITER = ";"
    
    def __init__(
        self,
        version: GKGVersion = GKGVersion.V2,
        fields_to_parse: Optional[List[str]] = None,
        skip_malformed: bool = True,
        max_errors: int = 1000,
        error_callback: Optional[Callable[[ParseError], None]] = None,
    ):
        """
        Initialize GKG parser.
        
        Args:
            version: GKG format version
            fields_to_parse: List of fields to parse (None = all)
            skip_malformed: Skip malformed rows instead of raising
            max_errors: Maximum errors before aborting
            error_callback: Optional callback for parse errors
        """
        self.version = version
        self.fields_to_parse = set(fields_to_parse) if fields_to_parse else None
        self.skip_malformed = skip_malformed
        self.max_errors = max_errors
        self.error_callback = error_callback
        
        # Parsing stats
        self._reset_stats()
    
    def _reset_stats(self) -> None:
        """Reset parsing statistics."""
        self.stats = {
            "lines_processed": 0,
            "records_parsed": 0,
            "errors": 0,
            "skipped": 0,
        }
    
    def _should_parse(self, field_name: str) -> bool:
        """Check if field should be parsed."""
        if self.fields_to_parse is None:
            return True
        return field_name in self.fields_to_parse
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse GDELT date format (YYYYMMDDHHMMSS)."""
        if not date_str:
            return None
        
        try:
            # Handle various lengths
            if len(date_str) >= 14:
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:
                return datetime.strptime(date_str[:8], "%Y%m%d")
            return None
        except ValueError:
            return None
    
    def _parse_list(self, value: str, delimiter: str = ";") -> List[str]:
        """Parse semicolon-separated list."""
        if not value:
            return []
        return [item.strip() for item in value.split(delimiter) if item.strip()]
    
    def _parse_locations(self, locations_str: str) -> List[Location]:
        """Parse locations field."""
        if not locations_str:
            return []
        
        locations = []
        for loc_str in locations_str.split(self.LIST_DELIMITER):
            loc = Location.from_string(loc_str.strip())
            if loc:
                locations.append(loc)
        return locations
    
    def _parse_line(self, line: str, line_number: int) -> Optional[GKGRecord]:
        """
        Parse a single GKG line into a record.
        
        Args:
            line: Raw tab-separated line
            line_number: Line number for error reporting
            
        Returns:
            GKGRecord or None if parsing failed
        """
        if not line.strip():
            return None
        
        fields = line.split(self.FIELD_DELIMITER)
        
        # Validate minimum field count
        expected_fields = len(GKG_V2_FIELDS)
        if len(fields) < 5:  # Minimum viable fields
            raise ParseError(
                f"Too few fields: {len(fields)}, expected at least 5",
                line_number,
                line,
            )
        
        # Pad fields if necessary
        while len(fields) < expected_fields:
            fields.append("")
        
        record = GKGRecord(line_number=line_number)
        
        try:
            # Core fields (always parsed)
            record.gkg_record_id = fields[0].strip()
            record.date_str = fields[1].strip()
            record.date = self._parse_date(record.date_str)
            
            try:
                record.source_collection_id = int(fields[2]) if fields[2].strip() else 0
            except ValueError:
                record.source_collection_id = 0
            
            record.source_common_name = fields[3].strip()
            record.document_id = fields[4].strip()
            
            # Store raw fields for lazy parsing
            for i, field_name in enumerate(GKG_V2_FIELDS):
                if i < len(fields):
                    record._raw_fields[field_name] = fields[i]
            
            # Parse optional fields based on config
            if self._should_parse("themes"):
                record.themes = self._parse_list(fields[7] if len(fields) > 7 else "")
            
            if self._should_parse("locations") and len(fields) > 10:
                record.locations = self._parse_locations(fields[10])
            
            if self._should_parse("persons") and len(fields) > 11:
                record.persons = self._parse_list(fields[11])
            
            if self._should_parse("organizations") and len(fields) > 13:
                record.organizations = self._parse_list(fields[13])
            
            if self._should_parse("tone") and len(fields) > 15:
                record.tone = ToneMetrics.from_string(fields[15])
            
        except Exception as e:
            record.parse_errors.append(str(e))
            if not self.skip_malformed:
                raise ParseError(str(e), line_number, line)
        
        return record
    
    def parse_file(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> Iterator[GKGRecord]:
        """
        Stream-parse a GKG file.
        
        Memory-safe: reads and processes line by line.
        
        Args:
            file_path: Path to GKG CSV file
            encoding: File encoding (default: utf-8)
            
        Yields:
            GKGRecord for each valid line
        """
        self._reset_stats()
        file_path = Path(file_path)
        
        logger.info(f"Parsing GKG file: {file_path}")
        
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            yield from self._parse_text_stream(f)
        
        logger.info(
            f"Parsed {self.stats['records_parsed']} records, "
            f"{self.stats['errors']} errors, "
            f"{self.stats['skipped']} skipped"
        )
    
    def parse_stream(
        self,
        byte_stream: Iterator[bytes],
        encoding: str = "utf-8",
    ) -> Iterator[GKGRecord]:
        """
        Parse a stream of bytes (e.g., from HTTP response).
        
        Memory-safe: buffers only current line.
        
        Args:
            byte_stream: Iterator of byte chunks
            encoding: Character encoding
            
        Yields:
            GKGRecord for each valid line
        """
        self._reset_stats()
        
        # Buffer for incomplete lines
        buffer = ""
        
        for chunk in byte_stream:
            # Decode chunk
            try:
                text = chunk.decode(encoding, errors="replace")
            except Exception as e:
                logger.warning(f"Decode error: {e}")
                continue
            
            buffer += text
            
            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                
                self.stats["lines_processed"] += 1
                
                if not line.strip():
                    continue
                
                try:
                    record = self._parse_line(line, self.stats["lines_processed"])
                    if record and record.is_valid:
                        self.stats["records_parsed"] += 1
                        yield record
                    else:
                        self.stats["skipped"] += 1
                
                except ParseError as e:
                    self.stats["errors"] += 1
                    
                    if self.error_callback:
                        self.error_callback(e)
                    
                    if self.stats["errors"] >= self.max_errors:
                        raise ParseError(
                            f"Max errors ({self.max_errors}) exceeded",
                            self.stats["lines_processed"],
                        )
                    
                    if not self.skip_malformed:
                        raise
        
        # Process remaining buffer
        if buffer.strip():
            self.stats["lines_processed"] += 1
            try:
                record = self._parse_line(buffer, self.stats["lines_processed"])
                if record and record.is_valid:
                    self.stats["records_parsed"] += 1
                    yield record
            except ParseError:
                self.stats["errors"] += 1
    
    def _parse_text_stream(self, text_stream: TextIO) -> Iterator[GKGRecord]:
        """Parse a text stream (file handle)."""
        for line_number, line in enumerate(text_stream, start=1):
            self.stats["lines_processed"] = line_number
            
            if not line.strip():
                continue
            
            try:
                record = self._parse_line(line.rstrip("\n\r"), line_number)
                if record and record.is_valid:
                    self.stats["records_parsed"] += 1
                    yield record
                else:
                    self.stats["skipped"] += 1
            
            except ParseError as e:
                self.stats["errors"] += 1
                
                if self.error_callback:
                    self.error_callback(e)
                
                if self.stats["errors"] >= self.max_errors:
                    raise ParseError(
                        f"Max errors ({self.max_errors}) exceeded",
                        line_number,
                    )
                
                if not self.skip_malformed:
                    raise
    
    def parse_string(self, content: str) -> Iterator[GKGRecord]:
        """Parse GKG content from a string (for testing)."""
        self._reset_stats()
        yield from self._parse_text_stream(io.StringIO(content))
    
    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics."""
        return dict(self.stats)


class GKGRecordBatcher:
    """
    Batch GKG records for efficient downstream processing.
    
    Groups records by date partition for batch inserts.
    """
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self._batches: Dict[str, List[GKGRecord]] = {}
        self._count = 0
    
    def add(self, record: GKGRecord) -> Optional[Tuple[str, List[GKGRecord]]]:
        """
        Add record to batcher.
        
        Returns:
            (partition_key, records) if batch is full, None otherwise
        """
        partition = record.year_month or "unknown"
        
        if partition not in self._batches:
            self._batches[partition] = []
        
        self._batches[partition].append(record)
        self._count += 1
        
        if len(self._batches[partition]) >= self.batch_size:
            batch = self._batches.pop(partition)
            return (partition, batch)
        
        return None
    
    def flush(self) -> Iterator[Tuple[str, List[GKGRecord]]]:
        """Flush all remaining batches."""
        for partition, records in self._batches.items():
            if records:
                yield (partition, records)
        self._batches.clear()
    
    @property
    def total_count(self) -> int:
        """Total records processed."""
        return self._count
