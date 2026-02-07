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
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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
    from ..ingestion.manifest import FileStatus, ManifestDB
    from ..ingestion.worker import IngestionWorker
    from ..parsing.gkg_parser import GKGStreamParser
    from ..embedding.batch import BatchEmbedder
    from ..vectorstore.lancedb_store import LanceDBStore
    
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
        logger.info("Step 1: Syncing and downloading GKG files...")
        manifest = ManifestDB(manifest_path)
        worker = IngestionWorker(
            manifest_db=manifest,
            data_dir=str(data_dir / "raw"),
        )
        
        # Sync masterfilelist
        sync_stats = worker.sync_masterfilelist(
            source="gkg",
            file_type_filter="gkg",
            date_min=date_from,
            date_max=date_to,
        )
        stats["sync"] = sync_stats
        logger.info(f"Synced: {sync_stats}")
        
        # Download pending files
        download_count = 0
        download_errors = 0
        max_downloads = args.limit if args.limit else None
        
        for result in worker.download_pending(batch_size=10, max_files=max_downloads):
            if result["status"] == "downloaded":
                download_count += 1
            else:
                download_errors += 1
        
        stats["download"] = {
            "files_downloaded": download_count,
            "errors": download_errors,
        }
        logger.info(f"Downloaded: {download_count} files, {download_errors} errors")
        worker.close()
    
    # Step 2: Parse (unless embed-only)
    if not args.embed_only:
        logger.info("Step 2: Parsing GKG files...")
        manifest = ManifestDB(manifest_path)
        parser = GKGStreamParser(
            fields_to_parse=["themes", "locations", "persons", "organizations", "tone"]
        )
        parse_count = 0
        parse_errors = 0
        
        for file_record in manifest.get_files_by_status(FileStatus.DOWNLOADED, limit=args.limit or 1000):
            try:
                # Find the downloaded file
                date = file_record.date_partition
                if not date:
                    continue
                
                year = date[:4]
                month = date[4:6]
                raw_path = data_dir / "raw" / year / month
                
                # Look for the CSV file
                csv_files = list(raw_path.glob(f"{date}*.gkg.csv")) if raw_path.exists() else []
                if not csv_files:
                    logger.warning(f"No CSV file found for {date}")
                    continue
                
                csv_path = csv_files[0]
                
                # Parse and save as JSONL
                records = list(parser.parse_file(csv_path))
                output_path = parsed_dir / f"{date}.gkg.jsonl"
                
                with open(output_path, "w") as f:
                    for r in records:
                        # Convert record to JSON-serializable dict
                        record_dict = {
                            "gkg_record_id": r.gkg_record_id,
                            "date_str": r.date_str,
                            "source_common_name": r.source_common_name,
                            "document_id": r.document_id,
                            "themes": r.themes,
                            "persons": r.persons,
                            "organizations": r.organizations,
                            "locations": [
                                {
                                    "full_name": loc.full_name,
                                    "country_code": loc.country_code,
                                }
                                for loc in r.locations
                            ],
                            "tone": {
                                "tone": r.tone.tone if r.tone else 0.0,
                            } if r.tone else None,
                        }
                        f.write(json.dumps(record_dict) + "\n")
                
                manifest.update_status(
                    file_record.file_url,
                    FileStatus.COMPLETED,
                    record_count=len(records),
                )
                parse_count += 1
                logger.info(f"Parsed {len(records)} records from {date}")
                
            except Exception as e:
                logger.error(f"Parse error for {file_record.file_url}: {e}")
                manifest.update_status(file_record.file_url, FileStatus.FAILED, error_message=str(e))
                parse_errors += 1
        
        stats["parsed_files"] = parse_count
        stats["parse_errors"] = parse_errors
        logger.info(f"Parsed {parse_count} files, {parse_errors} errors")
    
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
        print(f"\nPipeline stats: {json.dumps(stats, indent=2)}")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
