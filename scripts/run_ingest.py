"""Run the ingestion pipeline over the bank document corpus.
 
Usage:
    python scripts/run_ingest.py                 # incremental ingest of dataset/raw_documents
    python scripts/run_ingest.py --rebuild       # wipe the index and re-ingest everything
    python scripts/run_ingest.py --limit 50      # small pilot run
    python scripts/run_ingest.py --dry-run       # chunk without indexing
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.container import get_services  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="corpus directory (default: dataset/raw_documents)")
    parser.add_argument("--rebuild", action="store_true", help="clear the index before ingesting")
    parser.add_argument("--dry-run", action="store_true", help="load and chunk but do not index")
    parser.add_argument("--limit", type=int, default=None, help="ingest at most N files")
    args = parser.parse_args()

    services = get_services()
    report = services.ingester.run(
        source_dir=Path(args.source) if args.source else None,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
        limit=args.limit,)
    services.acl.set_domains(services.ingester.domains())
    print(json.dumps(report.to_dict(), indent=2))
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())