from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.cambodia_dataset import load_listing_snapshot, write_corpus  # noqa: E402
from carwise.config import (  # noqa: E402
    CAMBODIA_SNAPSHOT_CSV,
    CURATED_CSV,
    DOCUMENTS_DIR,
    LISTING_IMAGES_CSV,
    REFERENCE_SPECS_CSV,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Cambodian marketplace CarWise document corpus."
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=CAMBODIA_SNAPSHOT_CSV,
        help="Cambodian listing snapshot CSV (defaults to the included snapshot).",
    )
    parser.add_argument(
        "--specs-csv",
        type=Path,
        default=REFERENCE_SPECS_CSV,
        help="Reference vehicle specifications CSV.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=LISTING_IMAGES_CSV,
        help="Listing image URL CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv.resolve()
    if not source_csv.exists():
        raise SystemExit(f"Snapshot CSV not found: {source_csv}")
    specs_csv = args.specs_csv.resolve()
    if not specs_csv.exists():
        raise SystemExit(f"Reference specifications CSV not found: {specs_csv}")
    images_csv = args.images_csv.resolve()
    if not images_csv.exists():
        raise SystemExit(f"Listing image CSV not found: {images_csv}")
    rows = load_listing_snapshot(source_csv, specs_csv, images_csv)
    documents = write_corpus(rows, CURATED_CSV, DOCUMENTS_DIR)
    print(
        f"Prepared {len(documents)} Cambodian listing documents in "
        f"{DOCUMENTS_DIR} from {source_csv}."
    )


if __name__ == "__main__":
    main()
