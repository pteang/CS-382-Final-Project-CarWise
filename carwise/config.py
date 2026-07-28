from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_local_env()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CAMBODIA_SNAPSHOT_CSV = RAW_DIR / "khmer24_cars_2026-07-28.csv"
REFERENCE_SPECS_CSV = RAW_DIR / "reference_vehicle_specs.csv"
LISTING_IMAGES_CSV = RAW_DIR / "listing_images.csv"
CURATED_CSV = DATA_DIR / "curated" / "cambodia_car_listings.csv"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"

# Default CarWise corpus: a timestamped snapshot of public Cambodian listings.
CAMBODIA_DATASET_PAGE_URL = "https://www.khmer24.com/c-cars-for-sale"

# Kept for the optional legacy EPA ingestion module in carwise/dataset.py.
DATASET_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"
DATASET_PAGE_URL = "https://www.fueleconomy.gov/feg/ws/index.shtml"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
DEFAULT_LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)
