from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import CAMBODIA_DATASET_PAGE_URL
from .corpus import load_documents
from .models import Document


REQUIRED_COLUMNS = {
    "listing_id",
    "title",
    "make",
    "model",
    "model_year",
    "price_usd",
    "condition",
    "registration",
    "location",
    "province",
    "body_type",
    "fuel_type",
    "observed_at",
    "source_url",
}

REQUIRED_METADATA = {
    "listing_id",
    "make",
    "model",
    "price_usd",
    "condition",
    "province",
    "body_type",
    "fuel_economy",
    "cylinders",
    "displacement_l",
    "seats",
    "transmission",
    "image_url",
}

SPEC_COLUMNS = {
    "listing_id",
    "fuel_economy",
    "cylinders",
    "displacement_l",
    "seats",
    "transmission",
    "spec_source_name",
    "spec_source_url",
    "spec_confidence",
    "spec_note",
}

IMAGE_COLUMNS = {"listing_id", "image_url"}


def _clean(value: object, fallback: str = "Not reported") -> str:
    if pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def _positive_int(value: object) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_listing_snapshot(
    source_csv: Path,
    specs_csv: Path | None = None,
    images_csv: Path | None = None,
) -> pd.DataFrame:
    """Load and validate the included Cambodian marketplace snapshot."""
    rows = pd.read_csv(source_csv)
    missing = REQUIRED_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(
            "Cambodian listing snapshot is missing columns: "
            + ", ".join(sorted(missing))
        )
    rows = rows.drop_duplicates(subset=["listing_id"]).copy()
    rows["price_usd"] = pd.to_numeric(
        rows["price_usd"], errors="coerce"
    ).astype("Int64")
    rows["model_year"] = pd.to_numeric(
        rows["model_year"], errors="coerce"
    ).astype("Int64")
    rows = rows[rows["price_usd"].gt(0) & rows["source_url"].notna()]

    spec_fields = sorted(SPEC_COLUMNS - {"listing_id"})
    if specs_csv is not None:
        specs = pd.read_csv(specs_csv, dtype={"listing_id": str})
        missing_specs = SPEC_COLUMNS.difference(specs.columns)
        if missing_specs:
            raise ValueError(
                "Reference specification CSV is missing columns: "
                + ", ".join(sorted(missing_specs))
            )
        specs = specs.drop_duplicates(subset=["listing_id"]).copy()
        rows["listing_id"] = rows["listing_id"].astype(str)
        rows = rows.merge(
            specs[["listing_id", *spec_fields]],
            on="listing_id",
            how="left",
            validate="one_to_one",
        )
    for field in spec_fields:
        if field not in rows:
            rows[field] = "Not verified"
        rows[field] = rows[field].fillna("Not verified")

    if images_csv is not None:
        images = pd.read_csv(images_csv, dtype={"listing_id": str})
        missing_images = IMAGE_COLUMNS.difference(images.columns)
        if missing_images:
            raise ValueError(
                "Listing image CSV is missing columns: "
                + ", ".join(sorted(missing_images))
            )
        images = images.drop_duplicates(subset=["listing_id"]).copy()
        rows["listing_id"] = rows["listing_id"].astype(str)
        rows = rows.merge(
            images[["listing_id", "image_url"]],
            on="listing_id",
            how="left",
            validate="one_to_one",
        )
    if "image_url" not in rows:
        rows["image_url"] = ""
    rows["image_url"] = rows["image_url"].fillna("")
    return rows.reset_index(drop=True)


def row_to_document(row: pd.Series) -> Document:
    listing_id = _clean(row["listing_id"])
    title = _clean(row["title"])
    make = _clean(row["make"])
    model = _clean(row["model"])
    year = _positive_int(row["model_year"])
    price = _positive_int(row["price_usd"])
    condition = _clean(row["condition"])
    registration = _clean(row["registration"])
    location = _clean(row["location"])
    province = _clean(row["province"])
    body_type = _clean(row["body_type"])
    fuel_type = _clean(row["fuel_type"])
    observed_at = _clean(row["observed_at"])
    source_url = _clean(row["source_url"])
    fuel_economy = _clean(row.get("fuel_economy"), "Not verified")
    cylinders = _clean(row.get("cylinders"), "Not verified")
    displacement_l = _clean(row.get("displacement_l"), "Not verified")
    seats = _clean(row.get("seats"), "Not verified")
    transmission = _clean(row.get("transmission"), "Not verified")
    spec_source_name = _clean(row.get("spec_source_name"), "Not verified")
    spec_source_url = _clean(row.get("spec_source_url"), "")
    spec_confidence = _clean(row.get("spec_confidence"), "Not verified")
    spec_note = _clean(row.get("spec_note"), "Not verified")
    image_url = _clean(row.get("image_url"), "")

    identity = f"{year} {make} {model}" if year else f"{make} {model}"
    text = "\n".join(
        [
            f"# {identity}",
            "## Cambodian marketplace listing",
            f"- Listing title: {title}.",
            f"- Asking price: ${price:,.0f} USD.",
            f"- Condition label: {condition}.",
            f"- Location: {location}.",
            f"- Province or municipality: {province}.",
            f"- Registration or import-paper label: {registration}.",
            f"- Listing observed: {observed_at}.",
            "## Vehicle details",
            f"- Make: {make}.",
            f"- Model: {model}.",
            f"- Model year: {year if year else 'Not reported'}.",
            f"- Body type: {body_type}.",
            f"- Fuel or powertrain: {fuel_type}.",
            "## Reference specifications",
            f"- Fuel economy: {fuel_economy}.",
            f"- Cylinders: {cylinders}.",
            f"- Engine displacement: {displacement_l}.",
            f"- Seating capacity: {seats}.",
            f"- Transmission: {transmission}.",
            f"- Specification confidence: {spec_confidence}.",
            f"- Specification source: {spec_source_name}.",
            f"- Specification note: {spec_note.rstrip('.')}.",
            "## Evidence scope",
            (
                "This is a timestamped Khmer24 marketplace listing in Cambodia. "
                "The asking price, condition label, paperwork label, and availability "
                "are seller-provided and may change after the observation date. "
                "The listing does not establish crash safety, reliability, mechanical "
                "condition, legal ownership, or a fair market value. Open the source "
                "listing and independently inspect the vehicle and documents before "
                "making a purchase. Technical specifications are separate model-year "
                "references and may not match the listed vehicle's exact trim, engine, "
                "transmission, wheels, or import market."
            ),
        ]
    )
    metadata = {
        "listing_id": listing_id,
        "year": year,
        "model_year": year,
        "make": make,
        "model": model,
        "price_usd": price,
        "condition": condition,
        "registration": registration,
        "location": location,
        "province": province,
        "body_type": body_type,
        "fuel_type": fuel_type,
        "observed_at": observed_at,
        "market": "Cambodia",
        "fuel_economy": fuel_economy,
        "cylinders": cylinders,
        "displacement_l": displacement_l,
        "seats": seats,
        "transmission": transmission,
        "spec_source_name": spec_source_name,
        "spec_source_url": spec_source_url,
        "spec_confidence": spec_confidence,
        "spec_note": spec_note,
        "image_url": image_url,
    }
    return Document(
        document_id=f"khmer24-{listing_id}-{_slug(identity)}",
        title=identity,
        text=text,
        source_name="Khmer24 Cambodia listing (seller-provided)",
        source_url=source_url,
        metadata=metadata,
    )


def write_corpus(
    rows: pd.DataFrame,
    curated_csv: Path,
    documents_dir: Path,
) -> list[Document]:
    curated_csv.parent.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(curated_csv, index=False)

    for old_path in documents_dir.glob("*.json"):
        old_path.unlink()

    documents = [row_to_document(row) for _, row in rows.iterrows()]
    for document in documents:
        path = documents_dir / f"{document.document_id}.json"
        path.write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    observed_dates = sorted(
        {_clean(document.metadata["observed_at"]) for document in documents}
    )
    manifest = {
        "dataset_name": "CarWise Cambodian marketplace listing snapshot",
        "dataset_page": CAMBODIA_DATASET_PAGE_URL,
        "source": "Public Khmer24 cars-for-sale listing cards",
        "observed_dates": observed_dates,
        "document_count": len(documents),
        "method": (
            "One document per unique public listing with an identifiable vehicle. "
            "Seller contact details were excluded. Vehicle types and powertrains were "
            "normalized conservatively from the listing title."
        ),
        "limitations": (
            "Asking prices and availability are time-sensitive seller claims. The "
            "snapshot does not verify safety, reliability, ownership, mechanical "
            "condition, or fair market value."
        ),
    }
    (documents_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return documents


def ensure_cambodia_corpus(
    source_csv: Path,
    curated_csv: Path,
    documents_dir: Path,
    specs_csv: Path | None = None,
    images_csv: Path | None = None,
) -> tuple[list[Document], bool]:
    """Load the Cambodian corpus, rebuilding legacy or mixed document folders."""
    try:
        documents = load_documents(documents_dir)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        documents = []

    compatible = bool(documents) and all(
        REQUIRED_METADATA.issubset(document.metadata) for document in documents
    )
    if compatible:
        return documents, False

    if not source_csv.exists():
        raise FileNotFoundError(
            "The Cambodian listing snapshot is missing: "
            f"{source_csv}. Restore the included CSV or run the dataset preparation "
            "script with --source-csv."
        )
    rows = load_listing_snapshot(source_csv, specs_csv, images_csv)
    return write_corpus(rows, curated_csv, documents_dir), True
