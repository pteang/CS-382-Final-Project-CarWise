from __future__ import annotations

import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATASET_PAGE_URL, DATASET_URL
from .models import Document


SUPPORTED_MAKES = {
    "Acura",
    "Audi",
    "BMW",
    "Buick",
    "Cadillac",
    "Chevrolet",
    "Chrysler",
    "Dodge",
    "Ford",
    "Genesis",
    "GMC",
    "Honda",
    "Hyundai",
    "Jeep",
    "Kia",
    "Land Rover",
    "Lexus",
    "Mazda",
    "Mercedes-Benz",
    "MINI",
    "Mitsubishi",
    "Nissan",
    "Porsche",
    "Rivian",
    "Subaru",
    "Tesla",
    "Toyota",
    "Volkswagen",
    "Volvo",
}

USEFUL_CLASSES = (
    "Cars",
    "Sport Utility",
    "Station Wagons",
    "Minivan",
    "Pickup",
)


def download_dataset(destination_csv: Path, *, url: str = DATASET_URL) -> Path:
    """Download and extract the official FuelEconomy.gov vehicle CSV."""
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    archive_path = destination_csv.with_suffix(".csv.zip")
    urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("The downloaded archive did not contain a CSV file.")
        with archive.open(csv_names[0]) as source, destination_csv.open("wb") as target:
            shutil.copyfileobj(source, target)
    return destination_csv


def _clean_text(value: object, fallback: str = "Not reported") -> str:
    if pd.isna(value) or str(value).strip() in {"", "0", "0.0", "-1"}:
        return fallback
    return str(value).strip()


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def curate_vehicle_rows(
    raw_csv: Path,
    *,
    years: tuple[int, ...] = (2025, 2026),
    max_per_make: int = 8,
) -> pd.DataFrame:
    """Select a balanced, demo-sized subset with one efficient trim per model."""
    data = pd.read_csv(raw_csv, low_memory=False)
    selected = data[
        data["year"].isin(years)
        & data["make"].isin(SUPPORTED_MAKES)
        & data["VClass"].fillna("").str.contains("|".join(USEFUL_CLASSES), regex=True)
    ].copy()
    selected["baseModel"] = selected["baseModel"].fillna(selected["model"])

    # Keep the most recent and most efficient representative configuration for each
    # make/base-model/fuel combination, then cap each make for collection balance.
    selected = selected.sort_values(
        ["year", "comb08", "range"], ascending=[False, False, False]
    )
    selected = selected.drop_duplicates(
        subset=["make", "baseModel", "fuelType1"], keep="first"
    )
    selected = (
        selected.sort_values(["make", "baseModel", "fuelType1"])
        .groupby("make", group_keys=False)
        .head(max_per_make)
        .reset_index(drop=True)
    )
    return selected


def row_to_document(row: pd.Series) -> Document:
    year = int(row["year"])
    make = _clean_text(row["make"])
    model = _clean_text(row["model"])
    base_model = _clean_text(row.get("baseModel"), model)
    vehicle_id = int(row["id"])
    title = f"{year} {make} {model}"
    source_url = f"https://www.fueleconomy.gov/feg/Find.do?action=sbs&id={vehicle_id}"

    fuel = _clean_text(row.get("fuelType1"))
    combined = _number(row.get("comb08"))
    city = _number(row.get("city08"))
    highway = _number(row.get("highway08"))
    range_miles = _number(row.get("range"))
    range_alt = _number(row.get("rangeA"))
    electric_range = range_miles or range_alt
    charge_240 = _number(row.get("charge240"))
    annual_cost = _number(row.get("fuelCost08"))
    co2 = _number(row.get("co2TailpipeGpm"))

    unit = "MPGe" if fuel == "Electricity" else "MPG"
    efficiency_lines = [
        f"- City efficiency: {city:g} {unit}." if city else "- City efficiency: Not reported.",
        (
            f"- Highway efficiency: {highway:g} {unit}."
            if highway
            else "- Highway efficiency: Not reported."
        ),
        (
            f"- Combined efficiency: {combined:g} {unit}."
            if combined
            else "- Combined efficiency: Not reported."
        ),
    ]
    electric_lines = [
        (
            f"- EPA driving range: {electric_range:g} miles."
            if electric_range
            else "- EPA driving range: Not reported."
        ),
        (
            f"- 240-volt charge time: {charge_240:g} hours."
            if charge_240
            else "- 240-volt charge time: Not reported."
        ),
    ]

    text = "\n".join(
        [
            f"# {title}",
            "## Vehicle identity",
            f"- Base model: {base_model}.",
            f"- EPA vehicle class: {_clean_text(row.get('VClass'))}.",
            f"- Drive system: {_clean_text(row.get('drive'))}.",
            f"- Transmission: {_clean_text(row.get('trany'))}.",
            f"- Primary fuel: {fuel}.",
            f"- Engine displacement: {_clean_text(row.get('displ'))} liters.",
            f"- Cylinders: {_clean_text(row.get('cylinders'))}.",
            "## Fuel economy",
            *efficiency_lines,
            f"- EPA fuel-economy score: {_clean_text(row.get('feScore'))}.",
            "## Operating impact",
            (
                f"- Estimated annual fuel cost: ${annual_cost:,.0f}."
                if annual_cost
                else "- Estimated annual fuel cost: Not reported."
            ),
            (
                f"- Tailpipe CO2 emissions: {co2:g} grams per mile."
                if co2
                else "- Tailpipe CO2 emissions: zero or not reported."
            ),
            f"- EPA greenhouse-gas score: {_clean_text(row.get('ghgScore'))}.",
            "## Electric details",
            *electric_lines,
            f"- Electric motor: {_clean_text(row.get('evMotor'))}.",
            "## Evidence scope",
            (
                "This profile is derived from the official FuelEconomy.gov vehicle "
                "record. It supports comparisons about fuel, efficiency, EPA class, "
                "drive system, emissions, estimated fuel cost, and electric range "
                "when those fields are reported. It does not establish purchase "
                "price, reliability, crash safety, comfort, or local availability."
            ),
        ]
    )
    metadata = {
        "vehicle_id": vehicle_id,
        "year": year,
        "make": make,
        "model": model,
        "base_model": base_model,
        "vehicle_class": _clean_text(row.get("VClass")),
        "fuel_type": fuel,
        "drive": _clean_text(row.get("drive")),
        "combined_efficiency": combined,
        "efficiency_unit": unit,
    }
    return Document(
        document_id=f"epa-{vehicle_id}-{_slug(title)}",
        title=title,
        text=text,
        source_name="FuelEconomy.gov vehicle data (U.S. DOE/EPA)",
        source_url=source_url,
        metadata=metadata,
    )


def write_corpus(
    curated_rows: pd.DataFrame,
    curated_csv: Path,
    documents_dir: Path,
) -> list[Document]:
    curated_csv.parent.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    curated_rows.to_csv(curated_csv, index=False)
    for old_path in documents_dir.glob("*.json"):
        old_path.unlink()

    documents = [row_to_document(row) for _, row in curated_rows.iterrows()]
    for document in documents:
        path = documents_dir / f"{document.document_id}.json"
        path.write_text(
            json.dumps(document.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    manifest = {
        "dataset_name": "FuelEconomy.gov vehicle data",
        "dataset_page": DATASET_PAGE_URL,
        "dataset_download": DATASET_URL,
        "model_years": sorted(
            {int(document.metadata["year"]) for document in documents}
        ),
        "document_count": len(documents),
        "method": (
            "One representative high-efficiency configuration per make/base-model/"
            "fuel combination, capped per make for a balanced classroom corpus."
        ),
    }
    (documents_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return documents
