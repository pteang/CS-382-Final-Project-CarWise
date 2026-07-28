from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.config import (  # noqa: E402
    CAMBODIA_SNAPSHOT_CSV,
    RAW_DIR,
    REFERENCE_SPECS_CSV,
)
from carwise.dataset import download_dataset  # noqa: E402


DEFAULT_EPA_CSV = RAW_DIR / "epa_vehicles.csv"
EPA_SOURCE_PAGE = "https://www.fueleconomy.gov/feg/ws/index.shtml"

MAKE_ALIASES = {
    "mini": "mini",
    "mercedes benz": "mercedes benz",
    "rolls royce": "rolls royce",
    "land rover": "land rover",
    "mclaren automotive": "mclaren",
}

MODEL_STOP_WORDS = {
    "advance",
    "automatic",
    "edition",
    "executive",
    "full",
    "luxury",
    "new",
    "option",
    "premium",
    "special",
    "sport",
    "standard",
    "ultra",
    "used",
}

GENERIC_MODELS = {
    "convertible",
    "coupe",
    "hatchback",
    "mpv",
    "pickup",
    "sedan",
    "sport",
    "sports",
    "suv",
}


def normalize_make(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return MAKE_ALIASES.get(text, text)


def normalize_model(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b\d+\.\d+\s*l?\b", " ", text)
    text = re.sub(
        r"\b(?:awd|4wd|2wd|fwd|rwd|xdrive|sdrive|v[468]|i[346]|l[346]|turbo|"
        r"diesel|hybrid|electric|automatic|manual)\b",
        " ",
        text,
    )
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens = [token for token in tokens if token not in MODEL_STOP_WORDS]
    return " ".join(tokens).strip()


def model_match_score(listing_model: str, epa_model: object) -> float:
    left = normalize_model(listing_model)
    right = normalize_model(epa_model)
    if (
        not left
        or left in {"model not identified", "not identified"}
        or left in GENERIC_MODELS
        or not right
    ):
        return 0.0
    if left == right:
        return 1.0
    if min(len(left), len(right)) >= 3 and (left in right or right in left):
        return 0.94
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    sequence = SequenceMatcher(None, left, right).ratio()
    return max(sequence, overlap)


def fuel_matches(listing_fuel: str, row: pd.Series) -> bool:
    fuel = str(row.get("fuelType1", "")).lower()
    technology = str(row.get("atvType", "")).lower()
    if listing_fuel == "Electricity":
        return "electricity" in fuel or technology == "ev"
    if listing_fuel == "Plug-in Hybrid":
        return "plug-in hybrid" in technology
    if listing_fuel == "Hybrid":
        return technology == "hybrid"
    if listing_fuel == "Diesel":
        return "diesel" in fuel
    if listing_fuel in {"Gasoline", "LPG or Gasoline"}:
        return "gasoline" in fuel
    return True


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def numeric_values(rows: pd.DataFrame, column: str) -> list[float]:
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return sorted({float(value) for value in values if float(value) > 0})


def format_range(values: list[float], suffix: str) -> str:
    if not values:
        return "Not verified"
    if len(values) == 1:
        return f"{format_number(values[0])}{suffix}"
    return f"{format_number(values[0])}–{format_number(values[-1])}{suffix}"


def transmission_summary(rows: pd.DataFrame) -> str:
    transmissions = sorted(
        {
            str(value).strip()
            for value in rows["trany"].dropna()
            if str(value).strip()
        }
    )
    if not transmissions:
        return "Not verified"
    if len(transmissions) <= 2:
        return " or ".join(transmissions)
    return "Multiple EPA-listed transmissions"


def cylinders_summary(rows: pd.DataFrame, electric: bool) -> str:
    if electric:
        return "N/A (electric)"
    values = numeric_values(rows, "cylinders")
    if not values:
        return "Not verified"
    return " or ".join(format_number(value) for value in values)


def displacement_summary(rows: pd.DataFrame, electric: bool) -> str:
    if electric:
        return "N/A (electric)"
    values = numeric_values(rows, "displ")
    if not values:
        return "Not verified"
    return " or ".join(f"{format_number(value)} L" for value in values)


def efficiency_summary(rows: pd.DataFrame, electric: bool) -> str:
    combined = numeric_values(rows, "comb08")
    if electric:
        result = format_range(combined, " MPGe combined (EPA)")
        ranges = numeric_values(rows, "range")
        if ranges:
            result += f"; {format_range(ranges, ' miles EPA range')}"
        return result
    return format_range(combined, " mpg combined (EPA)")


def epa_source_url(rows: pd.DataFrame) -> str:
    ids = [
        str(int(value))
        for value in pd.to_numeric(rows["id"], errors="coerce").dropna().head(3)
    ]
    if not ids:
        return EPA_SOURCE_PAGE
    return (
        "https://www.fueleconomy.gov/feg/Find.do?action=sbs&"
        + "&".join(f"id={vehicle_id}" for vehicle_id in ids)
    )


def best_epa_rows(
    listing: pd.Series,
    epa: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    year = listing.get("model_year")
    if pd.isna(year):
        return epa.iloc[0:0], 0.0
    make = normalize_make(listing.get("make"))
    model = str(listing.get("model", ""))
    if not make or normalize_model(model) in {"", "model not identified"}:
        return epa.iloc[0:0], 0.0

    candidates = epa[
        epa["_normalized_make"].eq(make)
        & epa["year"].eq(int(float(year)))
    ].copy()
    if candidates.empty:
        return candidates, 0.0
    candidates = candidates[
        candidates.apply(
            lambda row: fuel_matches(str(listing.get("fuel_type")), row),
            axis=1,
        )
    ].copy()
    if candidates.empty:
        return candidates, 0.0

    candidates["_score"] = candidates.apply(
        lambda row: max(
            model_match_score(model, row.get("model")),
            model_match_score(model, row.get("baseModel")),
        ),
        axis=1,
    )
    best_score = float(candidates["_score"].max())
    if best_score < 0.82:
        return candidates.iloc[0:0], best_score
    return candidates[candidates["_score"].ge(best_score - 0.04)], best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich Cambodian listings with exact year/make/model "
            "FuelEconomy.gov references."
        )
    )
    parser.add_argument(
        "--epa-csv",
        type=Path,
        default=DEFAULT_EPA_CSV,
        help="Official FuelEconomy.gov vehicle CSV cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epa_csv = args.epa_csv.resolve()
    if not epa_csv.exists():
        print(f"Downloading official FuelEconomy.gov data to {epa_csv}...")
        download_dataset(epa_csv)

    listings = pd.read_csv(
        CAMBODIA_SNAPSHOT_CSV,
        dtype={"listing_id": str},
    )
    specs = pd.read_csv(
        REFERENCE_SPECS_CSV,
        dtype={"listing_id": str},
    )
    epa = pd.read_csv(epa_csv, low_memory=False)
    epa["_normalized_make"] = epa["make"].map(normalize_make)

    listings_by_id = listings.set_index("listing_id", drop=False)
    matched = 0
    seller_ranges = 0
    for spec_index, spec in specs.iterrows():
        listing_id = str(spec["listing_id"])
        if listing_id not in listings_by_id.index:
            continue
        listing = listings_by_id.loc[listing_id]
        if isinstance(listing, pd.DataFrame):
            listing = listing.iloc[0]
        was_epa_match = (
            str(spec.get("spec_confidence", "")).strip()
            == "Exact year/make/model EPA range"
        )
        confidence = str(spec.get("spec_confidence", "")).strip()
        fallback_efficiency = str(spec.get("fuel_economy", "")).startswith(
            ("Official figure unavailable:", "Range unavailable:")
        )
        generated_reference = confidence in {
            "Exact year/make/model EPA range",
            "Official manufacturer model-family range",
            "Seller did not identify exact configuration",
        }
        if (
            str(spec.get("fuel_economy", "")).strip() != "Not verified"
            and not fallback_efficiency
            and not generated_reference
        ):
            continue

        exact_rows, _ = best_epa_rows(listing, epa)
        if not exact_rows.empty:
            electric = str(listing.get("fuel_type")) == "Electricity"
            specs.at[spec_index, "fuel_economy"] = efficiency_summary(
                exact_rows,
                electric,
            )
            specs.at[spec_index, "cylinders"] = cylinders_summary(
                exact_rows,
                electric,
            )
            specs.at[spec_index, "displacement_l"] = displacement_summary(
                exact_rows,
                electric,
            )
            specs.at[spec_index, "transmission"] = transmission_summary(
                exact_rows
            )
            specs.at[spec_index, "spec_source_name"] = (
                "U.S. DOE/EPA FuelEconomy.gov"
            )
            specs.at[spec_index, "spec_source_url"] = epa_source_url(exact_rows)
            specs.at[spec_index, "spec_confidence"] = (
                "Exact year/make/model EPA range"
            )
            specs.at[spec_index, "spec_note"] = (
                "EPA values cover matching U.S.-market configurations. "
                "The imported Cambodian trim may use a different engine, "
                "transmission, battery, or test cycle."
            )
            matched += 1
            continue

        if was_epa_match:
            electric = str(listing.get("fuel_type")) == "Electricity"
            specs.at[spec_index, "fuel_economy"] = "Not verified"
            specs.at[spec_index, "cylinders"] = (
                "N/A (electric)" if electric else "Not verified"
            )
            specs.at[spec_index, "displacement_l"] = (
                "N/A (electric)" if electric else "Not verified"
            )
            specs.at[spec_index, "transmission"] = (
                "Single-speed electric drive"
                if electric
                else "Not verified"
            )
            specs.at[spec_index, "spec_source_name"] = "Khmer24 listing"
            specs.at[spec_index, "spec_source_url"] = str(
                listing.get("source_url")
            )
            specs.at[spec_index, "spec_confidence"] = (
                "Listing does not identify exact trim"
            )
            specs.at[spec_index, "spec_note"] = (
                "Technical fields are left unverified until the exact engine, "
                "battery, transmission, and trim can be confirmed."
            )

        if (
            str(listing.get("fuel_type")) == "Electricity"
            and str(spec.get("fuel_economy")) == "Not verified"
        ):
            range_match = re.search(
                r"\b(\d{3,4})\s*km\b",
                str(listing.get("title", "")),
                re.I,
            )
            if range_match and 100 <= int(range_match.group(1)) <= 1200:
                specs.at[spec_index, "fuel_economy"] = (
                    f"Seller advertises {int(range_match.group(1))} km range"
                )
                specs.at[spec_index, "spec_source_name"] = "Khmer24 listing"
                specs.at[spec_index, "spec_source_url"] = str(
                    listing.get("source_url")
                )
                specs.at[spec_index, "spec_confidence"] = (
                    "Seller-advertised range"
                )
                specs.at[spec_index, "spec_note"] = (
                    "The seller does not identify the range test cycle; "
                    "verify battery, trim, and real-world range."
                )
                seller_ranges += 1

    specs.to_csv(REFERENCE_SPECS_CSV, index=False)
    print(
        f"Matched {matched} listings to exact year/make/model EPA records; "
        f"captured {seller_ranges} additional seller-advertised EV ranges."
    )


if __name__ == "__main__":
    main()
