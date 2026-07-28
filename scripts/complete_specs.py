from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.config import (  # noqa: E402
    CAMBODIA_SNAPSHOT_CSV,
    REFERENCE_SPECS_CSV,
)


@dataclass(frozen=True)
class FamilySpec:
    make: str
    model_pattern: str
    fuel_economy: str
    cylinders: str
    displacement_l: str
    seats: str
    transmission: str
    source_name: str
    source_url: str
    note: str

    def matches(self, listing: pd.Series) -> bool:
        return (
            str(listing.get("make")) == self.make
            and re.search(
                self.model_pattern,
                str(listing.get("model", "")),
                re.I,
            )
            is not None
        )


FAMILY_SPECS = [
    FamilySpec(
        "Toyota",
        r"^(?:Alphard|Vellfire)$",
        "9.5–19.4 km/L (official Japanese test cycles; powertrain dependent)",
        "4 or 6",
        "2.4 L, 2.5 L, or 3.5 L",
        "6–8 depending on grade",
        "Super CVT-i, e-CVT, or 6-speed automatic",
        "Toyota Global Newsroom",
        "https://global.toyota/en/newsroom/toyota/23310520.html",
        "Official Alphard family figures span gasoline and hybrid grades. "
        "Confirm the imported vehicle's grade and test cycle.",
    ),
    FamilySpec(
        "Kia",
        r"^Morning$",
        "52.3–64.2 UK mpg combined (official Picanto family configurations)",
        "3 or 4",
        "1.0 L or 1.25 L",
        "5",
        "5-speed manual, 4-speed automatic, or 5-speed AMT",
        "Kia Picanto official specification",
        (
            "https://www.kia.com/content/dam/kwcms/kme/uk/en/assets/"
            "vehicles/All-New%20Picanto/Specifications/"
            "all-new-picanto-specification.PDF"
        ),
        "Kia Morning is sold as Picanto in other markets. Figures are a "
        "model-family range; confirm the Cambodian import's engine.",
    ),
    FamilySpec(
        "Kia",
        r"^Carnival$",
        "7.5 L/100 km combined for an official regional diesel configuration",
        "4 or 6",
        "2.2 L diesel or 3.5 L gasoline",
        "7, 8, 9, or 11 depending on market",
        "8-speed automatic (current family reference)",
        "Kia Carnival official specifications",
        "https://www.kia.com/hk/en/showroom/carnival/specification.html",
        "Carnival engines and seating layouts vary substantially by market "
        "and generation; confirm the listing's exact configuration.",
    ),
    FamilySpec(
        "Hyundai",
        r"^Staria$",
        "Official consumption varies by 2.2 diesel or 3.5 gasoline market trim",
        "4 or 6",
        "2.2 L diesel or 3.5 L gasoline",
        "3, 7, 10, or 11 depending on configuration",
        "6-speed manual or 8-speed automatic",
        "Hyundai STARIA official brochure",
        (
            "https://www.hyundai.com/content/dam/hyundai/ph/en/data/"
            "marketing/brochure/product/staria-2021/staria-0702622.pdf"
        ),
        "The seller must confirm whether the listing is a passenger or cargo "
        "version and identify its market specification.",
    ),
    FamilySpec(
        "Hyundai",
        r"^Custin$",
        "Official consumption varies by destination-market 1.5 T-GDI calibration",
        "4",
        "1.5 L turbo gasoline",
        "7",
        "8-speed automatic",
        "Hyundai CUSTIN official brochure",
        (
            "https://www.hyundai.com/content/dam/hyundai/ph/en/data/"
            "marketing/brochure/product/custin/Custin-brochure-v2.pdf"
        ),
        "Official Philippine-market family reference; verify the Cambodian "
        "import's calibration and equipment.",
    ),
    FamilySpec(
        "Mitsubishi",
        r"^Xpander$",
        "Manufacturer reports a low-consumption 1.5 L powertrain; exact market figure varies",
        "4",
        "1.5 L",
        "7",
        "5-speed manual or 4-speed automatic",
        "Mitsubishi Motors XPANDER release",
        (
            "https://www.mitsubishi-motors.com/en/newsroom/"
            "newsrelease/2017/20170810_1.html"
        ),
        "Official launch specifications are for an ASEAN-market XPANDER.",
    ),
    FamilySpec(
        "Mazda",
        r"^BT-50$",
        "6.3–9.2 L/100 km across official generation/powertrain references",
        "4 or 5",
        "1.9 L, 2.2 L, 3.0 L, or 3.2 L diesel",
        "2–5 depending on cab configuration",
        "6-speed manual/automatic or current 8-speed automatic",
        "Mazda BT-50 official specifications",
        (
            "https://www.mazda.com.au/mazda-news/"
            "mazda-bt-50-20-year-history/"
        ),
        "BT-50 figures span several generations and cab configurations. "
        "Confirm the listing's engine and cab before purchase.",
    ),
    FamilySpec(
        "Peugeot",
        r"^3008$",
        "Up to 12.6 km/L combined (official regional NEDC figure)",
        "4",
        "1.6 L turbo gasoline family reference",
        "5",
        "6-speed automatic",
        "Peugeot 3008 official brochure",
        (
            "https://ksa.peugeot.com/content/dam/peugeot/saudi_arabia/"
            "brochure/PEUGEOT-Passenger-CAR-Brochure-3008_eng.pdf"
        ),
        "Official regional specifications; earlier model years and imported "
        "engines can differ.",
    ),
    FamilySpec(
        "Suzuki",
        r"^Celerio$",
        "Up to 5.0 L/100 km (official regional model-family figure)",
        "3",
        "1.0 L",
        "5",
        "5-speed manual or Auto Gear Shift",
        "Suzuki Global Celerio specification",
        "https://www.globalsuzuki.com/globalnews/2021/1110.html",
        "Celerio figures vary by generation and destination market.",
    ),
    FamilySpec(
        "McLaren",
        r"^570S$",
        "26.6 UK mpg / 10.7 L/100 km combined (official NEDC reference)",
        "8",
        "3.8 L",
        "2",
        "7-speed Seamless Shift dual-clutch gearbox",
        "McLaren 570S official specifications",
        "https://cars.mclaren.com/gb-en/sports-series/570s/interior",
        "Manufacturer comparison figure; real driving and Spider/Coupe trim "
        "can differ.",
    ),
    FamilySpec(
        "BYD",
        r"^Seagull$",
        "300–380 km NEDC range (sold globally as DOLPHIN MINI)",
        "N/A (electric)",
        "N/A (electric)",
        "4–5 depending on market version",
        "Single-speed electric drive",
        "BYD DOLPHIN MINI official specification",
        (
            "https://www.byd.com/content/dam/byd-site/america-public/"
            "flyer/BYD-DOLPHIN-MINI-flyer-ES-20231116.pdf"
        ),
        "BYD Seagull is marketed as DOLPHIN MINI in several export markets.",
    ),
    FamilySpec(
        "BYD",
        r"^Atto 3$",
        "345–420 km WLTP range",
        "N/A (electric)",
        "N/A (electric)",
        "5",
        "Single-speed electric drive",
        "BYD ATTO 3 official brochure",
        (
            "https://www.byd.com/content/dam/byd-site/au/product/atto3/"
            "new-added%282024%29/24_atto3/"
            "20240603_2024%20ATTO%203%20brochure.pdf"
        ),
        "Range depends on battery version; verify the Cambodian vehicle's "
        "battery capacity.",
    ),
    FamilySpec(
        "Honda",
        r"^e:N Series$",
        "Up to 545 km CLTC range for the e:NP2/e:NS2 family",
        "N/A (electric)",
        "N/A (electric)",
        "5",
        "Single-speed electric drive",
        "Honda Global e:NP2/e:NS2 release",
        (
            "https://global.honda/en/newsroom/news/2024/"
            "c240425aeng.html"
        ),
        "The official range uses China's CLTC cycle. Confirm whether the "
        "listing is e:NP2 or e:NS2 and verify battery trim.",
    ),
    FamilySpec(
        "XPeng",
        r"^X9$",
        "535–580 km WLTP range",
        "N/A (electric)",
        "N/A (electric)",
        "7",
        "Single-speed electric drive",
        "XPENG X9 official configurator",
        "https://store.xpeng.com/au/configurator/New_X9",
        "Official export-market range depends on battery and drive version.",
    ),
    FamilySpec(
        "Zeekr",
        r"^9X$",
        "Up to 380 km CLTC pure-electric range",
        "4",
        "2.0 L turbo gasoline hybrid engine",
        "6",
        "SEA Super Hybrid electric-drive system",
        "ZEEKR 9X official release",
        "https://www.zeekrgroup.com/en/news/202509291",
        "The 9X is a plug-in hybrid, not a battery-only EV. Official range "
        "uses China's CLTC cycle and varies by battery/trim.",
    ),
    FamilySpec(
        "Ford",
        r"^Ranger(?: Wildtrak)?$",
        "8.9–10.7 km/L in official regional diesel tests",
        "4 or 6",
        "2.0 L or 3.0 L diesel",
        "5 for double-cab passenger configuration",
        "6-speed manual/automatic or 10-speed automatic",
        "Ford Ranger official specifications",
        (
            "https://media.ford.com/content/dam/fordmedia/img/"
            "MiddleEast/2023/15May/"
            "Next%20Generation%20Ranger%20Spec%20Sheet%20x%20English%20%281%29.pdf"
        ),
        "Ranger specifications vary by cab, engine, and market. Confirm that "
        "the listing is the double-cab version.",
    ),
]


def is_missing(value: object) -> bool:
    text = str(value).strip()
    lower = text.lower()
    return (
        not text
        or lower in {"nan", "none", "not verified"}
        or lower.startswith("official figure unavailable:")
        or lower.startswith("range unavailable:")
        or lower.startswith("varies by engine;")
        or lower.startswith("varies by trim;")
        or "exact model/trim not identified" in lower
        or "exact seating layout not identified" in lower
        or "exact cab configuration not identified" in lower
    )


def is_missing_for(field: str, value: object, listing: pd.Series) -> bool:
    if is_missing(value):
        return True
    non_electric = str(listing.get("fuel_type")) != "Electricity"
    text = str(value).strip().lower()
    return non_electric and (
        "n/a (electric)" in text
        or (field == "transmission" and "single-speed electric" in text)
    )


def seat_fallback(listing: pd.Series) -> str:
    body_type = str(listing.get("body_type", "vehicle"))
    ranges = {
        "Convertible": "Usually 2–4; exact model/trim not identified",
        "Coupe": "Usually 2–5; exact model/trim not identified",
        "Hatchback": "Usually 4–5; exact model/trim not identified",
        "MPV": "Usually 5–11; exact seating layout not identified",
        "Pickup": "Usually 2–5; exact cab configuration not identified",
        "Sedan": "Usually 4–5; exact model/trim not identified",
        "Sports": "Usually 2–4; exact model/trim not identified",
        "SUV": "Usually 5–8; exact model/trim not identified",
    }
    return ranges.get(
        body_type,
        "Capacity varies; exact model/trim not identified by seller",
    )


def explicit_fallback(field: str, listing: pd.Series) -> str:
    electric = str(listing.get("fuel_type")) == "Electricity"
    if field == "fuel_economy":
        if electric:
            return (
                "Range unavailable: seller did not identify the exact "
                "battery/model version"
            )
        return (
            "Official figure unavailable: seller did not identify the exact "
            "engine/trim"
        )
    if field == "cylinders":
        return (
            "N/A (electric)"
            if electric
            else "Varies by engine; exact trim not identified"
        )
    if field == "displacement_l":
        return (
            "N/A (electric)"
            if electric
            else "Varies by engine; exact trim not identified"
        )
    if field == "seats":
        return seat_fallback(listing)
    if field == "transmission":
        return (
            "Single-speed electric drive"
            if electric
            else "Varies by trim; exact gearbox not identified"
        )
    raise KeyError(field)


def main() -> None:
    listings = pd.read_csv(
        CAMBODIA_SNAPSHOT_CSV,
        dtype={"listing_id": str},
    )
    specs = pd.read_csv(
        REFERENCE_SPECS_CSV,
        dtype={"listing_id": str},
    )
    listings_by_id = listings.set_index("listing_id", drop=False)
    fields = [
        "fuel_economy",
        "cylinders",
        "displacement_l",
        "seats",
        "transmission",
    ]
    family_rows = 0
    fallback_values = 0

    for index, spec in specs.iterrows():
        listing_id = str(spec["listing_id"])
        if listing_id not in listings_by_id.index:
            continue
        listing = listings_by_id.loc[listing_id]
        if isinstance(listing, pd.DataFrame):
            listing = listing.iloc[0]

        original_confidence = str(spec.get("spec_confidence", "")).strip()
        family = next(
            (candidate for candidate in FAMILY_SPECS if candidate.matches(listing)),
            None,
        )
        if family is not None:
            changed = False
            for field in fields:
                if is_missing_for(field, specs.at[index, field], listing):
                    specs.at[index, field] = getattr(family, field)
                    changed = True
            if changed:
                specs.at[index, "spec_source_name"] = family.source_name
                specs.at[index, "spec_source_url"] = family.source_url
                specs.at[index, "spec_confidence"] = (
                    "Official manufacturer model-family range"
                )
                specs.at[index, "spec_note"] = family.note
                family_rows += 1

        for field in fields:
            if is_missing_for(field, specs.at[index, field], listing):
                specs.at[index, field] = explicit_fallback(field, listing)
                fallback_values += 1

        if any(
            "exact" in str(specs.at[index, field]).lower()
            and "not identified" in str(specs.at[index, field]).lower()
            for field in fields
        ):
            if original_confidence in {
                "",
                "Listing does not identify exact trim",
                "Seller did not identify exact configuration",
            }:
                specs.at[index, "spec_confidence"] = (
                    "Seller did not identify exact configuration"
                )
                specs.at[index, "spec_note"] = (
                    "The marketplace listing lacks enough model, engine, "
                    "battery, or trim detail for a responsible numeric claim. "
                    "The card states the limitation instead of inventing data."
                )

    specs.to_csv(REFERENCE_SPECS_CSV, index=False)
    print(
        f"Applied official manufacturer family references to {family_rows} "
        f"listings and replaced {fallback_values} empty display values with "
        "explicit configuration limitations."
    )


if __name__ == "__main__":
    main()
