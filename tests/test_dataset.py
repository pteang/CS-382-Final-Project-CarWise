import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from carwise.cambodia_dataset import (
    ensure_cambodia_corpus,
    load_listing_snapshot,
    row_to_document,
)
from carwise.config import (
    CAMBODIA_SNAPSHOT_CSV,
    LISTING_IMAGES_CSV,
    REFERENCE_SPECS_CSV,
)
from carwise.models import Document


class DatasetTests(unittest.TestCase):
    def test_included_snapshot_has_broad_electric_vehicle_coverage(self) -> None:
        rows = load_listing_snapshot(
            CAMBODIA_SNAPSHOT_CSV,
            REFERENCE_SPECS_CSV,
            LISTING_IMAGES_CSV,
        )
        electric = rows[rows["fuel_type"].eq("Electricity")]
        performance = rows[
            rows["body_type"].isin(["Sports", "Coupe", "Convertible"])
        ]

        self.assertGreaterEqual(len(rows), 600)
        self.assertGreaterEqual(rows["make"].nunique(), 45)
        self.assertGreaterEqual(len(electric), 60)
        self.assertGreaterEqual(
            len(rows[rows["fuel_type"].eq("Gasoline")]),
            450,
        )
        self.assertGreaterEqual(len(performance), 60)
        self.assertTrue(electric["image_url"].str.startswith("https://").all())
        self.assertTrue(electric["cylinders"].eq("N/A (electric)").all())
        self.assertTrue(electric["displacement_l"].eq("N/A (electric)").all())
        for field in [
            "fuel_economy",
            "cylinders",
            "displacement_l",
            "seats",
            "transmission",
        ]:
            values = rows[field].fillna("").astype(str).str.strip()
            self.assertTrue(values.ne("").all(), field)
            self.assertFalse(
                values.str.contains("not verified", case=False).any(),
                field,
            )

    def test_validates_and_converts_cambodian_listing(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "listing_id": "123",
                    "title": "Toyota Prius 2013",
                    "make": "Toyota",
                    "model": "Prius",
                    "model_year": 2013,
                    "price_usd": 17500,
                    "condition": "Used",
                    "registration": "Tax Paper",
                    "location": "Por Sen Chey, Phnom Penh",
                    "province": "Phnom Penh",
                    "body_type": "Hatchback",
                    "fuel_type": "Hybrid",
                    "observed_at": "2026-07-26",
                    "source_url": "https://example.com/listing-123",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "listings.csv"
            specs_path = Path(directory) / "specs.csv"
            images_path = Path(directory) / "images.csv"
            rows.to_csv(path, index=False)
            pd.DataFrame(
                [
                    {
                        "listing_id": "123",
                        "fuel_economy": "48 mpg combined",
                        "cylinders": "4",
                        "displacement_l": "1.8 L",
                        "seats": "5",
                        "transmission": "e-CVT",
                        "spec_source_name": "FuelEconomy.gov",
                        "spec_source_url": "https://example.com/specs-123",
                        "spec_confidence": "Exact model-year reference",
                        "spec_note": "Verify the imported trim.",
                    }
                ]
            ).to_csv(specs_path, index=False)
            pd.DataFrame(
                [
                    {
                        "listing_id": "123",
                        "image_url": "https://example.com/prius.jpg",
                    }
                ]
            ).to_csv(images_path, index=False)
            curated = load_listing_snapshot(path, specs_path, images_path)

        document = row_to_document(curated.iloc[0])
        self.assertIn("$17,500 USD", document.text)
        self.assertIn("48 mpg combined", document.text)
        self.assertIn("Seating capacity: 5", document.text)
        self.assertIn("Transmission: e-CVT", document.text)
        self.assertIn("seller-provided", document.text)
        self.assertEqual("Phnom Penh", document.metadata["province"])
        self.assertEqual(17500, document.metadata["price_usd"])
        self.assertEqual("1.8 L", document.metadata["displacement_l"])
        self.assertEqual(
            "https://example.com/prius.jpg",
            document.metadata["image_url"],
        )
        self.assertIn("Khmer24 Cambodia", document.source_name)

    def test_rebuilds_legacy_document_folder(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "listing_id": "123",
                    "title": "Toyota Prius 2013",
                    "make": "Toyota",
                    "model": "Prius",
                    "model_year": 2013,
                    "price_usd": 17500,
                    "condition": "Used",
                    "registration": "Tax Paper",
                    "location": "Por Sen Chey, Phnom Penh",
                    "province": "Phnom Penh",
                    "body_type": "Hatchback",
                    "fuel_type": "Hybrid",
                    "observed_at": "2026-07-26",
                    "source_url": "https://example.com/listing-123",
                }
            ]
        )
        legacy = Document(
            document_id="epa-legacy",
            title="Legacy EPA vehicle",
            text="Old corpus",
            source_name="EPA",
            source_url="https://example.com/legacy",
            metadata={"year": 2026, "make": "Example"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_csv = root / "snapshot.csv"
            curated_csv = root / "curated.csv"
            documents_dir = root / "documents"
            documents_dir.mkdir()
            rows.to_csv(source_csv, index=False)
            (documents_dir / "epa-legacy.json").write_text(
                json.dumps(legacy.to_dict()),
                encoding="utf-8",
            )

            documents, rebuilt = ensure_cambodia_corpus(
                source_csv,
                curated_csv,
                documents_dir,
            )
            remaining_names = {
                path.name for path in documents_dir.glob("*.json")
            }

        self.assertTrue(rebuilt)
        self.assertEqual(1, len(documents))
        self.assertEqual("Used", documents[0].metadata["condition"])
        self.assertFalse(any(name.startswith("epa-") for name in remaining_names))


if __name__ == "__main__":
    unittest.main()
