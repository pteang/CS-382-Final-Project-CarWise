import unittest

import pandas as pd

from scripts.complete_specs import FAMILY_SPECS, explicit_fallback


class CompleteSpecsTests(unittest.TestCase):
    def test_manufacturer_family_reference_matches_canonical_model(self) -> None:
        listing = pd.Series({"make": "Toyota", "model": "Alphard"})
        reference = next(item for item in FAMILY_SPECS if item.matches(listing))

        self.assertEqual("Toyota Global Newsroom", reference.source_name)
        self.assertIn("6–8", reference.seats)
        self.assertIn("2.5 L", reference.displacement_l)

    def test_unidentified_configuration_uses_explicit_nonblank_values(self) -> None:
        listing = pd.Series(
            {
                "fuel_type": "Gasoline",
                "body_type": "SUV",
            }
        )

        for field in [
            "fuel_economy",
            "cylinders",
            "displacement_l",
            "seats",
            "transmission",
        ]:
            value = explicit_fallback(field, listing)
            self.assertTrue(value.strip())
            self.assertNotIn("not verified", value.lower())


if __name__ == "__main__":
    unittest.main()
