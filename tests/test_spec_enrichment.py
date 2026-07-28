import unittest

import pandas as pd

from scripts.enrich_specs import (
    efficiency_summary,
    model_match_score,
    normalize_model,
)


class SpecEnrichmentTests(unittest.TestCase):
    def test_model_matcher_accepts_specific_model_variants(self) -> None:
        self.assertEqual("488 gtb", normalize_model("2018 488 GTB 3.9 V8"))
        self.assertGreaterEqual(
            model_match_score("488 GTB", "Ferrari 488 GTB"),
            0.9,
        )
        self.assertGreaterEqual(
            model_match_score("M240i", "M240i xDrive Coupe"),
            0.9,
        )

    def test_model_matcher_rejects_generic_body_styles(self) -> None:
        self.assertEqual(0.0, model_match_score("Coupe", "M240i xDrive Coupe"))
        self.assertEqual(0.0, model_match_score("SUV", "DBX"))
        self.assertEqual(0.0, model_match_score("Sedan", "Camry"))

    def test_efficiency_summary_reports_configuration_range(self) -> None:
        rows = pd.DataFrame(
            {
                "comb08": [22, 25],
                "range": [0, 0],
            }
        )
        self.assertEqual(
            "22–25 mpg combined (EPA)",
            efficiency_summary(rows, electric=False),
        )


if __name__ == "__main__":
    unittest.main()
