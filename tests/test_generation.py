import unittest

from carwise.generation import (
    GenerationError,
    RetrievalPreviewGenerator,
    build_grounded_prompt,
    extract_openai_output,
    require_source_citation,
)
from carwise.models import Chunk, RetrievedChunk


class GenerationTests(unittest.TestCase):
    def test_prompt_labels_sources(self) -> None:
        source = RetrievedChunk(
            Chunk(
                chunk_id="c1",
                document_id="d1",
                document_title="Example EV",
                text="EPA range is 300 miles.",
                source_name="EPA",
                source_url="https://example.com",
                metadata={"section": "Electric details"},
            ),
            0.8,
        )
        prompt = build_grounded_prompt("Which EV?", [source], "Concise")
        self.assertIn("[S1]", prompt)
        self.assertIn("using only the retrieved evidence", prompt.lower())

    def test_extracts_responses_api_text(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Grounded answer [S1]."}],
                }
            ]
        }
        self.assertEqual("Grounded answer [S1].", extract_openai_output(payload))

    def test_local_summary_reports_evidence_fields(self) -> None:
        source = RetrievedChunk(
            Chunk(
                chunk_id="c1",
                document_id="d1",
                document_title="2020 Lexus UX 300e",
                text="Cambodian marketplace evidence.",
                source_name="Khmer24",
                source_url="https://example.com",
                metadata={
                    "price_usd": 28800,
                    "condition": "Used",
                    "body_type": "SUV",
                    "fuel_type": "Electricity",
                    "location": "Tuol Kouk, Phnom Penh",
                    "fuel_economy": "16.8 kWh/100 km WLTP",
                    "cylinders": "N/A (electric)",
                    "displacement_l": "N/A (electric)",
                    "seats": "5",
                    "transmission": "Single-speed electric drive",
                },
            ),
            0.8,
        )
        answer = RetrievalPreviewGenerator().generate(
            "Recommend an electric SUV", [source], "Concise"
        )
        self.assertIn("$28,800 asking price", answer)
        self.assertIn("Electricity", answer)
        self.assertIn("16.8 kWh/100 km WLTP", answer)
        self.assertIn("Single-speed electric drive", answer)
        self.assertIn("seller-provided", answer)
        self.assertNotIn("matched the query", answer)

    def test_local_llm_requires_citations(self) -> None:
        with self.assertRaisesRegex(GenerationError, "did not cite"):
            require_source_citation("Uncited answer")
        self.assertEqual(
            "Grounded answer [S1].",
            require_source_citation("Grounded answer [S1]."),
        )


if __name__ == "__main__":
    unittest.main()
