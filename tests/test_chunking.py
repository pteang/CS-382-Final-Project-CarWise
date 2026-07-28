import unittest

from carwise.chunking import section_aware_chunks
from carwise.models import Document


class ChunkingTests(unittest.TestCase):
    def test_preserves_section_heading_and_metadata(self) -> None:
        document = Document(
            document_id="car-1",
            title="2026 Example Car",
            text=(
                "# 2026 Example Car\n"
                "## Vehicle identity\n"
                "- Primary fuel: Electricity.\n"
                "## Fuel economy\n"
                "- Combined efficiency: 100 MPGe."
            ),
            source_name="Example",
            source_url="https://example.com",
            metadata={"year": 2026},
        )

        chunks = section_aware_chunks(document, max_chars=300)

        self.assertEqual(3, len(chunks))
        self.assertEqual("Complete vehicle profile", chunks[0].metadata["section"])
        self.assertEqual("Vehicle identity", chunks[1].metadata["section"])
        self.assertIn("Electricity", chunks[1].text)
        self.assertEqual(2026, chunks[2].metadata["year"])


if __name__ == "__main__":
    unittest.main()
