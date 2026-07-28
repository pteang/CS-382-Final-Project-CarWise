import unittest

from carwise.models import Document
from carwise.recommendations import similar_price_documents


def vehicle(document_id: str, price: int, make: str = "Toyota") -> Document:
    return Document(
        document_id=document_id,
        title=f"{make} vehicle",
        text="Vehicle",
        source_name="Marketplace",
        source_url=f"https://example.com/{document_id}",
        metadata={"price_usd": price, "make": make},
    )


class SimilarPriceTests(unittest.TestCase):
    def test_returns_closest_distinct_prices_and_excludes_matches(self) -> None:
        documents = [
            vehicle("selected", 17_000),
            vehicle("nearer", 17_200),
            vehicle("near", 16_500),
            vehicle("far", 20_000),
        ]

        recommendations = similar_price_documents(
            documents,
            17_000,
            exclude_document_ids={"selected"},
            limit=2,
        )

        self.assertEqual(["nearer", "near"], [item.document_id for item in recommendations])

    def test_respects_active_metadata_filters(self) -> None:
        documents = [
            vehicle("toyota", 17_100, "Toyota"),
            vehicle("kia", 17_050, "Kia"),
        ]

        recommendations = similar_price_documents(
            documents,
            17_000,
            metadata_filters={"make": {"Toyota"}},
        )

        self.assertEqual(["toyota"], [item.document_id for item in recommendations])


if __name__ == "__main__":
    unittest.main()
