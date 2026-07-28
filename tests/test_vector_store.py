import unittest

import numpy as np

from carwise.models import Chunk
from carwise.vector_store import VectorIndex, inferred_price_range


class KeywordEmbedder:
    vocabulary = ("electric", "gasoline", "suv", "toyota", "kia")

    @property
    def model_name(self) -> str:
        return "test-keyword-embedder"

    def encode(self, texts):
        return np.asarray(
            [
                [float(text.lower().count(word)) for word in self.vocabulary]
                for text in texts
            ],
            dtype=np.float32,
        )


class VectorStoreTests(unittest.TestCase):
    def test_search_and_metadata_filter(self) -> None:
        chunks = [
            Chunk(
                "c1",
                "d1",
                "Electric SUV Cambodia",
                "electric suv listed in Cambodia",
                "Khmer24",
                "https://example.com/1",
                {
                    "fuel_type": "Electricity",
                    "body_type": "SUV",
                    "price_usd": 28800,
                    "condition": "Used",
                    "make": "Lexus",
                    "province": "Phnom Penh",
                },
            ),
            Chunk(
                "c2",
                "d2",
                "Gas Car",
                "gasoline compact car",
                "Khmer24",
                "https://example.com/2",
                {
                    "fuel_type": "Gasoline",
                    "body_type": "Sedan",
                    "price_usd": 5350,
                    "condition": "Used",
                    "make": "Toyota",
                    "province": "Phnom Penh",
                },
            ),
        ]
        embedder = KeywordEmbedder()
        index = VectorIndex.build(chunks, embedder)

        results = index.search("electric suv", embedder, top_k=1)
        self.assertEqual("Electric SUV Cambodia", results[0].chunk.document_title)

        filtered = index.search(
            "car",
            embedder,
            top_k=2,
            metadata_filters={"fuel_type": {"Gasoline"}},
        )
        self.assertEqual(["Gas Car"], [result.chunk.document_title for result in filtered])

    def test_budget_is_a_strict_filter(self) -> None:
        chunks = [
            Chunk(
                "c1",
                "d1",
                "Toyota under budget",
                "Toyota used car",
                "Khmer24",
                "https://example.com/1",
                {
                    "make": "Toyota",
                    "model": "Prius",
                    "price_usd": 17500,
                    "condition": "Used",
                },
            ),
            Chunk(
                "c2",
                "d2",
                "Toyota other model",
                "Toyota used car",
                "Khmer24",
                "https://example.com/2",
                {
                    "make": "Toyota",
                    "model": "Highlander",
                    "price_usd": 15000,
                    "condition": "Used",
                },
            ),
        ]
        embedder = KeywordEmbedder()
        index = VectorIndex.build(chunks, embedder)

        results = index.search("Toyota Prius under $20,000", embedder, top_k=5)

        self.assertEqual(
            ["Toyota under budget"],
            [result.chunk.document_title for result in results],
        )

    def test_sports_car_query_filters_to_sports_body_type(self) -> None:
        chunks = [
            Chunk(
                "sports",
                "sports-doc",
                "Sports car",
                "Performance vehicle",
                "Khmer24",
                "https://example.com/sports",
                {"body_type": "Sports", "price_usd": 45000},
            ),
            Chunk(
                "sedan",
                "sedan-doc",
                "Family sedan",
                "Family vehicle",
                "Khmer24",
                "https://example.com/sedan",
                {"body_type": "Sedan", "price_usd": 20000},
            ),
        ]
        embedder = KeywordEmbedder()
        index = VectorIndex.build(chunks, embedder)

        results = index.search("sports cars under $50,000", embedder, top_k=5)

        self.assertEqual(
            ["Sports car"],
            [result.chunk.document_title for result in results],
        )

    def test_fuel_word_is_not_mistaken_for_a_model_filter(self) -> None:
        chunks = [
            Chunk(
                "prius",
                "prius-doc",
                "2012 Toyota Prius",
                "2012 Toyota Prius hybrid",
                "Khmer24",
                "https://example.com/prius",
                {
                    "year": 2012,
                    "make": "Toyota",
                    "model": "Prius",
                    "fuel_type": "Hybrid",
                    "price_usd": 15600,
                },
            ),
            Chunk(
                "ambiguous",
                "ambiguous-doc",
                "2013 Hybrid listing",
                "2013 hybrid vehicle",
                "Khmer24",
                "https://example.com/hybrid",
                {
                    "year": 2013,
                    "make": "Not identified",
                    "model": "Hybrid",
                    "fuel_type": "Hybrid",
                    "price_usd": 12000,
                },
            ),
        ]
        embedder = KeywordEmbedder()
        index = VectorIndex.build(chunks, embedder)

        results = index.search(
            "Show 2012 hybrid cars under $18,000",
            embedder,
            top_k=5,
        )

        self.assertEqual(
            ["2012 Toyota Prius"],
            [result.chunk.document_title for result in results],
        )

    def test_parses_common_price_ranges(self) -> None:
        self.assertEqual((None, 20000), inferred_price_range("SUV under $20,000"))
        self.assertEqual((10000, 20000), inferred_price_range("between 10k and 20k"))


if __name__ == "__main__":
    unittest.main()
