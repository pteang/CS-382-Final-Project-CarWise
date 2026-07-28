import unittest

from carwise.models import Chunk, RetrievedChunk
from carwise.pipeline import RAGPipeline, unsupported_requirements


class NeverCalledIndex:
    def search(self, *args, **kwargs):
        raise AssertionError("Unsupported questions must be rejected before retrieval.")


class NeverCalledGenerator:
    @property
    def provider_name(self):
        return "test"

    def generate(self, *args, **kwargs):
        raise AssertionError("Unsupported questions must be rejected before generation.")


class AllMatchesIndex:
    def __init__(self) -> None:
        self.requested_top_k = "not-called"

    def search(self, *args, **kwargs):
        self.requested_top_k = kwargs["top_k"]
        return [
            RetrievedChunk(
                Chunk(
                    f"chunk-{index}",
                    f"document-{index}",
                    f"Car {index}",
                    "Cambodian car listing",
                    "Khmer24",
                    f"https://example.com/{index}",
                    {},
                ),
                0.9 - (index * 0.1),
            )
            for index in range(5)
        ]


class CapturingGenerator:
    def __init__(self) -> None:
        self.sources = []

    @property
    def provider_name(self):
        return "test"

    def generate(self, query, sources, answer_mode):
        self.sources = sources
        return "Grounded answer [S1]"


class PipelineTests(unittest.TestCase):
    def test_detects_safety_and_reliability_but_not_purchase_price(self) -> None:
        detected = unsupported_requirements(
            "What is the safest and most reliable car under $20,000?"
        )
        self.assertEqual(
            ["crash safety", "reliability"],
            detected,
        )

    def test_rejects_unsupported_query_without_retrieval(self) -> None:
        pipeline = RAGPipeline(
            NeverCalledIndex(),
            object(),
            NeverCalledGenerator(),
        )
        result = pipeline.answer(
            "What is the safest car under $20,000?",
            top_k=5,
            minimum_similarity=0.2,
            answer_mode="Concise",
        )
        self.assertFalse(result.grounded)
        self.assertEqual([], result.sources)
        self.assertIn("No vehicle recommendation was produced", result.answer)
        self.assertIn("crash safety", result.answer)
        self.assertIn("seller asking price", result.answer)

    def test_returns_all_matches_but_limits_generation_evidence(self) -> None:
        index = AllMatchesIndex()
        generator = CapturingGenerator()
        pipeline = RAGPipeline(index, object(), generator)

        result = pipeline.answer(
            "Show me Cambodian cars",
            top_k=2,
            minimum_similarity=0.2,
            answer_mode="Concise",
        )

        self.assertIsNone(index.requested_top_k)
        self.assertEqual(5, len(result.sources))
        self.assertEqual(2, len(generator.sources))
        self.assertTrue(result.grounded)


if __name__ == "__main__":
    unittest.main()
