import unittest

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


if __name__ == "__main__":
    unittest.main()
