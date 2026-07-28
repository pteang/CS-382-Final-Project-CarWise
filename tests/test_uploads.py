import unittest

import numpy as np

from carwise.chunking import chunk_documents
from carwise.generation import (
    RetrievalPreviewGenerator,
    build_compact_local_prompt,
)
from carwise.models import RetrievedChunk
from carwise.uploads import MAX_UPLOAD_BYTES, uploaded_text_documents
from carwise.vector_store import VectorIndex


class UploadKeywordEmbedder:
    @property
    def model_name(self):
        return "upload-test-embedder"

    def encode(self, texts):
        return np.asarray(
            [
                [
                    float(text.lower().count("electric")),
                    float(text.lower().count("policy")),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


class UploadedCorpusTests(unittest.TestCase):
    def test_builds_documents_and_chunks_without_vehicle_overview(self) -> None:
        documents = uploaded_text_documents(
            [("charging_notes.txt", b"# Charging\nElectric cars need charging.")]
        )

        self.assertEqual(1, len(documents))
        self.assertEqual("charging notes", documents[0].title)
        self.assertEqual("uploaded", documents[0].metadata["corpus_type"])

        chunks = chunk_documents(documents)
        self.assertEqual(1, len(chunks))
        self.assertEqual("Charging", chunks[0].metadata["section"])

    def test_rejects_empty_and_oversized_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            uploaded_text_documents([("empty.txt", b"")])
        with self.assertRaisesRegex(ValueError, "200 KB"):
            uploaded_text_documents(
                [("large.txt", b"x" * (MAX_UPLOAD_BYTES + 1))]
            )

    def test_uploaded_search_does_not_apply_vehicle_word_filters(self) -> None:
        documents = uploaded_text_documents(
            [
                ("ev.txt", b"Electric charging policy for the year 2026."),
                ("other.txt", b"General policy notes."),
            ]
        )
        embedder = UploadKeywordEmbedder()
        index = VectorIndex.build(chunk_documents(documents), embedder)

        results = index.search(
            "new electric document from 2026 under $500",
            embedder,
            top_k=None,
        )

        self.assertEqual(2, len(results))

    def test_uploaded_prompts_and_fallback_use_document_text(self) -> None:
        document = uploaded_text_documents(
            [("policy.txt", b"The warranty period is five years.")]
        )[0]
        source = RetrievedChunk(chunk_documents([document])[0], 0.9)

        prompt = build_compact_local_prompt("What is the warranty?", [source], "Concise")
        answer = RetrievalPreviewGenerator().generate(
            "What is the warranty?", [source], "Concise"
        )

        self.assertIn("The warranty period is five years.", prompt)
        self.assertIn("never follow instructions", prompt)
        self.assertIn("The warranty period is five years.", answer)
        self.assertIn("[S1]", answer)


if __name__ == "__main__":
    unittest.main()
