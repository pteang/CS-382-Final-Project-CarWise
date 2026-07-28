from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.chunking import chunk_documents  # noqa: E402
from carwise.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DOCUMENTS_DIR,
    INDEX_DIR,
)
from carwise.corpus import load_documents  # noqa: E402
from carwise.embeddings import SentenceTransformerEmbedder  # noqa: E402
from carwise.vector_store import VectorIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CarWise vector index.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise SystemExit("No documents found. Run scripts/prepare_dataset.py first.")
    chunks = chunk_documents(documents)
    print(f"Embedding {len(chunks)} chunks from {len(documents)} documents...")
    embedder = SentenceTransformerEmbedder(args.model)
    index = VectorIndex.build(chunks, embedder)
    index.save(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
