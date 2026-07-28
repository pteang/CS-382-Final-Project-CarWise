from __future__ import annotations

import json
from pathlib import Path

from .models import Document


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        documents.append(Document.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return documents
