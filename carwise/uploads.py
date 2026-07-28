from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from .models import Document


MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 200_000
MAX_TOTAL_UPLOAD_BYTES = 2_000_000


def uploaded_text_documents(
    files: Iterable[tuple[str, bytes]],
) -> list[Document]:
    """Validate uploaded UTF-8 text files and convert them into documents."""
    uploaded = list(files)
    if not uploaded:
        raise ValueError("Upload at least one .txt file.")
    if len(uploaded) > MAX_UPLOAD_FILES:
        raise ValueError(f"Upload no more than {MAX_UPLOAD_FILES} files at once.")

    total_bytes = sum(len(content) for _, content in uploaded)
    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
        raise ValueError("The uploaded corpus is larger than the 2 MB total limit.")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for original_name, content in uploaded:
        filename = Path(original_name).name.strip()
        if not filename.lower().endswith(".txt"):
            raise ValueError(f"{filename or 'Unnamed file'} must be a .txt file.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{filename} is larger than the 200 KB file limit.")
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{filename} must use UTF-8 text encoding.") from exc
        if not text:
            raise ValueError(f"{filename} is empty.")

        digest = hashlib.sha256(
            filename.encode("utf-8") + b"\0" + content
        ).hexdigest()[:20]
        document_id = f"upload-{digest}"
        if document_id in seen_ids:
            continue
        seen_ids.add(document_id)

        title = Path(filename).stem.replace("_", " ").strip() or filename
        documents.append(
            Document(
                document_id=document_id,
                title=title,
                text=text,
                source_name=f"Uploaded text: {filename}",
                source_url="",
                metadata={
                    "corpus_type": "uploaded",
                    "uploaded_filename": filename,
                },
            )
        )
    return documents
