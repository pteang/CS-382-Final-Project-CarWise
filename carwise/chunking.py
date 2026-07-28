from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Chunk, Document


HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_long_section(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a long section at sentence boundaries with a small text overlap."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    if len(sentences) == 1:
        step = max(1, max_chars - overlap_chars)
        return [text[start : start + max_chars] for start in range(0, len(text), step)]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            chunk = " ".join(current)
            chunks.append(chunk)
            overlap: list[str] = []
            overlap_length = 0
            for prior in reversed(current):
                if overlap and overlap_length + len(prior) + 1 > overlap_chars:
                    break
                overlap.insert(0, prior)
                overlap_length += len(prior) + 1
            current = overlap
            current_length = len(" ".join(current))
        current.append(sentence)
        current_length += len(sentence) + (1 if current_length else 0)

    if current:
        chunks.append(" ".join(current))
    return chunks


def section_aware_chunks(
    document: Document,
    *,
    max_chars: int = 1_100,
    overlap_chars: int = 160,
) -> list[Chunk]:
    """Create chunks that preserve Markdown section headings."""
    sections: list[tuple[str, list[str]]] = []
    heading = document.title
    body: list[str] = []

    for raw_line in document.text.splitlines():
        line = raw_line.strip()
        match = HEADING_RE.match(line)
        if match:
            if body:
                sections.append((heading, body))
            heading = match.group(2)
            body = []
        elif line:
            body.append(line)
    if body:
        sections.append((heading, body))

    chunks: list[Chunk] = []
    sequence = 0

    # Vehicle profiles are intentionally short. An overview chunk keeps constraints
    # such as class, fuel, drivetrain, and efficiency together for multi-criteria
    # recommendation queries, while section chunks retain precise citations.
    overview_text = " ".join(
        line.lstrip("#").strip()
        for line in document.text.splitlines()
        if line.strip()
    )
    if overview_text:
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::chunk-{sequence}",
                document_id=document.document_id,
                document_title=document.title,
                text=overview_text,
                source_name=document.source_name,
                source_url=document.source_url,
                metadata={
                    **document.metadata,
                    "section": "Complete vehicle profile",
                    "chunk_sequence": sequence,
                },
            )
        )
        sequence += 1

    for section_heading, lines in sections:
        section_text = f"{document.title}. Section: {section_heading}. {' '.join(lines)}"
        for piece in _split_long_section(section_text, max_chars, overlap_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}::chunk-{sequence}",
                    document_id=document.document_id,
                    document_title=document.title,
                    text=piece,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    metadata={
                        **document.metadata,
                        "section": section_heading,
                        "chunk_sequence": sequence,
                    },
                )
            )
            sequence += 1
    return chunks


def chunk_documents(documents: Iterable[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(section_aware_chunks(document))
    return chunks
