from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import Document


def similar_price_documents(
    documents: Iterable[Document],
    target_price: int | float,
    *,
    exclude_document_ids: set[str] | None = None,
    metadata_filters: dict[str, set[Any]] | None = None,
    limit: int = 3,
) -> list[Document]:
    """Return distinct vehicles whose asking prices are closest to the target."""
    excluded = exclude_document_ids or set()
    filters = metadata_filters or {}
    candidates: list[Document] = []
    seen: set[str] = set()

    for document in documents:
        if document.document_id in excluded or document.document_id in seen:
            continue
        if any(
            allowed and document.metadata.get(key) not in allowed
            for key, allowed in filters.items()
        ):
            continue
        price = document.metadata.get("price_usd")
        if not isinstance(price, (int, float)) or price <= 0:
            continue
        seen.add(document.document_id)
        candidates.append(document)

    candidates.sort(
        key=lambda document: (
            abs(float(document.metadata["price_usd"]) - float(target_price)),
            float(document.metadata["price_usd"]),
            document.title,
        )
    )
    return candidates[: max(0, limit)]
