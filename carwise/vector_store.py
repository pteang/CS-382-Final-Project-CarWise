from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import Embedder, normalize_rows
from .models import Chunk, RetrievedChunk


MONEY_NUMBER = r"(\d[\d,]*(?:\.\d+)?)\s*([kK]?)"
MONEY_TOKEN = rf"\$?\s*{MONEY_NUMBER}"
AMBIGUOUS_MODEL_TERMS = {
    "convertible",
    "coupe",
    "diesel",
    "electric",
    "gasoline",
    "hatchback",
    "hybrid",
    "mpv",
    "pickup",
    "sedan",
    "sport",
    "sports",
    "suv",
}


def _money_value(number: str, suffix: str) -> float:
    value = float(number.replace(",", ""))
    return value * 1_000 if suffix.lower() == "k" else value


def inferred_price_range(query: str) -> tuple[float | None, float | None]:
    """Extract common natural-language USD purchase-price constraints."""
    query_lower = query.lower()
    between = re.search(
        rf"\bbetween\s+{MONEY_TOKEN}\s+(?:and|to)\s+{MONEY_TOKEN}",
        query_lower,
    )
    if between:
        first = _money_value(between.group(1), between.group(2))
        second = _money_value(between.group(3), between.group(4))
        return min(first, second), max(first, second)

    maximum = re.search(
        rf"(?:under|below|less than|up to|maximum|max|budget(?:\s+of)?|"
        rf"no more than)\s*{MONEY_TOKEN}",
        query_lower,
    )
    minimum = re.search(
        rf"(?:over|above|more than|at least|minimum|min)\s*{MONEY_TOKEN}",
        query_lower,
    )
    max_value = (
        _money_value(maximum.group(1), maximum.group(2)) if maximum else None
    )
    min_value = (
        _money_value(minimum.group(1), minimum.group(2)) if minimum else None
    )

    # A dollar-denominated amount in a recommendation query is normally a budget.
    if max_value is None:
        dollar_amount = re.search(rf"\$\s*{MONEY_NUMBER}", query_lower)
        if dollar_amount:
            max_value = _money_value(dollar_amount.group(1), dollar_amount.group(2))
    return min_value, max_value


def corpus_fingerprint(chunks: list[Chunk], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


class VectorIndex:
    """A small persistent in-memory cosine-similarity vector store."""

    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str,
        fingerprint: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")
        self.chunks = chunks
        self.embeddings = normalize_rows(embeddings)
        self.model_name = model_name
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        if not chunks:
            raise ValueError("Cannot build an index with no chunks.")
        embeddings = embedder.encode([chunk.text for chunk in chunks])
        return cls(
            chunks,
            embeddings,
            embedder.model_name,
            corpus_fingerprint(chunks, embedder.model_name),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        payload = {
            "model_name": self.model_name,
            "fingerprint": self.fingerprint,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        (directory / "index.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        embeddings = np.load(directory / "embeddings.npy")
        chunks = [Chunk.from_dict(value) for value in payload["chunks"]]
        return cls(
            chunks,
            embeddings,
            payload["model_name"],
            payload["fingerprint"],
        )

    def is_current(self, chunks: list[Chunk], model_name: str) -> bool:
        return self.fingerprint == corpus_fingerprint(chunks, model_name)

    def search(
        self,
        query: str,
        embedder: Embedder,
        *,
        top_k: int = 5,
        metadata_filters: dict[str, set[Any]] | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        if embedder.model_name != self.model_name:
            raise ValueError(
                f"Index uses {self.model_name!r}, but query embedder uses "
                f"{embedder.model_name!r}."
            )

        query_vector = normalize_rows(embedder.encode([query]))[0]
        scores = np.sum(
            np.asarray(self.embeddings, dtype=np.float64)
            * np.asarray(query_vector, dtype=np.float64)[None, :],
            axis=1,
        )

        query_lower = query.lower()
        explicit_filters = metadata_filters or {}
        inferred_years = {
            int(value)
            for value in re.findall(r"\b(?:19|20)\d{2}\b", query_lower)
        }

        inferred_fuels: set[str] = set()
        if re.search(r"\bhybrid\b", query_lower) and "plug-in" not in query_lower:
            inferred_fuels.add("Hybrid")
        if re.search(r"plug[- ]?in hybrid|\bphev\b", query_lower):
            inferred_fuels.add("Plug-in Hybrid")
        if re.search(r"\bgas(?:oline)?\b|\bpetrol\b", query_lower):
            inferred_fuels.add("Gasoline")
        if "diesel" in query_lower:
            inferred_fuels.add("Diesel")
        if re.search(r"\belectric\b|\bev\b", query_lower):
            inferred_fuels.add("Electricity")

        body_terms: list[str] = []
        if re.search(r"\bsuvs?\b|sport utility", query_lower):
            body_terms.append("SUV")
        if re.search(r"\bpickups?\b|\btrucks?\b", query_lower):
            body_terms.append("Pickup")
        if re.search(r"\bsedans?\b", query_lower):
            body_terms.append("Sedan")
        if re.search(r"\bhatchbacks?\b", query_lower):
            body_terms.append("Hatchback")
        if re.search(r"\bminivans?\b|\bmpvs?\b", query_lower):
            body_terms.append("MPV")
        if re.search(r"\bcoupes?\b", query_lower):
            body_terms.append("Coupe")
        if re.search(r"\bsports?\s+cars?\b|\bsupercars?\b", query_lower):
            body_terms.append("Sports")
        if re.search(r"\bconvertibles?\b|\bcabriolets?\b|\broadsters?\b", query_lower):
            body_terms.append("Convertible")

        inferred_conditions: set[str] = set()
        if re.search(r"\bused\b|pre[- ]owned", query_lower):
            inferred_conditions.add("Used")
        if re.search(r"\bnew\b|brand[- ]new", query_lower):
            inferred_conditions.add("New")

        min_price, max_price = inferred_price_range(query_lower)
        known_makes = {
            str(chunk.metadata.get("make", ""))
            for chunk in self.chunks
            if chunk.metadata.get("make")
        }
        mentioned_makes = {
            make for make in known_makes if make.lower() in query_lower
        }
        known_models = {
            str(chunk.metadata.get("model", ""))
            for chunk in self.chunks
            if chunk.metadata.get("model")
            and len(str(chunk.metadata.get("model", "")).strip()) >= 3
            and str(chunk.metadata.get("model", "")).strip().lower()
            not in AMBIGUOUS_MODEL_TERMS
        }
        mentioned_models = {
            model for model in known_models if model.lower() in query_lower
        }
        known_provinces = {
            str(chunk.metadata.get("province", ""))
            for chunk in self.chunks
            if chunk.metadata.get("province")
        }
        mentioned_provinces = {
            province for province in known_provinces if province.lower() in query_lower
        }

        candidates: list[int] = []
        for index, chunk in enumerate(self.chunks):
            if metadata_filters and any(
                allowed and chunk.metadata.get(key) not in allowed
                for key, allowed in metadata_filters.items()
            ):
                continue
            metadata = chunk.metadata
            if (
                inferred_years
                and not explicit_filters.get("year")
                and metadata.get("year") not in inferred_years
            ):
                continue
            if (
                inferred_fuels
                and not explicit_filters.get("fuel_type")
                and metadata.get("fuel_type") not in inferred_fuels
            ):
                continue
            if (
                body_terms
                and not explicit_filters.get("body_type")
                and metadata.get("body_type") not in body_terms
            ):
                continue
            if (
                inferred_conditions
                and not explicit_filters.get("condition")
                and metadata.get("condition") not in inferred_conditions
            ):
                continue
            if (
                mentioned_makes
                and not explicit_filters.get("make")
                and metadata.get("make") not in mentioned_makes
            ):
                continue
            if mentioned_models and not any(
                str(metadata.get("model", "")).lower().startswith(model.lower())
                for model in mentioned_models
            ):
                continue
            if (
                mentioned_provinces
                and not explicit_filters.get("province")
                and metadata.get("province") not in mentioned_provinces
            ):
                continue
            price = metadata.get("price_usd")
            if min_price is not None and (price is None or float(price) < min_price):
                continue
            if max_price is not None and (price is None or float(price) > max_price):
                continue
            candidates.append(index)

        def adjusted_score(index: int) -> float:
            chunk = self.chunks[index]
            overview_bonus = (
                0.07
                if chunk.metadata.get("section") == "Complete vehicle profile"
                else 0.0
            )
            return float(scores[index]) + overview_bonus

        ranked = sorted(candidates, key=adjusted_score, reverse=True)

        # Recommendation results are more useful when top-k represents distinct
        # vehicles rather than several sections from the same profile. Return each
        # document's complete profile so price and technical specs stay together,
        # even when a narrower section produced the strongest semantic match.
        candidate_set = set(candidates)
        overview_by_document = {
            chunk.document_id: index
            for index, chunk in enumerate(self.chunks)
            if index in candidate_set
            and chunk.metadata.get("section") == "Complete vehicle profile"
        }
        unique_ranked: list[int] = []
        seen_documents: set[str] = set()
        for index in ranked:
            document_id = self.chunks[index].document_id
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)
            unique_ranked.append(overview_by_document.get(document_id, index))

        # If the query names multiple makes, reserve one result for each named make.
        mentioned_make_list = sorted(mentioned_makes)
        if len(mentioned_make_list) > 1:
            diversified: list[int] = []
            for make in mentioned_make_list:
                match = next(
                    (
                        index
                        for index in unique_ranked
                        if self.chunks[index].metadata.get("make") == make
                    ),
                    None,
                )
                if match is not None:
                    diversified.append(match)
            diversified.extend(
                index for index in unique_ranked if index not in diversified
            )
            unique_ranked = diversified

        return [
            RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            for index in unique_ranked[: max(1, top_k)]
        ]
