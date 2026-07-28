from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .embeddings import Embedder
from .generation import Generator, NO_EVIDENCE_ANSWER
from .models import RetrievedChunk
from .vector_store import VectorIndex


StepCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]
    grounded: bool


UNSUPPORTED_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    (r"\bsafe(?:st|ty)?\b|crash|ncap|nhtsa", "crash safety"),
    (r"\breliab(?:le|ility)\b|dependab", "reliability"),
    (r"maintenance|repair cost|service cost", "maintenance and repair cost"),
    (r"comfort|ride quality|seat comfort", "comfort"),
)


def unsupported_requirements(query: str) -> list[str]:
    query_lower = query.lower()
    return [
        label
        for pattern, label in UNSUPPORTED_REQUIREMENTS
        if re.search(pattern, query_lower)
    ]


def unsupported_answer(requirements: list[str]) -> str:
    if len(requirements) == 1:
        requirement_text = requirements[0]
    else:
        requirement_text = (
            ", ".join(requirements[:-1]) + f", and {requirements[-1]}"
        )
    return (
        f"I cannot recommend a vehicle for this request because it depends on "
        f"**{requirement_text}**, which the Cambodian marketplace snapshot does "
        "not verify.\n\n"
        "No vehicle recommendation was produced. CarWise can support questions about "
        "seller asking price, model year, make, body type, condition label, fuel type, "
        "and listing location in Cambodia.\n\n"
        "Try: *Which used SUVs in Phnom Penh are listed below $20,000?*"
    )


class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        generator: Generator,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.generator = generator

    def answer(
        self,
        query: str,
        *,
        top_k: int,
        minimum_similarity: float,
        answer_mode: str,
        metadata_filters: dict[str, set[Any]] | None = None,
        on_step: StepCallback | None = None,
    ) -> AnswerResult:
        def report(stage: str, message: str) -> None:
            if on_step is not None:
                on_step(stage, message)

        report("validation", "Checking the question and active search filters.")
        query = query.strip()
        if not query:
            report("stopped", "Search stopped because the question is empty.")
            return AnswerResult("Please enter a question.", [], False)

        unsupported = unsupported_requirements(query)
        if unsupported:
            report(
                "stopped",
                "The question needs evidence that this dataset cannot verify safely.",
            )
            return AnswerResult(unsupported_answer(unsupported), [], False)

        report(
            "embedding",
            "Creating a semantic embedding and running vector retrieval.",
        )
        sources = self.index.search(
            query,
            self.embedder,
            top_k=None,
            metadata_filters=metadata_filters,
        )
        report(
            "retrieval",
            f"Retrieved all {len(sources)} candidate vehicles that match the "
            "question and active filters.",
        )
        relevant = [source for source in sources if source.score >= minimum_similarity]
        report(
            "scoring",
            (
                f"Kept {len(relevant)} chunks at or above the "
                f"{minimum_similarity:.2f} similarity threshold."
            ),
        )
        if not relevant:
            report(
                "stopped",
                "No sufficiently relevant evidence was found; no answer was generated.",
            )
            return AnswerResult(NO_EVIDENCE_ANSWER, sources, False)

        report(
            "generation",
            f"Generating a grounded answer from the top {min(top_k, len(relevant))} "
            f"matches with {self.generator.provider_name}.",
        )
        # Keep generation focused while returning every qualifying match to the
        # interface for card rendering.
        answer = self.generator.generate(query, relevant[:top_k], answer_mode)
        report(
            "citations",
            "Grounded answer and citation-linked source evidence are ready.",
        )
        return AnswerResult(answer, relevant, True)
