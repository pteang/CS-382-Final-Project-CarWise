from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from carwise.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_LLM_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    INDEX_DIR,
)
from carwise.embeddings import SentenceTransformerEmbedder  # noqa: E402
from carwise.generation import (  # noqa: E402
    OllamaGenerator,
    OpenAIResponsesGenerator,
    RetrievalPreviewGenerator,
    TransformersGenerator,
)
from carwise.vector_store import VectorIndex  # noqa: E402


def make_generator(provider: str):
    if provider == "local-llm":
        return TransformersGenerator(DEFAULT_LOCAL_LLM_MODEL)
    if provider == "openai":
        return OpenAIResponsesGenerator(DEFAULT_OPENAI_MODEL)
    if provider == "ollama":
        return OllamaGenerator(DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_BASE_URL)
    if provider == "deterministic":
        return RetrievalPreviewGenerator()
    return None


def write_report(output: dict, output_path: Path) -> None:
    lines = [
        "# CarWise Retrieval and Generation Evaluation",
        "",
        f"Evaluated: {output['evaluated_at']}",
        "",
        "## Summary",
        "",
        f"- Queries: {output['total']}",
        f"- Retrieval passes: {output['retrieval_passed']}/{output['total']}",
        f"- Generation grounding passes: {output['generation_passed']}/{output['total']}",
        f"- Manual answer-quality passes: {output['quality_passed']}/{output['total']}",
        f"- Retrieval pass rate: {output['retrieval_pass_rate']:.1%}",
        f"- Generation grounding pass rate: {output['generation_pass_rate']:.1%}",
        f"- Manual answer-quality pass rate: {output['quality_pass_rate']:.1%}",
        f"- Generation provider: {output['generation_provider']}",
        f"- Mean retrieval latency: {output['mean_retrieval_latency_seconds']:.2f}s",
        f"- Mean generation latency: {output['mean_generation_latency_seconds']:.2f}s",
        "",
        "A retrieval pass means every expected evidence term appeared in the top-k "
        "chunks. A generation grounding pass means the answer was non-empty and every "
        "citation referred to one of the supplied chunks. The qualitative notes below "
        "also record whether each result answered the intended constraint.",
        "",
        "## Per-query results",
        "",
    ]
    for row in output["results"]:
        top_documents = ", ".join(
            f"{item['title']} ({item['score']:.3f})"
            for item in row["top_documents"]
        )
        lines.extend(
            [
                f"### {row['id']}: {row['query']}",
                "",
                f"- Retrieval: {'PASS' if row['retrieval_passed'] else 'NEEDS REVIEW'}",
                f"- Citation grounding: {'PASS' if row['generation_passed'] else 'NEEDS REVIEW'}",
                f"- Manual answer quality: {row['quality_status'].upper()}",
                f"- Top evidence: {top_documents or 'No relevant chunks'}",
                f"- Expected evidence found: {', '.join(row['matched_terms']) or 'None'}",
                f"- Performance: retrieval {row['retrieval_latency_seconds']:.2f}s; "
                f"generation {row['generation_latency_seconds']:.2f}s",
                f"- Qualitative assessment: {row['manual_assessment']}",
                "",
                "**Generated answer**",
                "",
                row["answer"] or "_No answer was generated._",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations and interpretation",
            "",
            "- The automatic retrieval metric checks expected evidence coverage, not "
            "whether the first result is always the best possible vehicle.",
            "- Citation validation confirms that labels point to supplied chunks; manual "
            "review is still required to detect subtle paraphrase errors.",
            "- Khmer24 prices, condition labels, and availability are time-sensitive "
            "seller claims. Technical specifications are model-year references and may "
            "not match an imported trim exactly.",
            "- Crash safety, reliability, mechanical condition, legal ownership, and fair "
            "market value remain outside the evidence collection.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CarWise retrieval evaluation.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--generation-provider",
        choices=["local-llm", "openai", "ollama", "deterministic", "none"],
        default="local-llm",
    )
    parser.add_argument("--minimum-similarity", type=float, default=0.22)
    args = parser.parse_args()

    queries = json.loads(
        (PROJECT_ROOT / "evaluation" / "queries.json").read_text(encoding="utf-8")
    )
    manual_assessments = json.loads(
        (PROJECT_ROOT / "evaluation" / "manual_assessments.json").read_text(
            encoding="utf-8"
        )
    )
    index = VectorIndex.load(INDEX_DIR)
    embedder = SentenceTransformerEmbedder(args.model)
    generator = make_generator(args.generation_provider)

    retrieval_passed = 0
    generation_passed = 0
    rows: list[dict] = []
    for case in queries:
        retrieval_started = time.perf_counter()
        results = index.search(case["query"], embedder, top_k=args.top_k)
        relevant = [
            result for result in results if result.score >= args.minimum_similarity
        ]
        retrieval_latency = time.perf_counter() - retrieval_started
        evidence = "\n".join(result.chunk.text for result in results).lower()
        matched = [
            term
            for term in case["expected_terms"]
            if term.lower() in evidence
        ]
        retrieval_success = len(matched) == len(case["expected_terms"])
        retrieval_passed += int(retrieval_success)

        answer = ""
        generation_latency = 0.0
        citations: list[int] = []
        generation_success = False
        if generator and relevant:
            generation_started = time.perf_counter()
            answer = generator.generate(case["query"], relevant[:3], "Concise")
            generation_latency = time.perf_counter() - generation_started
            citations = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
            generation_success = bool(
                answer.strip()
                and citations
                and all(1 <= citation <= len(relevant) for citation in citations)
            )
        elif generator is None:
            generation_success = True
            answer = "Generation skipped by command-line option."
        generation_passed += int(generation_success)

        manual = manual_assessments.get(
            case["id"],
            {
                "status": "needs_review",
                "assessment": "No manual assessment has been recorded.",
            },
        )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "retrieval_passed": retrieval_success,
                "generation_passed": generation_success,
                "matched_terms": matched,
                "top_score": round(results[0].score, 4) if results else None,
                "top_documents": [
                    {
                        "title": result.chunk.document_title,
                        "score": round(result.score, 4),
                    }
                    for result in results[:3]
                ],
                "citations": sorted(set(citations)),
                "answer": answer,
                "retrieval_latency_seconds": round(retrieval_latency, 3),
                "generation_latency_seconds": round(generation_latency, 3),
                "quality_status": manual["status"],
                "manual_assessment": manual["assessment"],
                "notes": case["notes"],
            }
        )

    generation_latencies = [
        row["generation_latency_seconds"]
        for row in rows
        if row["generation_latency_seconds"] > 0
    ]
    output = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_metric": (
            "Every expected evidence term appears in "
            f"the top-{args.top_k} retrieved chunks."
        ),
        "generation_metric": (
            "The answer is non-empty and every citation label refers to a supplied chunk."
        ),
        "generation_provider": (
            generator.provider_name if generator else "Skipped"
        ),
        "retrieval_passed": retrieval_passed,
        "generation_passed": generation_passed,
        "quality_passed": sum(
            row["quality_status"] == "pass" for row in rows
        ),
        "total": len(queries),
        "retrieval_pass_rate": round(retrieval_passed / len(queries), 3),
        "generation_pass_rate": round(generation_passed / len(queries), 3),
        "quality_pass_rate": round(
            sum(row["quality_status"] == "pass" for row in rows) / len(queries),
            3,
        ),
        "mean_retrieval_latency_seconds": round(
            sum(row["retrieval_latency_seconds"] for row in rows) / len(rows),
            3,
        ),
        "mean_generation_latency_seconds": round(
            sum(generation_latencies) / len(generation_latencies),
            3,
        )
        if generation_latencies
        else 0.0,
        "results": rows,
    }
    output_path = PROJECT_ROOT / "evaluation" / "latest_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    report_path = PROJECT_ROOT / "evaluation" / "report.md"
    write_report(output, report_path)
    print(json.dumps(output, indent=2))
    print(f"\nSaved evaluation to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
