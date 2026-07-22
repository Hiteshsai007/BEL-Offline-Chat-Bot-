"""
RAG orchestration pipeline — the single entry point for all queries.

Flow (PRD Section 5):
  exact code match → semantic retrieval → confidence filter
  → [LLM inference + guardrail] → structured response

  If retrieval returns nothing → NOT_FOUND_MSG returned directly.
  The LLM is NEVER invoked on an empty retrieval result.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from app.logger import get_logger
from app.rag.generator import (
    DEGRADED_MSG,
    NOT_FOUND_MSG,
    generate,
)
from app.rag.retriever import RetrievedChunk, get_retriever

log = get_logger(__name__)


@dataclass
class QueryResponse:
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    top_score: float = 0.0
    latency_ms: int = 0
    found: bool = True
    guardrail_triggered: bool = False
    error: Optional[str] = None


def query(question: str) -> QueryResponse:
    """
    Process a user question end-to-end and return a QueryResponse.
    Never raises — all errors are captured in QueryResponse.error.
    """
    t_start = time.perf_counter()
    question = question.strip()

    if not question:
        return QueryResponse(
            answer="Please enter a question.",
            found=False,
            latency_ms=0,
        )

    log.info("Query received: '%s'", question[:120])

    # ── Retrieval ──────────────────────────────────────────────────────────
    try:
        retriever = get_retriever()
        results: list[RetrievedChunk] = retriever.retrieve(question)
    except FileNotFoundError as e:
        log.error("Retriever init failed: %s", e)
        return QueryResponse(
            answer=(
                "Knowledge base not found. "
                "Please run the ingestion script: python -m app.ingestion.ingest"
            ),
            found=False,
            error="index_missing",
            latency_ms=int((time.perf_counter() - t_start) * 1000),
        )
    except Exception as e:
        log.error("Retrieval error: %s", e)
        return QueryResponse(
            answer=DEGRADED_MSG,
            found=False,
            error=str(e),
            latency_ms=int((time.perf_counter() - t_start) * 1000),
        )

    # ── Not-found path (FR-7) — LLM must NOT be called ───────────────────
    if not results:
        elapsed = int((time.perf_counter() - t_start) * 1000)
        log.info("No relevant chunks found — returning not-found message (no LLM call).")
        return QueryResponse(
            answer=NOT_FOUND_MSG,
            found=False,
            latency_ms=elapsed,
        )

    top_score = results[0].score

    # ── Inference (FR-4) — real LLM call, every time ──────────────────────
    try:
        gen_result = generate(question, results)
    except Exception as e:
        log.error("Generator error: %s", e)
        return QueryResponse(
            answer=DEGRADED_MSG,
            found=True,
            retrieved_chunks=[r.chunk for r in results],
            top_score=top_score,
            error=str(e),
            latency_ms=int((time.perf_counter() - t_start) * 1000),
        )

    elapsed = int((time.perf_counter() - t_start) * 1000)
    log.info(
        "Query complete in %dms | top_score=%.3f | guardrail=%s",
        elapsed, top_score, gen_result.get("guardrail_triggered"),
    )

    return QueryResponse(
        answer=gen_result["answer"],
        citations=gen_result["citations"],
        retrieved_chunks=[r.chunk for r in results],
        top_score=top_score,
        latency_ms=elapsed,
        found=True,
        guardrail_triggered=gen_result.get("guardrail_triggered", False),
        error=gen_result.get("error"),
    )
