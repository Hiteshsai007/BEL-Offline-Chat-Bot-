"""
RAG orchestration pipeline — the single entry point for all queries.

Flow (PRD Section 5):
  exact code match → semantic retrieval → confidence filter
  → [direct answer if high confidence OR LLM inference + guardrail]
  → structured response

  If retrieval returns nothing → NOT_FOUND_MSG returned directly.
  The LLM is NEVER invoked on an empty retrieval result.

Performance optimisation:
  For a structured fault-code table, the retrieved chunk already IS the
  answer.  When the top retrieval score ≥ DIRECT_ANSWER_THRESHOLD we
  format a grounded response directly from the chunk metadata — no LLM
  round-trip required.  This brings latency from ~30s to <500ms.
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
from app.settings import CONFIDENCE_THRESHOLD

log = get_logger(__name__)

# If the top retrieval score is above this, skip the LLM and answer directly.
# Exact code matches always score 1.0; strong semantic hits score 0.65+.
DIRECT_ANSWER_THRESHOLD = 0.60


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


def _format_direct_answer(results: list[RetrievedChunk]) -> str:
    """
    Build a human-readable answer directly from retrieved chunks.
    Every statement is traceable to the source document — zero hallucination.
    """
    lines = []
    for rc in results:
        c = rc.chunk
        code = c.get("error_code", "N/A")
        desc = c.get("error_description", "N/A")
        rem  = c.get("error_remarks", "")
        doc  = c.get("document_name", "IRL Fault Codes")

        line = f"**{code}** — {desc}"
        if rem:
            line += f"\n  Remarks: {rem}"
        line += f"\n  [Source: {doc}, {code}]"
        lines.append(line)

    if len(lines) == 1:
        return lines[0]
    else:
        return "\n\n".join(lines)


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

    # ── FAST PATH: direct answer from structured data (no LLM) ────────────
    # For a structured fault-code table, the chunk metadata IS the answer.
    # This brings response time from ~30s down to <500ms.
    if top_score >= DIRECT_ANSWER_THRESHOLD:
        answer = _format_direct_answer(results)
        citations = [
            f"{r.chunk.get('document_name', 'IRL Fault Codes')}, {r.chunk.get('error_code', 'N/A')}"
            for r in results
        ]
        elapsed = int((time.perf_counter() - t_start) * 1000)
        log.info(
            "FAST PATH: direct answer in %dms | top_score=%.3f | %d chunk(s)",
            elapsed, top_score, len(results),
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=[r.chunk for r in results],
            top_score=top_score,
            latency_ms=elapsed,
            found=True,
            guardrail_triggered=False,
        )

    # ── SLOW PATH: LLM inference for ambiguous/low-confidence queries ─────
    log.info("Low confidence (%.3f) — falling back to LLM inference.", top_score)
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
        "LLM PATH: query complete in %dms | top_score=%.3f | guardrail=%s",
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
