"""
RAG orchestration pipeline — the single entry point for all queries.

Routing strategy (hybrid lookup + LLM):
  exact code match → semantic retrieval → confidence filter
  → [direct answer if high confidence OR LLM inference + guardrail]
  → structured response

  If retrieval returns nothing → NOT_FOUND_MSG returned directly.
  The LLM is NEVER invoked on an empty retrieval result.

Fast-path routing (deterministic):
  When the top retrieval score ≥ DIRECT_ANSWER_THRESHOLD, the answer is
  formatted directly from chunk metadata without invoking the LLM.
  Exact code lookups always score 1.0 and therefore always take this
  path.  For the shipped fault-code corpus, most semantic queries also
  score above the threshold, making the fast path the dominant route.
  This is intentional: for a structured fault-code reference table, a
  deterministic metadata-derived answer is more auditable and reliable
  than an LLM paraphrase.

LLM path (generative):
  Queries that retrieve chunks below DIRECT_ANSWER_THRESHOLD are sent to
  the local LLM (Ollama) for synthesis with citation guardrails.  This
  path handles genuinely ambiguous or fuzzy phrasing where the retrieval
  confidence is too low for a direct template answer.
"""
import re
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
from app.session import get_session_store
from app.settings import DIRECT_ANSWER_THRESHOLD, MAX_HISTORY_TURNS

log = get_logger(__name__)

_ERROR_CODE_REGEX = re.compile(r"0x[0-9a-fA-F]{4}", re.IGNORECASE)
_PRONOUN_PATTERNS = [
    r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthey\b", r"\bthem\b",
    r"\bthe fault\b", r"\bthe error\b", r"\bhow does it\b", r"\bhow to fix\b",
    r"\bwhat is it\b", r"\bexplain it\b"
]


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
        rem = c.get("error_remarks", "")
        doc = c.get("document_name", "IRL Fault Codes")

        line = f"**{code}** — {desc}"
        if rem:
            line += f"\n  Remarks: {rem}"
        line += f"\n  [Source: {doc}, {code}]"
        lines.append(line)

    if len(lines) == 1:
        return lines[0]
    else:
        return "\n\n".join(lines)


def _expand_query_from_history(question: str, history: list[dict]) -> str:
    """
    Lightweight rule-based query context expansion.
    If the question contains pronouns/follow-up triggers or lacks an error code,
    extract recent error codes from history to aid FAISS retrieval.
    """
    if not history:
        return question

    q_lower = question.lower()
    has_pronoun = any(re.search(pat, q_lower) for pat in _PRONOUN_PATTERNS)
    has_code = bool(_ERROR_CODE_REGEX.search(question))
    is_short = len(question.split()) <= 6

    if not (has_pronoun or (not has_code and is_short)):
        return question

    # Scan history backwards for error codes
    for msg in reversed(history):
        content = msg.get("content", "")
        code_match = _ERROR_CODE_REGEX.search(content)
        if code_match:
            found_code = code_match.group(0)
            if found_code.lower() not in question.lower():
                expanded = f"{question} {found_code}"
                log.info("Rule-based context expansion: '%s' -> '%s'", question, expanded)
                return expanded

    return question


def query(question: str, session_id: Optional[str] = None) -> QueryResponse:
    """
    Process a user question end-to-end and return a QueryResponse.
    Never raises — all errors are captured in QueryResponse.error.
    Maintain backward compatibility: session_id is optional.
    """
    t_start = time.perf_counter()
    question = question.strip()

    if not question:
        return QueryResponse(
            answer="Please enter a question.",
            found=False,
            latency_ms=0,
        )

    log.info("Query received: '%s' | session_id=%s", question[:120], session_id)

    # ── 1. History loading ────────────────────────────────────────────────
    t_hist_start = time.perf_counter()
    history: list[dict] = []
    if session_id:
        try:
            store = get_session_store()
            history = store.get_history(session_id, max_turns=MAX_HISTORY_TURNS)
        except Exception as e:
            log.warning("Failed to fetch session history for %s: %s", session_id, e)
    t_hist_ms = (time.perf_counter() - t_hist_start) * 1000

    # ── 2. Query expansion ────────────────────────────────────────────────
    t_exp_start = time.perf_counter()
    search_query = _expand_query_from_history(question, history)
    t_exp_ms = (time.perf_counter() - t_exp_start) * 1000

    # ── 3. Retrieval ──────────────────────────────────────────────────────
    t_ret_start = time.perf_counter()
    try:
        retriever = get_retriever()
        results: list[RetrievedChunk] = retriever.retrieve(search_query)
        t_ret_ms = (time.perf_counter() - t_ret_start) * 1000
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
        log.info(
            "TIMING BREAKDOWN (No Chunks) | Total: %dms | History: %.2fms | Expansion: %.2fms | Retrieval: %.2fms",
            elapsed, t_hist_ms, t_exp_ms, t_ret_ms,
        )
        return QueryResponse(
            answer=NOT_FOUND_MSG,
            found=False,
            latency_ms=elapsed,
        )

    top_score = results[0].score

    # ── FAST PATH: direct answer from structured data (no LLM) ────────────
    if top_score >= DIRECT_ANSWER_THRESHOLD:
        answer = _format_direct_answer(results)
        citations = [
            f"{r.chunk.get('document_name', 'IRL Fault Codes')}, {r.chunk.get('error_code', 'N/A')}"
            for r in results
        ]
        elapsed = int((time.perf_counter() - t_start) * 1000)
        log.info(
            "FAST PATH TIMING BREAKDOWN | Total: %dms | History: %.2fms | "
            "Expansion: %.2fms | Retrieval: %.2fms | DirectAnswer: <1ms",
            elapsed, t_hist_ms, t_exp_ms, t_ret_ms,
        )

        if session_id:
            try:
                get_session_store().add_turn(session_id, question, answer)
            except Exception as e:
                log.warning("Failed to record turn for %s: %s", session_id, e)

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=[{**r.chunk, "score": r.score} for r in results],
            top_score=top_score,
            latency_ms=elapsed,
            found=True,
            guardrail_triggered=False,
        )

    # ── SLOW PATH: LLM inference for ambiguous/low-confidence queries ─────
    log.info("Low confidence (%.3f) — falling back to LLM inference.", top_score)
    try:
        gen_result = generate(question, results, history=history)
    except Exception as e:
        log.error("Generator error: %s", e)
        return QueryResponse(
            answer=DEGRADED_MSG,
            found=True,
            retrieved_chunks=[{**r.chunk, "score": r.score} for r in results],
            top_score=top_score,
            error=str(e),
            latency_ms=int((time.perf_counter() - t_start) * 1000),
        )

    elapsed = int((time.perf_counter() - t_start) * 1000)
    answer = gen_result["answer"]
    prompt_ms = gen_result.get("prompt_construction_ms", 0.0)
    ollama_ms = gen_result.get("ollama_inference_ms", 0.0)

    log.info(
        "LLM PATH TIMING BREAKDOWN | Total: %dms | History: %.2fms | "
        "Expansion: %.2fms | Retrieval: %.2fms | PromptBuild: %.2fms | Ollama: %.2fms",
        elapsed, t_hist_ms, t_exp_ms, t_ret_ms, prompt_ms, ollama_ms,
    )

    if session_id and answer and answer != DEGRADED_MSG:
        try:
            get_session_store().add_turn(session_id, question, answer)
        except Exception as e:
            log.warning("Failed to record turn for %s: %s", session_id, e)

    return QueryResponse(
        answer=answer,
        citations=gen_result["citations"],
        retrieved_chunks=[{**r.chunk, "score": r.score} for r in results],
        top_score=top_score,
        latency_ms=elapsed,
        found=True,
        guardrail_triggered=gen_result.get("guardrail_triggered", False),
        error=gen_result.get("error"),
    )
