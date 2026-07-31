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
_FOLLOWUP_PATTERNS = [
    r"\bfix\b", r"\bremedy\b", r"\bsolution\b", r"\bcorrective action\b",
    r"\bcorrective steps\b", r"\btroubleshooting\b", r"\bresolve\b",
    r"\bresolution\b", r"\baction\b", r"\bcause\b", r"\bwhy\b",
    r"\brepair\b", r"\brecommendation\b", r"\brecommended action\b",
    r"\bnext step\b", r"\bnext steps\b", r"\bit\b", r"\bthis\b",
    r"\bthat\b", r"\bthey\b", r"\bthem\b", r"\bthe fault\b",
    r"\bthe error\b", r"\bhow does it\b", r"\bhow to fix\b",
    r"\bwhat is it\b", r"\bexplain it\b"
]
_NEUTRAL_PATTERNS = [
    r"^\s*hi\s*$", r"^\s*hello\s*$", r"^\s*hey\s*$", r"^\s*good morning\s*$",
    r"^\s*good evening\s*$", r"^\s*thanks\s*$", r"^\s*thank you\s*$",
    r"^\s*okay\s*$", r"^\s*ok\s*$", r"^\s*alright\s*$", r"^\s*cool\s*$",
    r"^\s*got it\s*$", r"^\s*understood\s*$", r"^\s*makes sense\s*$",
    r"^\s*bye\s*$", r"^\s*goodbye\s*$"
]


def _is_neutral_conversation_turn(text: str) -> bool:
    """
    Check if a message is a neutral conversation turn (greeting, acknowledgement,
    thanks, or short confirmation).

    False Positive Protection:
    A message is considered neutral ONLY if:
    - It matches a neutral pattern, AND
    - It contains no question mark (?)
    - It contains no fault code (0xXXXX)
    - It contains no additional substantive request or instruction (word count <= 4)
    """
    if not text or "?" in text:
        return False

    if _ERROR_CODE_REGEX.search(text):
        return False

    t_clean = text.strip().lower()
    t_no_punct = re.sub(r"[^\w\s]", "", t_clean).strip()

    matches_pattern = any(re.match(pat, t_no_punct) for pat in _NEUTRAL_PATTERNS)
    if not matches_pattern:
        return False

    words = t_no_punct.split()
    if len(words) > 4:
        return False

    return True


def _get_conversational_response(text: str) -> Optional[str]:
    """
    Check if a text is a simple conversational message (greeting, acknowledgement,
    thanks, or farewell) and return a lightweight friendly response.
    Returns None if the message is not a simple conversational turn.
    """
    if not text or not _is_neutral_conversation_turn(text):
        return None

    t_clean = text.strip().lower()
    t_no_punct = re.sub(r"[^\w\s]", "", t_clean).strip()

    if t_no_punct in ("hi", "hello", "hey"):
        return "Hello! How can I help you today?"
    elif t_no_punct == "good morning":
        return "Good morning! How can I assist you today?"
    elif t_no_punct == "good evening":
        return "Good evening! How can I assist you today?"
    elif t_no_punct in ("thanks", "thank you"):
        return "You're welcome."
    elif t_no_punct in ("ok", "okay", "alright", "cool", "got it", "understood", "makes sense"):
        return "Got it."
    elif t_no_punct in ("bye", "goodbye"):
        return "Goodbye! Have a great day."

    return None


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


def _has_remediation(results: list[RetrievedChunk]) -> bool:
    """Check if retrieved chunks contain explicit remediation/corrective action information."""
    keywords = [
        "remedy", "remediation", "corrective action", "action required",
        "solution", "fix:", "repair", "replace", "troubleshooting steps",
        "corrective steps", "recommended action"
    ]
    for rc in results:
        c = rc.chunk
        text = f"{c.get('error_remarks', '')} {c.get('chunk_text', '')}".lower()
        if any(kw in text for kw in keywords):
            return True
    return False


def _format_missing_remediation_answer(
    results: list[RetrievedChunk], resolved_code: str
) -> str:
    """
    Format a grounded response when a follow-up query requests remediation for a fault code,
    but the source documentation contains no corrective steps.
    """
    target_chunk = None
    if resolved_code:
        for rc in results:
            if (rc.chunk.get("error_code") or "").lower() == resolved_code.lower():
                target_chunk = rc.chunk
                break
    if not target_chunk and results:
        target_chunk = results[0].chunk

    code = target_chunk.get("error_code") if target_chunk else resolved_code
    code = code or resolved_code or "N/A"
    desc = target_chunk.get("error_description", "") if target_chunk else ""
    rem = target_chunk.get("error_remarks", "") if target_chunk else ""
    doc = target_chunk.get("document_name", "IRL Fault Codes.pdf") if target_chunk else "IRL Fault Codes.pdf"

    bullets = []
    if desc and desc != "N/A":
        bullets.append(f"• {desc}")
    if rem and rem != "N/A":
        bullets.append(f"• {rem}")

    info_block = "\n".join(bullets) if bullets else "• No additional metadata documented."

    return (
        f"The available documentation does not provide corrective steps for error {code}.\n\n"
        f"Available documented information:\n"
        f"{info_block}\n\n"
        f"[Source: {doc}, {code}]"
    )


def _resolve_history_context(
    question: str, history: list[dict]
) -> tuple[bool, Optional[str], str, bool, int, str]:
    """
    Inspect question for follow-up intent and search conversation history
    (most recent turn first) for the active fault code reference (0xXXXX).

    Follow-up resolution priority:
    1. Active topic context: If a prior conversation turn represents a topic shift
       to non-fault-code documents or general topics without referencing a fault code,
       stop searching so older fault codes do not override newer topics.
    2. Neutral turn skipping: Greetings, acknowledgements, thanks, and short confirmations
       are skipped during context resolution and do NOT trigger topic shifts or clear fault codes.

    Returns:
        (has_followup_intent, resolved_fault_code, search_query,
         neutral_turn_detected, skipped_neutral_count, context_source)
    """
    if not question:
        return False, None, question, False, 0, "none"

    q_lower = question.lower()
    has_followup = any(re.search(pat, q_lower) for pat in _FOLLOWUP_PATTERNS)

    resolved_code: Optional[str] = None
    skipped_neutral_count = 0
    neutral_turn_detected = False
    context_source = "none"

    if history:
        # Group messages into turn pairs (user, assistant)
        turns = []
        i = 0
        while i < len(history):
            if history[i].get("role") == "user":
                u = history[i].get("content", "")
                a = history[i + 1].get("content", "") if (i + 1 < len(history) and history[i + 1].get("role") == "assistant") else ""
                turns.append((u, a))
                i += 2
            else:
                i += 1

        for u_msg, a_msg in reversed(turns):
            if _is_neutral_conversation_turn(u_msg):
                neutral_turn_detected = True
                skipped_neutral_count += 1
                continue

            combined = f"{u_msg} {a_msg}"
            match = _ERROR_CODE_REGEX.search(combined)
            if match:
                resolved_code = match.group(0)
                context_source = "fault_code_history"
                break

            u_has_followup = any(re.search(pat, u_msg.lower()) for pat in _FOLLOWUP_PATTERNS)
            u_has_code = bool(_ERROR_CODE_REGEX.search(u_msg))

            if not u_has_code and not u_has_followup:
                has_general_doc = any(
                    marker in a_msg for marker in ["page ", ".md", "Owner", "Manual", "Ninja"]
                ) or bool(a_msg and "0x" not in a_msg)
                if has_general_doc:
                    resolved_code = None
                    context_source = "general_topic"
                    break

    search_query = question
    if resolved_code and resolved_code.lower() not in q_lower and has_followup:
        search_query = f"{question} {resolved_code}"
        log.info(
            "Rule-based context expansion: '%s' -> '%s' (resolved code %s)",
            question, search_query, resolved_code
        )

    return (
        has_followup,
        resolved_code,
        search_query,
        neutral_turn_detected,
        skipped_neutral_count,
        context_source,
    )


def _expand_query_from_history(question: str, history: list[dict]) -> str:
    """Backward-compatible wrapper for query expansion."""
    _, _, search_query, _, _, _ = _resolve_history_context(question, history)
    return search_query


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

    # ── 2. Context resolution & Query expansion ───────────────────────────
    t_exp_start = time.perf_counter()
    (
        has_followup,
        resolved_code,
        search_query,
        neutral_turn_detected,
        skipped_neutral_count,
        context_source,
    ) = _resolve_history_context(question, history)
    t_exp_ms = (time.perf_counter() - t_exp_start) * 1000

    # ── Conversational Shortcut Path (Bypass RAG & LLM for greetings/thanks/acks)
    conv_response = _get_conversational_response(question)
    if conv_response:
        elapsed = int((time.perf_counter() - t_start) * 1000)
        selected_path = "Conversational Shortcut"
        log.info(
            "SESSION DIAGNOSTICS | session_id=%s | history_length=%d | "
            "expanded_query='%s' | detected_follow_up_intent=%s | "
            "neutral_turn_detected=%s | skipped_history_turns=%d | "
            "resolved_fault_code=%s | resolved_context_source=%s | selected_path=%s",
            session_id, len(history), search_query, has_followup,
            neutral_turn_detected, skipped_neutral_count,
            resolved_code or "None", context_source, selected_path
        )
        if session_id:
            try:
                get_session_store().add_turn(session_id, question, conv_response)
            except Exception as e:
                log.warning("Failed to record turn for %s: %s", session_id, e)

        return QueryResponse(
            answer=conv_response,
            citations=[],
            retrieved_chunks=[],
            top_score=1.0,
            latency_ms=elapsed,
            found=True,
            guardrail_triggered=False,
        )

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
        selected_path = "Fast Path (Not Found)"
        log.info(
            "SESSION DIAGNOSTICS | session_id=%s | history_length=%d | "
            "expanded_query='%s' | detected_follow_up_intent=%s | "
            "neutral_turn_detected=%s | skipped_history_turns=%d | "
            "resolved_fault_code=%s | resolved_context_source=%s | selected_path=%s",
            session_id, len(history), search_query, has_followup,
            neutral_turn_detected, skipped_neutral_count,
            resolved_code or "None", context_source, selected_path
        )
        return QueryResponse(
            answer=NOT_FOUND_MSG,
            found=False,
            latency_ms=elapsed,
        )

    top_score = results[0].score
    is_fault_code_match = results and bool(results[0].chunk.get("error_code"))

    # ── FAST PATH / FOLLOW-UP FAST PATH (Deterministic) ───────────────────
    if top_score >= DIRECT_ANSWER_THRESHOLD and is_fault_code_match:
        if has_followup:
            selected_path = "Follow-up Fast Path"
            code_ref = resolved_code or results[0].chunk.get("error_code") or ""
            if _has_remediation(results):
                answer = _format_direct_answer(results)
            else:
                answer = _format_missing_remediation_answer(results, code_ref)
        else:
            selected_path = "Fast Path"
            answer = _format_direct_answer(results)

        citations = [
            f"{r.chunk.get('document_name', 'IRL Fault Codes')}, {r.chunk.get('error_code', 'N/A')}"
            for r in results
        ]
        elapsed = int((time.perf_counter() - t_start) * 1000)

        log.info(
            "SESSION DIAGNOSTICS | session_id=%s | history_length=%d | "
            "expanded_query='%s' | detected_follow_up_intent=%s | "
            "neutral_turn_detected=%s | skipped_history_turns=%d | "
            "resolved_fault_code=%s | resolved_context_source=%s | selected_path=%s",
            session_id, len(history), search_query, has_followup,
            neutral_turn_detected, skipped_neutral_count,
            resolved_code or "None", context_source, selected_path
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
    selected_path = "LLM Path"
    log.info(
        "SESSION DIAGNOSTICS | session_id=%s | history_length=%d | "
        "expanded_query='%s' | detected_follow_up_intent=%s | "
        "neutral_turn_detected=%s | skipped_history_turns=%d | "
        "resolved_fault_code=%s | resolved_context_source=%s | selected_path=%s",
        session_id, len(history), search_query, has_followup,
        neutral_turn_detected, skipped_neutral_count,
        resolved_code or "None", context_source, selected_path
    )

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
