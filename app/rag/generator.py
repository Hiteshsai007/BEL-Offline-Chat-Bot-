"""
Ollama inference wrapper with citation guardrail.

This module is invoked by the pipeline only for queries that fall below
the DIRECT_ANSWER_THRESHOLD — i.e. ambiguous or low-confidence retrieval
results where the structured fast-path template is insufficient.

Guarantees when this module IS called:
  • The generator is NEVER called when retrieval returns nothing.
  • Output is checked for citations; one regeneration attempt is made
    if the first response fails the citation check.
  • If both attempts fail, a "documentation insufficient" fallback is
    returned.
  • Response latency is logged — sub-500ms is flagged as suspicious.
"""
import re
import time

import httpx

from app.logger import get_logger
from app.settings import (
    ERROR_CODE_PATTERN,
    MAX_MESSAGE_CHARS,
    MAX_TOKENS,
    MODEL_TAG,
    NUM_CTX,
    OLLAMA_URL,
    TEMPERATURE,
    TIMEOUT,
)

log = get_logger(__name__)

# Reuse the single source of truth for what an error code looks like
_CODE_RE = re.compile(ERROR_CODE_PATTERN, re.IGNORECASE)

# ── Fixed response strings (PRD Section 13) ────────────────────────────────
NOT_FOUND_MSG = "This information is not available in the current documentation."
DEGRADED_MSG = (
    "The AI inference service is temporarily unavailable. "
    "Please check that Ollama is running and try again."
)
INSUFFICIENT_MSG = (
    "The retrieved documentation was insufficient to produce a "
    "fully cited answer. Please consult the source document directly."
)

# ── System prompt (verbatim from PRD Section 13) ───────────────────────────
_SYSTEM_PROMPT = """You are a technical assistant restricted to the provided context.
Rules:
- Read ALL numbered context passages carefully before answering.
- Answer ONLY using information from the numbered context passages below.
- Synthesize information from MULTIPLE passages when relevant.
- If the answer is not present in the context, respond exactly with:
  "This information is not available in the current documentation."
- Cite every supported statement as [Document Name, Error Code] or
  [Document Name, page N] for documents without error codes.
- Do not cite Previous Conversation. Cite ONLY from the numbered Context passages.
- Do not use any knowledge beyond the provided context.
- Be concise and precise. Do not speculate.
- When multiple context passages are relevant, combine their information
  into a comprehensive answer rather than choosing only one."""


def _build_context_block(retrieved_chunks: list) -> str:
    """Format retrieved chunks as numbered context passages."""
    lines = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        chunk = rc.chunk
        doc = chunk.get("document_name", "IRL Fault Codes")
        code = chunk.get("error_code") or "N/A"
        page = chunk.get("page_number")
        text = chunk.get("chunk_text", "")
        # Include page number for general documents (no error codes)
        if code != "N/A":
            lines.append(f"[{i}] {text} (Source: {doc}, Error Code: {code})")
        elif page is not None:
            lines.append(f"[{i}] {text} (Source: {doc}, page {page})")
        else:
            lines.append(f"[{i}] {text} (Source: {doc})")
    return "\n".join(lines)


def _build_history_block(history: list | None) -> str:
    """Format previous conversation turns into a concise text block."""
    if not history:
        return ""
    lines = ["Previous Conversation:"]
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _citable_codes(retrieved_chunks: list) -> set:
    """Error codes actually present in the retrieved context, lower-cased."""
    codes = set()
    for rc in retrieved_chunks:
        code = (rc.chunk.get("error_code") or "").strip().lower()
        if code:
            codes.add(code)
    return codes


def _citable_documents(retrieved_chunks: list) -> set:
    """Document names actually present in the retrieved context, lower-cased."""
    docs = set()
    for rc in retrieved_chunks:
        doc = (rc.chunk.get("document_name") or "").strip().lower()
        if doc:
            docs.add(doc)
    return docs


def _citable_pages(retrieved_chunks: list) -> set:
    """Page numbers actually present in the retrieved context, as strings."""
    pages = set()
    for rc in retrieved_chunks:
        page = rc.chunk.get("page_number")
        if page is not None:
            pages.add(str(page))
    return pages


def _has_citation(text: str, retrieved_chunks: list) -> bool:
    """
    Validate that the response contains at least one *grounded* citation.
    """
    spans = re.findall(r"\[([^\]]+)\]", text)
    if not spans:
        spans = [text]

    valid_codes = _citable_codes(retrieved_chunks)

    if valid_codes:
        # Fault-code documents: require at least one error code citation
        for span in spans:
            for found in _CODE_RE.findall(span):
                if found.strip().lower() in valid_codes:
                    return True
        return False

    # General documents (no error codes): accept document name, page number, or source tag
    valid_docs = _citable_documents(retrieved_chunks)
    valid_pages = _citable_pages(retrieved_chunks)
    for span in spans:
        span_l = span.lower()
        has_doc = any(doc and doc in span_l for doc in valid_docs)
        has_page = any(
            re.search(rf"\bpage\s*{re.escape(p)}\b", span_l)
            or re.search(rf"\bp\.?\s*{re.escape(p)}\b", span_l)
            or re.search(rf"\b{re.escape(p)}\b", span_l)
            for p in valid_pages
        )
        if has_doc or has_page:
            return True

    return False


def _extract_citations(text: str, retrieved_chunks: list) -> list:
    """
    Return only the bracketed spans that pass the grounded-citation check.
    """
    valid_codes = _citable_codes(retrieved_chunks)
    valid_docs = _citable_documents(retrieved_chunks)
    valid_pages = _citable_pages(retrieved_chunks)
    out = []

    for span in re.findall(r"\[([^\]]+)\]", text):
        grounded = any(
            f.strip().lower() in valid_codes for f in _CODE_RE.findall(span)
        )
        if not grounded and not valid_codes:
            # General documents: accept document name or page number
            span_l = span.lower()
            has_doc = any(doc and doc in span_l for doc in valid_docs)
            has_page = any(
                re.search(rf"\bpage\s*{re.escape(p)}\b", span_l)
                or re.search(rf"\bp\.?\s*{re.escape(p)}\b", span_l)
                for p in valid_pages
            )
            grounded = has_doc or has_page
        if grounded and span not in out:
            out.append(span)

    return out


def _call_ollama(prompt: str, system: str) -> tuple[str, float]:
    """
    Make a blocking call to Ollama's /api/generate endpoint.
    Raises httpx.HTTPError or RuntimeError on failure.
    """
    payload = {
        "model": MODEL_TAG,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "num_ctx": NUM_CTX,
        },
    }
    url = f"{OLLAMA_URL}/api/generate"
    log.info("Calling Ollama model '%s' …", MODEL_TAG)
    t0 = time.perf_counter()

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    answer = result.get("response", "").strip()

    log.info("Ollama response in %.0fms (%d chars)", elapsed_ms, len(answer))

    if elapsed_ms < 500:
        log.warning(
            "SUSPICIOUS: LLM responded in %.0fms — "
            "this may indicate the model is not actually being invoked.",
            elapsed_ms,
        )

    return answer, elapsed_ms


def generate(question: str, retrieved_chunks: list, history: list | None = None) -> dict:
    """
    Generate a grounded answer from retrieved chunks and conversation history.
    """
    if not retrieved_chunks:
        raise ValueError(
            "generate() called with empty retrieved_chunks. "
            "The pipeline must return NOT_FOUND_MSG without calling the generator."
        )

    t_prompt_start = time.perf_counter()
    context_block = _build_context_block(retrieved_chunks)
    history_block = _build_history_block(history)

    prompt_parts = []
    if history_block:
        prompt_parts.append(history_block)
    prompt_parts.append(f"Context:\n{context_block}")
    prompt_parts.append(f"Question: {question}")
    prompt = "\n\n".join(prompt_parts)
    prompt_construction_ms = (time.perf_counter() - t_prompt_start) * 1000

    log.info("Prompt construction completed in %.2fms", prompt_construction_ms)

    guardrail_triggered = False
    answer = ""
    elapsed_ms = 0.0

    # ── Attempt 1 ─────────────────────────────────────────────────────────
    try:
        answer, elapsed_ms = _call_ollama(prompt, _SYSTEM_PROMPT)
    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s — is it running?", OLLAMA_URL)
        return {
            "answer": DEGRADED_MSG,
            "citations": [],
            "latency_ms": 0,
            "guardrail_triggered": False,
            "error": "ollama_unavailable",
        }
    except Exception as e:
        log.error("Ollama call failed: %s", e)
        return {
            "answer": DEGRADED_MSG,
            "citations": [],
            "latency_ms": 0,
            "guardrail_triggered": False,
            "error": str(e),
        }

    # ── Citation guardrail ─────────────────────────────────────────────────
    if not _has_citation(answer, retrieved_chunks):
        log.warning("Citation guardrail triggered — regenerating …")
        guardrail_triggered = True

        # Tailor the citation instruction to the document type
        has_codes = bool(_citable_codes(retrieved_chunks))
        if has_codes:
            cite_instruction = (
                "IMPORTANT: Your answer MUST include at least one citation "
                "in the format [Document Name, Error Code]. Do not omit "
                "citations."
            )
        else:
            cite_instruction = (
                "IMPORTANT: Your answer MUST include at least one citation "
                "in the format [Document Name, page N] where N is the page "
                "number from the source. Do not omit citations."
            )

        retry_parts = []
        if history_block:
            retry_parts.append(history_block)
        retry_parts.append(f"Context:\n{context_block}")
        retry_parts.append(f"Question: {question}")
        retry_parts.append(cite_instruction)
        retry_prompt = "\n\n".join(retry_parts)
        try:
            answer, elapsed_ms = _call_ollama(retry_prompt, _SYSTEM_PROMPT)
        except Exception as e:
            log.error("Retry call failed: %s", e)
            answer = INSUFFICIENT_MSG

        if not _has_citation(answer, retrieved_chunks):
            log.warning("Citation guardrail: retry also failed — using insufficient fallback.")
            answer = INSUFFICIENT_MSG

    citations = _extract_citations(answer, retrieved_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": round(elapsed_ms),
        "prompt_construction_ms": round(prompt_construction_ms, 3),
        "ollama_inference_ms": round(elapsed_ms, 3),
        "guardrail_triggered": guardrail_triggered,
    }
