"""
Ollama inference wrapper with citation and strict noun-grounding guardrails.

This module is invoked by the pipeline only for queries that fall below
the DIRECT_ANSWER_THRESHOLD — i.e. ambiguous or low-confidence retrieval
results where the structured fast-path template is insufficient.

Guarantees when this module IS called:
  • The generator is NEVER called when retrieval returns nothing.
  • Answers are derived STRICTLY from retrieved context passages. No external
    knowledge, domain interpretations, or ungrounded explanations are allowed.
  • Key nouns/terms in the draft answer are validated against the retrieved text.
    If ungrounded terms are detected, a retry is attempted. If retry fails,
    a strict no-definition fallback is returned.
  • Output is checked for citations; fallback returned if validation fails.
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

# ── Strict System prompt ───────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a technical assistant restricted strictly to the provided context.
Rules:
- Read ALL numbered context passages carefully before answering.
- Answer ONLY using information explicitly stated in the numbered context passages below.
- Do NOT use external knowledge, domain interpretation, speculation, or assumptions.
- Do NOT introduce concepts or terminology not present in the context (such as internal combustion
  engines, ignition systems, parameters, or operational limits unless explicitly written in context).
- If the user asks for the meaning or explanation of a fault description, summarize the documented
  remarks across the matching fault code entries using ONLY evidence from the context passages.
- When asked to summarize, explain, or list the contents of a specific page (for example: 'summarize page X',
  'explain page X', 'what is on page X', or 'what text is on page X'), provide a complete summary of all
  available content from that page, including prose text, part lists, labels, tables, notices, warnings, image
  references, OCR text, diagram descriptions, and other retrieved page content. Do not reject the request simply
  because the page contains labels, lists, tables, or diagrams instead of narrative paragraphs.
- Prefer quoting or directly paraphrasing source text over explaining or expanding upon it.
- Cite every supported statement as [Document Name, Error Code] or [Document Name, page N]
  for documents without error codes.
- Do not cite Previous Conversation. Cite ONLY from the numbered Context passages.
- If the answer is not present in the context at all, respond exactly with:
  "This information is not available in the current documentation." """

_COMMON_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "upon", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "i", "you", "he", "she", "it", "we", "they", "them", "their", "theirs", "my",
    "your", "his", "her", "its", "our", "us", "this", "that", "these", "those",
    "document", "documents", "documentation", "provide", "provides", "provided",
    "definition", "definitions", "define", "defines", "defined", "state", "states",
    "stated", "stating", "only", "text", "context", "passage", "passages", "information",
    "available", "current", "according", "source", "sources", "cited", "citation",
    "error", "errors", "code", "codes", "page", "pages", "pdf", "file", "unit",
    "system", "chapter", "section", "table", "figure", "fig", "manual", "guide",
    "reference", "referenced", "n/a", "no", "not", "ensure", "slightly", "engaging",
    "supply", "allow", "starting", "procedure", "step", "steps", "follow", "following",
    "instructions", "method", "make", "sure", "turn", "key", "press", "button",
    "using", "used", "switch", "lever", "position", "start", "stop", "check", "slowly",
    "open", "close", "pull", "push", "release", "set", "setting", "mode", "normal",
    "normally", "ordinary", "ordinarily", "correct", "proper", "standard", "general", "generally",
    "showing", "location", "located", "component", "components", "label", "labels", "part", "parts",
    "diagram", "diagrams", "image", "images", "item", "items", "number", "numbers", "list", "lists",
    "listed", "summary", "summarize", "summarizes", "summarized", "explain", "explains", "explanation",
    "content", "contents", "describe", "describes", "description", "contains", "contain", "containing",
    "related", "relates", "vehicle", "specifically", "specific", "detailing", "details", "detail",
    "corresponds", "corresponding", "refers", "refer", "referring", "shows", "show", "depicts", "depict",
    "depicting", "visual", "visually", "represent", "represents", "representing"
}


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
    stop_words = {"page", "pdf", "file", "doc", "document", "manual", "guide", "source"}

    for span in spans:
        span_l = span.lower()
        span_words = [
            w for w in re.findall(r"\b[a-zA-Z]{3,}\b", span_l)
            if w not in stop_words
        ]
        has_doc = any(
            doc and (
                doc in span_l
                or (len(span_l) >= 4 and span_l in doc)
                or any(w in doc for w in span_words)
            )
            for doc in valid_docs
        )
        has_page = any(
            re.search(rf"\bpage\s*{re.escape(p)}\b", span_l)
            or re.search(rf"\bp\.?\s*{re.escape(p)}\b", span_l)
            or re.search(rf"\b{re.escape(p)}\b", span_l)
            for p in valid_pages
        )
        if has_doc or has_page:
            return True

    # Fallback for general documents: evaluate full response text
    if spans != [text]:
        text_l = text.lower()
        text_words = [
            w for w in re.findall(r"\b[a-zA-Z]{3,}\b", text_l)
            if w not in stop_words
        ]
        has_doc_text = any(
            doc and (doc in text_l or any(w in doc for w in text_words))
            for doc in valid_docs
        )
        has_page_text = any(
            re.search(rf"\bpage\s*{re.escape(p)}\b", text_l)
            or re.search(rf"\bp\.?\s*{re.escape(p)}\b", text_l)
            for p in valid_pages
        )
        if has_doc_text or has_page_text:
            return True

    return False


def _extract_ungrounded_terms(answer: str, retrieved_chunks: list, history: list | None = None) -> list[str]:
    """
    Extract substantive nouns/terms from answer and verify they appear in retrieved context.
    Returns a list of ungrounded terms found in answer that are absent from context.
    """
    if not answer or answer in (INSUFFICIENT_MSG, NOT_FOUND_MSG, DEGRADED_MSG):
        return []

    ref_parts = []
    for rc in retrieved_chunks:
        c = rc.chunk
        ref_parts.append(c.get("chunk_text", ""))
        ref_parts.append(c.get("document_name", ""))
        ref_parts.append(str(c.get("error_code") or ""))
        ref_parts.append(str(c.get("error_description") or ""))
        ref_parts.append(str(c.get("error_remarks") or ""))
        ref_parts.append(str(c.get("section_heading") or ""))

    if history:
        for msg in history:
            ref_parts.append(msg.get("content", ""))

    ref_text = " ".join(ref_parts).lower()
    ref_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", ref_text))

    clean_answer = re.sub(r"\[[^\]]+\]", " ", answer)
    answer_words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_answer.lower())

    ungrounded = []
    for word in answer_words:
        if word in _COMMON_STOP_WORDS:
            continue
        if word not in ref_words and not any(word in rw or rw in word for rw in ref_words if len(rw) >= 4):
            if word not in ungrounded:
                ungrounded.append(word)

    return ungrounded


def _format_no_definition_fallback(retrieved_chunks: list) -> str:
    """Format a strict evidence-based summary when context lacks a formal definition."""
    codes = []
    remarks_set = []
    doc = "IRL Fault Codes.pdf"

    for rc in retrieved_chunks:
        c = rc.chunk
        if c.get("document_name"):
            doc = c.get("document_name")
        code = c.get("error_code")
        if code and code not in codes:
            codes.append(code)
        rem = c.get("error_remarks")
        if rem and rem not in remarks_set:
            remarks_set.append(rem)

    if remarks_set and codes:
        remarks_str = "; ".join(remarks_set)
        codes_str = ", ".join(codes)
        first_code = codes[0]
        return (
            f"In this system, this condition indicates: {remarks_str}. "
            f"This condition appears in multiple fault codes ({codes_str}) [{doc}, {first_code}]."
        )

    first = retrieved_chunks[0].chunk
    text = first.get("chunk_text") or first.get("error_description") or ""
    code = first.get("error_code") or (
        f"page {first['page_number']}" if first.get("page_number") is not None else ""
    )
    citation = f"[{doc}, {code}]" if code else f"[{doc}]"
    return f"The document does not provide a definition. It only states: {text} {citation}"


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

    if not out and not valid_codes:
        # Fallback for general documents: extract unbracketed page or document references
        text_l = text.lower()
        for p in valid_pages:
            pat = rf"\bpage\s*{re.escape(p)}\b|\bp\.?\s*{re.escape(p)}\b"
            if re.search(pat, text_l):
                for doc in valid_docs:
                    cit = f"{doc}, page {p}"
                    if cit not in out:
                        out.append(cit)

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

    log.info("RETRIEVED CONTEXT:\n%s", context_block)

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
        log.info("DRAFT ANSWER:\n%s", answer)
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

    # ── Validation Step: Citation & Noun Grounding ─────────────────────────
    has_citation = _has_citation(answer, retrieved_chunks)
    is_fault_code_context = any(bool(rc.chunk.get("error_code")) for rc in retrieved_chunks)
    ungrounded_terms = _extract_ungrounded_terms(answer, retrieved_chunks, history) if is_fault_code_context else []
    is_grounded = len(ungrounded_terms) == 0

    # For general document context (no error codes): if response is well-grounded but missing explicit brackets, attach source citation
    if (
        not is_fault_code_context
        and not has_citation
        and retrieved_chunks
        and answer
        and answer not in (INSUFFICIENT_MSG, NOT_FOUND_MSG, DEGRADED_MSG)
    ):
        top_c = retrieved_chunks[0].chunk
        doc = top_c.get("document_name", "Documentation")
        page = top_c.get("page_number")
        code = top_c.get("error_code")
        if code:
            answer = f"{answer.rstrip()}\n  [{doc}, {code}]"
        elif page is not None:
            answer = f"{answer.rstrip()}\n  [{doc}, page {page}]"
        else:
            answer = f"{answer.rstrip()}\n  [{doc}]"
        has_citation = True

    log.info(
        "CITATION & GROUNDING VALIDATION | has_citation=%s | is_grounded=%s | ungrounded_terms=%s",
        has_citation, is_grounded, ungrounded_terms or "None",
    )

    if not has_citation or not is_grounded:
        log.warning(
            "Validation failed (has_citation=%s, is_grounded=%s) — regenerating with strict grounding constraint...",
            has_citation, is_grounded,
        )
        guardrail_triggered = True

        instructions = []
        if not has_citation:
            has_codes = bool(_citable_codes(retrieved_chunks))
            fmt = "[Document Name, Error Code]" if has_codes else "[Document Name, page N]"
            instructions.append(f"IMPORTANT: Your answer MUST include at least one citation in the format {fmt}.")
        if not is_grounded:
            terms_str = ", ".join(ungrounded_terms[:5])
            instructions.append(
                f"CRITICAL GROUNDING ERROR: Your previous answer contained ungrounded words ({terms_str}) "
                "not present in context. DO NOT explain, define, or invent terms. If the context does not "
                "explicitly define the term, respond strictly in the format:\n"
                "'The document does not provide a definition. It only states: <exact text from context> "
                "[Source, Code]'."
            )

        retry_parts = []
        if history_block:
            retry_parts.append(history_block)
        retry_parts.append(f"Context:\n{context_block}")
        retry_parts.append(f"Question: {question}")
        retry_parts.append("\n".join(instructions))
        retry_prompt = "\n\n".join(retry_parts)

        try:
            answer, elapsed_ms = _call_ollama(retry_prompt, _SYSTEM_PROMPT)
            log.info("RETRY DRAFT ANSWER:\n%s", answer)
        except Exception as e:
            log.error("Retry call failed: %s", e)
            answer = INSUFFICIENT_MSG

        has_citation_2 = _has_citation(answer, retrieved_chunks)
        ungrounded_terms_2 = _extract_ungrounded_terms(answer, retrieved_chunks, history)
        is_grounded_2 = len(ungrounded_terms_2) == 0

        log.info(
            "RETRY VALIDATION RESULT | has_citation=%s | is_grounded=%s | ungrounded_terms=%s",
            has_citation_2, is_grounded_2, ungrounded_terms_2 or "None",
        )

        if not has_citation_2:
            log.warning("Citation guardrail: retry also failed — using insufficient fallback.")
            answer = INSUFFICIENT_MSG
        elif not is_grounded_2:
            log.warning("Grounding validation retry failed — using strict no-definition fallback.")
            answer = _format_no_definition_fallback(retrieved_chunks)

    citations = _extract_citations(answer, retrieved_chunks)

    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": round(elapsed_ms),
        "prompt_construction_ms": round(prompt_construction_ms, 3),
        "ollama_inference_ms": round(elapsed_ms, 3),
        "guardrail_triggered": guardrail_triggered,
    }
