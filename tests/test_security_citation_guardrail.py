"""
tests/test_security_citation_guardrail.py

Covers security finding S-6: the citation guardrail used
``re.search(r"\\[.+?,\\s*.+?\\]", text)`` -- any bracketed comma. "[1, 2]" or
"[TODO, fix]" satisfied it, so the control the PRD positions as the
anti-hallucination guarantee could be passed without citing anything real.

The guardrail now requires a bracketed span to reference an error code that
both matches the configured error-code pattern AND appears in the retrieved
context, so a fabricated-but-plausible code is rejected too.

Citation format under test is the one the codebase actually generates:
    app/rag/pipeline.py:66      -> "[Source: IRL Fault Codes.pdf, 0x0003]"
    app/rag/generator.py:52     -> "(Source: <doc>, Error Code: <code>)"
    system prompt               -> "[Document, Error Code/Section]"

Cross-platform notes
--------------------
* Pure string/regex assertions -- no I/O, no network, no model. Identical
  behaviour on Windows and Linux.
"""
from app.rag.generator import _extract_citations, _has_citation
from app.rag.retriever import RetrievedChunk

DOC = "IRL Fault Codes.pdf"


def _chunk(code: str = "0x0003", doc: str = DOC) -> RetrievedChunk:
    """Build a RetrievedChunk mirroring the real corpus schema."""
    return RetrievedChunk(
        chunk={
            "error_code": code,
            "document_name": doc,
            "error_description": "Fire aborted",
            "error_remarks": "Fire command aborted by operator",
            "chunk_text": "Error Code: 0x0003 | Error Description: Fire aborted",
            "chunk_type": "table",
        },
        score=1.0,
    )


CONTEXT = [_chunk("0x0003"), _chunk("0x0017")]


# ── Fake citations must be REJECTED ─────────────────────────────────────────

def test_rejects_generic_bracketed_comma() -> None:
    """The exact bypass named in the finding must no longer pass."""
    assert _has_citation("The answer is [not, real].", CONTEXT) is False


def test_rejects_numeric_list_artifact() -> None:
    """A markdown/reference artefact like '[1, 2]' is not a citation."""
    assert _has_citation("See references [1, 2] for details.", CONTEXT) is False


def test_rejects_todo_placeholder() -> None:
    assert _has_citation("Fix this later [TODO, fix].", CONTEXT) is False


def test_rejects_fabricated_error_code() -> None:
    """
    A code that is well-formed but absent from the retrieved context is a
    hallucination and must be rejected -- this is the case a pattern-only
    regex would wrongly accept.
    """
    answer = f"This indicates a launcher fault [{DOC}, 0x9999]."
    assert _has_citation(answer, CONTEXT) is False


def test_rejects_answer_with_no_brackets_at_all() -> None:
    assert _has_citation("Fire aborted by the operator.", CONTEXT) is False


def test_rejects_empty_answer() -> None:
    assert _has_citation("", CONTEXT) is False


# ── Real citations must be ACCEPTED ─────────────────────────────────────────

def test_accepts_real_pipeline_citation_format() -> None:
    """The exact format app/rag/pipeline.py emits must validate."""
    answer = f"**0x0003** — Fire aborted\n  [Source: {DOC}, 0x0003]"
    assert _has_citation(answer, CONTEXT) is True


def test_accepts_system_prompt_citation_format() -> None:
    """The format the system prompt asks the LLM for must validate."""
    answer = f"The fire was aborted [{DOC}, 0x0003]."
    assert _has_citation(answer, CONTEXT) is True


def test_accepts_second_retrieved_code() -> None:
    """Any code present in the context counts, not just the first."""
    answer = f"Depth setting failed [{DOC}, 0x0017]."
    assert _has_citation(answer, CONTEXT) is True


def test_accepts_case_insensitive_hex_code() -> None:
    """Hex codes are case-insensitive; 0X0003 must still match 0x0003."""
    answer = f"Fire aborted [{DOC}, 0X0003]."
    assert _has_citation(answer, CONTEXT) is True


def test_accepts_when_one_of_several_citations_is_real() -> None:
    """A real citation alongside noise is still a pass."""
    answer = f"See [1, 2] and [{DOC}, 0x0003]."
    assert _has_citation(answer, CONTEXT) is True


# ── Prose/footnote fallback ─────────────────────────────────────────────────

def test_accepts_document_citation_when_context_has_no_codes() -> None:
    """
    Footnote/prose chunks carry error_code=None. Requiring a code there would
    make the guardrail unsatisfiable, so document-name grounding is accepted.
    """
    prose = [RetrievedChunk(
        chunk={
            "error_code": None,
            "document_name": DOC,
            "chunk_text": "Note 6: throw range is advisory only.",
            "chunk_type": "prose",
        },
        score=0.8,
    )]
    assert _has_citation(f"Throw range is advisory [{DOC}, Note 6].", prose) is True


def test_rejects_unknown_document_when_context_has_no_codes() -> None:
    """The prose fallback must still reject an unrelated document name."""
    prose = [RetrievedChunk(
        chunk={"error_code": None, "document_name": DOC, "chunk_type": "prose"},
        score=0.8,
    )]
    assert _has_citation("[Some Other Manual.pdf, Note 6]", prose) is False


# ── Citation extraction must be grounded too ────────────────────────────────

def test_extract_citations_drops_ungrounded_spans() -> None:
    """
    The structured citation list must not be polluted with markdown artefacts
    or fabricated references.
    """
    answer = f"See [1, 2], [TODO, fix] and [{DOC}, 0x0003]."
    citations = _extract_citations(answer, CONTEXT)

    assert citations == [f"{DOC}, 0x0003"], (
        f"Expected only the grounded citation, got {citations}."
    )


def test_extract_citations_deduplicates() -> None:
    answer = f"[{DOC}, 0x0003] and again [{DOC}, 0x0003]."
    assert len(_extract_citations(answer, CONTEXT)) == 1


def test_extract_citations_empty_for_fabricated_only() -> None:
    assert _extract_citations(f"[{DOC}, 0x9999]", CONTEXT) == []


# ── Page-number citations for general documents ─────────────────────────────

def _general_chunk(
    doc: str = "Kawasaki Manual.pdf", page: int = 170,
) -> RetrievedChunk:
    """Build a RetrievedChunk for a general document (no error code)."""
    return RetrievedChunk(
        chunk={
            "error_code": None,
            "document_name": doc,
            "error_description": None,
            "error_remarks": None,
            "chunk_text": "Check tire pressure when tires are cold.",
            "chunk_type": "prose",
            "page_number": page,
        },
        score=0.75,
    )


GENERAL_CONTEXT = [_general_chunk("Kawasaki Manual.pdf", 170)]


def test_accepts_page_number_citation_for_general_doc() -> None:
    """General documents cite by page number, not error code."""
    answer = (
        "Check tire pressure when the tires are cold. "
        "[Kawasaki Manual.pdf, page 170]"
    )
    assert _has_citation(answer, GENERAL_CONTEXT) is True


def test_accepts_page_number_short_format() -> None:
    """Short page format like 'p. 170' should also be accepted."""
    answer = "Check tire pressure when cold. [Kawasaki Manual.pdf, p. 170]"
    assert _has_citation(answer, GENERAL_CONTEXT) is True


def test_rejects_wrong_page_number_for_general_doc() -> None:
    """A citation with wrong page BUT correct document is still accepted
    (document-level grounding is sufficient). Wrong page + wrong doc is rejected."""
    # Wrong page, right doc -> accepted (document name grounds it)
    answer_ok = "Check tire pressure. [Kawasaki Manual.pdf, page 999]"
    assert _has_citation(answer_ok, GENERAL_CONTEXT) is True
    # Wrong page, wrong doc -> rejected
    answer_bad = "Check tire pressure. [Honda Manual.pdf, page 999]"
    assert _has_citation(answer_bad, GENERAL_CONTEXT) is False


def test_accepts_document_name_only_for_general_doc() -> None:
    """A citation containing just the document name is accepted."""
    answer = "Check tire pressure when cold. [Kawasaki Manual.pdf]"
    assert _has_citation(answer, GENERAL_CONTEXT) is True


def test_extract_citations_page_number_format() -> None:
    """Page-number citations are extracted correctly."""
    answer = (
        "Check tire pressure when cold. "
        "[Kawasaki Manual.pdf, page 170]"
    )
    citations = _extract_citations(answer, GENERAL_CONTEXT)
    assert len(citations) == 1
    assert "page 170" in citations[0]


def test_build_context_block_includes_page_for_general_doc() -> None:
    """Context block includes page number for general documents."""
    from app.rag.generator import _build_context_block

    block = _build_context_block(GENERAL_CONTEXT)
    assert "page 170" in block
    assert "Error Code: N/A" not in block
