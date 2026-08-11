import json
from pathlib import Path

from app.rag.retriever import _extract_page_number_from_query, get_retriever


def _available_general_page() -> int:
    chunks_path = Path("data/index/general/chunks.jsonl")
    pages = {
        json.loads(line).get("page_number")
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return min(page for page in pages if page is not None)


def test_extract_page_number_from_query():
    assert _extract_page_number_from_query("What is on page 54?") == 54
    assert _extract_page_number_from_query("Summarize page 54.") == 54
    assert (
        _extract_page_number_from_query("Explain the diagram on page 54.")
        == 54
    )
    assert (
        _extract_page_number_from_query("Show the table on page 171.")
        == 171
    )
    assert _extract_page_number_from_query("Are there images on p.54?") == 54
    assert _extract_page_number_from_query("Check P. 54") == 54


def test_extract_page_number_ignores_fault_codes():
    assert _extract_page_number_from_query("Fault Code 54") is None
    assert _extract_page_number_from_query("code 54") is None
    assert (
        _extract_page_number_from_query("How to adjust brake lever?")
        is None
    )


def test_page_aware_retrieval_active_index():
    retriever = get_retriever()
    retriever.reload()

    # Use a page present in the checked-in active index.  The corpus is
    # replaceable, so a fixed page from an older manual is not reliable.
    page = _available_general_page()
    q1 = f"What is on page {page}?"
    res1 = retriever.retrieve(q1)
    assert len(res1) > 0
    assert all(r.chunk.get("page_number") == page for r in res1)
    assert res1[0].score >= 0.90

    q2 = f"Summarize page {page}."
    res2 = retriever.retrieve(q2)
    assert len(res2) > 0
    assert all(r.chunk.get("page_number") == page for r in res2)

    q3 = f"Are there images on page {page}?"
    res3 = retriever.retrieve(q3)
    assert len(res3) > 0
    assert all(r.chunk.get("page_number") == page for r in res3)

    q4 = f"Explain the diagram on page {page}."
    res4 = retriever.retrieve(q4)
    assert len(res4) > 0
    assert all(r.chunk.get("page_number") == page for r in res4)

    # Non-existent page query
    q_nonexistent = "What is on page 9999?"
    res_nonexistent = retriever.retrieve(q_nonexistent)
    assert len(res_nonexistent) == 0
