from app.rag.retriever import _extract_page_number_from_query, get_retriever


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

    # Queries asking for page 54
    q1 = "What is on page 54?"
    res1 = retriever.retrieve(q1)
    assert len(res1) > 0
    assert all(r.chunk.get("page_number") == 54 for r in res1)
    assert res1[0].score >= 0.90

    q2 = "Summarize page 54."
    res2 = retriever.retrieve(q2)
    assert len(res2) > 0
    assert all(r.chunk.get("page_number") == 54 for r in res2)

    q3 = "Are there images on page 54?"
    res3 = retriever.retrieve(q3)
    assert len(res3) > 0
    assert all(r.chunk.get("page_number") == 54 for r in res3)

    q4 = "Explain the diagram on page 54."
    res4 = retriever.retrieve(q4)
    assert len(res4) > 0
    assert all(r.chunk.get("page_number") == 54 for r in res4)

    # Non-existent page query
    q_nonexistent = "What is on page 9999?"
    res_nonexistent = retriever.retrieve(q_nonexistent)
    assert len(res_nonexistent) == 0
