import json
import re

import pytest

from app.ingestion.image_captioner import (
    generate_image_caption_and_metadata,
    save_image_metadata_store,
)
from app.rag.pipeline import query
from app.rag.retriever import _is_diagram_or_image_query, get_retriever


@pytest.fixture(autouse=True)
def mock_ollama_generator(monkeypatch):
    """Autouse fixture to mock Ollama LLM calls for offline unit testing."""
    def _mock_call_ollama(prompt: str, system: str) -> tuple[str, float]:
        doc_match = re.search(r"Source:\s*([^,\)\n]+)", prompt)
        page_match = re.search(r"page\s*(\d+)", prompt, re.IGNORECASE)
        code_match = re.search(r"Error Code:\s*(0x[0-9a-fA-F]{4})", prompt)

        doc = doc_match.group(1).strip() if doc_match else "Documentation"
        page = page_match.group(1) if page_match else "1"
        code = code_match.group(1) if code_match else None

        if code:
            citation = f"[{doc}, {code}]"
        else:
            citation = f"[{doc}, page {page}]"

        p_lower = prompt.lower()
        if "intake" in p_lower:
            text = f"Image on page {page} shows intake air meter details {citation}."
        elif "diagram" in p_lower or "image" in p_lower:
            text = f"Diagram on page {page} details {citation}."
        else:
            text = f"Document content for page {page} {citation}."

        return (text, 10.0)

    monkeypatch.setattr("app.rag.generator._call_ollama", _mock_call_ollama)


def test_is_diagram_or_image_query():
    assert _is_diagram_or_image_query("Explain the diagram on page 55") is True
    assert _is_diagram_or_image_query("What does the image show?") is True
    assert _is_diagram_or_image_query("Describe figure 2") is True
    assert _is_diagram_or_image_query("Show the chart") is True
    assert (
        _is_diagram_or_image_query("What is the error code 0x1234?") is False
    )


def test_generate_image_caption_and_metadata():
    raw_item = {
        "image_id": "page_55_img_0",
        "page_number": 55,
        "document": "test.pdf",
        "image_file_path": "/tmp/page_55_0.png",
        "caption": "Coolant temperature gauge",
    }
    page_text = "GENERAL INFORMATION\nCoolant temperature shows HI state."
    meta = generate_image_caption_and_metadata(raw_item, page_text)

    assert meta["image_id"] == "page_55_img_0"
    assert meta["page_number"] == 55
    assert meta["document_name"] == "test.pdf"
    assert "Coolant temperature gauge" in meta["image_caption"]
    assert "GENERAL INFORMATION" in meta["nearby_text_context"]


def test_save_image_metadata_store(tmp_path):
    images = [
        {
            "image_id": "page_1_img_0",
            "page_number": 1,
            "document_name": "test.pdf",
            "image_caption": "Caption 1",
        }
    ]
    out_file = tmp_path / "image_metadata.jsonl"
    save_image_metadata_store(images, out_file)
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["image_id"] == "page_1_img_0"


def test_diagram_retrieval_no_image_on_page():
    # Page 5 has 0 images in our manual index
    res = query("Explain the diagram on page 5.")
    assert "No image or diagram was found on page 5." in res.answer
    assert res.found is False


def test_diagram_retrieval_page_with_images():
    retriever = get_retriever()
    retriever.reload()
    # Query an extracted image page or general diagram query
    res = query("Explain the image on page 29.")
    assert res.answer is not None


def test_ocr_query_with_text():
    res = query("What text is visible inside image page_55_img_0?")
    assert res.found is True
    assert "page 55" in res.answer.lower() or "intake" in res.answer.lower()


def test_ocr_query_no_text_on_page():
    res = query("Extract text from image on page 5.")
    assert "No readable text was detected in the image." in res.answer
    assert res.found is False
