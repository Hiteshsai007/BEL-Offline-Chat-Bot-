"""
Image Captioner and Metadata Processor module for PDF ingestion.
Extracts rich descriptions, nearby text context, and layout metadata for
extracted images.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("bel.app.ingestion.image_captioner")


def generate_image_caption_and_metadata(
    image_item: dict[str, Any],
    page_text: str | None = None,
) -> dict[str, Any]:
    """
    Generate rich image metadata and caption for an extracted PDF image.

    Args:
        image_item: Raw image dict containing page_number, image_id, etc.
        page_text: Plain text extracted from the same page as the image.

    Returns:
        Enriched image dict containing metadata fields.
    """
    page_num = image_item.get("page_number", 0)
    img_id = image_item.get("image_id") or f"page_{page_num}_img_0"
    doc_name = (
        image_item.get("document")
        or image_item.get("document_name")
        or "Document.pdf"
    )
    img_path = (
        image_item.get("image_file_path")
        or image_item.get("image_path")
        or ""
    )

    existing_caption = image_item.get("caption")

    # Extract nearby text context from page text
    nearby_context = ""
    if page_text:
        lines = [
            line.strip()
            for line in page_text.splitlines()
            if line.strip() and not line.strip().startswith("<!--")
        ]
        # Look for lines mentioning figure/diagram/chart or select surrounding text
        fig_pat = r"\b(figure|fig|diagram|chart|table|notice|warning)\b"
        fig_lines = [
            line for line in lines
            if re.search(fig_pat, line, re.IGNORECASE)
        ]
        if fig_lines:
            nearby_context = " ".join(fig_lines[:4])
        else:
            nearby_context = " ".join(lines[:4])

    # Perform OCR on image file if present
    from app.ingestion.ocr_engine import perform_ocr_on_image
    ocr_text = perform_ocr_on_image(img_path) if img_path else ""

    # Build structured caption
    caption_parts = []
    if existing_caption:
        caption_parts.append(f"Caption: {existing_caption}.")
    else:
        caption_parts.append(
            f"Diagram/Figure '{img_id}' on Page {page_num} of {doc_name}."
        )

    if ocr_text:
        caption_parts.append(f"OCR Text: {ocr_text}")

    if nearby_context:
        caption_parts.append(f"Nearby Context: {nearby_context}")

    full_caption = " ".join(caption_parts)

    metadata = {
        "image_id": img_id,
        "page_number": page_num,
        "document_name": doc_name,
        "image_path": str(img_path),
        "image_caption": full_caption,
        "ocr_text": ocr_text,
        "nearby_text_context": nearby_context,
    }

    return metadata


def save_image_metadata_store(
    images: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save image metadata records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for img in images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")
    log.info("Saved %d image metadata records -> %s", len(images), output_path)
