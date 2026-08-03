"""
Chunker for general (unstructured) documents.

Converts parsed content items from general_parser into embedding-ready chunk
records.  Uses a paragraph/section-aware strategy with configurable overlap
so that adjacent chunks share context at their boundaries.

Chunk schema is a superset of the fault-code chunker's schema, ensuring
forward compatibility when the two indexes are eventually merged.  Every
chunk includes ``source_document`` so retrieval and citations can
distinguish origin once combined.

Table chunks from the parser are kept intact (not split row-by-row).
Image items produce lightweight metadata chunks that embed the caption
text and reference the saved image file path.
"""
import json
import re
import uuid
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)

# Defaults for the paragraph-aware chunking strategy.
DEFAULT_MAX_TOKENS = 600      # ~500-800 token target
DEFAULT_OVERLAP_RATIO = 0.12  # ~10-15 % overlap between adjacent chunks
_TOKEN_APPROX_CHARS = 4       # rough: 1 token ~ 4 chars in English


def _estimate_tokens(text: str) -> int:
    """Fast token-count approximation (word-based, matching existing chunker)."""
    return len(text.split())


def _split_paragraphs(text: str) -> list[str]:
    """
    Split text on double-newlines (paragraph boundaries), stripping empty
    fragments.  Preserves single-newline wraps within a paragraph.
    """
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _make_chunk_record(
    chunk_text: str,
    document_name: str,
    source_hash: str,
    page_number: int | None,
    chunk_type: str,
    section_heading: str | None = None,
    image_file_path: str | None = None,
    figure_references: list[str] | None = None,
    image_id: str | None = None,
    image_caption: str | None = None,
    nearby_text_context: str | None = None,
    ocr_text: str | None = None,
) -> dict[str, Any]:
    """
    Build a single chunk dict.

    Schema mirrors the fault-code chunker with additional general-document
    fields.  Fault-code-specific fields (error_code, sl_no, etc.) are set
    to None so the record is structurally compatible.
    """
    return {
        "chunk_id": str(uuid.uuid4()),
        "chunk_text": chunk_text,
        # -- Shared fields (compatible with fault-code schema) --
        "document_name": document_name,
        "document_version": None,
        "system_name": None,
        "subsystem": None,
        "chapter": section_heading,
        "section": section_heading,
        "page_number": page_number,
        "error_code": None,
        "chunk_type": chunk_type,       # 'prose', 'table', or 'image'
        "token_count": _estimate_tokens(chunk_text),
        "source_hash": source_hash,
        # -- Fault-code convenience fields (null for general docs) --
        "sl_no": None,
        "error_description": None,
        "error_remarks": None,
        # -- General-document-specific fields --
        "section_heading": section_heading,
        "image_file_path": image_file_path,
        "figure_references": figure_references or [],
        "image_id": image_id,
        "image_caption": image_caption,
        "ocr_text": ocr_text,
        "nearby_text_context": nearby_text_context,
    }


def _chunk_prose_item(
    item: dict[str, Any],
    document_name: str,
    source_hash: str,
    max_tokens: int,
    overlap_ratio: float,
) -> list[dict[str, Any]]:
    """
    Chunk a prose content item into segments of approximately ``max_tokens``
    words, splitting on paragraph boundaries where possible and inserting
    overlap text between adjacent chunks.
    """
    text = item["text"]
    page = item.get("page_number")
    heading = item.get("section_heading")

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    overlap_tokens = int(max_tokens * overlap_ratio)
    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If a single paragraph exceeds max, split it on sentences
        if para_tokens > max_tokens * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent_tokens = _estimate_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current_parts:
                    chunk_text = "\n\n".join(current_parts)
                    chunks.append(_make_chunk_record(
                        chunk_text, document_name, source_hash, page,
                        "prose", section_heading=heading,
                    ))
                    # Overlap: keep the tail of the current chunk
                    current_parts = _overlap_tail(current_parts, overlap_tokens)
                    current_tokens = _estimate_tokens("\n\n".join(current_parts))
                current_parts.append(sent)
                current_tokens += sent_tokens
        elif current_tokens + para_tokens > max_tokens and current_parts:
            # Flush current chunk
            chunk_text = "\n\n".join(current_parts)
            chunks.append(_make_chunk_record(
                chunk_text, document_name, source_hash, page,
                "prose", section_heading=heading,
            ))
            # Overlap
            current_parts = _overlap_tail(current_parts, overlap_tokens)
            current_tokens = _estimate_tokens("\n\n".join(current_parts))
            current_parts.append(para)
            current_tokens += para_tokens
        else:
            current_parts.append(para)
            current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunk_text = "\n\n".join(current_parts)
        chunks.append(_make_chunk_record(
            chunk_text, document_name, source_hash, page,
            "prose", section_heading=heading,
        ))

    return chunks


def _overlap_tail(parts: list[str], overlap_tokens: int) -> list[str]:
    """
    Return the last few paragraphs/sentences from ``parts`` that together
    contain roughly ``overlap_tokens`` words, for context overlap.
    """
    if overlap_tokens <= 0:
        return []
    collected: list[str] = []
    total = 0
    for part in reversed(parts):
        t = _estimate_tokens(part)
        if total + t > overlap_tokens and collected:
            break
        collected.append(part)
        total += t
    collected.reverse()
    return collected


def _chunk_table_item(
    item: dict[str, Any],
    document_name: str,
    source_hash: str,
) -> dict[str, Any]:
    """
    Convert a table item into a single chunk (table rows stay together).
    """
    data = item["data"]
    page = item.get("page_number")
    # Format table as markdown-ish text for embedding
    lines = []
    for i, row in enumerate(data):
        lines.append(" | ".join(str(c) for c in row))
        if i == 0:
            # Separator after header row
            lines.append(" | ".join("---" for _ in row))
    chunk_text = "\n".join(lines)
    return _make_chunk_record(
        chunk_text, document_name, source_hash, page, "table",
    )


def _chunk_image_item(
    item: dict[str, Any],
    document_name: str,
    source_hash: str,
) -> dict[str, Any] | None:
    """
    Create a metadata chunk for an extracted image.

    The chunk text contains the image ID, caption, nearby text context, and file path.
    Only text/metadata is embedded -- raw image bytes are never part of FAISS vectors.
    """
    caption = item.get("image_caption") or item.get("caption")
    page = item.get("page_number")
    img_path = item.get("image_file_path") or item.get("image_path") or ""
    img_id = item.get("image_id") or f"page_{page}_img"
    nearby = item.get("nearby_text_context")
    ocr_text = item.get("ocr_text")

    parts = [
        f"[Image: {img_id}] Diagram / Figure / Chart / Image on Page {page}"
    ]
    if ocr_text:
        parts.append(f"OCR Text: {ocr_text}")
    if caption:
        parts.append(f"Caption / Description: {caption}")
    if nearby:
        parts.append(f"Nearby Context: {nearby}")
    if img_path:
        parts.append(f"Source image file: {img_path}")
    chunk_text = "\n".join(parts)

    return _make_chunk_record(
        chunk_text,
        document_name,
        source_hash,
        page,
        "image",
        image_file_path=img_path,
        image_id=img_id,
        image_caption=caption,
        nearby_text_context=nearby,
        ocr_text=ocr_text,
    )


def build_general_chunks(
    parse_result: dict[str, Any],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[dict[str, Any]]:
    """
    Convert parse_result (from general_parser.parse_general_pdf) into a
    list of chunk dicts ready for embedding.

    Args:
        parse_result: Output of parse_general_pdf.
        max_tokens: Target max words per prose chunk.
        overlap_ratio: Fraction of max_tokens to overlap between adjacent
            prose chunks (0.0 to disable overlap).

    Returns:
        List of chunk dicts.
    """
    items = parse_result["items"]
    document_name = parse_result["source_document"]
    source_hash = parse_result["source_hash"]

    chunks: list[dict[str, Any]] = []

    for item in items:
        item_type = item.get("type", "prose")

        if item_type == "prose":
            chunks.extend(_chunk_prose_item(
                item, document_name, source_hash, max_tokens, overlap_ratio,
            ))
        elif item_type == "table":
            chunks.append(_chunk_table_item(
                item, document_name, source_hash,
            ))
        elif item_type == "image":
            img_chunk = _chunk_image_item(
                item, document_name, source_hash,
            )
            if img_chunk:
                chunks.append(img_chunk)
        else:
            log.warning("Unknown item type '%s' -- skipping", item_type)

    # Enrich prose chunks with figure references from image items on the
    # same page (e.g. "see Figure 2-1" in prose should link to the image).
    _link_figure_references(chunks)

    log.info(
        "Built %d general chunks from %d items (source: %s)",
        len(chunks), len(items), document_name,
    )
    return chunks


def _link_figure_references(chunks: list[dict[str, Any]]) -> None:
    """
    For each prose chunk that mentions a figure (e.g. 'see Figure 2-1'),
    find the image chunk on the same page with a matching caption and add
    the image_file_path to the prose chunk's figure_references list.
    """
    # Build lookup: page_number -> list of image chunks on that page
    images_by_page: dict[int, list[dict]] = {}
    for c in chunks:
        if c.get("chunk_type") == "image" and c.get("page_number"):
            images_by_page.setdefault(c["page_number"], []).append(c)

    fig_pattern = re.compile(
        r"(?:Figure|Fig\.?)\s+([\d\-]+)", re.IGNORECASE,
    )

    for c in chunks:
        if c.get("chunk_type") != "prose":
            continue
        text = c.get("chunk_text", "")
        refs = fig_pattern.findall(text)
        if not refs:
            continue
        page = c.get("page_number")
        page_images = images_by_page.get(page, [])
        for img_chunk in page_images:
            caption = img_chunk.get("chunk_text", "")
            img_path = img_chunk.get("image_file_path", "")
            for ref_id in refs:
                if ref_id in caption and img_path:
                    if img_path not in c["figure_references"]:
                        c["figure_references"].append(img_path)


def save_general_chunks(
    chunks: list[dict[str, Any]], path: Path,
) -> None:
    """Persist chunks to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Saved %d general chunks -> %s", len(chunks), path)


def load_general_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# -- Post-chunk boilerplate deduplication ----------------------------------

# A chunk whose first-paragraph word set has overlap coefficient >= this
# with another chunk's first paragraph, and the cluster spans >= this many
# pages, is considered boilerplate and deduplicated.
_BOILERPLATE_OVERLAP = 0.80
_BOILERPLATE_MIN_PAGES = 6
_BOILERPLATE_LEAD_WORDS = 80


def _overlap_coeff(a: set[str], b: set[str]) -> float:
    """|A ∩ B| / min(|A|, |B|).  1.0 when one is a subset of the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _lead_word_set(text: str, max_words: int = _BOILERPLATE_LEAD_WORDS) -> set[str]:
    """First N words of text, lowercased, as a set."""
    return set(text.lower().split()[:max_words])


def deduplicate_boilerplate_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Remove prose chunks whose lead text repeats across many pages.

    The first paragraph of each page often contains a repeated safety
    warning or legal disclaimer.  After chunking, this boilerplate sits
    at the start of the first chunk from that page, mixed with unique
    content.  Jaccard on full chunk text fails (unique content dilutes
    the signal).  Instead, we compare only the first ~80 words of each
    chunk (the boilerplate region) using overlap coefficient, which
    measures what fraction of the smaller set is contained in the larger.

    If a cluster of chunks with similar lead text spans >=
    ``_BOILERPLATE_MIN_PAGES`` distinct pages, keep only the first
    occurrence and drop the rest.  Table and image chunks are never
    dropped.

    Returns a new list with boilerplate duplicates removed.
    """
    min_pages = _BOILERPLATE_MIN_PAGES

    # Build lead word sets for prose chunks
    prose_indices: list[int] = []
    prose_leads: list[set[str]] = []
    for i, chunk in enumerate(chunks):
        if chunk.get("chunk_type") != "prose":
            continue
        text = chunk.get("chunk_text", "")
        lead = _lead_word_set(text)
        if lead:
            prose_indices.append(i)
            prose_leads.append(lead)

    # Cluster by overlap coefficient on lead text (greedy)
    clustered: list[tuple[set[str], list[int]]] = []
    for idx, lead in zip(prose_indices, prose_leads):
        matched = False
        for canon_lead, members in clustered:
            if not canon_lead or not lead:
                continue
            if _overlap_coeff(lead, canon_lead) >= _BOILERPLATE_OVERLAP:
                members.append(idx)
                matched = True
                break
        if not matched:
            clustered.append((lead, [idx]))

    # Find clusters spanning too many pages -> boilerplate
    drop_indices: set[int] = set()
    boilerplate_clusters = 0
    for canon_lead, members in clustered:
        if len(members) < min_pages:
            continue
        pages = {chunks[m].get("page_number") for m in members}
        if len(pages) >= min_pages:
            boilerplate_clusters += 1
            for m in members[1:]:
                drop_indices.add(m)

    if not drop_indices:
        return chunks

    filtered = [
        chunk for i, chunk in enumerate(chunks) if i not in drop_indices
    ]

    sample_text = ""
    for idx in sorted(drop_indices):
        t = chunks[idx].get("chunk_text", "")
        if t:
            sample_text = t[:80].replace("\n", " ")
            break

    log.info(
        "Boilerplate dedup: removed %d chunk(s) across %d cluster(s) "
        "(sample: '%s...')",
        len(drop_indices), boilerplate_clusters, sample_text,
    )
    return filtered
