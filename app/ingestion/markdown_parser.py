"""
Markdown parser for general document ingestion.

Converts a .md file into the same parse_result dict format that
general_parser.parse_general_pdf() produces, allowing
general_chunker.build_general_chunks() to work unchanged for
both PDF and Markdown source documents.

Supported Markdown features:
  - ATX headings (#, ##, ###, etc.): H1 headings create synthetic
    page boundaries; sub-headings (## and deeper) update the
    section_heading metadata of subsequent items on the same page.
  - GFM pipe tables (| col | ... | with a separator row): converted
    to table items matching the PDF parser's table schema.
  - All other non-empty content: treated as prose items.

No external dependencies beyond the Python standard library.
"""
import hashlib
import re
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)

# GFM separator row: | :--- | --- | ---: | etc.  (only dashes, colons, pipes, spaces)
_TABLE_SEP_RE = re.compile(r"^\|[\s\-:| ]+\|$")

# GFM table row: starts and ends with |
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")

# ATX heading of any depth
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# H1-only variant used for page-boundary splitting
_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file, read in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _parse_table_block(lines: list) -> list | None:
    """
    Convert a list of consecutive GFM table lines into a list-of-lists.

    Returns None when:
      - fewer than 3 lines (need header + separator + at least one data row),
      - the second line is not a valid separator row.

    The separator row itself is dropped from the output.
    """
    if len(lines) < 3:
        return None
    if not _TABLE_SEP_RE.match(lines[1].strip()):
        return None

    rows = []
    for i, line in enumerate(lines):
        if i == 1:          # skip separator
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows if rows else None


# ---------------------------------------------------------------------------
# Section splitting
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE:(\d+)\s*-->")
_IMAGE_REF_RE = re.compile(r"\[Image:\s*([^\]]+)\]")


def _split_into_sections(text: str) -> list:
    """
    Split Markdown text into sections based on explicit <!-- PAGE:N --> markers
    if present, otherwise fall back to splitting on H1 (single-#) headings.
    """
    sections: list = []

    if _PAGE_MARKER_RE.search(text):
        parts = _PAGE_MARKER_RE.split(text)
        # parts alternates: [pre_content, page_num_str_1, body_1, page_num_str_2, body_2, ...]
        pre = parts[0].strip()
        if pre:
            sections.append({"heading": None, "content": pre, "page_number": 1})

        it = iter(parts[1:])
        for page_str in it:
            try:
                p_num = int(page_str)
            except ValueError:
                p_num = 1
            body = next(it, "")
            if body.strip():
                sections.append({
                    "heading": None,
                    "content": body,
                    "page_number": p_num,
                })
        return sections

    parts = _H1_RE.split(text)
    # _H1_RE.split alternates: [pre_content, h1_title, body, h1_title, body, ...]

    page = 0
    pre = parts[0].strip()
    if pre:
        page += 1
        sections.append({"heading": None, "content": pre, "page_number": page})

    it = iter(parts[1:])
    for title in it:
        body = next(it, "")
        page += 1
        sections.append({
            "heading": title.strip(),
            "content": body,
            "page_number": page,
        })

    if not sections:
        # No H1 headings at all — the whole file is one section.
        sections.append({"heading": None, "content": text, "page_number": 1})

    return sections


# ---------------------------------------------------------------------------
# Section content parsing
# ---------------------------------------------------------------------------

def _parse_section_content(
    content: str,
    page_number: int,
    section_heading: str | None,
) -> list:
    """
    Parse a single section's content into a list of prose and table items.

    Sub-headings (## H2, ### H3, …) within the section update
    ``section_heading`` for all items that follow them on the same page
    but do NOT create new page boundaries.

    Returns a list of content-item dicts, each with at least::

        {"type": "prose"|"table", "page_number": int, "section_heading": str|None, ...}
    """
    items: list = []
    current_heading = section_heading
    table_lines: list = []
    prose_lines: list = []

    def _flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        parsed = _parse_table_block(table_lines)
        if parsed:
            items.append({
                "type": "table",
                "page_number": page_number,
                "data": parsed,
                "section_heading": current_heading,
            })
        else:
            # Not a valid table — fall back to prose
            prose_lines.extend(table_lines)
        table_lines = []

    def _flush_prose() -> None:
        text = "\n".join(prose_lines).strip()
        prose_lines.clear()
        if text:
            items.append({
                "type": "prose",
                "page_number": page_number,
                "text": text,
                "section_heading": current_heading,
            })

    for line in content.split("\n"):
        heading_match = _HEADING_RE.match(line)
        if heading_match and len(heading_match.group(1)) >= 2:
            # Sub-heading (## or deeper): flush buffers, update heading.
            _flush_table()
            _flush_prose()
            current_heading = heading_match.group(2).strip()
            continue

        stripped = line.strip()
        if _TABLE_ROW_RE.match(stripped):
            # Entering or continuing a table block.
            _flush_prose()
            table_lines.append(stripped)
        else:
            # Non-table line: close any open table first.
            if table_lines:
                _flush_table()
            prose_lines.append(line)

    # Flush any remaining content.
    _flush_table()
    _flush_prose()

    return items


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_markdown_file(path: Path) -> dict[str, Any]:
    """
    Parse a Markdown (.md) file into the parse_result format expected by
    ``general_chunker.build_general_chunks()``.

    The returned dict is structurally identical to the one produced by
    ``general_parser.parse_general_pdf()``, so the entire downstream
    pipeline (chunking, deduplication, validation, embedding, FAISS)
    works without modification.

    Args:
        path: Path to the ``.md`` file.

    Returns:
        A dict with keys:
            ``items``           — list of prose/table content-item dicts.
            ``source_hash``     — hex SHA-256 of the source file bytes.
            ``page_count``      — number of synthetic pages (H1 sections).
            ``source_document`` — the filename (``path.name``).

    Raises:
        FileNotFoundError: if *path* does not exist.
        RuntimeError:      if no content items are extracted (empty file).
    """
    log.info("Parsing Markdown file: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    source_hash = _sha256_file(path)
    log.info("Source SHA-256: %s", source_hash)

    text = path.read_text(encoding="utf-8")
    document_name = path.name

    sections = _split_into_sections(text)
    items: list = []
    for section in sections:
        section_items = _parse_section_content(
            section["content"],
            section["page_number"],
            section["heading"],
        )
        items.extend(section_items)

    if not items:
        raise RuntimeError(
            f"No content extracted from Markdown file: {path}. "
            "Check that the file is not empty."
        )

    page_count = max(s["page_number"] for s in sections) if sections else 0

    log.info(
        "Markdown parse complete: %d items, %d synthetic page(s) (source: %s)",
        len(items), page_count, document_name,
    )
    return {
        "items": items,
        "source_hash": source_hash,
        "page_count": page_count,
        "source_document": document_name,
    }
