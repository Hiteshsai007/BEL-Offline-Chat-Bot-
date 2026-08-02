"""
PDF-to-Markdown Converter for Hybrid RAG Ingestion.

Converts structured items extracted from a PDF (via general_parser.py)
into a structured Markdown representation (generated.md) with explicit
page boundary markers (<!-- PAGE:N -->).

The original PDF remains the authoritative source of truth. The Markdown
representation is generated solely for retrieval optimization while preserving
all page numbers, table data, and image metadata.
"""
from typing import Any
from app.logger import get_logger

log = get_logger(__name__)


def _format_table_as_markdown(table_data: list[list[str]]) -> str:
    """
    Convert a 2D list of strings into a Markdown pipe table.
    Falls back to raw text lines if table formatting fails.
    """
    if not table_data or not any(table_data):
        return ""

    try:
        # Find maximum columns
        max_cols = max(len(row) for row in table_data)
        if max_cols == 0:
            return ""

        # Normalize rows to max_cols
        norm_rows = []
        for row in table_data:
            clean_row = [
                str(cell or "").replace("\n", " ").replace("|", "\\|").strip()
                for cell in row
            ]
            while len(clean_row) < max_cols:
                clean_row.append("")
            norm_rows.append(clean_row)

        header = norm_rows[0]
        divider = ["---"] * max_cols

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |",
        ]

        for row in norm_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
    except Exception as e:
        log.warning(
            "Markdown table conversion failed (%s), using raw text fallback", e
        )
        fallback_lines = []
        for row in table_data:
            row_str = " | ".join(str(c or "").strip() for c in row if c)
            if row_str:
                fallback_lines.append(row_str)
        return "\n".join(fallback_lines)


def generate_markdown_from_pdf_parse(parse_result: dict[str, Any]) -> str:
    """
    Generate a Markdown document string from parse_general_pdf() result.

    Emits explicit page markers: <!-- PAGE:N -->.
    Preserves headings (#, ##, ###), tables, and image references.
    """
    items = parse_result.get("items", [])
    doc_name = parse_result.get("source_document", "Document")

    # Group items by page number
    page_items: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        p_num = item.get("page_number", 1)
        page_items.setdefault(p_num, []).append(item)

    sorted_pages = sorted(page_items.keys())
    md_lines: list[str] = [f"# {doc_name}\n"]

    for page_num in sorted_pages:
        md_lines.append(f"\n<!-- PAGE:{page_num} -->\n")
        items_on_page = page_items[page_num]

        for item in items_on_page:
            itype = item.get("type")

            if itype == "prose":
                text = (item.get("text") or "").strip()
                heading = item.get("section_heading")
                if heading and heading in text:
                    # Format section heading as markdown level 2 if present
                    md_lines.append(f"## {heading}\n")
                    text_without_heading = text.replace(heading, "", 1).strip()
                    if text_without_heading:
                        md_lines.append(f"{text_without_heading}\n")
                elif text:
                    md_lines.append(f"{text}\n")

            elif itype == "table":
                tdata = item.get("data")
                if tdata:
                    tbl_md = _format_table_as_markdown(tdata)
                    if tbl_md:
                        md_lines.append(f"\n{tbl_md}\n")
                elif item.get("raw_text"):
                    md_lines.append(f"\n{item['raw_text']}\n")

            elif itype == "image":
                img_id = item.get("image_id") or f"page_{page_num}_img"
                caption = item.get("caption") or ""
                img_ref = f"[Image: {img_id}]"
                if caption:
                    img_ref += f" - {caption}"
                md_lines.append(f"\n{img_ref}\n")

    return "\n".join(md_lines)
