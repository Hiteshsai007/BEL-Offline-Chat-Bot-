"""
PDF parser for IRL_Fault_Codes.pdf.

Extracts the fault-code table using pdfplumber (primary) with PyMuPDF as
fallback. Returns a list of raw row dicts — text is verbatim from the source,
never paraphrased or invented (PRD Section 9 fidelity requirement).
"""
import hashlib
import re
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())  # collapse whitespace, strip edges


# ── pdfplumber extraction ──────────────────────────────────────────────────

def _extract_with_pdfplumber(path: Path) -> list[dict[str, Any]]:
    import pdfplumber  # type: ignore

    rows: list[dict[str, Any]] = []
    footnotes: list[str] = []

    # Persist column indices across pages
    idx_sl, idx_desc, idx_rem, idx_code = None, None, None, None

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue

                header = [_clean(c) for c in (table[0] or [])]

                # Check if this row looks like a header
                is_header_row = any(
                    kw in " ".join(header).lower()
                    for kw in ("error code", "sl", "error description")
                )

                start_idx = 0
                if is_header_row:
                    start_idx = 1

                    # Map header names to column indices
                    def col(keywords: list[str]) -> int | None:
                        for i, h in enumerate(header):
                            if any(k.lower() in h.lower() for k in keywords):
                                return i
                        return None

                    idx_sl = col(["sl", "s.no", "sno", "sr"])
                    idx_desc = col(["description", "error desc"])
                    idx_rem = col(["remark", "remarks"])
                    idx_code = col(["code"])
                elif idx_code is None and len(header) >= 4:
                    # Fallback if no header was found on the first page
                    idx_sl, idx_desc, idx_rem, idx_code = 0, 1, 2, 3

                for raw_row in table[start_idx:]:
                    if not raw_row:
                        continue
                    cells = [_clean(c) for c in raw_row]

                    def get(idx: int | None) -> str:
                        return cells[idx] if idx is not None and idx < len(cells) else ""

                    sl = get(idx_sl)
                    desc = get(idx_desc)
                    rem = get(idx_rem)
                    code = get(idx_code)

                    # Skip blank or header-repeat rows
                    if not desc and not code:
                        continue
                    if desc.lower() in ("error description", ""):
                        continue

                    rows.append({
                        "sl_no": sl,
                        "error_description": desc,
                        "error_remarks": rem,
                        "error_code": code,
                        "page_number": page_num,
                    })

            # Collect footnotes (lines starting with "Note")
            text = page.extract_text() or ""
            for line in text.splitlines():
                line = line.strip()
                if re.match(r"Note\s*\d+", line, re.IGNORECASE):
                    footnotes.append(line)

    return rows, footnotes


# ── PyMuPDF fallback ───────────────────────────────────────────────────────

def _extract_with_pymupdf(path: Path) -> list[dict[str, Any]]:
    """
    Fallback: extracts text blocks and heuristically finds the table.
    Less accurate than pdfplumber for tables but better than nothing.
    """
    import fitz  # PyMuPDF  # type: ignore

    rows: list[dict[str, Any]] = []
    footnotes: list[str] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Look for lines that match the pattern: digits  text  text  0x####
            code_pattern = re.compile(r"(0x[0-9A-Fa-f]{4})")
            for i, line in enumerate(lines):
                if code_pattern.search(line):
                    # Try to reconstruct: sl | desc | remarks | code
                    parts = line.split()
                    code_match = code_pattern.search(line)
                    code = code_match.group(1) if code_match else ""
                    # Attempt to extract sl_no (leading integer)
                    sl = parts[0] if parts and parts[0].isdigit() else ""
                    rows.append({
                        "sl_no": sl,
                        "error_description": line,
                        "error_remarks": "",
                        "error_code": code,
                        "page_number": page_num,
                    })

                if re.match(r"Note\s*\d+", line, re.IGNORECASE):
                    footnotes.append(line)

    return rows, footnotes


# ── Public API ─────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> dict[str, Any]:
    """
    Parse the fault-code PDF and return:
        {
            "rows": [...],       # list of row dicts (verbatim text)
            "footnotes": [...],  # any "Note N" lines found
            "source_hash": str,  # SHA-256 of the PDF bytes
            "page_count": int,
        }

    Raises RuntimeError if no rows are extracted.
    """
    log.info("Parsing PDF: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")

    source_hash = _sha256(path)
    log.info("Source SHA-256: %s", source_hash)

    rows, footnotes = [], []

    # Try pdfplumber first
    try:
        rows, footnotes = _extract_with_pdfplumber(path)
        log.info("pdfplumber extracted %d rows, %d footnotes", len(rows), len(footnotes))
    except Exception as e:
        log.warning("pdfplumber failed (%s) — trying PyMuPDF fallback", e)

    # Fallback if pdfplumber returned nothing
    if not rows:
        try:
            rows, footnotes = _extract_with_pymupdf(path)
            log.info("PyMuPDF fallback extracted %d rows", len(rows))
        except Exception as e:
            log.error("PyMuPDF also failed: %s", e)

    if not rows:
        raise RuntimeError(
            "No rows extracted from PDF. Check that the PDF is text-native "
            "(not a scanned image). OCR is out of scope (PRD Section 2)."
        )

    # Count pages using PyMuPDF (lightweight)
    try:
        import fitz  # type: ignore
        with fitz.open(str(path)) as doc:
            page_count = doc.page_count
    except Exception:
        page_count = -1

    log.info("Parse complete: %d rows, %d footnotes, %d pages", len(rows), len(footnotes), page_count)
    return {
        "rows": rows,
        "footnotes": footnotes,
        "source_hash": source_hash,
        "page_count": page_count,
    }
