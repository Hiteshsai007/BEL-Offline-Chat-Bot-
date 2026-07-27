"""
General-purpose PDF parser for unstructured documents (manuals, guides, etc.).

Extracts both prose text and tables from a PDF without assuming any specific
document structure (unlike the fault-code parser which expects specific column
names). Also extracts embedded images and captures nearby caption text.

Uses pdfplumber (primary) for text and table extraction with PyMuPDF as
fallback for pages pdfplumber cannot parse cleanly. PyMuPDF is also used
for embedded image extraction since pdfplumber does not support this well.

Output is a list of page-level content items, each tagged as 'prose',
'table', or 'image'. Page numbers are preserved for citation purposes
(a general manual cites by page number + section heading, not error code).
"""
import hashlib
import re
from pathlib import Path
from typing import Any

from app.logger import get_logger

log = get_logger(__name__)

# Minimum pixel dimension for extracted images.
# Images smaller than this in BOTH width and height are skipped — they are
# almost certainly decorative icons, bullets, or page-corner graphics, not
# informative diagrams.  48px is a conservative threshold: the smallest
# genuine diagram in a typical manual is at least 100-150px on each side.
_MIN_IMAGE_DIM = 48


# -- Helpers ---------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(text: str | None) -> str:
    """Collapse internal whitespace, normalize concatenated words, and strip edges."""
    if not text:
        return ""
    # Insert space between lowercase and uppercase letter (e.g. FuelTank -> Fuel Tank)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space between multiple uppercase and CamelCase (e.g. GENERALINFORMATION -> GENERAL INFORMATION)
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    # Insert space between digits and letters (e.g. 17L -> 17 L, 4.5USgal -> 4.5 US gal)
    text = re.sub(r'(\d+(?:\.\d+)?)([a-zA-Z]+)', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z]+)(\d+(?:\.\d+)?)', r'\1 \2', text)
    return " ".join(text.split())


def _detect_section_heading(line: str) -> str | None:
    """
    Heuristic: return the line text if it looks like a section heading,
    else None.  Matches patterns like 'Chapter N: ...', 'Section N.N',
    or lines in ALL CAPS (common in manuals).
    """
    stripped = line.strip()
    if not stripped:
        return None
    if re.match(
        r"^(Chapter|Section|PART|Appendix)\s+[\dA-Z]",
        stripped,
        re.IGNORECASE,
    ):
        return stripped
    # Short all-caps line (likely a heading, not prose)
    if stripped.isupper() and len(stripped) < 80:
        return stripped
    return None


# -- pdfplumber extraction -------------------------------------------------

def _extract_with_pdfplumber(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Extract page content using pdfplumber.

    Returns a list of content items, each with:
      type: 'prose' | 'table'
      page_number: 1-based
      text: str (for prose) or list[list[str]] (for table)
      section_heading: str | None
    """
    import pdfplumber  # type: ignore

    items: list[dict[str, Any]] = []
    current_heading: str | None = None

    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # -- Tables first --
            tables = page.extract_tables()
            for table in tables:
                if not table or not any(table):
                    continue
                # Clean cells
                cleaned = []
                for row in table:
                    cleaned.append([_clean(c) for c in (row or [])])
                # Skip if all cells empty
                if not any(cell for row in cleaned for cell in row):
                    continue
                items.append({
                    "type": "table",
                    "page_number": page_num,
                    "data": cleaned,
                })

            # -- Prose text --
            text = page.extract_text() or ""
            if text.strip():
                # Track section headings
                for line in text.splitlines():
                    heading = _detect_section_heading(line)
                    if heading:
                        current_heading = heading

                items.append({
                    "type": "prose",
                    "page_number": page_num,
                    "text": _clean(text),
                    "section_heading": current_heading,
                })

    return items


# -- PyMuPDF fallback ------------------------------------------------------

def _extract_with_pymupdf(path: Path) -> list[dict[str, Any]]:
    """
    Fallback extraction using PyMuPDF when pdfplumber fails on a page.
    Extracts plain text only (no table detection).
    """
    import fitz  # type: ignore

    items: list[dict[str, Any]] = []
    current_heading: str | None = None

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text or not text.strip():
                continue

            for line in text.splitlines():
                heading = _detect_section_heading(line)
                if heading:
                    current_heading = heading

            items.append({
                "type": "prose",
                "page_number": page_num,
                "text": _clean(text),
                "section_heading": current_heading,
            })

    return items


# -- Image extraction (always via PyMuPDF) ---------------------------------

def _extract_images(
    path: Path,
    output_dir: Path,
    source_slug: str,
) -> list[dict[str, Any]]:
    """
    Extract embedded images from each page using PyMuPDF.

    Saves images to output_dir/<source_slug>/page_<N>_<idx>.png
    and returns metadata dicts for each image.
    """
    import fitz  # type: ignore

    images: list[dict[str, Any]] = []
    skipped_count = 0

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            img_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    # Convert CMYK or other colourspaces to RGB
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                except Exception as e:
                    log.warning(
                        "Could not extract image xref=%d on page %d: %s",
                        xref, page_num, e,
                    )
                    continue

                # Skip decorative/icon-sized images
                if pix.width < _MIN_IMAGE_DIM and pix.height < _MIN_IMAGE_DIM:
                    log.debug(
                        "Skipped icon-sized image: page %d, idx %d "
                        "(%dx%d < %dpx threshold)",
                        page_num, img_idx, pix.width, pix.height,
                        _MIN_IMAGE_DIM,
                    )
                    skipped_count += 1
                    continue

                img_name = f"page_{page_num}_{img_idx}.png"
                img_dir = output_dir / source_slug
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / img_name
                pix.save(str(img_path))

                # Try to find nearby caption text on the same page
                caption = _find_nearby_caption(page, img_info)

                images.append({
                    "type": "image",
                    "page_number": page_num,
                    "image_file_path": str(img_path),
                    "caption": caption,
                })
                log.info(
                    "Extracted image: page %d, idx %d -> %s",
                    page_num, img_idx, img_path,
                )

    if skipped_count:
        log.info(
            "Skipped %d icon-sized images (<%dpx on both axes)",
            skipped_count, _MIN_IMAGE_DIM,
        )

    return images


def _find_nearby_caption(page: Any, img_info: tuple) -> str | None:
    """
    Heuristic: look for text containing 'Figure', 'Fig.', or 'Table'
    near the image on the same page.
    """
    text = page.get_text("text") or ""
    # Look for lines matching figure/table caption patterns
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(
            r"^(Figure|Fig\.?|Table|Diagram)\s+[\d\-]+",
            stripped,
            re.IGNORECASE,
        ):
            return stripped
    return None


# -- Public API ------------------------------------------------------------

def parse_general_pdf(
    path: Path,
    image_output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Parse a general PDF and return extracted content.

    Args:
        path: Path to the PDF file.
        image_output_dir: Where to save extracted images.
            Defaults to data/index/general/images/.

    Returns:
        {
            "items": [...],        # list of prose/table/image items
            "source_hash": str,    # SHA-256 of the PDF bytes
            "page_count": int,
            "source_document": str, # filename
        }

    Raises RuntimeError if no text content is extracted.
    """
    log.info("Parsing general PDF: %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Source PDF not found: {path}")

    source_hash = sha256_file(path)
    log.info("Source SHA-256: %s", source_hash)

    # -- Primary extraction via pdfplumber (detects tables) --
    items: list[dict[str, Any]] = []
    try:
        items = _extract_with_pdfplumber(path)
        log.info("pdfplumber extracted %d content items", len(items))
    except Exception as e:
        log.warning("pdfplumber failed (%s) -- trying PyMuPDF fallback", e)

    # Fallback if pdfplumber returned nothing
    if not items:
        try:
            items = _extract_with_pymupdf(path)
            log.info("PyMuPDF fallback extracted %d items", len(items))
        except Exception as e:
            log.error("PyMuPDF also failed: %s", e)

    if not items:
        raise RuntimeError(
            "No text content extracted from PDF. Check that the PDF is "
            "text-native (not a scanned image)."
        )

    # -- Image extraction via PymuPDF --
    if image_output_dir is None:
        from app.settings import FAISS_INDEX_PATH
        image_output_dir = FAISS_INDEX_PATH.parent / "general" / "images"

    source_slug = path.stem.replace(" ", "_")
    try:
        image_items = _extract_images(path, image_output_dir, source_slug)
        items.extend(image_items)
        log.info("Extracted %d images", len(image_items))
    except Exception as e:
        log.warning("Image extraction failed (non-fatal): %s", e)

    # -- Page count --
    try:
        import fitz  # type: ignore
        with fitz.open(str(path)) as doc:
            page_count = doc.page_count
    except Exception:
        page_count = -1

    # -- Boilerplate deduplication --
    # NOTE: Removed from parser — now runs post-chunk in
    # general_chunker.deduplicate_boilerplate_chunks(), where the
    # smaller chunk size makes Jaccard similarity effective.

    log.info(
        "Parse complete: %d items, %d pages",
        len(items), page_count,
    )
    return {
        "items": items,
        "source_hash": source_hash,
        "page_count": page_count,
        "source_document": path.name,
    }
