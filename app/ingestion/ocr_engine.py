"""
OCR Engine module for PDF image text extraction.
Runs OCR on extracted images using RapidOCR (or fallback mechanisms).
"""
import logging
from pathlib import Path

log = logging.getLogger("bel.app.ingestion.ocr_engine")

_RAPID_OCR_ENGINE = None


def _get_ocr_engine():
    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_ENGINE = RapidOCR()
            log.info("RapidOCR engine initialized successfully.")
        except Exception as e:
            log.warning(
                "Could not initialize RapidOCR engine: %s. "
                "OCR will fallback to empty results.", e
            )
            _RAPID_OCR_ENGINE = False
    return _RAPID_OCR_ENGINE if _RAPID_OCR_ENGINE is not False else None


def perform_ocr_on_image(image_path: str | Path) -> str:
    """
    Perform OCR on a local image file and return extracted text string.

    Args:
        image_path: Path to the PNG/JPG image on disk.

    Returns:
        Extracted text joined by spaces, or empty string if no text found.
    """
    if not image_path:
        return ""

    p = Path(image_path)
    if not p.exists():
        log.warning("OCR image path does not exist: %s", p)
        return ""

    engine = _get_ocr_engine()
    if not engine:
        return ""

    try:
        results, _ = engine(str(p))
        if not results:
            return ""

        lines = [
            line[1].strip()
            for line in results
            if len(line) >= 2 and line[1] and line[1].strip()
        ]
        extracted = " ".join(lines)
        log.debug(
            "OCR extracted %d characters from %s", len(extracted), p.name
        )
        return extracted
    except Exception as e:
        log.warning("OCR processing failed for %s: %s", p, e)
        return ""
