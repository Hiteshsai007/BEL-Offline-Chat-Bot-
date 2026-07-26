"""
Generate a small synthetic PDF fixture for testing the general document
ingestion pipeline.

Creates a 3-page PDF:
  Page 1: Title page with prose
  Page 2: Prose section with an embedded image and figure reference
  Page 3: A simple table (torque specs)
"""
import struct
import sys
import zlib
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF (fitz) is required. Install with: pip install pymupdf")
    sys.exit(1)


def _make_tiny_png() -> bytes:
    """Create a small 100x100 blue PNG without any image library dependency."""
    w, h = 100, 100
    raw = b""
    for _ in range(h):
        raw += b"\x00" + bytes([30, 80, 160]) * w
    compressed = zlib.compress(raw)

    def _chunk(ctype: bytes, data: bytes) -> bytes:
        c = ctype + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def build_fixture(output_path: Path) -> None:
    doc = fitz.open()

    # -- Page 1: Title + prose --------------------------------------------
    page1 = doc.new_page(width=595, height=842)  # A4
    page1.insert_text(
        (72, 100), "Kawasaki Motorcycle Owner's Manual",
        fontname="helv", fontsize=18,
    )
    page1.insert_text(
        (72, 130), "Model: ZX-10R SE 2018",
        fontname="helv", fontsize=12,
    )
    page1.insert_textbox(
        fitz.Rect(72, 170, 523, 400),
        (
            "Chapter 1: General Information\n\n"
            "This manual is designed to help you get the most enjoyment "
            "from your motorcycle. Read this manual thoroughly before "
            "operating the motorcycle. Proper maintenance and care will "
            "help ensure the reliability and longevity of your vehicle.\n\n"
            "The motorcycle is equipped with a 998 cc inline-four engine "
            "producing 200 horsepower. The chassis features electronic "
            "suspension adjustment and multiple riding modes."
        ),
        fontname="helv", fontsize=11,
    )

    # -- Page 2: Prose + embedded image + caption -------------------------
    page2 = doc.new_page(width=595, height=842)
    page2.insert_textbox(
        fitz.Rect(72, 50, 523, 200),
        (
            "Chapter 2: Controls and Instruments\n\n"
            "The instrument panel displays speed, RPM, gear position, "
            "and various warning indicators. See Figure 2-1 for the "
            "instrument cluster layout.\n\n"
            "WARNING: Do not operate the motorcycle if any warning "
            "indicator remains illuminated after startup."
        ),
        fontname="helv", fontsize=11,
    )
    # Embed a real image so PyMuPDF can extract it
    img_rect = fitz.Rect(150, 250, 445, 450)
    page2.insert_image(img_rect, stream=_make_tiny_png())
    # Caption below image
    page2.insert_text(
        (150, 465), "Figure 2-1: Instrument Cluster Layout",
        fontname="helv", fontsize=10,
    )

    # -- Page 3: Table (torque specs) -------------------------------------
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text(
        (72, 50), "Chapter 3: Torque Specifications",
        fontname="helv", fontsize=14,
    )
    page3.insert_textbox(
        fitz.Rect(72, 80, 523, 120),
        "The following table lists the recommended torque values "
        "for common fasteners. Always use a calibrated torque wrench.",
        fontname="helv", fontsize=11,
    )
    # Insert a real PDF table using insert_textbox in a grid layout
    # that pdfplumber can detect as a table.
    table_data = [
        ["Fastener", "Torque (Nm)", "Location"],
        ["Oil drain plug", "30", "Engine bottom"],
        ["Front axle nut", "108", "Front wheel"],
        ["Rear axle nut", "108", "Rear wheel"],
        ["Spark plug", "13", "Cylinder head"],
        ["Caliper bolt", "34", "Front brake"],
    ]
    # Draw the table using page3.new_table if available, else draw_grid
    try:
        tbl = page3.new_table(
            rows=len(table_data),
            cols=3,
            col_widths=(180, 100, 150),
            bbox=fitz.Rect(72, 140, 523, 340),
        )
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                tbl.set_cell_text((i, j), cell)
        tbl.show()
    except Exception:
        # Fallback: draw with grid lines
        x0, y0 = 72, 140
        col_w = [180, 100, 150]
        row_h = 28
        # Draw cells
        for i, row in enumerate(table_data):
            x = x0
            for j, cell in enumerate(row):
                rect = fitz.Rect(x, y0 + i * row_h, x + col_w[j], y0 + (i + 1) * row_h)
                shape = page3.new_shape()
                shape.draw_rect(rect)
                shape.finish(color=(0, 0, 0), width=0.5)
                shape.commit()
                page3.insert_textbox(
                    rect, cell,
                    fontname="helv", fontsize=10,
                    align=fitz.TEXT_ALIGN_LEFT,
                )
                x += col_w[j]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    print(f"Fixture written: {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    out = (
        Path(__file__).resolve().parent.parent.parent
        / "tests" / "fixtures" / "test_manual.pdf"
    )
    build_fixture(out)
