"""
tests/test_favicon.py

Verifies that GET /favicon.ico is served correctly (HTTP 200, image/x-icon)
and that the static asset exists on disk.

Cross-platform notes
--------------------
* STATIC_DIR resolution uses pathlib.Path, which normalises separators on both
  Windows and Linux -- no OS-specific branching is needed here.
* The ICO magic-bytes check (first 4 bytes == b'\x00\x00\x01\x00') is a
  file-format invariant that holds regardless of the OS that created the file.
* The route under test uses FastAPI's FileResponse with an explicit
  media_type="image/x-icon", so Content-Type is deterministic.
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app, STATIC_DIR

client = TestClient(app)


def test_favicon_status_ok() -> None:
    """GET /favicon.ico must return 200 OK -- not 404."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200, (
        f"Expected 200 for /favicon.ico, got {response.status_code}. "
        "Server logs would be cluttered with 404 noise on every browser launch."
    )


def test_favicon_content_type() -> None:
    """Response must carry image/x-icon so browsers render it as a tab icon."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "image/x-icon" in content_type, (
        f"Expected 'image/x-icon' in Content-Type, got '{content_type}'."
    )


def test_favicon_is_valid_ico() -> None:
    """ICO files begin with the 4-byte reserved+type header 00 00 01 00."""
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.content[:4] == b"\x00\x00\x01\x00", (
        "favicon.ico does not start with the expected ICO magic bytes. "
        "The file may be corrupt or not a real ICO."
    )


def test_favicon_file_exists_on_disk() -> None:
    """app/static/favicon.ico must be present so the route can serve it."""
    favicon_path: Path = STATIC_DIR / "favicon.ico"
    assert favicon_path.exists(), (
        f"favicon.ico not found at {favicon_path}. "
        "Add the file to app/static/ and commit it."
    )
    assert favicon_path.stat().st_size > 0, "favicon.ico exists but is empty."


def test_index_html_references_favicon() -> None:
    """The served index.html must include a link rel=icon pointing to /favicon.ico."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'rel="icon"' in body, (
        'link rel="icon" tag is missing from index.html. '
        "Add it so browsers explicitly request /favicon.ico."
    )
    assert "/favicon.ico" in body, (
        "index.html does not reference /favicon.ico in a link tag. "
        "Browsers may fall back to their own broken-icon UI."
    )


def test_favicon_excluded_from_openapi_schema() -> None:
    """The favicon route must not appear in the generated OpenAPI spec."""
    response = client.get("/openapi.json")
    if response.status_code == 200:
        paths = response.json().get("paths", {})
        assert "/favicon.ico" not in paths, (
            "/favicon.ico must use include_in_schema=False so it does not "
            "pollute the API documentation."
        )
