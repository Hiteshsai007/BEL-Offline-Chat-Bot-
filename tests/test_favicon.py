"""
tests/test_favicon.py

Comprehensive test suite for the GET /favicon.ico route added in PR #6.

Coverage map
------------
Area 1  - Route ordering / catch-all shadowing protection
Area 2  - HTTP method correctness (405 for POST/PUT/DELETE)
Area 3  - Caching headers observation (informational, non-asserting)
Area 4  - Concurrent / repeated requests (byte-identical responses)
Area 5  - File integrity (byte-for-byte comparison with on-disk asset)
Area 6  - Missing-file resilience (monkeypatched STATIC_DIR -> 404 not 500)
Area 7  - Static-file-mount non-interference (/static/favicon.ico vs /favicon.ico)
Area 8  - index.html regression guard (exactly ONE <link rel="icon"> tag)

Cross-platform notes
--------------------
* All path resolution uses pathlib.Path (via STATIC_DIR imported from app.main),
  which normalises separators on both Windows and Linux automatically.
* No OS-specific branching, shell commands, or line-ending assumptions are used.
* No real network calls are made -- TestClient exercises the ASGI app in-process.
* The ICO magic-bytes constant (b'\x00\x00\x01\x00') is a file-format invariant
  independent of the OS that created or reads the file.
"""
import re
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import STATIC_DIR, app

# ---------------------------------------------------------------------------
# Shared client -- reused across all tests (no state leaks between requests)
# ---------------------------------------------------------------------------
client: TestClient = TestClient(app)

# ICO file-format constant: bytes 0-3 = 0x0000 (reserved) + 0x0001 (type=icon)
_ICO_MAGIC: bytes = b"\x00\x00\x01\x00"


# ===========================================================================
# ORIGINAL TESTS (preserved and unchanged)
# ===========================================================================


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
    assert response.content[:4] == _ICO_MAGIC, (
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


def test_favicon_excluded_from_openapi_schema() -> None:
    """The favicon route must not appear in the generated OpenAPI spec."""
    response = client.get("/openapi.json")
    if response.status_code == 200:
        paths = response.json().get("paths", {})
        assert "/favicon.ico" not in paths, (
            "/favicon.ico must use include_in_schema=False so it does not "
            "pollute the API documentation."
        )


# ===========================================================================
# AREA 1 -- Route ordering / catch-all shadowing protection
# ===========================================================================


def test_no_catchall_route_can_shadow_favicon() -> None:
    """
    Guard against a future catch-all route (e.g. @app.get('/{path:path}'))
    being registered above /favicon.ico and hijacking the request.

    Strategy: inspect the app's route table and assert that no parameterised
    catch-all route with a path that matches '/favicon.ico' is registered
    *before* the dedicated favicon handler.  Since no catch-all exists today,
    this test will fail loudly if one is added later without considering ordering.

    FastAPI stores routes in app.routes in registration order.  We walk the
    list and confirm the favicon route appears before any route whose path
    template could match '/favicon.ico' (i.e. contains '{').
    """
    from fastapi.routing import APIRoute

    favicon_index: int = -1
    catchall_before_favicon: list[str] = []

    for i, route in enumerate(app.routes):
        if not isinstance(route, APIRoute):
            continue  # skip mounts and other non-API entries
        path: str = route.path
        if path == "/favicon.ico":
            favicon_index = i
        elif "{" in path and favicon_index == -1:
            # A parameterised route registered before we found /favicon.ico
            catchall_before_favicon.append(path)

    assert favicon_index != -1, (
        "No route registered for /favicon.ico -- the dedicated handler is missing."
    )
    assert catchall_before_favicon == [], (
        f"The following parameterised routes are registered BEFORE /favicon.ico "
        f"and could shadow it: {catchall_before_favicon}. "
        "Move /favicon.ico above them or ensure they do not match 'favicon.ico'."
    )


def test_favicon_response_is_binary_not_html() -> None:
    """
    Even if a catch-all were somehow routing /favicon.ico, the response body
    must be ICO binary data -- not index.html HTML content.
    This cross-checks that the correct handler is actually executing.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    body: bytes = response.content
    # Must NOT start with the HTML doctype string
    assert not body.lstrip().startswith(b"<!DOCTYPE"), (
        "/favicon.ico returned HTML content instead of ICO binary data. "
        "A catch-all or SPA-fallback route is intercepting the request."
    )
    # Must start with the ICO magic bytes
    assert body[:4] == _ICO_MAGIC, (
        f"Response body starts with {body[:4]!r} -- expected ICO magic bytes {_ICO_MAGIC!r}."
    )


# ===========================================================================
# AREA 2 -- HTTP method correctness
# ===========================================================================


@pytest.mark.parametrize("method", ["post", "put", "delete"])
def test_favicon_wrong_methods_return_405(method: str) -> None:
    """
    POST, PUT, and DELETE to /favicon.ico must return 405 Method Not Allowed.
    A 404 would mean the route was not found at all; a 500 would mean a crash.
    Neither is acceptable -- 405 proves the route exists but rejects the method.
    """
    response = getattr(client, method)("/favicon.ico")
    assert response.status_code == 405, (
        f"{method.upper()} /favicon.ico returned {response.status_code}, "
        f"expected 405 Method Not Allowed."
    )


# ===========================================================================
# AREA 3 -- Caching headers (observational -- non-asserting)
# ===========================================================================


def test_favicon_caching_headers_observation() -> None:
    """
    Observe which caching headers FastAPI's FileResponse sets automatically.
    This test never fails -- it documents the current behaviour.

    Finding: FastAPI's FileResponse sets 'last-modified' and 'etag' from the
    file's mtime and size, and sets 'cache-control: no-cache' by default
    (as of Starlette >= 0.20).  We assert at minimum that 'last-modified'
    is present, since FileResponse always derives it from the file's mtime.
    If this assertion fails it means Starlette's behaviour changed and the
    team should consider adding an explicit Cache-Control header to the route.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    headers = {k.lower(): v for k, v in response.headers.items()}

    # Informational log of what is actually present (visible in pytest -s output)
    caching_fields = ["cache-control", "etag", "last-modified", "expires"]
    present = {f: headers[f] for f in caching_fields if f in headers}
    absent = [f for f in caching_fields if f not in headers]
    # Print findings without failing; useful during code review
    print(f"\n[caching headers] present={present} absent={absent}")

    # Minimum expectation: FileResponse derives last-modified from file mtime
    assert "last-modified" in headers, (
        "FileResponse did not set a Last-Modified header. "
        "Starlette behaviour may have changed -- consider an explicit Cache-Control."
    )


# ===========================================================================
# AREA 4 -- Concurrent / repeated requests
# ===========================================================================


def test_favicon_repeated_requests_are_identical() -> None:
    """
    Fire 10 sequential GET /favicon.ico requests and confirm:
    - Every response has status 200.
    - Every response body is byte-identical to the first.
    - Content-Type stays image/x-icon throughout.

    Catches accidental stateful bugs (e.g. a file handle left open),
    generator exhaustion, or any per-request mutation of the response.
    """
    responses = [client.get("/favicon.ico") for _ in range(10)]

    first_body: bytes = responses[0].content
    assert len(first_body) > 0, "First response body is empty."

    for i, resp in enumerate(responses):
        assert resp.status_code == 200, (
            f"Request #{i + 1} returned {resp.status_code}, expected 200."
        )
        assert "image/x-icon" in resp.headers.get("content-type", ""), (
            f"Request #{i + 1} has wrong Content-Type: {resp.headers.get('content-type')}."
        )
        assert resp.content == first_body, (
            f"Request #{i + 1} body differs from request #1 "
            f"({len(resp.content)} bytes vs {len(first_body)} bytes). "
            "The favicon handler may have a stateful bug or file-handle leak."
        )


# ===========================================================================
# AREA 5 -- File integrity (byte-for-byte comparison)
# ===========================================================================


def test_favicon_served_bytes_match_disk() -> None:
    """
    The bytes served by GET /favicon.ico must be byte-for-byte identical
    to the raw file at app/static/favicon.ico.

    This goes beyond the magic-byte check: it catches silent corruption,
    encoding re-interpretation, or a wrong file being served.
    Uses pathlib.Path (STATIC_DIR from app.main) -- cross-platform.
    """
    disk_bytes: bytes = (STATIC_DIR / "favicon.ico").read_bytes()
    response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.content == disk_bytes, (
        f"Served content ({len(response.content)} bytes) does not match "
        f"on-disk file ({len(disk_bytes)} bytes). The wrong file may be served, "
        "or the content was modified in transit by a middleware."
    )


# ===========================================================================
# AREA 6 -- Missing-file resilience (monkeypatch STATIC_DIR)
# ===========================================================================


@pytest.fixture()
def favicon_absent(tmp_path: Path) -> Generator[None, None, None]:
    """
    Temporarily redirect STATIC_DIR inside app.main to a scratch directory
    that contains NO favicon.ico, then restore it afterwards.

    Why patch app.main.STATIC_DIR rather than the file system?
    The route handler reads:  icon_path = STATIC_DIR / "favicon.ico"
    Patching STATIC_DIR at the module level is the correct seam -- it does
    not require deleting real files, works on both Windows and Linux, and
    leaves no temporary files behind.
    """
    # tmp_path is a pytest built-in; guaranteed empty and auto-cleaned
    with patch("app.main.STATIC_DIR", new=tmp_path):
        yield


def test_favicon_missing_file_returns_404(favicon_absent: None) -> None:
    """
    When favicon.ico does not exist in STATIC_DIR, the route must return
    404 Not Found (exercising the `if not icon_path.exists()` guard in main.py).
    A 500 would mean the guard is broken or an unhandled exception escapes.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == 404, (
        f"Expected 404 when favicon.ico is absent, got {response.status_code}. "
        "The `if not icon_path.exists()` guard in app/main.py may be broken."
    )
    # Confirm it's a clean JSON error, not a crash traceback
    data = response.json()
    assert "detail" in data, (
        "404 response should include a JSON 'detail' field (FastAPI HTTPException format)."
    )


# ===========================================================================
# AREA 7 -- Static-file-mount non-interference
# ===========================================================================


def test_favicon_not_served_via_static_mount() -> None:
    """
    The app mounts app/static/ at /static (StaticFiles).  If someone requests
    /static/favicon.ico they will get the file through the generic mount --
    that path bypasses our dedicated route and returns the raw file without
    the explicit media_type='image/x-icon' header we set.

    This test confirms that the CANONICAL path (/favicon.ico) is served by
    our dedicated handler (verified by checking the media_type header), while
    the mount path (/static/favicon.ico) is a separate concern.

    It also protects against a future scenario where the /favicon.ico route
    is removed but favicon.ico accidentally continues to be served (silently,
    without the correct headers) via the StaticFiles mount.
    """
    # Our dedicated route -- must have the explicit media_type we set
    canonical = client.get("/favicon.ico")
    assert canonical.status_code == 200
    assert "image/x-icon" in canonical.headers.get("content-type", ""), (
        "The dedicated /favicon.ico route did not set image/x-icon Content-Type."
    )

    # Via the StaticFiles mount -- this path is intentionally different
    via_mount = client.get("/static/favicon.ico")
    # favicon.ico IS in the static directory, so the mount can serve it too --
    # but the Content-Type may differ (mount uses mime-type sniffing, not explicit).
    # The assertion we care about: the canonical route works independently of the mount.
    if via_mount.status_code == 200:
        # If the mount serves it, the bytes must still match the disk file
        disk_bytes: bytes = (STATIC_DIR / "favicon.ico").read_bytes()
        assert via_mount.content == disk_bytes, (
            "/static/favicon.ico served different bytes than the on-disk file."
        )
    # Whether the mount returns 200 or 404 is acceptable -- we do not mandate
    # that the mount exposes the file; we only mandate the canonical path works.


# ===========================================================================
# AREA 8 -- index.html regression guard (exactly ONE <link rel="icon"> tag)
# ===========================================================================


def test_index_html_references_favicon() -> None:
    """
    The served index.html must include a link rel=icon pointing to /favicon.ico.
    (Preserved from original suite.)
    """
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


def test_index_html_has_exactly_one_favicon_link() -> None:
    """
    index.html must contain EXACTLY ONE <link rel='icon'> tag.

    Zero means the favicon reference was accidentally removed.
    Two or more means a duplicate was added (e.g. during a merge), which
    causes browsers to use whichever href they encounter last -- potentially
    an empty or wrong path.

    Uses a regex that matches both single and double-quoted attribute values
    and is insensitive to attribute order within the <link> tag.
    """
    response = client.get("/")
    assert response.status_code == 200
    body: str = response.text

    # Match any <link ...> tag that contains rel="icon" or rel='icon'
    pattern = re.compile(r'<link\b[^>]*\brel=["\']icon["\'][^>]*/?\s*>', re.IGNORECASE)
    matches = pattern.findall(body)

    assert len(matches) == 1, (
        f"Expected exactly 1 <link rel='icon'> tag in index.html, found {len(matches)}. "
        f"Matches: {matches}. "
        "Zero means the favicon link was removed; >1 means a duplicate was introduced."
    )
