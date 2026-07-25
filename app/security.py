"""
Same-origin enforcement for state-changing routes (security finding S-1).

Threat model
------------
The server binds to loopback only (127.0.0.1 -- PRD Section 12), which blocks
remote *network* attackers. It does NOT block the operator's own browser: any
web page the operator visits can issue a cross-origin ``POST`` to
``http://127.0.0.1:8000/reload`` or ``/query``. Simple form posts are not
preflighted, so CORS never gets a chance to stop the request reaching the
handler -- CORS only stops the attacker *reading* the response, which a
fire-and-forget CSRF does not care about.

Why an Origin/Referer check rather than a shared-secret token
-------------------------------------------------------------
Both were considered. The origin check wins for this application because:

* **Zero friction.** The brief requires no added friction for normal use and no
  login system. This needs no token file, no client change, and no extra round
  trip -- ``app/static/app.js`` keeps working untouched.
* **Browsers always send ``Origin`` on POST.** Per the Fetch standard, ``Origin``
  is attached to every request whose method is not GET/HEAD, including
  cross-origin form posts. A malicious page cannot suppress or forge it.
* **A token file adds no real protection here.** A secret written to a local
  file readable by the operator is equally readable by any local process running
  as that operator, so it does not raise the bar against the local-malware case
  while it does add operational friction.

Residual risk (accepted, documented)
------------------------------------
This defends against *browser-driven* CSRF only. A non-browser local client
(curl, a script, local malware running as the operator) can omit ``Origin``
entirely and will be allowed through -- see ``_is_browser_request``. That is a
deliberate trade-off: requests with no ``Origin``/``Referer`` are not reachable
from a hostile web page, and rejecting them would break the CLI, health probes
and the test suite. Genuine protection against a local attacker who already has
code execution as the operator is out of scope for a single-operator local tool.
"""
from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request

from app.logger import get_logger
from app.settings import SERVER_HOST, SERVER_PORT

log = get_logger(__name__)


def _build_allowed_origins() -> frozenset[str]:
    """
    Build the set of origins considered same-origin for this server.

    The configured host is included, plus the loopback aliases a browser may
    legitimately use to reach it. ``Launch.bat`` opens ``127.0.0.1`` while an
    operator typing the address by hand often uses ``localhost``; both must work.
    """
    hosts = {SERVER_HOST, "127.0.0.1", "localhost", "[::1]"}
    return frozenset(f"http://{h}:{SERVER_PORT}" for h in hosts)


ALLOWED_ORIGINS: frozenset[str] = _build_allowed_origins()


def _origin_of(url: str) -> Optional[str]:
    """Reduce an absolute URL to its ``scheme://host:port`` origin form."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        return None
    if parsed.port is not None:
        return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    return f"{parsed.scheme}://{parsed.hostname}"


def _is_browser_request(origin: Optional[str], referer: Optional[str]) -> bool:
    """
    True when the request carries browser-supplied provenance headers.

    Requests with neither header did not come from a web page, so they are not
    the CSRF threat this guard addresses (see module docstring).
    """
    return bool(origin or referer)


def verify_same_origin(request: Request) -> None:
    """
    FastAPI dependency: reject cross-origin state-changing requests with 403.

    Applied to ``POST /query`` and ``POST /reload``. Read-only routes (``/``,
    ``/health``, ``/favicon.ico``, ``/static/*``) are unaffected -- they change
    no state, and gating them would break plain browser navigation.
    """
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")

    if not _is_browser_request(origin, referer):
        # Non-browser client (CLI, curl, health probe, tests). Not CSRF-reachable.
        return

    # Origin is authoritative when present; fall back to Referer for the rare
    # browser/proxy configuration that strips Origin but keeps Referer.
    candidate = origin if origin else _origin_of(referer or "")

    if candidate in ALLOWED_ORIGINS:
        return

    log.warning(
        "Rejected cross-origin %s %s | origin=%r referer=%r",
        request.method, request.url.path, origin, referer,
    )
    raise HTTPException(
        status_code=403,
        detail="Cross-origin requests are not permitted on this endpoint.",
    )
