"""
tests/test_security_ollama_install.py

Covers security finding S-4: bootstrap.py installed Ollama with
``curl -fsSL https://ollama.com/install.sh | sh`` -- unverified remote code
executed through a shell, unpinned, with no checksum or signature.

Ollama publishes a ``sha256sum.txt`` manifest as a release asset alongside the
platform tarballs, so a verified install is possible. These tests cover the
verification logic (digest parsing, match, mismatch, opt-in gating) without
touching the network.

What these tests CANNOT cover
-----------------------------
The end-to-end download-and-extract path is not exercised: it requires network
egress to github.com and root to untar into /usr. The HTTP fetch and the
extraction are therefore mocked. See the PR description for what was verified
manually against the upstream API and what remains unverified.

Cross-platform notes
--------------------
* The digest/parsing helpers under test are pure and OS-independent.
* Tests that touch the Linux-only install helper are skipped on Windows.
"""
import hashlib
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bootstrap  # noqa: E402

IS_WINDOWS = platform.system() == "Windows"

MANIFEST = (
    "6f1e7c1e2d3a4b5c6d7e8f90112233445566778899aabbccddeeff0011223344  "
    "ollama-linux-amd64.tgz\n"
    "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888  "
    "ollama-linux-arm64.tgz\n"
)


# ── The insecure pattern must be gone ───────────────────────────────────────

def test_unconditional_pipe_to_shell_removed() -> None:
    """The bare unverified pipe-to-shell must no longer be reachable by default."""
    source = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    assert 'curl -fsSL https://ollama.com/install.sh | sh"' not in source, (
        "The unconditional 'curl | sh' invocation is still present."
    )


def test_install_is_version_pinned() -> None:
    """The install must target a specific reviewed version, not floating latest."""
    assert bootstrap.OLLAMA_PINNED_VERSION.startswith("v")
    source = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    assert "OLLAMA_VERSION=" in source, (
        "Even the fallback script path must pin a version for reproducibility."
    )


def test_script_fallback_requires_explicit_opt_in(monkeypatch) -> None:
    """Without the opt-in env var, the unverified path must refuse to run."""
    monkeypatch.delenv("BEL_ALLOW_OLLAMA_SCRIPT", raising=False)

    with patch.object(bootstrap, "_run") as mock_run:
        result = bootstrap._ollama_script_fallback("test reason")

    assert result is not None and result.after == bootstrap.Status.FAILED
    mock_run.assert_not_called(), (
        "The unverified install script must never execute without opt-in."
    )
    assert "BEL_ALLOW_OLLAMA_SCRIPT=1" in result.detail


def test_script_fallback_runs_only_when_opted_in(monkeypatch) -> None:
    """With the opt-in set, the pinned script is permitted."""
    monkeypatch.setenv("BEL_ALLOW_OLLAMA_SCRIPT", "1")

    class _OK:
        returncode = 0
        stderr = ""

    with patch.object(bootstrap, "_run", return_value=_OK()) as mock_run:
        result = bootstrap._ollama_script_fallback("test reason")

    assert result is None
    cmd = mock_run.call_args[0][0][-1]
    assert "OLLAMA_VERSION=" in cmd, "Fallback must pin the version."


# ── Checksum helpers ────────────────────────────────────────────────────────

def test_parse_sha256sum_finds_matching_asset() -> None:
    digest = bootstrap._parse_sha256sum(MANIFEST, "ollama-linux-amd64.tgz")
    assert digest == (
        "6f1e7c1e2d3a4b5c6d7e8f90112233445566778899aabbccddeeff0011223344"
    )


def test_parse_sha256sum_returns_none_for_absent_asset() -> None:
    assert bootstrap._parse_sha256sum(MANIFEST, "ollama-linux-riscv.tgz") is None


def test_parse_sha256sum_handles_path_prefixed_entries() -> None:
    """Manifests sometimes prefix a path; the basename must still match."""
    manifest = "abc123  ./dist/ollama-linux-amd64.tgz\n"
    assert bootstrap._parse_sha256sum(manifest, "ollama-linux-amd64.tgz") == "abc123"


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    payload = b"bel offline ai assistant test payload"
    f = tmp_path / "blob.bin"
    f.write_bytes(payload)

    assert bootstrap._sha256_file(f) == hashlib.sha256(payload).hexdigest()


# ── Integrity enforcement ───────────────────────────────────────────────────

@pytest.mark.skipif(IS_WINDOWS, reason="Linux-only install path")
def test_checksum_mismatch_aborts_without_extracting(tmp_path, monkeypatch) -> None:
    """
    A digest mismatch is a genuine integrity failure: it must abort loudly and
    must NOT silently fall back to the unverified script.
    """
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap.tempfile, "mkdtemp", lambda **kw: str(tmp_path))

    def fake_download(url: str, dest, timeout: int = 300):
        if url.endswith("sha256sum.txt"):
            Path(dest).write_text(MANIFEST, encoding="utf-8")
        else:
            Path(dest).write_bytes(b"tampered payload")  # hashes to something else
        return None

    with patch.object(bootstrap, "_download", side_effect=fake_download), \
            patch.object(bootstrap, "_run") as mock_run:
        result = bootstrap._install_ollama_linux_verified()

    assert result is not None
    assert result.after == bootstrap.Status.FAILED
    assert "SHA-256" in result.detail
    mock_run.assert_not_called(), (
        "Nothing may be extracted when the checksum does not match."
    )


@pytest.mark.skipif(IS_WINDOWS, reason="Linux-only install path")
def test_matching_checksum_proceeds_to_extract(tmp_path, monkeypatch) -> None:
    """A correct digest must allow the install to proceed."""
    payload = b"genuine ollama tarball"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = f"{digest}  ollama-linux-amd64.tgz\n"

    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(bootstrap.tempfile, "mkdtemp", lambda **kw: str(tmp_path))

    def fake_download(url: str, dest, timeout: int = 300):
        if url.endswith("sha256sum.txt"):
            Path(dest).write_text(manifest, encoding="utf-8")
        else:
            Path(dest).write_bytes(payload)
        return None

    class _OK:
        returncode = 0
        stderr = ""

    with patch.object(bootstrap, "_download", side_effect=fake_download), \
            patch.object(bootstrap, "_run", return_value=_OK()) as mock_run:
        result = bootstrap._install_ollama_linux_verified()

    assert result is None, "A verified tarball must install cleanly."
    assert mock_run.called, "Extraction should run once the digest matches."
    assert "tar" in mock_run.call_args[0][0]


@pytest.mark.skipif(IS_WINDOWS, reason="Linux-only install path")
def test_unsupported_architecture_fails_cleanly(monkeypatch) -> None:
    """An architecture with no pinned build must fail with a clear message."""
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "riscv64")

    result = bootstrap._install_ollama_linux_verified()

    assert result is not None
    assert result.after == bootstrap.Status.FAILED
    assert "riscv64" in result.detail
