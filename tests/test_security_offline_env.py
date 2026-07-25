"""
tests/test_security_offline_env.py

Covers security finding S-5: the offline guarantee was enforced with
TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE, but modern huggingface_hub (which
sentence-transformers >=3.x uses for all resolution) keys off HF_HUB_OFFLINE,
which was set nowhere -- not in Python, not in the launch scripts, not in CI.
The "no runtime network calls" claim was asserted in a docstring but never
actually enforced.

These tests pin the Python-side guarantee and additionally assert the launcher
scripts and CI workflow set the variable, since an air-gapped guarantee that
holds only in-process is not worth much.

Cross-platform notes
--------------------
* The env assertions are OS-independent.
* The script assertions read the files as text rather than executing them, so
  they hold when running the suite on either Windows or Linux (the .bat files
  cannot be executed on Linux and vice versa for the .sh files).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_hf_hub_offline_set_after_embedder_import() -> None:
    """Importing the embedder must enforce HF_HUB_OFFLINE=1."""
    import app.rag.embedder  # noqa: F401  (imported for its side effect)

    assert os.environ["HF_HUB_OFFLINE"] == "1", (
        "HF_HUB_OFFLINE must be '1' after importing app.rag.embedder -- this is "
        "the variable huggingface_hub actually honours. Without it, a cache "
        "miss can still trigger an outbound request to huggingface.co."
    )


def test_legacy_offline_vars_still_set() -> None:
    """The pre-existing switches must remain -- the new one supplements them."""
    import app.rag.embedder  # noqa: F401

    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"


def test_embedder_uses_setdefault_not_assignment() -> None:
    """
    The offline vars must be set with setdefault so the setup scripts can
    legitimately export '0' for the one-time model download. A hard assignment
    would silently break first-time setup.
    """
    source = (ROOT / "app" / "rag" / "embedder.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in source, (
        "HF_HUB_OFFLINE must be set via setdefault, not direct assignment."
    )


# ── The guarantee must hold outside the Python process too ──────────────────

def test_windows_launchers_set_hf_hub_offline() -> None:
    """Both .bat launchers must export HF_HUB_OFFLINE before starting Python."""
    for name in ("Launch.bat", "Launch-CLI.bat"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert "set HF_HUB_OFFLINE=1" in content, (
            f"{name} sets the legacy offline vars but not HF_HUB_OFFLINE."
        )


def test_linux_launchers_set_hf_hub_offline() -> None:
    """Both .sh launchers must export HF_HUB_OFFLINE before starting Python."""
    for name in ("launch.sh", "launch-cli.sh"):
        content = (ROOT / name).read_text(encoding="utf-8")
        assert "export HF_HUB_OFFLINE=1" in content, (
            f"{name} does not enforce HF_HUB_OFFLINE, so the Linux launch path "
            "has a weaker offline guarantee than the Windows one."
        )


def test_setup_ps1_sets_hf_hub_offline() -> None:
    """setup.ps1 must arm HF_HUB_OFFLINE alongside the other offline flags."""
    content = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    assert '$env:HF_HUB_OFFLINE' in content


def test_setup_ps1_lifts_hf_hub_offline_for_download() -> None:
    """
    The one-time download step must lift HF_HUB_OFFLINE too, otherwise setting
    it would break first-time model fetching.
    """
    content = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    assert '$env:HF_HUB_OFFLINE = "0"' in content, (
        "setup.ps1 arms HF_HUB_OFFLINE but never lifts it for the model "
        "download -- first-time setup would fail."
    )


def test_bootstrap_lifts_hf_hub_offline_for_download() -> None:
    """bootstrap.py's embedding download must also run with the switch lifted."""
    content = (ROOT / "bootstrap.py").read_text(encoding="utf-8")
    assert '"HF_HUB_OFFLINE": "0"' in content, (
        "bootstrap.py lifts TRANSFORMERS_OFFLINE for the model download but "
        "not HF_HUB_OFFLINE -- the download would be blocked."
    )


# NOTE: .github/workflows/ci.yml also needs HF_HUB_OFFLINE: "1" adding beside
# the existing TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE entries, so CI runs the
# suite under the same offline posture as production. That one-line change is
# NOT in this PR: the GitHub App pushing it lacks the `workflows` permission and
# the push is rejected outright. It must be applied by a human with write access
# to the workflow file. Deliberately not asserted here -- a test that fails
# until an out-of-band manual step happens would just be a broken build.
