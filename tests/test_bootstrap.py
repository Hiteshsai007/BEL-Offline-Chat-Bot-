import sys
from pathlib import Path

import pytest

# Ensure we can import bootstrap
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import bootstrap
except ImportError:
    pytest.skip("Could not import bootstrap.py (syntax error or missing dependencies)", allow_module_level=True)


def test_os_detection():
    os_info = bootstrap.detect_os()
    assert os_info["system"] in ("Windows", "Linux"), f"Unsupported OS detected: {os_info['system']}"
    assert "release" in os_info
    assert "machine" in os_info
    if os_info["system"] == "Linux":
        assert os_info["pkg_manager"] in ("apt", "dnf", "yum", "pacman", None)
    elif os_info["system"] == "Windows":
        assert os_info["pkg_manager"] in ("winget", "choco", None)


def test_python_version_check():
    res = bootstrap.check_python()
    assert res.name == "Python"
    # The runner must be at least Python 3.11 for this test to even run correctly
    # or if it's running on an older version, we expect an OUTDATED result.
    if sys.version_info >= (3, 11):
        assert res.before == bootstrap.Status.PRESENT
        assert res.after == bootstrap.Status.PRESENT
    else:
        assert res.before == bootstrap.Status.OUTDATED
        assert res.after == bootstrap.Status.OUTDATED


def test_venv_check_mocked(monkeypatch):
    """Test that check_venv correctly identifies presence based on pathlib."""
    def mock_exists(self):
        return True

    # Mocking Path.exists to always return True for this test
    monkeypatch.setattr(Path, "exists", mock_exists)

    res = bootstrap.check_venv()
    assert res.name == "Virtual Env"
    assert res.before == bootstrap.Status.PRESENT


def test_read_config_tag():
    """Ensure we parse the config file correctly without pyyaml."""
    tag = bootstrap._read_config_tag()
    assert isinstance(tag, str)
    assert len(tag) > 0
    # Should match whatever is in config.yaml
    assert tag == "qwen2.5:7b"


def test_read_config_embed():
    model = bootstrap._read_config_embed_model()
    assert model == "BAAI/bge-small-en-v1.5"


def test_summary_does_not_crash(capsys):
    """Test that print_summary formats text without crashing."""
    results = [
        bootstrap.Result("Test1", bootstrap.Status.ABSENT, bootstrap.Status.INSTALLED, detail="installed fine"),
        bootstrap.Result("Test2", bootstrap.Status.PRESENT, bootstrap.Status.PRESENT, version="1.0"),
    ]
    bootstrap.print_summary(results)
    captured = capsys.readouterr()
    assert "Test1" in captured.out
    assert "Test2" in captured.out
    assert "All checks passed" in captured.out


def test_summary_failure_does_not_crash(capsys):
    """Test summary with a failed component."""
    results = [
        bootstrap.Result("Test1", bootstrap.Status.ABSENT, bootstrap.Status.FAILED, detail="broken"),
    ]
    bootstrap.print_summary(results)
    captured = capsys.readouterr()
    assert "Test1" in captured.out
    assert "1 component(s) need attention" in captured.out


def test_bootstrap_build_faiss_index(tmp_path, monkeypatch):
    import shutil
    import subprocess
    # Mock ROOT to point to tmp_path
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)

    # Copy IRL Fault Codes.pdf to tmp_path
    real_pdf = ROOT / "IRL Fault Codes.pdf"
    tmp_pdf = tmp_path / "IRL Fault Codes.pdf"
    shutil.copy(real_pdf, tmp_pdf)

    # Mock _venv_python to return standard Python
    monkeypatch.setattr(bootstrap, "_venv_python", lambda: Path(sys.executable))

    # Mock _run so that instead of running the subprocess, it creates the index and chunks files,
    # and returns a CompletedProcess with returncode 0.
    def mock_run(cmd, **kw):
        idx_dir = tmp_path / "data" / "index"
        idx_dir.mkdir(parents=True, exist_ok=True)
        (idx_dir / "faiss.index").write_text("mock index content")
        (idx_dir / "chunks.jsonl").write_text("mock chunks content")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bootstrap, "_run", mock_run)

    # Call build_faiss_index
    res = bootstrap.build_faiss_index()

    # Assert FAISS index file and chunks file exist in the tmp_path/data/index/
    idx_file = tmp_path / "data" / "index" / "faiss.index"
    chunks_file = tmp_path / "data" / "index" / "chunks.jsonl"

    assert res.after == bootstrap.Status.INSTALLED
    assert idx_file.exists()
    assert chunks_file.exists()
