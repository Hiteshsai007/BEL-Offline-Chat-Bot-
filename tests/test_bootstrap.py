import subprocess
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
    if (sys.version_info.major, sys.version_info.minor) >= bootstrap.MIN_PYTHON and (sys.version_info.major, sys.version_info.minor) <= bootstrap.MAX_PYTHON:
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
    assert tag == "qwen2.5:3b"


def test_check_pip_packages_reports_outdated_version(tmp_path, monkeypatch):
    """Bootstrap should treat installed packages that do not satisfy the requirements as outdated."""
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("rapidocr_onnxruntime>=1.3.0\n", encoding="utf-8")

    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "_venv_pip", lambda: tmp_path / "pip.exe")

    def mock_run(cmd, **kw):
        if len(cmd) >= 3 and cmd[1] == "list" and cmd[2] == "--format=json":
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='[{"name": "rapidocr_onnxruntime", "version": "1.2.0"}]', stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(bootstrap, "_run", mock_run)

    res = bootstrap.check_pip_packages()
    assert res.before == bootstrap.Status.OUTDATED
    assert res.after == bootstrap.Status.OUTDATED


def test_run_bootstrap_reexecutes_with_supported_python(monkeypatch):
    """Bootstrap should restart itself with a supported interpreter when the current one is unsupported."""
    monkeypatch.setattr(bootstrap, "detect_os", lambda: {"system": "Windows", "release": "11", "machine": "AMD64", "pkg_manager": "winget"})
    monkeypatch.setattr(bootstrap, "check_python", lambda: bootstrap.Result("Python", bootstrap.Status.OUTDATED, bootstrap.Status.OUTDATED, "3.14.2", detail="unsupported"))
    monkeypatch.setattr(bootstrap, "install_python_windows", lambda: None)
    monkeypatch.setattr(bootstrap, "_find_supported_python_command", lambda: ["C:/Python313/python.exe"])
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py", "--check-only"])

    called = {}

    def fake_call(cmd, **kwargs):
        called["cmd"] = cmd
        called["env"] = kwargs.get("env")
        raise SystemExit(0)

    monkeypatch.setattr(bootstrap.subprocess, "call", fake_call)

    with pytest.raises(SystemExit) as exc:
        bootstrap.run_bootstrap(check_only=True)

    assert exc.value.code == 0
    assert called["cmd"][0] == "C:/Python313/python.exe"


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
