#!/usr/bin/env python3
"""
BEL Offline AI Technical Assistant -- Cross-Platform Bootstrap
==============================================================

Automated, idempotent environment setup for Windows and Linux.

Usage:
    python bootstrap.py              # Full setup (check + install + verify)
    python bootstrap.py --check-only # Dependency audit only (no changes)
    python bootstrap.py --verbose    # Detailed logging to console

PRD Requirements Implemented:
    FR-1  OS detection (Windows / Linux)
    FR-2  Dependency inventory (present / absent / outdated)
    FR-3  Idempotent -- skip already-present components
    FR-4  Auto-install missing components
    FR-5  Cross-platform code (shared logic, thin OS adapter)
    FR-6  Human-readable summary
    FR-7  Clear, actionable errors on failure
    NFR-1 Re-run safety (no changes if everything satisfied)
    NFR-2 One shared codebase for both OSes
    NFR-3 All actions logged with source
    NFR-4 Offline after bootstrap completes
    NFR-5 Single command to reach working state
"""

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
MIN_PYTHON = (3, 11)
TOTAL_STEPS = 8

# ANSI colours (safe on Win10+ and all modern Linux terminals)
class _C:
    RST = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
    BLU = "\033[94m"; CYN = "\033[96m"

if IS_WINDOWS:
    os.system("")  # enable VT100 on Windows 10+

# ---------------------------------------------------------------------------
# Logging -- file + console
# ---------------------------------------------------------------------------
_log_dir = ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
_LOG_FILE = _log_dir / "bootstrap.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("bootstrap")

def _print(icon: str, colour: str, msg: str):
    print(f"  {colour}{icon}{_C.RST} {msg}")
    log.info(msg)

def ok(msg: str):   _print("OK", _C.GRN, msg)
def warn(msg: str): _print("!!", _C.YLW, msg)
def fail(msg: str): _print("FAIL", _C.RED, msg)

def step(n: int, msg: str):
    print(f"\n{_C.CYN}[{n}/{TOTAL_STEPS}]{_C.RST} {_C.B}{msg}{_C.RST}")
    log.info("Step %d/%d: %s", n, TOTAL_STEPS, msg)

# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------
class Status(Enum):
    PRESENT   = "present"
    ABSENT    = "absent"
    OUTDATED  = "outdated"
    INSTALLED = "installed"
    UPDATED   = "updated"
    FAILED    = "failed"

@dataclass
class Result:
    name: str
    before: Status
    after: Status
    version: str = ""
    action: str = ""
    detail: str = ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    """Run a command, log it, return CompletedProcess."""
    log.debug("Running: %s", " ".join(str(c) for c in cmd))
    # Use UTF-8 with replace to handle Ollama's progress bar characters
    # that fail on Windows cp1252 default encoding.
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, capture_output=True, **kw)

def _read_config_tag() -> str:
    """Read ollama_tag from config.yaml without PyYAML (stdlib only)."""
    cfg = ROOT / "app" / "config.yaml"
    if cfg.exists():
        m = re.search(r'ollama_tag:\s*["\']?([^"\'\n]+)', cfg.read_text("utf-8"))
        if m:
            return m.group(1).strip()
    return "qwen2.5:3b"

def _read_config_embed_model() -> str:
    cfg = ROOT / "app" / "config.yaml"
    if cfg.exists():
        m = re.search(r'model_name:\s*["\']?([^"\'\n]+)', cfg.read_text("utf-8"))
        if m:
            return m.group(1).strip()
    return "BAAI/bge-small-en-v1.5"

def _http_get_json(url: str, timeout: int = 5) -> Optional[dict]:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None

def _venv_python() -> Path:
    venv = ROOT / ".venv"
    if IS_WINDOWS:
        return venv / "Scripts" / "python.exe"
    p = venv / "bin" / "python3"
    return p if p.exists() else venv / "bin" / "python"

def _venv_pip() -> Path:
    venv = ROOT / ".venv"
    if IS_WINDOWS:
        return venv / "Scripts" / "pip.exe"
    return venv / "bin" / "pip3" if (venv / "bin" / "pip3").exists() else venv / "bin" / "pip"

def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix

def _find_executable(name: str) -> Optional[str]:
    """Find an executable in PATH or common fallback locations."""
    path = shutil.which(name)
    if path:
        return path
    
    # Fallback to standard installation paths that might not be in the current process PATH
    if IS_LINUX:
        for base in ("/usr/local/bin", "/usr/bin", "/bin", "/opt/bin"):
            p = Path(base) / name
            if p.exists() and os.access(p, os.X_OK):
                return str(p)
        p = Path.home() / ".local" / "bin" / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    elif IS_WINDOWS:
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            # winget standard install locations
            for sub in (["Programs", name.capitalize()], ["Microsoft", "WindowsApps"]):
                p = Path(local_app).joinpath(*sub) / f"{name}.exe"
                if p.exists() and os.access(p, os.X_OK):
                    return str(p)
    return None

# ---------------------------------------------------------------------------
# FR-1  OS Detection
# ---------------------------------------------------------------------------
def detect_os() -> dict:
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "pkg_manager": None,
    }
    if IS_LINUX:
        for pm in ("apt", "dnf", "yum", "pacman"):
            if shutil.which(pm):
                info["pkg_manager"] = pm
                break
    elif IS_WINDOWS:
        for pm in ("winget", "choco"):
            if shutil.which(pm):
                info["pkg_manager"] = pm
                break
    return info

# ---------------------------------------------------------------------------
# Dependency checks  (FR-2 / FR-3)
# ---------------------------------------------------------------------------

def check_python() -> Result:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        return Result("Python", Status.PRESENT, Status.PRESENT, ver)
    return Result("Python", Status.OUTDATED, Status.OUTDATED, ver,
                  detail=f"Need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")

def check_venv() -> Result:
    venv = ROOT / ".venv"
    py = _venv_python()
    if venv.exists() and (py.exists() or (venv / "Scripts" / "python.exe").exists()
                          or (venv / "bin" / "python").exists()):
        return Result("Virtual Env", Status.PRESENT, Status.PRESENT, str(venv))
    return Result("Virtual Env", Status.ABSENT, Status.ABSENT)

def check_pip_packages() -> Result:
    """Check whether all packages from requirements.txt are installed in venv."""
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return Result("pip Packages", Status.ABSENT, Status.ABSENT, detail="requirements.txt missing")
    pip = _venv_pip()
    if not pip.exists():
        return Result("pip Packages", Status.ABSENT, Status.ABSENT, detail="venv pip not found")
    r = _run([str(pip), "list", "--format=json"])
    if r.returncode != 0:
        return Result("pip Packages", Status.ABSENT, Status.ABSENT, detail="pip list failed")
    installed = {p["name"].lower() for p in json.loads(r.stdout)}
    needed = set()
    for line in req_file.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[><=!;\[]", line)[0].strip().lower()
        if name:
            needed.add(name)
    missing = needed - installed
    if not missing:
        return Result("pip Packages", Status.PRESENT, Status.PRESENT, f"{len(needed)} packages")
    return Result("pip Packages", Status.ABSENT, Status.ABSENT,
                  detail=f"Missing: {', '.join(sorted(missing))}")

def check_pytorch() -> Result:
    py = _venv_python()
    if not py.exists():
        return Result("PyTorch", Status.ABSENT, Status.ABSENT)
    r = _run([str(py), "-c", "import torch; print(torch.__version__)"])
    if r.returncode == 0:
        return Result("PyTorch", Status.PRESENT, Status.PRESENT, r.stdout.strip())
    return Result("PyTorch", Status.ABSENT, Status.ABSENT)

def check_ollama_binary() -> Result:
    path = _find_executable("ollama")
    if path:
        r = _run([path, "--version"])
        ver = r.stdout.strip() if r.returncode == 0 else "unknown"
        return Result("Ollama Binary", Status.PRESENT, Status.PRESENT, ver)
    return Result("Ollama Binary", Status.ABSENT, Status.ABSENT)

def check_ollama_service() -> Result:
    data = _http_get_json("http://127.0.0.1:11434/api/tags")
    if data is not None:
        return Result("Ollama Service", Status.PRESENT, Status.PRESENT, "running")
    return Result("Ollama Service", Status.ABSENT, Status.ABSENT)

def check_model() -> Result:
    tag = _read_config_tag()
    data = _http_get_json("http://127.0.0.1:11434/api/tags")
    if data is None:
        return Result(f"Model ({tag})", Status.ABSENT, Status.ABSENT,
                      detail="Ollama not reachable")
    names = [m.get("name", "") for m in data.get("models", [])]
    if any(tag in n for n in names):
        return Result(f"Model ({tag})", Status.PRESENT, Status.PRESENT, tag)
    return Result(f"Model ({tag})", Status.ABSENT, Status.ABSENT,
                  detail=f"Available: {', '.join(names) or 'none'}")

def check_embedding_model() -> Result:
    model_id = _read_config_embed_model()
    cache_name = f"models--{model_id.replace('/', '--')}"
    candidates = []
    for env in ("SENTENCE_TRANSFORMERS_HOME", "HF_HOME"):
        val = os.environ.get(env)
        if val:
            candidates.append(Path(val) / cache_name)
    home = Path.home()
    candidates += [
        home / ".cache" / "huggingface" / "hub" / cache_name,
        home / ".cache" / "torch" / "sentence_transformers" / model_id.replace("/", "_"),
    ]
    for c in candidates:
        if c.exists():
            return Result("Embedding Model", Status.PRESENT, Status.PRESENT, model_id)
    return Result("Embedding Model", Status.ABSENT, Status.ABSENT, detail=model_id)

def check_faiss_index() -> Result:
    idx = ROOT / "data" / "index" / "faiss.index"
    chunks = ROOT / "data" / "index" / "chunks.jsonl"
    if idx.exists() and chunks.exists():
        return Result("FAISS Index", Status.PRESENT, Status.PRESENT,
                      f"{idx.stat().st_size / 1024:.0f} KB")
    return Result("FAISS Index", Status.ABSENT, Status.ABSENT)

# ---------------------------------------------------------------------------
# Install actions  (FR-4)
# ---------------------------------------------------------------------------

def install_venv() -> Result:
    venv = ROOT / ".venv"
    if venv.exists():
        return Result("Virtual Env", Status.PRESENT, Status.PRESENT, str(venv))
    log.info("Creating virtual environment at %s", venv)
    r = _run([sys.executable, "-m", "venv", str(venv)])
    if r.returncode == 0:
        return Result("Virtual Env", Status.ABSENT, Status.INSTALLED, str(venv),
                      action="python -m venv .venv")
    return Result("Virtual Env", Status.ABSENT, Status.FAILED, detail=r.stderr)

def install_pytorch() -> Result:
    chk = check_pytorch()
    if chk.before == Status.PRESENT:
        return chk
    pip = _venv_pip()
    log.info("Installing PyTorch (CPU-only) ...")
    r = _run([str(pip), "install", "torch",
              "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"])
    if r.returncode == 0:
        v = check_pytorch()
        return Result("PyTorch", Status.ABSENT, Status.INSTALLED, v.version,
                      action="pip install torch (CPU)")
    return Result("PyTorch", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])

def install_pip_packages() -> Result:
    chk = check_pip_packages()
    if chk.before == Status.PRESENT:
        return chk
    pip = _venv_pip()
    req = ROOT / "requirements.txt"
    log.info("Installing pip packages from %s ...", req)
    r = _run([str(pip), "install", "-r", str(req), "--quiet"])
    if r.returncode == 0:
        v = check_pip_packages()
        return Result("pip Packages", Status.ABSENT, Status.INSTALLED, v.version,
                      action="pip install -r requirements.txt")
    return Result("pip Packages", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])

def install_ollama(os_info: dict) -> Result:
    chk = check_ollama_binary()
    if chk.before == Status.PRESENT:
        return chk
    
    success = False
    action = ""
    if IS_LINUX:
        log.info("Installing Ollama via install script ...")
        r = _run(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"])
        if r.returncode == 0:
            success = True
            action = "curl install.sh | sh"
        else:
            return Result("Ollama Binary", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])
    elif IS_WINDOWS:
        pm = os_info.get("pkg_manager")
        if pm == "winget":
            log.info("Installing Ollama via winget ...")
            r = _run(["winget", "install", "Ollama.Ollama",
                       "--accept-source-agreements", "--accept-package-agreements"])
            if r.returncode == 0:
                success = True
                action = "winget install Ollama.Ollama"
            else:
                return Result("Ollama Binary", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])
        else:
            return Result("Ollama Binary", Status.ABSENT, Status.FAILED,
                          detail="Install Ollama manually from https://ollama.com/download")
    else:
        return Result("Ollama Binary", Status.ABSENT, Status.FAILED, detail="Unsupported OS")
        
    if success:
        if not _find_executable("ollama"):
            return Result("Ollama Binary", Status.ABSENT, Status.FAILED,
                          detail="installed but not found - PATH issue?")
        return Result("Ollama Binary", Status.ABSENT, Status.INSTALLED, action=action)
    return Result("Ollama Binary", Status.ABSENT, Status.FAILED, detail="Unknown failure")

def start_ollama_service() -> Result:
    chk = check_ollama_service()
    if chk.before == Status.PRESENT:
        return chk
    
    ollama_bin = _find_executable("ollama")
    if not ollama_bin:
        return Result("Ollama Service", Status.ABSENT, Status.FAILED,
                      detail="Ollama binary not found")
        
    log.info("Starting Ollama service ...")
    if IS_WINDOWS:
        subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    # Wait for service to become ready
    for _ in range(240):
        time.sleep(1)
        if check_ollama_service().before == Status.PRESENT:
            return Result("Ollama Service", Status.ABSENT, Status.INSTALLED,
                          "running", action="ollama serve")
    return Result("Ollama Service", Status.ABSENT, Status.FAILED,
                  detail="Service did not start within 4 minutes")

def pull_model() -> Result:
    chk = check_model()
    if chk.before == Status.PRESENT:
        return chk
    tag = _read_config_tag()
    if check_ollama_service().before != Status.PRESENT:
        return Result(f"Model ({tag})", Status.ABSENT, Status.FAILED,
                      detail="Ollama service not running")
                      
    ollama_bin = _find_executable("ollama")
    if not ollama_bin:
        return Result(f"Model ({tag})", Status.ABSENT, Status.FAILED,
                      detail="Ollama binary not found")
                      
    log.info("Pulling model %s ...", tag)
    r = _run([ollama_bin, "pull", tag])
    if r.returncode == 0:
        return Result(f"Model ({tag})", Status.ABSENT, Status.INSTALLED, tag,
                      action=f"ollama pull {tag}")
    return Result(f"Model ({tag})", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])

def download_embedding_model() -> Result:
    chk = check_embedding_model()
    if chk.before == Status.PRESENT:
        return chk
    model_id = _read_config_embed_model()
    py = _venv_python()
    if not py.exists():
        return Result("Embedding Model", Status.ABSENT, Status.FAILED,
                      detail="venv python not found")
    log.info("Downloading embedding model %s ...", model_id)
    code = (
        "from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{model_id}', device='cpu')"
    )
    # One-time model download: both offline switches must be lifted, or
    # huggingface_hub refuses the fetch (finding S-5).
    download_env = {**os.environ, "TRANSFORMERS_OFFLINE": "0", "HF_HUB_OFFLINE": "0"}
    r = _run([str(py), "-c", code], env=download_env)
    if r.returncode == 0:
        return Result("Embedding Model", Status.ABSENT, Status.INSTALLED, model_id,
                      action=f"Downloaded {model_id}")
    return Result("Embedding Model", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])

def build_faiss_index() -> Result:
    chk = check_faiss_index()
    if chk.before == Status.PRESENT:
        return chk
    pdf = ROOT / "IRL Fault Codes.pdf"
    if not pdf.exists():
        return Result("FAISS Index", Status.ABSENT, Status.FAILED,
                      detail="Source PDF not found: IRL Fault Codes.pdf")
    py = _venv_python()
    if not py.exists():
        return Result("FAISS Index", Status.ABSENT, Status.FAILED,
                      detail="venv python not found")
    log.info("Building FAISS index from %s ...", pdf)
    r = _run([str(py), "-m", "app.ingestion.ingest", "--pdf", str(pdf)], cwd=str(ROOT))
    if r.returncode == 0:
        return Result("FAISS Index", Status.ABSENT, Status.INSTALLED,
                      action="python -m app.ingestion.ingest")
    return Result("FAISS Index", Status.ABSENT, Status.FAILED, detail=r.stderr[:300])

# ---------------------------------------------------------------------------
# FR-6  Summary
# ---------------------------------------------------------------------------
_STATUS_ICON = {
    Status.PRESENT:   f"{_C.GRN}PRESENT  {_C.RST}",
    Status.ABSENT:    f"{_C.RED}ABSENT   {_C.RST}",
    Status.OUTDATED:  f"{_C.YLW}OUTDATED {_C.RST}",
    Status.INSTALLED: f"{_C.GRN}INSTALLED{_C.RST}",
    Status.UPDATED:   f"{_C.GRN}UPDATED  {_C.RST}",
    Status.FAILED:    f"{_C.RED}FAILED   {_C.RST}",
}

def print_summary(results: List[Result]):
    print(f"\n{'=' * 64}")
    print(f"  {'Component':<22} {'Status':<20} {'Action / Detail'}")
    print(f"  {'-' * 20:<22} {'-' * 16:<20} {'-' * 24}")
    for r in results:
        icon = _STATUS_ICON.get(r.after, r.after.value)
        detail = r.action or r.version or r.detail or ""
        if len(detail) > 36:
            detail = detail[:33] + "..."
        print(f"  {r.name:<22} {icon}  {detail}")
    print(f"{'=' * 64}")
    passed = all(r.after in (Status.PRESENT, Status.INSTALLED, Status.UPDATED)
                 for r in results)
    if passed:
        print(f"\n  {_C.GRN}{_C.B}All checks passed. System is ready.{_C.RST}")
    else:
        failed = [r for r in results if r.after in (Status.FAILED, Status.ABSENT, Status.OUTDATED)]
        print(f"\n  {_C.RED}{_C.B}{len(failed)} component(s) need attention:{_C.RST}")
        for r in failed:
            print(f"    - {r.name}: {r.detail}")
    # Log to file
    for r in results:
        log.info("SUMMARY | %-22s | %-10s | %s", r.name, r.after.value,
                 r.action or r.detail or r.version)

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_bootstrap(check_only: bool = False) -> bool:
    start = time.time()
    print(f"\n{_C.CYN}{'=' * 64}{_C.RST}")
    print(f"  {_C.B}BEL Offline AI Technical Assistant -- Bootstrap{_C.RST}")
    print(f"{_C.CYN}{'=' * 64}{_C.RST}")

    # FR-1: OS detection
    os_info = detect_os()
    print(f"\n  OS: {_C.B}{os_info['system']} {os_info['release']}{_C.RST}"
          f"  ({os_info['machine']})")
    if os_info["pkg_manager"]:
        print(f"  Package manager: {os_info['pkg_manager']}")
    log.info("OS detected: %s", os_info)

    if os_info["system"] not in ("Windows", "Linux"):
        fail(f"Unsupported OS: {os_info['system']}. Only Windows and Linux are supported.")
        return False

    results: List[Result] = []

    # Step 1: Python version
    step(1, "Checking Python version")
    r = check_python()
    results.append(r)
    if r.after == Status.PRESENT:
        ok(f"Python {r.version}")
    else:
        fail(f"Python {r.version} -- need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        fail("Install Python 3.11+ from https://python.org and re-run.")
        print_summary(results)
        return False

    # Step 2: Virtual environment
    step(2, "Checking virtual environment")
    r = check_venv()
    if r.before == Status.PRESENT:
        ok("Virtual environment exists")
        results.append(r)
    elif check_only:
        warn("Virtual environment not found")
        results.append(r)
    else:
        r = install_venv()
        results.append(r)
        if r.after == Status.INSTALLED:
            ok("Virtual environment created")
        else:
            fail(f"Could not create venv: {r.detail}")
            print_summary(results)
            return False

    # Re-exec inside venv if not already there
    if not _in_venv() and not check_only:
        py = _venv_python()
        if py.exists():
            log.info("Re-executing inside venv: %s", py)
            ret = subprocess.call([str(py), str(Path(__file__).resolve())] +
                                  sys.argv[1:])
            sys.exit(ret)

    # Step 3: PyTorch (CPU)
    step(3, "Checking PyTorch")
    if check_only:
        r = check_pytorch()
        results.append(r)
    else:
        r = install_pytorch()
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok(f"PyTorch {r.version}")
    else:
        fail(f"PyTorch: {r.detail}")

    # Step 4: pip packages
    step(4, "Checking pip packages (requirements.txt)")
    if check_only:
        r = check_pip_packages()
        results.append(r)
    else:
        r = install_pip_packages()
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok(f"All packages satisfied ({r.version})")
    else:
        fail(f"pip packages: {r.detail}")

    # Step 5: Ollama binary
    step(5, "Checking Ollama runtime")
    if check_only:
        r = check_ollama_binary()
        results.append(r)
    else:
        r = install_ollama(os_info)
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok(f"Ollama binary found ({r.version})")
    else:
        fail(f"Ollama: {r.detail}")

    # Step 6: Ollama service
    step(6, "Checking Ollama service")
    if check_only:
        r = check_ollama_service()
        results.append(r)
    else:
        r = start_ollama_service()
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok("Ollama service is running")
    else:
        warn(f"Ollama service: {r.detail}")

    # Step 7: LLM model
    step(7, f"Checking LLM model ({_read_config_tag()})")
    if check_only:
        r = check_model()
        results.append(r)
    else:
        r = pull_model()
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok(f"Model ready: {r.version}")
    else:
        warn(f"Model: {r.detail}")

    # Step 8: Embedding model
    step(8, f"Checking embedding model")
    if check_only:
        r = check_embedding_model()
        results.append(r)
    else:
        r = download_embedding_model()
        results.append(r)
    if r.after in (Status.PRESENT, Status.INSTALLED):
        ok(f"Embedding model cached: {r.version}")
    else:
        warn(f"Embedding model: {r.detail}")

    # Summary (FR-6)
    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.1f}s")
    log.info("Bootstrap completed in %.1fs", elapsed)
    print_summary(results)

    # Log to bootstrap log
    log.info("Bootstrap log saved to %s", _LOG_FILE)
    print(f"\n  Log: {_LOG_FILE}")

    return all(r.after in (Status.PRESENT, Status.INSTALLED, Status.UPDATED)
               for r in results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BEL Offline AI -- cross-platform environment bootstrap"
    )
    parser.add_argument("--check-only", action="store_true",
                        help="Audit dependencies without installing anything")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed log messages to console")
    args = parser.parse_args()

    if args.verbose:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
        log.addHandler(console)

    success = run_bootstrap(check_only=args.check_only)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
