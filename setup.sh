#!/usr/bin/env bash
# BEL Offline AI Technical Assistant -- Linux Setup
# Usage: chmod +x setup.sh && ./setup.sh
#
# This thin wrapper locates Python 3.11+, then delegates to bootstrap.py.
# All real logic lives in bootstrap.py (shared with Windows).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================================"
echo "  BEL Offline AI Technical Assistant -- Setup (Linux)"
echo "============================================================"
echo ""

# --- Locate Python 3.11+ ----------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            # Verify venv is fully functional (Debian/Ubuntu strips ensurepip from base)
            if "$candidate" -c "import ensurepip" &>/dev/null; then
                PYTHON="$candidate"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.11+ not found. Attempting to install via package manager..."
    if command -v apt-get &>/dev/null; then
        SUDO=""
        if [ "$EUID" -ne 0 ]; then
            if command -v sudo &>/dev/null; then
                SUDO="sudo"
            else
                echo "ERROR: Root privileges (sudo) required to install Python."
                exit 1
            fi
        fi
        $SUDO apt-get update -yq
        $SUDO apt-get install -yq python3 python3-venv curl zstd
        PYTHON="python3"
    elif command -v dnf &>/dev/null; then
        SUDO=""
        [ "$EUID" -ne 0 ] && command -v sudo &>/dev/null && SUDO="sudo"
        $SUDO dnf install -yq python3 curl zstd
        PYTHON="python3"
    else
        echo "ERROR: Python 3.11+ is required. Package manager not recognized."
        echo "Please install Python manually and re-run."
        exit 1
    fi
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# --- Run the bootstrap -------------------------------------------------------
"$PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@"
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SETUP COMPLETE!"
    echo ""
    echo "  Launch options:"
    echo "    ./launch.sh          (Web UI -- opens browser)"
    echo "    ./launch-cli.sh      (Terminal interface)"
    echo "============================================================"
fi

exit $exit_code
