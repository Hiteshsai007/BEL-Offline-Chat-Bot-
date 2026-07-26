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
        $SUDO apt-get install -yq python3 python3-venv curl zstd lsof
        PYTHON="python3"
    elif command -v dnf &>/dev/null; then
        SUDO=""
        [ "$EUID" -ne 0 ] && command -v sudo &>/dev/null && SUDO="sudo"
        $SUDO dnf install -yq python3 curl zstd lsof
        PYTHON="python3"
    else
        echo "ERROR: Python 3.11+ is required. Package manager not recognized."
        echo "Please install Python manually and re-run."
        exit 1
    fi
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# --- Check memory and swap (recommended for 4 GB RAM machines) ---------------
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
TOTAL_SWAP_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
TOTAL_MEM_MB=$((TOTAL_MEM_KB / 1024))
TOTAL_SWAP_MB=$((TOTAL_SWAP_KB / 1024))

if [ "$TOTAL_MEM_MB" -lt 6000 ] && [ "$TOTAL_SWAP_MB" -lt 2000 ]; then
    echo ""
    echo "WARNING: This system has ${TOTAL_MEM_MB} MB RAM and ${TOTAL_SWAP_MB} MB swap."
    echo "For systems with less than 6 GB RAM, at least 2 GB of swap is recommended."
    echo "To create a 2 GB swap file:"
    echo "  sudo fallocate -l 2G /swapfile"
    echo "  sudo chmod 600 /swapfile"
    echo "  sudo mkswap /swapfile"
    echo "  sudo swapon /swapfile"
    echo "  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
    echo ""
    echo "Continuing setup (press Ctrl+C to abort and add swap first)..."
    sleep 5
fi

# --- Run the bootstrap -------------------------------------------------------
exit_code=0
"$PYTHON" "$SCRIPT_DIR/bootstrap.py" "$@" || exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SETUP COMPLETE!"
    echo ""
    echo "  Launch options:"
    echo "    ./launch.sh          (Web UI -- opens browser)"
    echo "    ./launch-cli.sh      (Terminal interface)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "  SETUP FAILED!"
    echo ""
    echo "  Please check the log above for errors."
    echo "============================================================"
fi

exit $exit_code
