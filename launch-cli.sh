#!/usr/bin/env bash
# BEL Offline AI Technical Assistant -- CLI Launch Script (Linux)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Launch CLI
python -m app.cli
