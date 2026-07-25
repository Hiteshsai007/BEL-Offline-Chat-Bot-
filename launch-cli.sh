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

# Enforce offline mode -- no runtime network calls (PRD Section 12, finding S-5).
# HF_HUB_OFFLINE is the variable modern huggingface_hub actually honours.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Launch CLI
python -m app.cli
