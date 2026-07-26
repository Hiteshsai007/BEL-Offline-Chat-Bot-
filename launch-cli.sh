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

# Ensure log directory exists
mkdir -p logs

# Verify Ollama is running (CLI needs it for LLM inference)
if ! curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is not running. Attempting to start..."
    if command -v ollama >/dev/null 2>&1; then
        nohup ollama serve > logs/ollama.log 2>&1 &
        echo -n "Waiting for Ollama"
        for i in $(seq 1 30); do
            if curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                echo " Ready!"
                break
            fi
            echo -n "."
            sleep 1
        done
        if ! curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
            echo ""
            echo "WARNING: Ollama may not be ready yet. CLI will retry on first query."
        fi
    else
        echo "WARNING: Ollama not found. LLM inference will be unavailable."
        echo "Run ./setup.sh to install all dependencies."
    fi
fi

# Launch CLI
python -m app.cli
