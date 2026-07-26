#!/usr/bin/env bash
# BEL Offline AI Technical Assistant -- Web UI Launch Script (Linux)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting BEL Offline AI Technical Assistant (Web UI)..."

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

# Check if server is already running on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Server is already running on port 8000."
else
    # Start FastAPI server in background
    echo "Starting backend server..."
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 &
    SERVER_PID=$!
    
    # Wait for server to start
    echo -n "Waiting for server to become ready"
    READY=0
    for i in {1..15}; do
        # curl -s -f returns non-zero on 503, so the loop correctly
        # keeps retrying during slow model loads or startup failures.
        if curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1; then
            READY=1
            break
        fi
        echo -n "."
        sleep 1
    done
    if [ "$READY" -eq 1 ]; then
        echo " Ready!"
    else
        echo " Timed out -- server may still be starting. Check logs/startup.log"
    fi
fi

# Open browser
echo "Opening Web UI in default browser..."
URL="http://127.0.0.1:8000"
if command -v xdg-open >/dev/null; then
    xdg-open "$URL"
elif command -v gnome-open >/dev/null; then
    gnome-open "$URL"
elif command -v python3 >/dev/null; then
    python3 -m webbrowser "$URL"
else
    echo "Could not detect web browser. Please open $URL manually."
fi

echo "Press Ctrl+C to stop the server."
wait
