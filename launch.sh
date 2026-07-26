#!/usr/bin/env bash
# BEL Offline AI Technical Assistant -- Web UI Launch Script (Linux)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting BEL Offline AI Technical Assistant (Web UI)..."

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running first-time setup via setup.sh..."
    chmod +x ./setup.sh
    ./setup.sh
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

# -- Verify Ollama is running ------------------------------------------------
if ! curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is not running. Attempting to start..."
    if command -v ollama >/dev/null 2>&1; then
        nohup ollama serve > logs/ollama.log 2>&1 &
        echo -n "Waiting for Ollama to start"
        OLLAMA_READY=0
        for i in $(seq 1 30); do
            if curl -s -f http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                OLLAMA_READY=1
                break
            fi
            echo -n "."
            sleep 1
        done
        if [ "$OLLAMA_READY" -eq 1 ]; then
            echo " Ready!"
        else
            echo ""
            echo "ERROR: Ollama did not start within 30 seconds."
            echo "Please start Ollama manually: ollama serve"
            exit 1
        fi
    else
        echo "ERROR: Ollama is not installed. Please run ./setup.sh first."
        exit 1
    fi
fi

# -- Graceful shutdown -------------------------------------------------------
SERVER_PID=""
cleanup() {
    echo ""
    echo "Shutting down server..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Server stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# -- Stop any stale server process on port 8000 ------------------------------
if command -v lsof >/dev/null 2>&1; then
    OLD_PID=$(lsof -t -i:8000 2>/dev/null || true)
    if [ -n "$OLD_PID" ]; then
        echo "Stopping existing process ($OLD_PID) on port 8000..."
        kill -9 $OLD_PID 2>/dev/null || true
        sleep 1
    fi
fi

# Start FastAPI server in background
echo "Starting backend server..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs/startup.log 2>&1 &
SERVER_PID=$!

    # Wait for server to start (60 seconds, matching Windows launcher)
    echo -n "Waiting for server to become ready"
    READY=0
    for i in $(seq 1 60); do
        # curl -s -f returns non-zero on 503, so the loop correctly
        # keeps retrying during slow model loads or startup failures.
        if curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1; then
            READY=1
            break
        fi
        # Check if server process died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "ERROR: Server process exited unexpectedly. Check logs/startup.log"
            exit 1
        fi
        echo -n "."
        sleep 1
    done
    if [ "$READY" -eq 1 ]; then
        echo " Ready!"
    else
        echo ""
        echo "WARNING: Server may still be starting. Check logs/startup.log"
        echo "On low-memory systems, startup can take up to 2 minutes."
    fi

# Open browser
echo "Opening Web UI in default browser..."
URL="http://127.0.0.1:8000"
if command -v xdg-open >/dev/null; then
    xdg-open "$URL" 2>/dev/null || true
elif command -v gnome-open >/dev/null; then
    gnome-open "$URL" 2>/dev/null || true
elif command -v python3 >/dev/null; then
    python3 -m webbrowser "$URL" 2>/dev/null || true
else
    echo "Could not detect web browser. Please open $URL manually."
fi

echo "Press Ctrl+C to stop the server."
wait "$SERVER_PID" 2>/dev/null || true
