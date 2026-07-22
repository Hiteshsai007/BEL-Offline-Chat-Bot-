# BEL Offline AI Interface — Phase 1
**Offline AI fault code lookup powered by local LLM inference**

Prepared for: **Bharat Electronics Limited (BEL)** — Technical Mentorship Review  
Authors: B V Hitesh Sai, Charan Gowda M D, Harsha B, Akshay

---

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.11 or later |
| GPU | ≥ 4 GB VRAM (NVIDIA recommended) |
| Ollama | Installed and running |
| OS | Windows 10/11 |
| Network | Not required after first setup |

---

## First-Time Setup (runs once, needs internet only this once)

```powershell
# From the project root (e:\BEL\)
.\setup.ps1
```

This script will:
1. Create a `.venv` virtual environment
2. Install all Python dependencies
3. Download the BGE embedding model (cached locally forever after)
4. Check Ollama and pull `qwen2.5:3b` if not already present
5. Run the knowledge base ingestion script

---

## How to Use (For Ship Workers)

The system has been configured for fully offline, one-click startup. No coding or terminal experience is required.

### Option 1: Desktop Shortcuts (Easiest)
There are two shortcuts located directly on the **Desktop**:
1. **BEL AI Assistant**: Double-click this to open the beautiful Chatbot graphical interface (requires a web browser to be installed on the machine).
2. **BEL AI Assistant (Terminal)**: Double-click this to open the command-line CLI interface. Use this if the machine is "bare metal" and has no web browser installed.

### Option 2: Batch Files (From the Folder)
If the shortcuts are missing, you can run the application directly from the `E:\BEL\` folder:
1. Double-click `Launch.bat` (Starts the server and opens the Chatbot UI).
2. Double-click `Launch-CLI.bat` (Opens the Terminal UI directly).

> Both options automatically activate the isolated Python environment and enforce offline security flags. No manual setup is needed.

---

## Updating the Knowledge Base

When a new version of `IRL Fault Codes.pdf` is available:

```powershell
# Place the new PDF in e:\BEL\
# Then re-run ingestion:
python -m app.ingestion.ingest
```

The index is rebuilt atomically — the running server is unaffected until the reload endpoint is called:

```powershell
# Hot-reload the index without restarting the server:
Invoke-RestMethod -Uri "http://127.0.0.1:8000/reload" -Method Post
```

---

## Project Structure

```
e:\BEL\
├── app\
│   ├── main.py              ← FastAPI server
│   ├── config.yaml          ← all tunables
│   ├── settings.py          ← config loader
│   ├── logger.py            ← rotating file logger
│   ├── rag\
│   │   ├── embedder.py      ← BGE embedding wrapper (CPU)
│   │   ├── retriever.py     ← FAISS search + confidence filter
│   │   ├── generator.py     ← Ollama inference + citation guardrail
│   │   └── pipeline.py      ← orchestration (retrieve → generate)
│   ├── ingestion\
│   │   ├── parser.py        ← PDF extraction (pdfplumber + PyMuPDF)
│   │   ├── chunker.py       ← row → chunk with metadata
│   │   ├── validator.py     ← fidelity check (verbatim source match)
│   │   └── ingest.py        ← CLI entrypoint
│   └── static\
│       ├── index.html       ← single-page UI
│       ├── style.css
│       └── app.js
├── data\
│   └── index\
│       ├── faiss.index      ← built by ingestion
│       └── chunks.jsonl     ← chunk store
├── logs\
│   └── app.log              ← rotating local log
├── requirements.txt
├── setup.ps1
└── README.md
```

---

## Tuning

All parameters are in [`app/config.yaml`](app/config.yaml) — no code changes needed:

| Parameter | Default | Effect |
|---|---|---|
| `retrieval.confidence_threshold` | `0.50` | Raise to reduce false positives |
| `retrieval.return_n` | `3` | Chunks passed to the LLM as context |
| `model.temperature` | `0.1` | Lower = more deterministic answers |
| `model.ollama_tag` | `qwen2.5:3b` | Change to upgrade the LLM |

---

## Security

- Ollama listens on `127.0.0.1` only — no remote inference
- FastAPI server listens on `127.0.0.1:8000` only
- No outbound network calls at runtime
- `TRANSFORMERS_OFFLINE=1` prevents any Hugging Face telemetry
- All logs are local, rotating, in `logs/app.log`

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Index not found" error | Run `python -m app.ingestion.ingest` |
| Status shows "Ollama not running" | Start Ollama: `ollama serve` |
| "qwen2.5:3b not found" | Run `ollama pull qwen2.5:3b` |
| Very slow responses | Check GPU is being used by Ollama: `nvidia-smi` |
| Sub-500ms response (suspicious) | Check logs — model may not be loaded |
