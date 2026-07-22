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

## First-Time Setup (Runs once, requires internet)

Before you can use the system completely offline on the ship, you must run the setup script **one time** while connected to the internet. This will download the required AI models and install the necessary software.

**Step-by-Step Instructions:**
1. Click the Windows Start Menu and type **PowerShell**.
2. Right-click on **Windows PowerShell** and select **Run as Administrator**.
3. In the blue terminal window, type the following command to navigate to the project folder (assuming you placed the folder in your E: drive):
   ```powershell
   cd E:\BEL
   ```
4. Now, run the setup script by typing:
   ```powershell
   .\setup.ps1
   ```
   *(Note: If you get a red error about "Execution of scripts is disabled", type `Set-ExecutionPolicy Bypass -Scope Process -Force`, press Enter, and then try `.\setup.ps1` again).*

**What happens next?**
The script will run automatically and do the following:
* Create an isolated Python environment (`.venv`) so it doesn't break your computer's other software.
* Install all required libraries (like FastAPI and PyTorch).
* Download the AI models (`bge-small-en-v1.5` and `qwen2.5:3b`).
* Read the `IRL Fault Codes.pdf` file and build the offline vector database.

Once it says "Setup Complete", you can disconnect from the internet forever.

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
