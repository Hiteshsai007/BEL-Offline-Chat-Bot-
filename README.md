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
| RAM | ≥ 8 GB system memory |
| Disk Space | ≥ 6 GB free on the drive where BEL is installed (for the AI model, Python packages, and temporary download files) |
| Ollama | Installed and running |
| OS | Windows 10/11 or Linux (Ubuntu/Debian) |
| Network | Not required after first setup |

---

## First-Time Setup (Runs once, requires internet)

Before you can use the system completely offline on the ship, you must run the setup script **one time** while connected to the internet. This will download the required AI models and install the necessary software.

### Windows Setup (Zero-Typing Instructions)
1. Open the `E:\BEL\` folder.
2. Double-click the file named **`First-Time-Setup.bat`**.
3. A black terminal window will open and ask you to press any key to continue. Press any key.

### Linux Setup (Step-by-Step)

1. **Open a terminal.**
   Most Linux desktops: right-click on the desktop or in the file manager and
   choose "Open Terminal here," or find "Terminal" in the applications menu.

2. **Navigate to the BEL folder.**
```bash
   cd /path/to/BEL
```
   (Replace `/path/to/BEL` with wherever the folder was copied to, e.g. `~/BEL`
   or `/opt/BEL`.)

3. **Check free disk space before starting.**
   The setup needs at least **6 GB free** on this drive (for the AI model,
   the software packages, and temporary files during download).
```bash
   df -h .
```
   Look at the "Avail" column for the drive the BEL folder is on. If it shows
   less than 6 GB, free up space first — the setup will fail partway through
   and may need to be re-run from a clean state if it doesn't.

4. **Make the setup script runnable.**
```bash
   chmod +x setup.sh
```
   (This only needs to be done once — it gives the script permission to run.)

5. **Run the setup script.**
```bash
   ./setup.sh
```
   You may be asked for your password partway through — this is normal, since
   installing some components (like Ollama) requires administrator
   permission. Type your password and press Enter (nothing will appear on
   screen as you type — that's expected).

6. **Wait for it to finish.**
   The script runs automatically and will:
   - Install any missing system packages
   - Set up an isolated Python environment
   - Download the offline AI model and embedding model
   - Build the fault-code search index

   This can take several minutes depending on internet speed — do not close
   the terminal window during this step.

7. **Look for the completion message.**
   When you see `SETUP COMPLETE!`, the setup finished successfully. You can
   now disconnect from the internet permanently — the assistant will keep
   working fully offline from here on.

**If setup fails partway through:** re-check step 3 (free disk space) first —
this is the most common cause of a failed setup. If space was the issue, free
up space and simply run `./setup.sh` again; it will skip anything that
already installed successfully and only retry what failed.

**What happens next?**
The script will run automatically in the background and do the following:
* **Set up the Engine**: It installs the background software needed to run the chatbot safely on your machine.
* **Download the Brain**: It pulls down the offline AI models so they can run locally without the internet.

Once it says "SETUP COMPLETE!", you can close the window and disconnect from the internet forever.

---

## How to Use (For Ship Workers)

The system has been configured for fully offline, one-click startup. No coding or terminal experience is required.

### Option 1: Desktop Shortcuts (Easiest)
There are two shortcuts located directly on the **Desktop**:
1. **BEL AI Assistant**: Double-click this to open the beautiful Chatbot graphical interface (requires a web browser to be installed on the machine).
2. **BEL AI Assistant (Terminal)**: Double-click this to open the command-line CLI interface. Use this if the machine is "bare metal" and has no web browser installed.

### Option 2: Launcher Scripts (From the Folder)
If the shortcuts are missing, you can run the application directly from the folder:

**Windows:**
1. Double-click `Launch.bat` (Starts the server and opens the Chatbot UI).
2. Double-click `Launch-CLI.bat` (Opens the Terminal UI directly).

**Linux:**
1. Run `./launch.sh` in the terminal (Starts the server and opens the Chatbot UI).
2. Run `./launch-cli.sh` in the terminal (Opens the Terminal UI directly).

> All options automatically activate the isolated Python environment and enforce offline security flags. No manual setup is needed.

---

## Updating the Knowledge Base

When a new version of `IRL Fault Codes.pdf` is available:

**Windows:**
```powershell
# Place the new PDF in the BEL folder
# Then re-run ingestion from an activated environment:
.\.venv\Scripts\python -m app.ingestion.ingest
```

**Linux:**
```bash
# Place the new PDF in the BEL folder
# Then re-run ingestion from an activated environment:
source .venv/bin/activate
python3 -m app.ingestion.ingest
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
├── setup.ps1                ← Windows setup script
├── setup.sh                 ← Linux setup script
├── bootstrap.py             ← Cross-platform setup logic
├── Launch.bat               ← Windows UI launcher
├── Launch-CLI.bat           ← Windows CLI launcher
├── launch.sh                ← Linux UI launcher
├── launch-cli.sh            ← Linux CLI launcher
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
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` prevent any Hugging Face network access or telemetry at runtime
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
