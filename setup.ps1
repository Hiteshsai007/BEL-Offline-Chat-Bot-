# BEL Offline AI Interface — Setup Script
# Run once as: .\setup.ps1
# Requires: Python 3.11+, Ollama installed
# All operations are local. No internet access required after first model pull.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  BEL Offline AI Interface — Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Create virtual environment ───────────────────────────────────────
$VenvPath = Join-Path $Root ".venv"
if (-not (Test-Path $VenvPath)) {
    Write-Host "[1/4] Creating virtual environment at .venv ..." -ForegroundColor Yellow
    python -m venv $VenvPath
    Write-Host "      Done." -ForegroundColor Green
} else {
    Write-Host "[1/6] Virtual environment already exists — skipping." -ForegroundColor DarkGray
}

# Activate venv
$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Error "Virtual environment activation script not found. Re-run setup."
}
& $Activate

# ── Step 2: Set offline flags (telemetry prevention) ────────────────────────
Write-Host "[2/6] Setting offline environment flags ..." -ForegroundColor Yellow
$env:TRANSFORMERS_OFFLINE    = "1"
$env:HF_DATASETS_OFFLINE     = "1"
$env:TOKENIZERS_PARALLELISM  = "false"
Write-Host "      TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1" -ForegroundColor Green

# ── Step 3: Install Python dependencies ──────────────────────────────────────
Write-Host "[3/6] Installing Python dependencies ..." -ForegroundColor Yellow
# Temporarily allow online for package install (first-time setup only)
$env:TRANSFORMERS_OFFLINE = "0"
pip install -r requirements.txt --quiet
$env:TRANSFORMERS_OFFLINE = "1"
Write-Host "      Dependencies installed." -ForegroundColor Green

# ── Step 4: Download BGE embedding model ─────────────────────────────────────
Write-Host "[4/6] Downloading BGE embedding model (BAAI/bge-small-en-v1.5) ..." -ForegroundColor Yellow
Write-Host "      This runs once. The model is cached locally for all future offline use." -ForegroundColor DarkGray
$env:TRANSFORMERS_OFFLINE = "0"
python -c "
from sentence_transformers import SentenceTransformer
print('  Downloading bge-small-en-v1.5 ...')
m = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
print('  Model cached at:', m._model_card_data)
print('  Done.')
"
$env:TRANSFORMERS_OFFLINE = "1"
Write-Host "      Embedding model ready." -ForegroundColor Green

# ── Step 5: Check Ollama ──────────────────────────────────────────────────────
Write-Host "[5/6] Checking Ollama ..." -ForegroundColor Yellow
try {
    $ollamaResp = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    $modelNames = $ollamaResp.models | ForEach-Object { $_.name }
    if ($modelNames -like "*qwen2.5:3b*") {
        Write-Host "      Ollama running. qwen2.5:3b is available." -ForegroundColor Green
    } else {
        Write-Host "      Ollama running but qwen2.5:3b not found. Pulling now ..." -ForegroundColor Yellow
        ollama pull qwen2.5:3b
        Write-Host "      Model pulled." -ForegroundColor Green
    }
} catch {
    Write-Host "      WARNING: Cannot reach Ollama at 127.0.0.1:11434." -ForegroundColor Red
    Write-Host "      Make sure Ollama is installed and running: https://ollama.com" -ForegroundColor Red
    Write-Host "      Then run: ollama pull qwen2.5:3b" -ForegroundColor Red
    Write-Host "      Re-run this setup script after starting Ollama." -ForegroundColor Red
}

# ── Step 6: Run ingestion ─────────────────────────────────────────────────────
Write-Host "[6/6] Running knowledge base ingestion ..." -ForegroundColor Yellow
$PdfPath = Join-Path $Root "IRL Fault Codes.pdf"
if (-not (Test-Path $PdfPath)) {
    Write-Host "      WARNING: 'IRL Fault Codes.pdf' not found in $Root" -ForegroundColor Red
    Write-Host "      Place the PDF there and run: python -m app.ingestion.ingest" -ForegroundColor Red
} else {
    python -m app.ingestion.ingest
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Ingestion complete. FAISS index built." -ForegroundColor Green
    } else {
        Write-Host "      Ingestion failed. Check logs/app.log for details." -ForegroundColor Red
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start the server:" -ForegroundColor White
Write-Host "    .venv\Scripts\python.exe -m app.main" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Then open in your browser:" -ForegroundColor White
Write-Host "    http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
