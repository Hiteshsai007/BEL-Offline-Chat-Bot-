@echo off
title BEL Offline AI CLI
echo Starting BEL Offline AI CLI Interface...

:: Set offline flags
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Run the CLI directly in this window using the virtual environment
.venv\Scripts\python.exe -m app.cli

pause
