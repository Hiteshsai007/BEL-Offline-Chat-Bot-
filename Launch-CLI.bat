@echo off
title BEL AI Assistant (Terminal)
echo Starting BEL Offline AI CLI Interface...

:: Navigate to the folder where this .bat file lives (handles any install location)
cd /d "%~dp0"

:: Set offline flags
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Run the CLI directly in this window using the virtual environment
.venv\Scripts\python.exe -m app.cli

pause
