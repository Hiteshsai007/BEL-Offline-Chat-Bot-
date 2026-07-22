@echo off
title BEL AI Assistant
echo Starting BEL Offline AI Interface...

:: Navigate to the folder where this .bat file lives (handles any install location)
cd /d "%~dp0"

:: Create logs folder if it doesn't exist
if not exist "logs" mkdir logs

:: Set offline flags
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Start the server in the background
start /B .venv\Scripts\python.exe -m app.main > logs\startup.log 2>&1

:: Wait for the server to boot up (loading AI models takes a few seconds)
echo Waiting for the AI engine to start...
timeout /t 8 /nobreak > NUL

:: Try to open in Edge App Mode first, fall back to default browser
where msedge >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start msedge --app=http://127.0.0.1:8000
) else (
    start http://127.0.0.1:8000
)

echo Interface launched!
exit
