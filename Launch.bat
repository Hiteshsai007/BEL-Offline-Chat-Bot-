@echo off
title BEL AI Assistant
echo Starting BEL Offline AI Interface...

:: Navigate to the folder where this .bat file lives (handles any install location)
cd /d "%~dp0"

:: Create logs folder if it doesn't exist
if not exist "logs" mkdir logs

:: Check if setup has been run; if not, automatically run first-time setup
if not exist ".venv" (
    echo Virtual environment not found. Running first-time setup...
    call First-Time-Setup.bat
    if %errorlevel% neq 0 (
        echo Setup failed. Aborting launch.
        pause
        exit /b 1
    )
)

:: Set offline flags
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Stop any existing server running on port 8000 so new code & config takes effect
powershell -NoProfile -Command "$p = (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess; if ($p) { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }"
timeout /t 1 /nobreak > NUL

:: Start the server in the background
start /B .venv\Scripts\python.exe -m app.main > logs\startup.log 2>&1

:: Poll /health until the server reports ready (up to 60 seconds).
:: A 503 during slow model load is not "ready" -- the loop keeps retrying.
echo Waiting for the AI engine to start...
set ATTEMPTS=0
:HEALTH_LOOP
set /a ATTEMPTS+=1
if %ATTEMPTS% gtr 60 (
    echo WARNING: Server did not become ready within 60 seconds. Opening anyway...
    goto OPEN_BROWSER
)
:: PowerShell checks for HTTP 200 specifically; non-200 (incl. 503) means not ready yet.
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% neq 0 (
    timeout /t 1 /nobreak > NUL
    goto HEALTH_LOOP
)
echo Server is ready!

:OPEN_BROWSER

:: Try to open in Edge App Mode first, fall back to default browser
where msedge >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start msedge --app=http://127.0.0.1:8000
) else (
    start http://127.0.0.1:8000
)

echo Interface launched!
exit
