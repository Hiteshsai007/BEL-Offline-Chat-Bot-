@echo off
echo Starting BEL Offline AI Interface...

:: Set offline flags
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

:: Start the server in the background
start /B .venv\Scripts\python.exe -m app.main > logs\startup.log 2>&1

:: Wait a few seconds for the server to boot up
timeout /t 5 /nobreak > NUL

:: Launch using Edge in App Mode (looks like a native desktop app, no browser UI)
start msedge --app=http://127.0.0.1:8000

echo Interface launched! 
exit
