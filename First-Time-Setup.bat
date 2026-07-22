@echo off
title BEL Offline AI - First Time Setup
color 0B

echo =======================================================
echo          BEL Offline AI - First Time Setup
echo =======================================================
echo.
echo IMPORTANT: You must be connected to the internet for this step.
echo This script will download the AI models and set up the Python environment.
echo Once this finishes, the system will be 100%% offline forever.
echo.
pause

echo.
echo Running setup... Please wait, this may take a few minutes...
echo.

:: Run the PowerShell setup script and bypass execution policies automatically
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
echo =======================================================
echo SETUP COMPLETE!
echo =======================================================
echo You can now disconnect from the internet forever.
echo Close this window and double-click the "BEL AI Assistant" 
echo shortcut on your Desktop to launch the system!
echo.
pause
