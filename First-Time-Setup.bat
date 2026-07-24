@echo off
title BEL Offline AI - Cross-Platform Bootstrap
color 0B

echo ============================================================
echo   BEL Offline AI Technical Assistant -- Setup (Windows)
echo ============================================================
echo.

:: Try to find Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python was not found in your PATH.
    echo.
    echo Please install Python 3.11 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Run the bootstrap script
python "%~dp0bootstrap.py" %*
set "exit_code=%errorlevel%"

if %exit_code% equ 0 (
    echo.
    echo ============================================================
    echo   SETUP COMPLETE!
    echo.
    echo   You can now disconnect from the internet forever.
    echo.
    echo   Launch options:
    echo     Launch.bat          (Web UI -- opens browser)
    echo     Launch-CLI.bat      (Terminal interface)
    echo ============================================================
    echo.
    pause
) else (
    echo.
    echo Setup failed! Please review the error messages above.
    echo.
    pause
)

exit /b %exit_code%
