@echo off
title BEL Offline AI - Cross-Platform Bootstrap
color 0B

echo ============================================================
echo   BEL Offline AI Technical Assistant -- Setup (Windows)
echo ============================================================
echo.

:: Find an available Python interpreter; bootstrap will auto-install a compatible one if needed.
set "PYTHON_EXE="
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo ERROR: Python was not found in your PATH.
    echo.
    echo Bootstrap will attempt to install Python automatically if possible.
    echo.
    pause
    exit /b 1
)

:: Run the bootstrap script
%PYTHON_EXE% "%~dp0bootstrap.py" %*
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
