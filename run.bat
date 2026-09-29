@echo off
title JARVIS Voice Assistant (Terminal)
cd /d "%~dp0"

echo ========================================================
echo         Starting JARVIS AI Voice Assistant
echo ========================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

:: Launch JARVIS Terminal Voice Assistant
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" app.py %*
) else (
    python app.py %*
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] App closed with error code %errorlevel%.
    pause
)
