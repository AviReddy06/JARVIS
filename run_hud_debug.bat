@echo off
title JARVIS HUD Service (Debug Console)
cd /d "%~dp0"

echo ========================================================
echo        JARVIS Background HUD Service (Debug Mode)
echo ========================================================
echo.
echo Logs and real-time audio/speech events will show here.
echo The HUD will display on your screen when you speak "Hey Jarvis".
echo Press Ctrl+C or close this window to stop JARVIS.
echo.

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0jarvis_ui.pyw" %*
) else (
    python "%~dp0jarvis_ui.pyw" %*
)

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Process exited with code %errorlevel%.
    pause
)
