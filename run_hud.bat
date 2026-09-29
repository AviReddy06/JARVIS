@echo off
title Launch JARVIS Background HUD Service
cd /d "%~dp0"

:: Terminate any stale lingering background instances so fresh code is always loaded
taskkill /F /IM pythonw.exe >nul 2>&1 || rem

:: Launch JARVIS silently in background with pythonw (no console window)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0jarvis_ui.pyw" %*
) else (
    start "" pythonw "%~dp0jarvis_ui.pyw" %*
)
exit /b 0
