@echo off
title Stop JARVIS Background Service
echo Stopping all JARVIS background processes...
taskkill /F /IM pythonw.exe >nul 2>&1
echo [SUCCESS] JARVIS background service stopped.
timeout /t 2 >nul
exit
