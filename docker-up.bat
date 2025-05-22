@echo off
echo Starting Docker environment for AI Analyst...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0docker-up.ps1"
pause
