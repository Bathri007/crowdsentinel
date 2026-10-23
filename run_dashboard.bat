@echo off
REM Start the CrowdSentinel control room on http://localhost:8050
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  exit /b 1
)
start "" http://localhost:8050
.venv\Scripts\python -m uvicorn backend.main:app --port 8050
