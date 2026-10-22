@echo off
REM Start the CrowdSentinel control room on http://localhost:8000
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  exit /b 1
)
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn backend.main:app --port 8000
