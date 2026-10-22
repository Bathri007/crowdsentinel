@echo off
REM One-time setup: virtual environment + packages
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from python.org and tick "Add python.exe to PATH".
    exit /b 1
  )
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
echo.
echo Setup done. Start the control room with: run_dashboard.bat
