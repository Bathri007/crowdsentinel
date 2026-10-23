@echo off
REM One double-click setup: Python environment, packages, tests, PyTorch, model weights, dataset.
REM Progress is shown here; full output goes to logs\complete_setup.log
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\complete_setup.log
echo started %date% %time% > %LOG%
echo RUNNING > logs\setup_status.txt
echo.
echo CrowdSentinel setup. Keep this window open, it closes by itself when done.
echo.
echo [1/5] Creating the Python environment...
if exist .venv\Scripts\python.exe goto have_venv
where py >nul 2>&1 && py -3 -m venv .venv >> %LOG% 2>&1
if not exist .venv\Scripts\python.exe python -m venv .venv >> %LOG% 2>&1
if not exist .venv\Scripts\python.exe (echo FAILED step 1: Python 3.10+ not found > logs\setup_status.txt & goto fail)
:have_venv
.venv\Scripts\python --version >> %LOG% 2>&1
echo [2/5] Installing packages, 2-3 minutes...
.venv\Scripts\python -m pip install --upgrade pip >> %LOG% 2>&1
.venv\Scripts\python -m pip install -r requirements.txt >> %LOG% 2>&1
if errorlevel 1 (echo FAILED step 2: package install > logs\setup_status.txt & goto fail)
echo [3/5] Running the tests...
.venv\Scripts\python -m pytest -q -p no:cacheprovider >> %LOG% 2>&1
if errorlevel 1 (echo FAILED step 3: tests > logs\setup_status.txt & goto fail)
echo [4/5] Installing PyTorch CPU, about 200 MB...
.venv\Scripts\python -m pip install -r requirements-vision.txt gdown >> %LOG% 2>&1
if errorlevel 1 (echo FAILED step 4: PyTorch install > logs\setup_status.txt & goto fail)
echo [5/5] Downloading DM-Count weights and ShanghaiTech images, about 400 MB...
.venv\Scripts\python -u -m tools.download_vision_data >> %LOG% 2>&1
if errorlevel 1 (echo PARTIAL step 5: some downloads failed, see log > logs\setup_status.txt & goto end)
echo OK > logs\setup_status.txt
echo.
echo Setup finished. Start the dashboard with run_dashboard.bat
goto end
:fail
echo.
type logs\setup_status.txt
echo See logs\complete_setup.log for details.
:end
echo finished %date% %time% >> %LOG%
timeout /t 20
