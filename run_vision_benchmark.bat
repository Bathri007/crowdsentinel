@echo off
REM Measure DM-Count accuracy on ShanghaiTech Part A and Part B (CPU, about 15 minutes).
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\vision_benchmark.log
echo started %date% %time% > %LOG%
echo RUNNING > logs\vision_status.txt
echo Measuring crowd-counting accuracy on ShanghaiTech. This takes about 15 minutes on CPU.
echo [1/2] Part A (dense crowds)...
.venv\Scripts\python -u -m vision.evaluate_density --root data\raw\ShanghaiTech\part_A_final --weights vision\weights\model_sh_A.pth >> %LOG% 2>&1
if errorlevel 1 (echo FAILED part A > logs\vision_status.txt & goto end)
echo [2/2] Part B (sparser street scenes)...
.venv\Scripts\python -u -m vision.evaluate_density --root data\raw\ShanghaiTech\part_B_final --weights vision\weights\model_sh_B.pth >> %LOG% 2>&1
if errorlevel 1 (echo FAILED part B > logs\vision_status.txt & goto end)
echo OK > logs\vision_status.txt
echo Done. Results are in experiments\results\density_part_A_final.json and density_part_B_final.json
:end
echo finished %date% %time% >> %LOG%
timeout /t 20
