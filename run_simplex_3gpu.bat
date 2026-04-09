@echo off
REM ============================================================================
REM  run_simplex_3gpu.bat
REM  Distributed Stochastic Multi-GPU Hyperparameter Optimization (Algorithm 1)
REM
REM  Usage:
REM    run_simplex_3gpu.bat                         (default: best.pt, 30 iters)
REM    run_simplex_3gpu.bat path\to\model.pt        (custom model path)
REM    run_simplex_3gpu.bat path\to\model.pt 50     (custom model + max iters)
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Configuration ---
set MODEL=%~1
if "%MODEL%"=="" set MODEL=models\best_vgg19.pt

set MAX_ITERS=%~2
if "%MAX_ITERS%"=="" set MAX_ITERS=30

set EPOCHS_PER_ITER=3
set GPU_IDS=0,1,2
set WORK_DIR=simplex_work
set COST_GOAL=1.05

REM --- Check prerequisites ---
if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    echo Usage: run_simplex_3gpu.bat [model_path] [max_iters]
    exit /b 1
)

if not exist "sanity_yolov2_googlenet.py" (
    echo [ERROR] sanity_yolov2_googlenet.py not found in current directory.
    echo         Please run from the project root.
    exit /b 1
)

REM --- Check GPU availability ---
echo ============================================================
echo  Checking GPU availability...
echo ============================================================
python -c "import torch; n=torch.cuda.device_count(); print(f'  Found {n} GPU(s)'); [print(f'    GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(n)]"
if errorlevel 1 (
    echo [ERROR] PyTorch CUDA not available.
    exit /b 1
)

REM --- Launch ---
echo.
echo ============================================================
echo  Launching Distributed Simplex Optimizer
echo    Model       : %MODEL%
echo    Max Iters   : %MAX_ITERS%
echo    Epochs/Iter : %EPOCHS_PER_ITER%
echo    GPUs        : %GPU_IDS%
echo    Cost Goal   : %COST_GOAL%
echo    Work Dir    : %WORK_DIR%
echo ============================================================
echo.

python distributed_simplex_3gpu.py ^
    --model "%MODEL%" ^
    --max-iters %MAX_ITERS% ^
    --epochs-per-iter %EPOCHS_PER_ITER% ^
    --gpu-ids %GPU_IDS% ^
    --work-dir %WORK_DIR% ^
    --cost-goal %COST_GOAL% ^
    --training-data tr1_fix.csv tr2_fix.csv tr3_fix.csv ^
    --validation-data valid1_fix.csv

if errorlevel 1 (
    echo.
    echo [ERROR] Optimization failed. Check logs in %WORK_DIR%\
    exit /b 1
)

echo.
echo ============================================================
echo  Optimization Complete!
echo  Results saved in: %WORK_DIR%\
echo  Best model: %WORK_DIR%\global_best.pt
echo ============================================================

endlocal
