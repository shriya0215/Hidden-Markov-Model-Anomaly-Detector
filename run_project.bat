@echo off
title Hidden Markov Model - Anomaly Detection System
color 0A
cd /d "%~dp0"

echo ===============================================================================
echo         HIDDEN MARKOV MODEL REAL-TIME ANOMALY DETECTOR
echo ===============================================================================
echo.

:: Set Matplotlib cache dir to prevent permission warnings
set MPLCONFIGDIR=%TEMP%\mpl

:: Detect Python environment
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo [*] Using Virtual Environment Python: .venv
) else (
    set "PYTHON_EXE=python"
    echo [*] Using System Python
)

echo [*] Current Directory: %CD%
echo.
echo -------------------------------------------------------------------------------
echo  Select an option:
echo   [1] Run Full Anomaly Detection Pipeline (Train, Detect & Open Plots)
echo   [2] Run Unit Tests (pytest)
echo   [3] Open Visualizations Folder
echo   [4] Exit
echo -------------------------------------------------------------------------------
echo.

set /p choice="Enter choice [1, 2, 3, or 4] (Default is 1): "
if "%choice%"=="" set choice=1

if "%choice%"=="1" goto RUN_PIPELINE
if "%choice%"=="2" goto RUN_TESTS
if "%choice%"=="3" goto OPEN_PLOTS
if "%choice%"=="4" goto EXIT_SCRIPT

:RUN_PIPELINE
echo.
echo ===============================================================================
echo [*] Executing Main Pipeline (main.py)...
echo ===============================================================================
echo.
"%PYTHON_EXE%" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Error occurred while running the pipeline.
    goto FINISH
)

echo.
echo [*] Opening telemetry detection dashboard...
if exist "visualizations\hmm_anomaly_detection_dashboard.png" (
    start "" "visualizations\hmm_anomaly_detection_dashboard.png"
)
goto FINISH

:RUN_TESTS
echo.
echo ===============================================================================
echo [*] Running Pytest Suite...
echo ===============================================================================
echo.
"%PYTHON_EXE%" -m pytest tests -v
goto FINISH

:OPEN_PLOTS
echo.
echo [*] Opening visualizations folder...
start "" "visualizations"
goto FINISH

:FINISH
echo.
echo ===============================================================================
echo [*] Execution completed.
echo ===============================================================================
echo.
pause
exit /b

:EXIT_SCRIPT
exit /b
