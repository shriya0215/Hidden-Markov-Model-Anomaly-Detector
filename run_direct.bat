@echo off
title Hidden Markov Model - Anomaly Detection Pipeline
color 0B
cd /d "%~dp0"

echo ===============================================================================
echo          STARTING HMM ANOMALY DETECTION PIPELINE (ONE-CLICK)
echo ===============================================================================
echo.

set MPLCONFIGDIR=%TEMP%\mpl

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [*] Running main.py... Please wait...
echo.

"%PYTHON_EXE%" main.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [*] Pipeline finished successfully!
    echo [*] Opening the Detection Dashboard visualization...
    if exist "visualizations\hmm_anomaly_detection_dashboard.png" (
        start "" "visualizations\hmm_anomaly_detection_dashboard.png"
    )
) else (
    echo.
    echo [!] An error occurred during execution.
)

echo.
echo ===============================================================================
pause
