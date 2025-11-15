@echo off
REM ========================================
REM Speed Climbing Performance Analysis
REM Full Analytics Pipeline - 188 Races
REM ========================================
REM
REM This script runs the complete Phase 3 analytics pipeline on all 188 races.
REM Estimated total runtime: ~5 minutes
REM
REM Prerequisites:
REM   - All pose files in data/processed/poses/ (188 races)
REM   - Python environment activated
REM   - All Phase 3 dependencies installed
REM
REM ========================================

echo.
echo ========================================
echo Speed Climbing Analytics Pipeline
echo ========================================
echo.
echo Starting full pipeline for 188 races...
echo Estimated time: ~5 minutes
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please activate your environment.
    pause
    exit /b 1
)

echo [1/6] Calculating metrics for all races...
echo Time estimate: ~30 seconds
python scripts/batch_calculate_metrics.py --resume
if errorlevel 1 (
    echo ERROR: Metrics calculation failed!
    pause
    exit /b 1
)
echo Step 1 complete!
echo.

echo [2/6] Aggregating competition statistics...
echo Time estimate: ~5 seconds
python scripts/aggregate_competition_stats.py
if errorlevel 1 (
    echo ERROR: Aggregation failed!
    pause
    exit /b 1
)
echo Step 2 complete!
echo.

echo [3/6] Comparing all races...
echo Time estimate: ~30 seconds
python scripts/compare_races.py --all --competition all
if errorlevel 1 (
    echo WARNING: Some comparisons may have failed (non-critical)
)
echo Step 3 complete!
echo.

echo [4/6] Generating visualizations...
echo Time estimate: ~20 seconds
python src/visualization/race_plots.py
if errorlevel 1 (
    echo WARNING: Some plots may have failed (non-critical)
)
echo Step 4 complete!
echo.

echo [5/6] Creating interactive dashboard...
echo Time estimate: ~5 seconds
python scripts/generate_html_dashboard.py
if errorlevel 1 (
    echo ERROR: Dashboard generation failed!
    pause
    exit /b 1
)
echo Step 5 complete!
echo.

echo [6/6] Exporting ML-ready datasets...
echo Time estimate: ~10 seconds
python scripts/export_ml_data.py --test-size 0.2 --random-seed 42
if errorlevel 1 (
    echo ERROR: ML export failed!
    pause
    exit /b 1
)
echo Step 6 complete!
echo.

echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo Output locations:
echo   - Metrics: data\processed\metrics\
echo   - Aggregates: data\processed\aggregates\
echo   - Plots: data\processed\plots\
echo   - Dashboard: data\processed\dashboard\index.html
echo   - ML Data: data\processed\ml_ready\
echo.
echo Next steps:
echo   1. Open dashboard: start data\processed\dashboard\index.html
echo   2. Review leaderboard: type data\processed\aggregates\leaderboard_top20.csv
echo   3. Check statistics: type data\processed\aggregates\overall_statistics.json
echo.
echo Press any key to open the dashboard...
pause >nul

REM Open dashboard in default browser
start data\processed\dashboard\index.html

echo.
echo Thank you for using the Speed Climbing Analytics Pipeline!
echo.
pause
