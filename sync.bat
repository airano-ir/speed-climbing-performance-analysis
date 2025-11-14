@echo off
REM ============================================
REM Sync Script for Gitea <-> GitHub
REM Windows Batch File
REM ============================================

setlocal enabledelayedexpansion

echo.
echo ================================================
echo    Speed Climbing Project - Sync Tool
echo ================================================
echo.

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a git repository!
    echo Please run this from the project root directory.
    pause
    exit /b 1
)

REM Parse command line argument
set MODE=%1
if "%MODE%"=="" set MODE=full

echo [INFO] Mode: %MODE%
echo.

REM ============================================
REM PULL from Gitea
REM ============================================

if "%MODE%"=="full" goto :pull
if "%MODE%"=="pull" goto :pull
goto :skip_pull

:pull
echo ----------------------------------------
echo [1/3] Pulling from Gitea (origin)...
echo ----------------------------------------

git fetch origin
if errorlevel 1 (
    echo [ERROR] Failed to fetch from Gitea!
    echo Check your network connection.
    pause
    exit /b 1
)

echo Current branch:
git branch --show-current

git pull origin main
if errorlevel 1 (
    echo [WARNING] Pull failed - you may have local changes.
    echo Run 'git status' to check.
    pause
)

echo [SUCCESS] Pulled from Gitea.
echo.

:skip_pull

REM ============================================
REM PUSH to GitHub
REM ============================================

if "%MODE%"=="full" goto :push
if "%MODE%"=="push" goto :push
goto :skip_push

:push
echo ----------------------------------------
echo [2/3] Pushing to GitHub...
echo ----------------------------------------

git push github main
if errorlevel 1 (
    echo [WARNING] Push to GitHub failed!
    echo You may need to pull first or force push.
    echo Try: git push github main --force-with-lease
    pause
)

echo [SUCCESS] Pushed to GitHub.
echo.

:skip_push

REM ============================================
REM Verify Sync
REM ============================================

if "%MODE%"=="full" goto :verify
if "%MODE%"=="verify" goto :verify
goto :end

:verify
echo ----------------------------------------
echo [3/3] Verifying sync...
echo ----------------------------------------

echo Gitea (origin):
git log origin/main --oneline -1

echo.
echo GitHub:
git log github/main --oneline -1

echo.

REM Compare commits
for /f "tokens=1" %%a in ('git log origin/main --oneline -1') do set GITEA_HASH=%%a
for /f "tokens=1" %%b in ('git log github/main --oneline -1') do set GITHUB_HASH=%%b

if "%GITEA_HASH%"=="%GITHUB_HASH%" (
    echo [SUCCESS] ✓ Gitea and GitHub are in sync!
) else (
    echo [WARNING] ✗ Gitea and GitHub are NOT in sync!
    echo   Gitea:  %GITEA_HASH%
    echo   GitHub: %GITHUB_HASH%
)

echo.

:end
echo ================================================
echo                   DONE
echo ================================================
echo.

if "%MODE%"=="full" (
    echo Summary:
    echo  - Pulled from Gitea
    echo  - Pushed to GitHub
    echo  - Verified sync
)

echo.
echo Usage:
echo   sync.bat         - Full sync (pull + push + verify)
echo   sync.bat pull    - Pull from Gitea only
echo   sync.bat push    - Push to GitHub only
echo   sync.bat verify  - Verify sync status only
echo.

pause
