@echo off
REM ===========================================================================
REM  One-click runner for the IndoBERT ABSA pipeline.
REM
REM  Double-click this file, or run it from a terminal:
REM      run_all.bat              full pipeline except retraining
REM      run_all.bat --list       show the stages
REM      run_all.bat --include-train
REM                                  also retrain the 10 folds (hours, ~80 GB)
REM
REM  See RUN_GUIDE.md for what each stage does.
REM ===========================================================================

setlocal
cd /d "%~dp0"

title IndoBERT ABSA Pipeline

REM Keep the console readable regardless of the system code page.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual environment not found at .venv
    echo.
    echo Create it first:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" run_all.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
    echo Pipeline finished successfully.
) else (
    echo Pipeline stopped with errors ^(exit %EXITCODE%^).
    echo Check the output above, and docs\BUG_TRACKER.md.
)

if not defined CI pause
endlocal & exit /b %EXITCODE%