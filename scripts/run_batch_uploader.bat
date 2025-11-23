@echo off
REM Standalone batch uploader script that can run outside Cursor
REM This prevents crashes from interrupting the upload process

echo ========================================
echo Batch Uploader - Standalone Mode
echo ========================================
echo.
echo This script runs independently of Cursor IDE
echo You can close Cursor without interrupting the upload
echo.
echo Press Ctrl+C to pause and save progress
echo The upload will resume from checkpoint if interrupted
echo.

cd /d "%~dp0\.."

REM Activate virtual environment if it exists
if exist ".venv-meetara\Scripts\activate.bat" (
    call .venv-meetara\Scripts\activate.bat
)

python scripts\batch_uploader.py --domain academic_tutoring

pause

