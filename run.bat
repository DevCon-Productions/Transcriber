@echo off
REM Launch the transcriber using the project's 3.13 venv in isolated mode (-E).
REM -E is required: a global PYTHONPATH points at the 3.14 site-packages and
REM would otherwise corrupt this 3.13 environment.
cd /d "%~dp0"
".venv\Scripts\python.exe" -E transcriber.py %*
