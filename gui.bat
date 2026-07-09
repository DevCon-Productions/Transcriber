@echo off
REM Launch the transcriber GUI using the project's 3.13 venv in isolated mode (-E).
REM -E is required: a global PYTHONPATH points at the 3.14 site-packages and
REM would otherwise corrupt this 3.13 environment. pythonw = no console window.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -E gui.py
