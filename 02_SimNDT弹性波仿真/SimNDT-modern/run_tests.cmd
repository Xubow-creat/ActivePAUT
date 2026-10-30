@echo off
setlocal
set "SIMNDT_MODERN_ROOT=D:\SimNDTModern"
cd /d "%SIMNDT_MODERN_ROOT%"
set "PYTHONPATH=%SIMNDT_MODERN_ROOT%\src"
set "CUPY_CACHE_DIR=%SIMNDT_MODERN_ROOT%\.cupy_cache"
set "QT_QPA_PLATFORM=offscreen"
"%SIMNDT_MODERN_ROOT%\.venv\Scripts\python.exe" -m pytest -v
pause
