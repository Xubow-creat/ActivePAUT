@echo off
setlocal
set "SIMNDT_MODERN_ROOT=D:\SimNDTModern"
cd /d "%SIMNDT_MODERN_ROOT%"
set "PYTHONPATH=%SIMNDT_MODERN_ROOT%\src"
set "CUPY_CACHE_DIR=%SIMNDT_MODERN_ROOT%\.cupy_cache"
"%SIMNDT_MODERN_ROOT%\.venv\Scripts\python.exe" "%SIMNDT_MODERN_ROOT%\tools\benchmark.py"
pause
