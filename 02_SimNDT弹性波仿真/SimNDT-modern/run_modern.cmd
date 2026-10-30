@echo off
setlocal
set "SIMNDT_MODERN_ROOT=D:\SimNDTModern"
cd /d "%SIMNDT_MODERN_ROOT%"
set "PYTHONPATH=%SIMNDT_MODERN_ROOT%\src"
set "CUPY_CACHE_DIR=%SIMNDT_MODERN_ROOT%\.cupy_cache"
start "SimNDT Modern" /D "%SIMNDT_MODERN_ROOT%" "%SIMNDT_MODERN_ROOT%\.venv\Scripts\pythonw.exe" -m simndt_modern.app
