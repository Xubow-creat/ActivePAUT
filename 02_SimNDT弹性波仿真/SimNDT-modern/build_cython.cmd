@echo off
setlocal
set "SIMNDT_MODERN_ROOT=D:\SimNDTModern"
cd /d "%SIMNDT_MODERN_ROOT%"
"%SIMNDT_MODERN_ROOT%\.venv\Scripts\python.exe" "%SIMNDT_MODERN_ROOT%\tools\build_cython.py"
if errorlevel 1 pause & exit /b 1
echo Cython native extension built successfully.
pause
