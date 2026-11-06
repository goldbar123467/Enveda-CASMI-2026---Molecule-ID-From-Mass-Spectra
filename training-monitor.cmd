@echo off
setlocal
pushd "%~dp0"
set "MONITOR_ARGS=%*"
if "%~1"=="" set "MONITOR_ARGS=--web --open"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "scripts\training_monitor.py" %MONITOR_ARGS%
) else if exist ".venv-gpu\Scripts\python.exe" (
    ".venv-gpu\Scripts\python.exe" "scripts\training_monitor.py" %MONITOR_ARGS%
) else (
    python "scripts\training_monitor.py" %MONITOR_ARGS%
)
set "MONITOR_EXIT=%ERRORLEVEL%"
popd
exit /b %MONITOR_EXIT%
