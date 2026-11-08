@echo off
setlocal
pushd "%~dp0"
".venv\Scripts\python.exe" "scripts\supervise_compatible_training.py" %*
set "TRAIN_EXIT=%ERRORLEVEL%"
popd
exit /b %TRAIN_EXIT%
