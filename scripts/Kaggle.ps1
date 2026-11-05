# Pass arguments directly; no shell interpolation of credentials or commands.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Run .\scripts\Setup.ps1 first.'
}
& $pythonExe (Join-Path $PSScriptRoot 'kaggle_cloud.py') @args
exit $LASTEXITCODE

