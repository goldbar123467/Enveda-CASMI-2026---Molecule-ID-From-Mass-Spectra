param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    if (-not $PythonPath) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundledPython) { $PythonPath = $bundledPython }
        else { $PythonPath = (Get-Command python -ErrorAction Stop).Source }
    }
    & $PythonPath -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local orchestration environment.' }
}
& $pythonExe -m pip install -r (Join-Path $projectRoot 'requirements-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Could not install Kaggle CLI.' }
& $pythonExe (Join-Path $PSScriptRoot 'kaggle_cloud.py') check
exit $LASTEXITCODE

