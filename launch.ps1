$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$reservoirPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $reservoirPython)) {
    throw 'Create the virtual environment and install requirements.txt as described in README.md.'
}
& $reservoirPython -m streamlit run app.py
