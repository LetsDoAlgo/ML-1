<#
.SYNOPSIS
	Launch the LogiBridge Ansible Deployment UI (Streamlit dashboard).

.DESCRIPTION
	- Uses the workspace .venv312 Python.
	- Installs Streamlit into .venv312 if not present (~15 s first time).
	- Starts the Streamlit server and opens your default browser.
	- The dashboard shows the fleet deploying live, with idempotency proof.

.EXAMPLE
	PS> .\Launch-AnsibleUI.ps1
#>

[CmdletBinding()]
param(
	[int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$scriptDir = $PSScriptRoot
$repoRoot  = Split-Path $scriptDir -Parent
$wsRoot    = Split-Path $repoRoot  -Parent
$venvPy    = Join-Path $wsRoot '.venv312\Scripts\python.exe'
$uiScript  = Join-Path $scriptDir 'ansible_ui.py'

Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host '  LogiBridge - Ansible Deployment UI (Streamlit)' -ForegroundColor Cyan
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host ''

if (-not (Test-Path $venvPy)) {
	throw "Cannot find $venvPy. Activate/rebuild your .venv312 first."
}
if (-not (Test-Path $uiScript)) {
	throw "Cannot find $uiScript."
}

Write-Host "[1/3] Checking Streamlit in .venv312 ..." -ForegroundColor Yellow
$check = & $venvPy -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)"
if ($LASTEXITCODE -ne 0) {
	Write-Host "     Streamlit not installed. Installing (one-time, ~15 s) ..." -ForegroundColor Yellow
	& $venvPy -m pip install --quiet --disable-pip-version-check streamlit
	if ($LASTEXITCODE -ne 0) { throw "pip install streamlit failed." }
	Write-Host "     Streamlit installed." -ForegroundColor Green
} else {
	Write-Host "     Streamlit already installed." -ForegroundColor Green
}

Write-Host ''
Write-Host "[2/3] Starting Streamlit server on http://localhost:$Port ..." -ForegroundColor Yellow
Write-Host "     Press Ctrl+C in this window to stop." -ForegroundColor DarkGray
Write-Host ''

$env:STREAMLIT_SERVER_HEADLESS   = 'false'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'

Write-Host "[3/3] Opening browser ..." -ForegroundColor Yellow
Write-Host ''

& $venvPy -m streamlit run $uiScript `
	--server.port $Port `
	--server.headless false `
	--browser.gatherUsageStats false `
	--theme.base dark
