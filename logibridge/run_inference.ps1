param(
    [string]$TruckId = "TRUCK_001"
)

$batchFile = Join-Path $PSScriptRoot "run_inference.bat"

if (-not (Test-Path $batchFile)) {
    Write-Error "Missing launcher: $batchFile"
    exit 2
}

Set-Location $PSScriptRoot
& $batchFile demo combined $TruckId
exit $LASTEXITCODE