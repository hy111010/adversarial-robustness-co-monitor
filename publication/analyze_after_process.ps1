param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid,
    [Parameter(Mandatory = $true)]
    [string]$InputCsv,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$analysisScriptPath = Join-Path $workspacePath "analyze_collapse_forecast.py"

$targetProcess = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
if ($null -ne $targetProcess) {
    Write-Output "waiting_for_pid=$WaitForPid"
    Wait-Process -Id $WaitForPid
}

Set-Location -LiteralPath $workspacePath
if (-not (Test-Path -LiteralPath $InputCsv)) {
    throw "Expected batch trace does not exist: $InputCsv"
}
& $pythonPath $analysisScriptPath $InputCsv --mode describe --output-dir $OutputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Forecast analysis failed with exit code $LASTEXITCODE"
}
