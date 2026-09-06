param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$fastScriptPath = Join-Path $workspacePath "train_fast_at_publication.py"

$targetProcess = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
if ($null -ne $targetProcess) {
    Write-Output "waiting_for_pid=$WaitForPid"
    Wait-Process -Id $WaitForPid
}

Set-Location -LiteralPath $workspacePath
Write-Output "starting=fast_fgsm_rs_preact_seed17"
& $pythonPath $fastScriptPath `
    --run-name fast_fgsm_rs_preact_seed17 `
    --seed 17 `
    --epochs 30 `
    --architecture preact_resnet18 `
    --trace-interval 20 `
    --batch-pgd-label-batches 1

if ($LASTEXITCODE -ne 0) {
    throw "Fast FGSM-RS trace failed with exit code $LASTEXITCODE"
}

$analysisScriptPath = Join-Path $workspacePath "analyze_collapse_forecast.py"
$batchMetricsPath = Join-Path $workspacePath "outputs_publication\fast_fgsm_rs_preact_seed17\batch_metrics.csv"
$analysisOutputPath = Join-Path $workspacePath "publication\forecast_analysis\seed17_descriptive"
Write-Output "starting=seed17_descriptive_forecast_analysis"
& $pythonPath $analysisScriptPath $batchMetricsPath `
    --mode describe `
    --output-dir $analysisOutputPath

if ($LASTEXITCODE -ne 0) {
    throw "Forecast analysis failed with exit code $LASTEXITCODE"
}
