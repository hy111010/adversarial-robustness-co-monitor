param(
    [Parameter(Mandatory = $true)]
    [int]$Seed
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$trainingScriptPath = Join-Path $workspacePath "train_fast_at_publication.py"
$analysisScriptPath = Join-Path $workspacePath "analyze_collapse_forecast.py"
$runName = "dev_collapse_full_zero_eps8_lr03_seed$Seed"
$batchMetricsPath = Join-Path $workspacePath "outputs_publication\$runName\batch_metrics.csv"
$analysisOutputPath = Join-Path $workspacePath "publication\forecast_analysis\collapse_full_zero_eps8_lr03_seed$Seed"

Set-Location -LiteralPath $workspacePath
Write-Output "starting_run=$runName"
Write-Output "started_at=$((Get-Date).ToString('o'))"

& $pythonPath $trainingScriptPath `
    --run-name $runName `
    --architecture preact_resnet18 `
    --device cuda `
    --seed $Seed `
    --epochs 30 `
    --batch-size 128 `
    --workers 4 `
    --max-lr 0.3 `
    --momentum 0.9 `
    --weight-decay 0.0005 `
    --epsilon 0.03137254901960784 `
    --attack-step-size 0.03137254901960784 `
    --attack-init zero `
    --pgd-step-size 0.00784313725490196 `
    --pgd-steps 10 `
    --pgd-monitor-batches 10 `
    --diagnostic-batches 2 `
    --trace-interval 20 `
    --batch-pgd-label-batches 1 `
    --val-size 5000

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $batchMetricsPath)) {
    throw "Expected batch trace does not exist: $batchMetricsPath"
}

Write-Output "starting_analysis=$analysisOutputPath"
& $pythonPath $analysisScriptPath $batchMetricsPath --mode describe --output-dir $analysisOutputPath
if ($LASTEXITCODE -ne 0) {
    throw "Forecast analysis failed with exit code $LASTEXITCODE"
}

Write-Output "completed_at=$((Get-Date).ToString('o'))"
