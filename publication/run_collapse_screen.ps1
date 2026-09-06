$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$trainScriptPath = Join-Path $workspacePath "train_fast_at_publication.py"
$analysisScriptPath = Join-Path $workspacePath "analyze_collapse_forecast.py"

Set-Location -LiteralPath $workspacePath

Write-Output "starting=dev_collapse_screen_eps12_seed17"
& $pythonPath $trainScriptPath `
    --run-name dev_collapse_screen_eps12_seed17 `
    --seed 17 `
    --epochs 15 `
    --architecture preact_resnet18 `
    --limit-train 10000 `
    --limit-val 512 `
    --epsilon 0.047058823529411764 `
    --attack-step-size 0.058823529411764705 `
    --pgd-step-size 0.011764705882352941 `
    --pgd-monitor-batches 4 `
    --diagnostic-batches 1 `
    --trace-interval 20 `
    --batch-pgd-label-batches 1
if ($LASTEXITCODE -ne 0) {
    throw "epsilon 12/255 collapse screen failed with exit code $LASTEXITCODE"
}

Write-Output "starting=dev_collapse_screen_eps16_seed17"
& $pythonPath $trainScriptPath `
    --run-name dev_collapse_screen_eps16_seed17 `
    --seed 17 `
    --epochs 15 `
    --architecture preact_resnet18 `
    --limit-train 10000 `
    --limit-val 512 `
    --epsilon 0.06274509803921569 `
    --attack-step-size 0.07843137254901960 `
    --pgd-step-size 0.01568627450980392 `
    --pgd-monitor-batches 4 `
    --diagnostic-batches 1 `
    --trace-interval 20 `
    --batch-pgd-label-batches 1
if ($LASTEXITCODE -ne 0) {
    throw "epsilon 16/255 collapse screen failed with exit code $LASTEXITCODE"
}

$eps12Metrics = Join-Path $workspacePath "outputs_publication\dev_collapse_screen_eps12_seed17\batch_metrics.csv"
$eps16Metrics = Join-Path $workspacePath "outputs_publication\dev_collapse_screen_eps16_seed17\batch_metrics.csv"
$analysisOutput = Join-Path $workspacePath "publication\forecast_analysis\collapse_screen_seed17"
Write-Output "starting=collapse_screen_analysis"
& $pythonPath $analysisScriptPath $eps12Metrics $eps16Metrics `
    --mode describe `
    --output-dir $analysisOutput
if ($LASTEXITCODE -ne 0) {
    throw "collapse screen analysis failed with exit code $LASTEXITCODE"
}
