$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$trainScript = Join-Path $workspacePath "train_fast_at_publication.py"
$analysisScript = Join-Path $workspacePath "analyze_adaptive_forecast.py"
Set-Location -LiteralPath $workspacePath

foreach ($seed in @(101, 202)) {
    $runName = "resnet18_unprotected_fgsm_holdout_seed$seed"
    $metrics = Join-Path $workspacePath "outputs_publication\$runName\metrics.csv"
    $complete = $false
    if (Test-Path $metrics) {
        $rows = Import-Csv $metrics
        $complete = $rows.Count -gt 0 -and [int]$rows[-1].epoch -ge 30
    }
    if (-not $complete) {
        Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
        & $pythonPath $trainScript `
            --run-name $runName --architecture resnet18 --device cuda `
            --seed $seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.3 `
            --epsilon 0.03137254901960784 --attack-step-size 0.03137254901960784 `
            --attack-init zero --pgd-step-size 0.00784313725490196 --pgd-steps 10 `
            --pgd-monitor-batches 10 --diagnostic-batches 2 --trace-interval 20 `
            --batch-pgd-label-batches 1 --val-size 5000
        if ($LASTEXITCODE -ne 0) { throw "$runName failed with exit code $LASTEXITCODE" }
    }
}

& $pythonPath $analysisScript `
    outputs_publication\resnet18_unprotected_fgsm_holdout_seed101\batch_metrics.csv `
    outputs_publication\resnet18_unprotected_fgsm_holdout_seed202\batch_metrics.csv `
    --mode eval `
    --thresholds publication\forecast_analysis\adaptive_dev3_fit\thresholds.json `
    --output-dir publication\forecast_analysis\adaptive_resnet18_holdout_eval
if ($LASTEXITCODE -ne 0) { throw "ResNet-18 detector evaluation failed" }
Write-Output "resnet18_detector_completed=$((Get-Date).ToString('o'))"
