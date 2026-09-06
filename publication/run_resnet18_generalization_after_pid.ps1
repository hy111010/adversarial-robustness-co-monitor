param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$traceScript = Join-Path $workspacePath "train_fast_at_publication.py"
$interventionScript = Join-Path $workspacePath "train_adaptive_intervention.py"
$analysisScript = Join-Path $workspacePath "analyze_adaptive_forecast.py"
$summaryScript = Join-Path $workspacePath "summarize_adaptive_intervention.py"
$plotScript = Join-Path $workspacePath "plot_adaptive_intervention.py"

$prior = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
if ($null -ne $prior) {
    Write-Output "waiting_for_pid=$WaitForPid"
    Wait-Process -Id $WaitForPid
}
Set-Location -LiteralPath $workspacePath

foreach ($seed in @(101, 202)) {
    $runName = "resnet18_unprotected_fgsm_holdout_seed$seed"
    Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
    & $pythonPath $traceScript `
        --run-name $runName --architecture resnet18 --device cuda `
        --seed $seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.3 `
        --epsilon 0.03137254901960784 --attack-step-size 0.03137254901960784 `
        --attack-init zero --pgd-step-size 0.00784313725490196 --pgd-steps 10 `
        --pgd-monitor-batches 10 --diagnostic-batches 2 --trace-interval 20 `
        --batch-pgd-label-batches 1 --val-size 5000
    if ($LASTEXITCODE -ne 0) { throw "$runName failed with exit code $LASTEXITCODE" }
}

& $pythonPath $analysisScript `
    outputs_publication\resnet18_unprotected_fgsm_holdout_seed101\batch_metrics.csv `
    outputs_publication\resnet18_unprotected_fgsm_holdout_seed202\batch_metrics.csv `
    --mode eval `
    --thresholds publication\forecast_analysis\adaptive_dev3_fit\thresholds.json `
    --output-dir publication\forecast_analysis\adaptive_resnet18_holdout_eval
if ($LASTEXITCODE -ne 0) { throw "ResNet-18 detector evaluation failed" }

foreach ($seed in @(101, 202)) {
    $runName = "resnet18_adaptive_rollback_pgd2_holdout_seed$seed"
    Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
    & $pythonPath $interventionScript `
        --run-name $runName --architecture resnet18 --device cuda `
        --seed $seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.3 `
        --epsilon 0.03137254901960784 --fgsm-step-size 0.03137254901960784 `
        --fgsm-init zero --recovery-step-size 0.01568627450980392 --recovery-steps 2 `
        --trace-interval 20 --detector adaptive_cosine --detector-window 5 `
        --detector-min-history 3 --detector-threshold 0.16632988750934596 `
        --pgd-step-size 0.00784313725490196 --pgd-steps 10 `
        --pgd-monitor-batches 10 --val-size 5000
    if ($LASTEXITCODE -ne 0) { throw "$runName failed with exit code $LASTEXITCODE" }
}

& $pythonPath $summaryScript `
    --interventions `
        outputs_publication\resnet18_adaptive_rollback_pgd2_holdout_seed101\metrics.csv `
        outputs_publication\resnet18_adaptive_rollback_pgd2_holdout_seed202\metrics.csv `
    --baselines `
        outputs_publication\resnet18_unprotected_fgsm_holdout_seed101\metrics.csv `
        outputs_publication\resnet18_unprotected_fgsm_holdout_seed202\metrics.csv `
    --output publication\resnet18_adaptive_intervention_summary.csv `
    --aggregate-output publication\resnet18_adaptive_intervention_aggregate.csv
if ($LASTEXITCODE -ne 0) { throw "ResNet-18 intervention summary failed" }

& $pythonPath $plotScript `
    publication\resnet18_adaptive_intervention_summary.csv `
    paper_assets\publication\resnet18_adaptive_intervention.png
if ($LASTEXITCODE -ne 0) { throw "ResNet-18 plot failed" }
Write-Output "resnet18_completed=$((Get-Date).ToString('o'))"
