$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$trainingScript = Join-Path $workspacePath "train_adaptive_intervention.py"
$summaryScript = Join-Path $workspacePath "summarize_adaptive_intervention.py"
$plotScript = Join-Path $workspacePath "plot_adaptive_intervention.py"

Set-Location -LiteralPath $workspacePath
foreach ($seed in @(17, 23, 42)) {
    $runName = "adaptive_rollback_pgd2_dev_seed$seed"
    Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
    & $pythonPath $trainingScript `
        --run-name $runName `
        --architecture preact_resnet18 `
        --device cuda `
        --seed $seed `
        --epochs 30 `
        --batch-size 128 `
        --workers 4 `
        --max-lr 0.3 `
        --epsilon 0.03137254901960784 `
        --fgsm-step-size 0.03137254901960784 `
        --fgsm-init zero `
        --recovery-step-size 0.01568627450980392 `
        --recovery-steps 2 `
        --trace-interval 20 `
        --detector-window 5 `
        --detector-min-history 3 `
        --detector-threshold 0.16632988750934596 `
        --pgd-step-size 0.00784313725490196 `
        --pgd-steps 10 `
        --pgd-monitor-batches 10 `
        --val-size 5000
    if ($LASTEXITCODE -ne 0) {
        throw "$runName failed with exit code $LASTEXITCODE"
    }
}

& $pythonPath $summaryScript `
    --interventions `
        outputs_publication\adaptive_rollback_pgd2_dev_seed17\metrics.csv `
        outputs_publication\adaptive_rollback_pgd2_dev_seed23\metrics.csv `
        outputs_publication\adaptive_rollback_pgd2_dev_seed42\metrics.csv `
        outputs_publication\adaptive_rollback_pgd2_holdout_seed101\metrics_repaired.csv `
        outputs_publication\adaptive_rollback_pgd2_holdout_seed202\metrics_repaired.csv `
    --baselines `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed17\metrics.csv `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed23\metrics.csv `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed42\metrics.csv `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed101\metrics.csv `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed202\metrics.csv `
    --output publication\adaptive_intervention_five_seed_summary.csv `
    --aggregate-output publication\adaptive_intervention_five_seed_aggregate.csv
if ($LASTEXITCODE -ne 0) {
    throw "Five-seed summary failed with exit code $LASTEXITCODE"
}

& $pythonPath $plotScript `
    publication\adaptive_intervention_five_seed_summary.csv `
    paper_assets\publication\adaptive_intervention_five_seed.png
if ($LASTEXITCODE -ne 0) {
    throw "Plot generation failed with exit code $LASTEXITCODE"
}
Write-Output "queue_completed=$((Get-Date).ToString('o'))"
