$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspacePath ".venv\Scripts\python.exe"
$trainingScript = Join-Path $workspacePath "train_adaptive_intervention.py"
$summaryScript = Join-Path $workspacePath "summarize_intervention_controls.py"
$plotScript = Join-Path $workspacePath "plot_intervention_controls.py"
Set-Location -LiteralPath $workspacePath

$common = @(
    "--architecture", "preact_resnet18", "--device", "cuda", "--epochs", "30",
    "--batch-size", "128", "--workers", "4", "--max-lr", "0.3",
    "--epsilon", "0.03137254901960784", "--fgsm-step-size", "0.03137254901960784",
    "--recovery-step-size", "0.01568627450980392", "--recovery-steps", "2",
    "--pgd-step-size", "0.00784313725490196", "--pgd-steps", "10",
    "--pgd-monitor-batches", "10", "--val-size", "5000"
)

$runName = "pgd_monitor_rollback_pgd2_holdout_seed202"
Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
& $pythonPath $trainingScript --run-name $runName --seed 202 @common `
    --fgsm-init zero --trace-interval 20 --detector pgd_validation `
    --pgd-detector-drop 0.10 --detector-pgd-batches 1
if ($LASTEXITCODE -ne 0) { throw "$runName failed" }

foreach ($seed in @(101, 202)) {
    $runName = "full_pgd2_holdout_seed$seed"
    Write-Output "starting=$runName at=$((Get-Date).ToString('o'))"
    & $pythonPath $trainingScript --run-name $runName --seed $seed @common `
        --start-in-rescue-mode
    if ($LASTEXITCODE -ne 0) { throw "$runName failed" }
}

& $pythonPath $summaryScript `
    --adaptive `
        outputs_publication\adaptive_rollback_pgd2_holdout_seed101\metrics_repaired.csv `
        outputs_publication\adaptive_rollback_pgd2_holdout_seed202\metrics_repaired.csv `
    --pgd-monitor `
        outputs_publication\pgd_monitor_rollback_pgd2_holdout_seed101\metrics.csv `
        outputs_publication\pgd_monitor_rollback_pgd2_holdout_seed202\metrics.csv `
    --full-pgd2 `
        outputs_publication\full_pgd2_holdout_seed101\metrics.csv `
        outputs_publication\full_pgd2_holdout_seed202\metrics.csv `
    --unprotected `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed101\metrics.csv `
        outputs_publication\dev_collapse_full_zero_eps8_lr03_seed202\metrics.csv `
    --output publication\intervention_controls_holdout.csv `
    --aggregate-output publication\intervention_controls_holdout_aggregate.csv
if ($LASTEXITCODE -ne 0) { throw "Control summary failed" }

$env:MPLCONFIGDIR = Join-Path $workspacePath ".matplotlib"
& $pythonPath $plotScript `
    publication\intervention_controls_holdout_aggregate.csv `
    paper_assets\publication\intervention_accuracy_compute.png
if ($LASTEXITCODE -ne 0) { throw "Control plot failed" }
Write-Output "controls_completed=$((Get-Date).ToString('o'))"
