$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$seeds = @(17, 23, 42, 101, 202)

foreach ($seed in $seeds) {
    $runName = "v5_immediate_no_replay_seed$seed"
    $metrics = Join-Path $root "outputs_publication\$runName\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = (Import-Csv -LiteralPath $metrics).Count
        if ($rows -eq 30) {
            Write-Output "skip completed seed=$seed"
            continue
        }
        throw "Incomplete existing run: $runName"
    }
    Write-Output "start seed=$seed time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $runName `
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
        --pgd-monitor-batches 10 `
        --immediate-switch-no-replay
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for seed $seed"
    }
    Write-Output "complete seed=$seed time=$((Get-Date).ToString('o'))"
}
