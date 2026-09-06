$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\supplemental_v2'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
Set-Content -LiteralPath (Join-Path $statusDir 'pid.txt') -Value $PID -Encoding ascii
$started = Get-Date -Format o
Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$started" -Encoding utf8

function Invoke-Run {
    param(
        [string]$Name,
        [int]$Seed,
        [string[]]$ExtraArgs
    )
    $metrics = Join-Path $root "outputs_publication\$Name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $count = (Import-Csv -LiteralPath $metrics).Count
        if ($count -eq 30) {
            Write-Output "skip_complete=$Name"
            return
        }
        throw "Partial run requires manual review: $Name ($count rows)"
    }
    Write-Output "start=$Name seed=$Seed time=$((Get-Date).ToString('o'))"
    $baseArgs = @(
        'train_adaptive_intervention.py',
        '--run-name', $Name,
        '--device', 'cuda',
        '--seed', "$Seed",
        '--epochs', '30',
        '--batch-size', '128',
        '--workers', '4',
        '--pgd-monitor-batches', '10'
    )
    & $python -u @baseArgs @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $Name" }
    Write-Output "finish=$Name time=$((Get-Date).ToString('o'))"
}

try {
    foreach ($seed in @(303, 404, 505, 606, 707)) {
        Invoke-Run -Name "v2_collapse_observe_seed$seed" -Seed $seed -ExtraArgs @(
            '--observe-only', '--fgsm-init', 'zero', '--max-lr', '0.3',
            '--fgsm-step-size', '0.03137254901960784',
            '--observer-pgd-period', '5', '--observer-pgd-followup-windows', '3'
        )
    }

    foreach ($seed in @(23, 42)) {
        Invoke-Run -Name "v2_stable_fgsm_rs_observe_seed$seed" -Seed $seed -ExtraArgs @(
            '--observe-only', '--fgsm-init', 'random_uniform', '--max-lr', '0.2',
            '--fgsm-step-size', '0.0392156862745098'
        )
    }

    foreach ($seed in @(17, 23, 42, 101, 202)) {
        Invoke-Run -Name "v2_adaptive_no_rollback_seed$seed" -Seed $seed -ExtraArgs @(
            '--disable-rollback', '--fgsm-init', 'zero', '--max-lr', '0.3',
            '--fgsm-step-size', '0.03137254901960784'
        )
    }

    foreach ($seed in @(17, 101, 202)) {
        Invoke-Run -Name "v2_fixed_early13_seed$seed" -Seed $seed -ExtraArgs @(
            '--fixed-trigger-epoch', '13', '--fgsm-init', 'zero', '--max-lr', '0.3'
        )
        Invoke-Run -Name "v2_fixed_late21_seed$seed" -Seed $seed -ExtraArgs @(
            '--fixed-trigger-epoch', '21', '--fgsm-init', 'zero', '--max-lr', '0.3'
        )
        Invoke-Run -Name "v2_full_pgd2_seed$seed" -Seed $seed -ExtraArgs @(
            '--start-in-rescue-mode', '--max-lr', '0.3'
        )
    }

    $checkpoint = 'outputs_publication\v2_adaptive_no_rollback_seed101\best_robust.pt'
    $checkpointDir = Split-Path -Parent (Join-Path $root $checkpoint)
    if (-not (Test-Path (Join-Path $checkpointDir 'attack_results.csv'))) {
        & $python -u eval_attacks.py --checkpoint $checkpoint --batch-size 128 --workers 4
        if ($LASTEXITCODE -ne 0) { throw 'Basic final evaluation failed' }
    }
    if (-not (Test-Path (Join-Path $checkpointDir 'strong_attack_results.csv'))) {
        & $python -u eval_strong_attacks.py --checkpoint $checkpoint --batch-size 128 --workers 4
        if ($LASTEXITCODE -ne 0) { throw 'Strong final evaluation failed' }
    }
    if (-not (Test-Path (Join-Path $checkpointDir 'official_autoattack_1000_results.csv'))) {
        & $python -u eval_official_autoattack.py `
            --checkpoint $checkpoint --batch-size 128 --loader-batch-size 256 --workers 4 `
            --max-samples 1000 --output-name official_autoattack_1000_results.csv `
            --state-name official_autoattack_1000_state.json --log-name official_autoattack_1000.log
        if ($LASTEXITCODE -ne 0) { throw 'AutoAttack final evaluation failed' }
    }

    & $python -u publication\summarize_supplemental_v2.py
    if ($LASTEXITCODE -ne 0) { throw 'Supplemental summary failed' }
    $finished = Get-Date -Format o
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nstarted=$started`nfinished=$finished" -Encoding utf8
}
catch {
    $failed = Get-Date -Format o
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nstarted=$started`nfailed=$failed`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
