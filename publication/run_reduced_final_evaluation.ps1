$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
$checkpoints = @(
    'outputs_publication\adaptive_rollback_pgd2_holdout_seed101\best_robust.pt',
    'outputs_publication\pgd10_preact_seed17\best_robust.pt',
    'outputs_publication\fast_fgsm_rs_preact_seed17\best_robust.pt'
)

$statusDir = Join-Path $root 'publication\reduced_final_evaluation'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
$started = Get-Date -Format o
Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$started" -Encoding utf8

try {
    foreach ($checkpoint in $checkpoints) {
        $modelDir = Split-Path -Parent (Join-Path $root $checkpoint)
        if (-not (Test-Path (Join-Path $modelDir 'attack_results.csv'))) {
            & $python -u eval_attacks.py --checkpoint $checkpoint --batch-size 128 --workers 4
            if ($LASTEXITCODE -ne 0) { throw "Basic evaluation failed: $checkpoint" }
        }

        if (-not (Test-Path (Join-Path $modelDir 'strong_attack_results.csv'))) {
            & $python -u eval_strong_attacks.py --checkpoint $checkpoint --batch-size 128 --workers 4
            if ($LASTEXITCODE -ne 0) { throw "Strong evaluation failed: $checkpoint" }
        }

        if (-not (Test-Path (Join-Path $modelDir 'official_autoattack_1000_results.csv'))) {
            & $python -u eval_official_autoattack.py `
                --checkpoint $checkpoint `
                --batch-size 128 `
                --loader-batch-size 256 `
                --workers 4 `
                --max-samples 1000 `
                --output-name official_autoattack_1000_results.csv `
                --state-name official_autoattack_1000_state.json `
                --log-name official_autoattack_1000.log
            if ($LASTEXITCODE -ne 0) { throw "AutoAttack evaluation failed: $checkpoint" }
        }
    }

    & $python -u publication\summarize_reduced_final_evaluation.py
    if ($LASTEXITCODE -ne 0) { throw 'Summary generation failed' }
    $finished = Get-Date -Format o
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nstarted=$started`nfinished=$finished" -Encoding utf8
}
catch {
    $failed = Get-Date -Format o
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nstarted=$started`nfailed=$failed`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
