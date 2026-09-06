$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$checkpoints = @(
    'outputs_publication\v3_fastadvplus_seed101\best_robust.pt',
    'outputs_publication\v3_gradalign_seed101\best_robust.pt'
)

foreach ($checkpoint in $checkpoints) {
    $folder = Split-Path -Parent (Join-Path $root $checkpoint)
    $result = Join-Path $folder 'official_autoattack_1000_results.csv'
    if (Test-Path -LiteralPath $result) {
        Write-Output "skip completed $checkpoint"
        continue
    }
    & $python -u eval_official_autoattack.py `
        --checkpoint $checkpoint `
        --batch-size 128 `
        --loader-batch-size 256 `
        --workers 4 `
        --seed 17 `
        --epsilon 0.03137254901960784 `
        --version standard `
        --max-samples 1000 `
        --output-name official_autoattack_1000_results.csv `
        --state-name official_autoattack_1000_state.json `
        --log-name official_autoattack_1000.log
    if ($LASTEXITCODE -ne 0) {
        throw "AutoAttack failed for $checkpoint"
    }
}
