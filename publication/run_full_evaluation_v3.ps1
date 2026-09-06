$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\full_evaluation_v3'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
Set-Content -LiteralPath (Join-Path $statusDir 'pid.txt') -Value $PID -Encoding ascii

$predecessor = Join-Path $root 'publication\crossdomain_v3\status.txt'
while ($true) {
    if (Test-Path -LiteralPath $predecessor) {
        $state = (Get-Content -LiteralPath $predecessor -First 1).Trim()
        if ($state -in @('complete', 'failed')) { break }
    }
    Start-Sleep -Seconds 30
}
Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

$models = [ordered]@{
    detector = 'outputs_publication\v2_adaptive_no_rollback_seed17\best_robust.pt'
    fast_fgsm_rs = 'outputs_publication\fast_fgsm_rs_preact_seed17\best_robust.pt'
    fastadvplus = 'outputs_publication\v3_fastadvplus_seed17\best_robust.pt'
    gradalign = 'outputs_publication\v3_gradalign_seed17\best_robust.pt'
    pgd10 = 'outputs_publication\pgd10_preact_seed17\best_robust.pt'
}

try {
    foreach ($entry in $models.GetEnumerator()) {
        $checkpoint = Join-Path $root $entry.Value
        if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing frozen checkpoint: $($entry.Key) $checkpoint" }
        $folder = Split-Path -Parent $checkpoint
        if (-not (Test-Path -LiteralPath (Join-Path $folder 'strong_attack_results.csv'))) {
            Write-Output "strong_eval=$($entry.Key) time=$((Get-Date).ToString('o'))"
            & $python -u eval_strong_attacks.py --checkpoint $checkpoint --batch-size 128 --workers 4
            if ($LASTEXITCODE -ne 0) { throw "Strong evaluation failed: $($entry.Key)" }
        }
        $aaOutput = Join-Path $folder 'official_autoattack_10000_results.csv'
        if (-not (Test-Path -LiteralPath $aaOutput)) {
            Write-Output "autoattack_10000=$($entry.Key) time=$((Get-Date).ToString('o'))"
            & $python -u eval_official_autoattack.py `
                --checkpoint $checkpoint --batch-size 128 --loader-batch-size 256 --workers 4 `
                --max-samples 10000 --output-name official_autoattack_10000_results.csv `
                --state-name official_autoattack_10000_state.json --log-name official_autoattack_10000.log
            if ($LASTEXITCODE -ne 0) { throw "Full AutoAttack failed: $($entry.Key)" }
        }
    }
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
