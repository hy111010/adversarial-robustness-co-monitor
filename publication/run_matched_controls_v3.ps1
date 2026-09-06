$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$status = Join-Path $root 'publication\matched_controls_v3'
New-Item -ItemType Directory -Force -Path $status | Out-Null
Set-Content -LiteralPath (Join-Path $status 'pid.txt') -Value $PID -Encoding ascii
Set-Content -LiteralPath (Join-Path $status 'status.txt') -Value "running`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

function Invoke-Control {
    param([string]$Name, [int]$Seed, [string[]]$ExtraArgs)
    $metrics = Join-Path $root "outputs_publication\$Name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) {
            Write-Output "skip_complete=$Name"
            return
        }
        throw "Partial run requires review: $Name"
    }
    Write-Output "start=$Name time=$((Get-Date).ToString('o'))"
    $base = @(
        'train_adaptive_intervention.py', '--run-name', $Name, '--device', 'cuda',
        '--seed', "$Seed", '--epochs', '30', '--batch-size', '128', '--workers', '4',
        '--pgd-monitor-batches', '10'
    )
    & $python -u @base @ExtraArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $Name" }
    Write-Output "finish=$Name time=$((Get-Date).ToString('o'))"
}

try {
    foreach ($seed in @(23, 42)) {
        Invoke-Control "v2_fixed_early13_seed$seed" $seed @(
            '--fixed-trigger-epoch', '13', '--fgsm-init', 'zero', '--max-lr', '0.3'
        )
        Invoke-Control "v2_fixed_late21_seed$seed" $seed @(
            '--fixed-trigger-epoch', '21', '--fgsm-init', 'zero', '--max-lr', '0.3'
        )
        Invoke-Control "v2_full_pgd2_seed$seed" $seed @(
            '--start-in-rescue-mode', '--max-lr', '0.3'
        )
    }
    Set-Content -LiteralPath (Join-Path $status 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $status 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
