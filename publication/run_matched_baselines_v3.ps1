$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\matched_baselines_v3'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
Set-Content -LiteralPath (Join-Path $statusDir 'pid.txt') -Value $PID -Encoding ascii

$controlStatus = Join-Path $root 'publication\matched_controls_v3\status.txt'
while ($true) {
    if (Test-Path -LiteralPath $controlStatus) {
        $state = (Get-Content -LiteralPath $controlStatus -First 1).Trim()
        if ($state -eq 'complete') { break }
        if ($state -eq 'failed' -or $state -eq 'paused_by_user') {
            throw "Control queue did not complete: $state"
        }
    }
    Start-Sleep -Seconds 30
}

Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

function Invoke-Baseline {
    param([string]$Method, [int]$Seed)
    $name = "v3_${Method}_seed$Seed"
    $runDir = Join-Path $root "outputs_publication\$name"
    $metrics = Join-Path $runDir 'metrics.csv'
    $extra = @()
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) {
            Write-Output "skip_complete=$name"
            return
        }
        if (Test-Path -LiteralPath (Join-Path $runDir 'last.pt')) {
            $extra = @('--resume')
        }
        else { throw "Partial baseline has no checkpoint: $name" }
    }
    Write-Output "start=$name time=$((Get-Date).ToString('o'))"
    & $python -u train_matched_baseline.py `
        --method $Method --run-name $name --device cuda --architecture preact_resnet18 `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.3 `
        --epsilon 0.03137254901960784 --fgsm-step-size 0.0392156862745098 `
        --pgd-step-size 0.00784313725490196 --pgd-steps 10 --pgd-monitor-batches 10 `
        @extra
    if ($LASTEXITCODE -ne 0) { throw "Baseline failed: $name" }
    Write-Output "finish=$name time=$((Get-Date).ToString('o'))"
}

try {
    foreach ($method in @('fastadvplus', 'gradalign')) {
        foreach ($seed in @(17, 23, 42, 101, 202)) {
            Invoke-Baseline $method $seed
        }
    }
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
