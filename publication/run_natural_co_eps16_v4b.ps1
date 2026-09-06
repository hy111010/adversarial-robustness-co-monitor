$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\natural_co_eps16_v4b'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
Set-Content -LiteralPath (Join-Path $statusDir 'pid.txt') -Value $PID -Encoding ascii
Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

function Invoke-Observation {
    param([int]$Seed)
    $name = "v4_natural_eps16_seed$Seed"
    $metrics = Join-Path $root "outputs_publication\$name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$name"; return }
        throw "Partial epsilon-16 observation run: $name"
    }
    Write-Output "start=$name time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $name --architecture preact_resnet18 --dataset cifar10 --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.2 `
        --epsilon 0.06274509803921569 --fgsm-init random_uniform `
        --fgsm-step-size 0.0784313725490196 --observe-only `
        --pgd-step-size 0.01568627450980392 --observer-pgd-period 5 `
        --observer-pgd-followup-windows 3 --pgd-monitor-batches 10
    if ($LASTEXITCODE -ne 0) { throw "Epsilon-16 observation failed: $name" }
    Write-Output "finish=$name time=$((Get-Date).ToString('o'))"
}

function Invoke-Immediate {
    param([int]$Seed)
    $name = "v4_natural_eps16_immediate_seed$Seed"
    $metrics = Join-Path $root "outputs_publication\$name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$name"; return }
        throw "Partial epsilon-16 intervention run: $name"
    }
    Write-Output "start=$name time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $name --architecture preact_resnet18 --dataset cifar10 --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.2 `
        --epsilon 0.06274509803921569 --fgsm-init random_uniform `
        --fgsm-step-size 0.0784313725490196 --recovery-step-size 0.03137254901960784 `
        --pgd-step-size 0.01568627450980392 --detector-threshold 0.16632988750934596 `
        --pgd-monitor-batches 10
    if ($LASTEXITCODE -ne 0) { throw "Epsilon-16 intervention failed: $name" }
    Write-Output "finish=$name time=$((Get-Date).ToString('o'))"
}

try {
    foreach ($seed in @(101, 202, 303)) { Invoke-Observation $seed }
    & $python publication\analyze_natural_co.py
    if ($LASTEXITCODE -ne 0) { throw 'Epsilon-16 natural-CO analysis failed' }
    $events = Import-Csv publication\natural_co_timing_v4\natural_co_observations.csv | Where-Object {
        $_.recipe -eq 'eps16' -and [int]$_.collapse_event -eq 1 -and [int]$_.detected_before_collapse -eq 1
    } | Select-Object -First 3
    foreach ($event in $events) { Invoke-Immediate ([int]$event.seed) }
    & $python publication\analyze_natural_co.py
    if ($LASTEXITCODE -ne 0) { throw 'Final epsilon-16 timing analysis failed' }
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
