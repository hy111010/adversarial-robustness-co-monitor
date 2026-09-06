$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\natural_co_timing_v4'
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

function Invoke-Observation {
    param([string]$Recipe, [int]$Seed, [double]$MaxLr)
    $name = "v4_natural_${Recipe}_seed$Seed"
    $metrics = Join-Path $root "outputs_publication\$name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$name"; return }
        throw "Partial natural-CO observation run: $name"
    }
    Write-Output "start=$name time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $name --architecture preact_resnet18 --dataset cifar10 --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr $MaxLr `
        --fgsm-init random_uniform --fgsm-step-size 0.0392156862745098 --observe-only `
        --observer-pgd-period 5 --observer-pgd-followup-windows 3 --pgd-monitor-batches 10
    if ($LASTEXITCODE -ne 0) { throw "Natural-CO observation failed: $name" }
    Write-Output "finish=$name time=$((Get-Date).ToString('o'))"
}

function Invoke-TimingIntervention {
    param([string]$Recipe, [int]$Seed, [double]$MaxLr, [string]$Timing)
    $name = "v4_natural_${Recipe}_${Timing}_seed$Seed"
    $metrics = Join-Path $root "outputs_publication\$name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$name"; return }
        throw "Partial natural-CO intervention run: $name"
    }
    $timingArg = @()
    if ($Timing -eq 'deferred') { $timingArg = @('--disable-rollback') }
    Write-Output "start=$name time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $name --architecture preact_resnet18 --dataset cifar10 --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr $MaxLr `
        --fgsm-init random_uniform --fgsm-step-size 0.0392156862745098 `
        --detector-threshold 0.16632988750934596 --pgd-monitor-batches 10 @timingArg
    if ($LASTEXITCODE -ne 0) { throw "Natural-CO intervention failed: $name" }
    Write-Output "finish=$name time=$((Get-Date).ToString('o'))"
}

try {
    # Existing standard random-start runs (seeds 17/23/42) provide the stable
    # pass-through reference. New held-out runs use random starts and differ
    # only by the reasonable peak cyclic learning rate of 0.3.
    foreach ($seed in @(101, 202, 303, 404, 505)) {
        Invoke-Observation 'highlr' $seed 0.3
    }
    & $python publication\analyze_natural_co.py
    if ($LASTEXITCODE -ne 0) { throw 'Natural-CO analysis failed' }

    $events = Import-Csv publication\natural_co_timing_v4\natural_co_observations.csv | Where-Object {
        [int]$_.collapse_event -eq 1 -and [int]$_.detected_before_collapse -eq 1
    } | Select-Object -First 3
    foreach ($event in $events) {
        $lr = if ($event.recipe -eq 'standard') { 0.2 } else { 0.3 }
        Invoke-TimingIntervention $event.recipe ([int]$event.seed) $lr 'immediate'
    }
    & $python publication\analyze_natural_co.py
    if ($LASTEXITCODE -ne 0) { throw 'Final natural-CO timing analysis failed' }
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
