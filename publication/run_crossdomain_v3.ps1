$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root '.venv\Scripts\python.exe'
$statusDir = Join-Path $root 'publication\crossdomain_v3'
New-Item -ItemType Directory -Force -Path $statusDir | Out-Null
Set-Content -LiteralPath (Join-Path $statusDir 'pid.txt') -Value $PID -Encoding ascii

$predecessor = Join-Path $root 'publication\matched_baselines_v3\status.txt'
while ($true) {
    if (Test-Path -LiteralPath $predecessor) {
        $state = (Get-Content -LiteralPath $predecessor -First 1).Trim()
        if ($state -in @('complete', 'failed')) { break }
    }
    Start-Sleep -Seconds 30
}
Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "running`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

function Invoke-Observe {
    param([string]$Name, [int]$Seed, [string]$Architecture, [string]$Dataset, [bool]$Stable)
    $metrics = Join-Path $root "outputs_publication\$Name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$Name"; return }
        throw "Partial generalization run: $Name"
    }
    $init = if ($Stable) { 'random_uniform' } else { 'zero' }
    $lr = if ($Stable) { '0.2' } else { '0.3' }
    $step = if ($Stable) { '0.0392156862745098' } else { '0.03137254901960784' }
    Write-Output "start=$Name time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $Name --architecture $Architecture --dataset $Dataset --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr $lr `
        --fgsm-init $init --fgsm-step-size $step --observe-only `
        --observer-pgd-period 5 --observer-pgd-followup-windows 3 `
        --pgd-monitor-batches 10
    if ($LASTEXITCODE -ne 0) { throw "Observation failed: $Name" }
    Write-Output "finish=$Name time=$((Get-Date).ToString('o'))"
}

function Invoke-Intervention {
    param([string]$Name, [int]$Seed, [string]$Architecture, [string]$Dataset, [double]$Threshold)
    $metrics = Join-Path $root "outputs_publication\$Name\metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        if ($rows.Count -eq 30 -and [int]$rows[-1].epoch -eq 30) { Write-Output "skip=$Name"; return }
        throw "Partial intervention run: $Name"
    }
    Write-Output "start=$Name threshold=$Threshold time=$((Get-Date).ToString('o'))"
    & $python -u train_adaptive_intervention.py `
        --run-name $Name --architecture $Architecture --dataset $Dataset --device cuda `
        --seed $Seed --epochs 30 --batch-size 128 --workers 4 --max-lr 0.3 `
        --fgsm-init zero --fgsm-step-size 0.03137254901960784 `
        --detector-threshold $Threshold --disable-rollback --pgd-monitor-batches 10
    if ($LASTEXITCODE -ne 0) { throw "Intervention failed: $Name" }
    Write-Output "finish=$Name time=$((Get-Date).ToString('o'))"
}

try {
    # New ResNet-18 seeds are held out from the two exploratory runs (101 and 202).
    $resCollapseSeeds = @(303, 404, 505, 606, 707)
    $resStableSeeds = @(17, 23, 42)
    foreach ($seed in $resCollapseSeeds) { Invoke-Observe "v3_resnet18_collapse_seed$seed" $seed 'resnet18' 'cifar10' $false }
    foreach ($seed in $resStableSeeds) { Invoke-Observe "v3_resnet18_stable_seed$seed" $seed 'resnet18' 'cifar10' $true }
    $resCollapse = $resCollapseSeeds | ForEach-Object { "outputs_publication\v3_resnet18_collapse_seed$_\detector_traces.csv" }
    $resStable = $resStableSeeds | ForEach-Object { "outputs_publication\v3_resnet18_stable_seed$_\detector_traces.csv" }
    & $python analyze_crossdomain_detector.py --collapse @resCollapse --stable @resStable `
        --mode eval --threshold 0.16632988750934596 `
        --output-dir publication\forecast_analysis\v3_resnet18_heldout5
    if ($LASTEXITCODE -ne 0) { throw 'ResNet-18 analysis failed' }
    $res = Import-Csv publication\forecast_analysis\v3_resnet18_heldout5\summary.csv
    if ([int]$res.observed_events -ge 4 -and [double]$res.event_recall_any_lead -ge 0.8 -and [double]$res.stable_false_alarms_per_run -le 0.34) {
        foreach ($seed in @(303, 404, 505)) {
            Invoke-Intervention "v3_resnet18_intervention_seed$seed" $seed 'resnet18' 'cifar10' 0.16632988750934596
        }
    }

    # CIFAR-100 domain development: threshold selection uses only seeds 17/23/42.
    $c100Dev = @(17, 23, 42)
    foreach ($seed in $c100Dev) {
        Invoke-Observe "v3_cifar100_collapse_dev_seed$seed" $seed 'preact_resnet18' 'cifar100' $false
        Invoke-Observe "v3_cifar100_stable_dev_seed$seed" $seed 'preact_resnet18' 'cifar100' $true
    }
    $c100DevCollapse = $c100Dev | ForEach-Object { "outputs_publication\v3_cifar100_collapse_dev_seed$_\detector_traces.csv" }
    $c100DevStable = $c100Dev | ForEach-Object { "outputs_publication\v3_cifar100_stable_dev_seed$_\detector_traces.csv" }
    & $python analyze_crossdomain_detector.py --collapse @c100DevCollapse --stable @c100DevStable `
        --mode fit --output-dir publication\forecast_analysis\v3_cifar100_dev3_fit
    if ($LASTEXITCODE -ne 0) { throw 'CIFAR-100 fit failed' }
    $frozen = Get-Content publication\forecast_analysis\v3_cifar100_dev3_fit\frozen_detector.json | ConvertFrom-Json

    # Quick-plan held-out set: three collapse-prone and two random-start runs.
    # Cross-dataset intervention is omitted because ResNet-18 already supplies
    # the transfer-to-intervention experiment; this stage tests detection transfer.
    $c100Holdout = @(101, 202, 303)
    $c100Stable = @(101, 202)
    foreach ($seed in $c100Holdout) { Invoke-Observe "v3_cifar100_collapse_holdout_seed$seed" $seed 'preact_resnet18' 'cifar100' $false }
    foreach ($seed in $c100Stable) { Invoke-Observe "v3_cifar100_stable_holdout_seed$seed" $seed 'preact_resnet18' 'cifar100' $true }
    $c100CollapseFiles = $c100Holdout | ForEach-Object { "outputs_publication\v3_cifar100_collapse_holdout_seed$_\detector_traces.csv" }
    $c100StableFiles = $c100Stable | ForEach-Object { "outputs_publication\v3_cifar100_stable_holdout_seed$_\detector_traces.csv" }
    & $python analyze_crossdomain_detector.py --collapse @c100CollapseFiles --stable @c100StableFiles `
        --mode eval --frozen publication\forecast_analysis\v3_cifar100_dev3_fit\frozen_detector.json `
        --output-dir publication\forecast_analysis\v3_cifar100_quick_heldout3
    if ($LASTEXITCODE -ne 0) { throw 'CIFAR-100 held-out analysis failed' }
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
}
catch {
    Set-Content -LiteralPath (Join-Path $statusDir 'status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))`nerror=$($_.Exception.Message)" -Encoding utf8
    throw
}
