$ErrorActionPreference = 'Stop'

$PythonExe = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$TrainingScript = Join-Path $ProjectRoot 'train_adaptive_intervention.py'
$OutputRoot = Join-Path $ProjectRoot 'outputs_publication'

function Test-CompletedRun {
    param([string]$RunName)
    $MetricsPath = Join-Path (Join-Path $OutputRoot $RunName) 'metrics.csv'
    if (-not (Test-Path -LiteralPath $MetricsPath)) { return $false }
    $Rows = Import-Csv -LiteralPath $MetricsPath
    if ($Rows.Count -lt 30) { return $false }
    $Last = $Rows[-1]
    $LastCheckpoint = Join-Path (Join-Path $OutputRoot $RunName) 'last.pt'
    return ([int]$Last.epoch -eq 30 -and (Test-Path -LiteralPath $LastCheckpoint))
}

function Invoke-TrainingRun {
    param(
        [string]$RunName,
        [int]$Seed,
        [double]$MaxLr,
        [double]$Epsilon,
        [double]$FgsmStep,
        [ValidateSet('zero', 'random_uniform')][string]$FgsmInit,
        [double]$RecoveryStep,
        [ValidateSet('switch', 'replay', 'next_epoch')][string]$Response
    )

    if (Test-CompletedRun -RunName $RunName) {
        Write-Output "SKIP_COMPLETE $RunName"
        return
    }

    $Arguments = @(
        $TrainingScript,
        '--data-root', (Join-Path $ProjectRoot 'data'),
        '--output-root', $OutputRoot,
        '--run-name', $RunName,
        '--architecture', 'preact_resnet18',
        '--device', 'cuda',
        '--seed', $Seed,
        '--epochs', '30',
        '--batch-size', '128',
        '--workers', '4',
        '--max-lr', $MaxLr,
        '--momentum', '0.9',
        '--weight-decay', '0.0005',
        '--epsilon', $Epsilon,
        '--fgsm-step-size', $FgsmStep,
        '--fgsm-init', $FgsmInit,
        '--recovery-step-size', $RecoveryStep,
        '--recovery-steps', '2',
        '--trace-interval', '20',
        '--detector-window', '5',
        '--detector-min-history', '3',
        '--detector-threshold', '0.16632988750934596',
        '--detector', 'adaptive_cosine',
        '--pgd-detector-drop', '0.1',
        '--detector-pgd-batches', '1',
        '--observer-pgd-period', '0',
        '--observer-pgd-followup-windows', '3',
        '--pgd-step-size', '0.00784313725490196',
        '--pgd-steps', '10',
        '--pgd-monitor-batches', '10',
        '--val-size', '5000'
    )

    if ($Response -eq 'switch') {
        $Arguments += '--immediate-switch-no-replay'
    } elseif ($Response -eq 'next_epoch') {
        $Arguments += '--disable-rollback'
    }

    Write-Output "START $RunName"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $RunName" }
    if (-not (Test-CompletedRun -RunName $RunName)) {
        throw "Training exited without a complete 30-epoch metrics file: $RunName"
    }
    Write-Output "COMPLETE $RunName"
}

# Primary CIFAR-10/PreActResNet-18 held-out intervention completion.
foreach ($Seed in 303, 404, 505) {
    Invoke-TrainingRun -RunName "v6_heldout_immediate_switch_seed$Seed" -Seed $Seed `
        -MaxLr 0.3 -Epsilon (8.0 / 255.0) -FgsmStep (8.0 / 255.0) `
        -FgsmInit zero -RecoveryStep (4.0 / 255.0) -Response switch
}
foreach ($Seed in 303, 404, 505) {
    Invoke-TrainingRun -RunName "v6_heldout_immediate_replay_seed$Seed" -Seed $Seed `
        -MaxLr 0.3 -Epsilon (8.0 / 255.0) -FgsmStep (8.0 / 255.0) `
        -FgsmInit zero -RecoveryStep (4.0 / 255.0) -Response replay
}
foreach ($Seed in 303, 404, 505) {
    Invoke-TrainingRun -RunName "v6_heldout_next_epoch_seed$Seed" -Seed $Seed `
        -MaxLr 0.3 -Epsilon (8.0 / 255.0) -FgsmStep (8.0 / 255.0) `
        -FgsmInit zero -RecoveryStep (4.0 / 255.0) -Response next_epoch
}

# Epsilon=16/255 random-start stress test for the main no-replay response.
foreach ($Seed in 101, 202, 303) {
    Invoke-TrainingRun -RunName "v6_eps16_immediate_switch_seed$Seed" -Seed $Seed `
        -MaxLr 0.2 -Epsilon (16.0 / 255.0) -FgsmStep (20.0 / 255.0) `
        -FgsmInit random_uniform -RecoveryStep (8.0 / 255.0) -Response switch
}

Write-Output 'ALL_MINIMAL_RUNS_COMPLETE'
