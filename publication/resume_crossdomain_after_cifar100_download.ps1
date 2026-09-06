$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$archive = Join-Path $root 'data\cifar-100-python.tar.gz'
$logDir = Join-Path $root 'publication\crossdomain_v3'
Set-Content -LiteralPath (Join-Path $logDir 'download_status.txt') -Value "downloading`nstarted=$((Get-Date).ToString('o'))" -Encoding utf8

& curl.exe -L --retry 20 --retry-delay 5 -C - -o $archive 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath (Join-Path $logDir 'download_status.txt') -Value "failed`nfailed=$((Get-Date).ToString('o'))" -Encoding utf8
    throw 'CIFAR-100 download failed'
}
& tar.exe -tzf $archive | Select-Object -First 3 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Set-Content -LiteralPath (Join-Path $logDir 'download_status.txt') -Value "failed_validation`nfailed=$((Get-Date).ToString('o'))" -Encoding utf8
    throw 'CIFAR-100 archive validation failed'
}
Set-Content -LiteralPath (Join-Path $logDir 'download_status.txt') -Value "complete`nfinished=$((Get-Date).ToString('o'))" -Encoding utf8
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'publication\run_crossdomain_v3.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Cross-domain queue failed after download' }
