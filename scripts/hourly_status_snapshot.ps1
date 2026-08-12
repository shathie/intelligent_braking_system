param(
    [string]$Root = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = (Split-Path -Parent $PSScriptRoot)
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $Root "output\monitoring"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logsDir = Join-Path $Root "output\logs"

$latestLog = $null
if (Test-Path $logsDir) {
    $latestLog = Get-ChildItem $logsDir -File |
        Where-Object { $_.Name -like "pipeline*.log" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

$logPath = if ($latestLog) { $latestLog.FullName } else { "" }
$logTail = @()
if ($latestLog) {
    $logTail = Get-Content $latestLog.FullName -Tail 40
}

$progressLine = $null
if ($logTail.Count -gt 0) {
    $progressLine = ($logTail | Where-Object { $_ -match "Epoch\s+\d+/\d+" } | Select-Object -Last 1)
}

$epochCurrent = $null
$epochTotal = $null
$iterCurrent = $null
$iterTotal = $null
$secPerIt = $null
$etaHoursCurrentEpoch = $null

if ($progressLine -and $progressLine -match "Epoch\s+(\d+)/(\d+).*?\|\s*(\d+)/(\d+)\s*\[[^\]]*,\s*([0-9.]+)s/it\]") {
    $epochCurrent = [int]$matches[1]
    $epochTotal = [int]$matches[2]
    $iterCurrent = [int]$matches[3]
    $iterTotal = [int]$matches[4]
    $secPerIt = [double]$matches[5]
    if ($iterTotal -gt $iterCurrent) {
        $etaHoursCurrentEpoch = [math]::Round((($iterTotal - $iterCurrent) * $secPerIt) / 3600.0, 2)
    }
}

$gpuLine = ""
try {
    $gpuLine = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | Select-Object -First 1)
} catch {
    $gpuLine = "nvidia-smi unavailable"
}

$pythonTop = ""
try {
    $pythonTop = (Get-Process -Name python -ErrorAction Stop |
        Sort-Object CPU -Descending |
        Select-Object -First 3 Id, CPU, WS, StartTime |
        ConvertTo-Json -Compress)
} catch {
    $pythonTop = "[]"
}

$snapshot = [ordered]@{
    timestamp = $timestamp
    pipeline_log = $logPath
    progress_line = $progressLine
    epoch_current = $epochCurrent
    epoch_total = $epochTotal
    iter_current = $iterCurrent
    iter_total = $iterTotal
    sec_per_iter = $secPerIt
    eta_hours_current_epoch = $etaHoursCurrentEpoch
    gpu = $gpuLine
    python_top = $pythonTop
}

$jsonPath = Join-Path $OutDir "hourly_snapshots.jsonl"
($snapshot | ConvertTo-Json -Compress) | Add-Content -Path $jsonPath -Encoding UTF8

$txtPath = Join-Path $OutDir "latest_hourly_snapshot.txt"
@(
    "Timestamp: $timestamp",
    "PipelineLog: $logPath",
    "Progress: $progressLine",
    "Epoch: $epochCurrent/$epochTotal",
    "Iter: $iterCurrent/$iterTotal",
    "SecPerIter: $secPerIt",
    "ETA(CurrentEpoch,hrs): $etaHoursCurrentEpoch",
    "GPU: $gpuLine",
    "PythonTop: $pythonTop",
    "",
    "--- Log Tail ---"
) + $logTail | Set-Content -Path $txtPath -Encoding UTF8

Write-Host "Snapshot captured at $timestamp"
Write-Host "JSONL: $jsonPath"
Write-Host "TEXT:  $txtPath"
