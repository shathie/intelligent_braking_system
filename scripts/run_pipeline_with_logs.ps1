param(
    [switch]$All,
    [string]$Step = "",
    [string]$EpochOverride = "",
    [string]$TrainCap = "",
    [string]$ValCap = "",
    [string]$EvalCap = "",
    [string]$SacEpochs = "",
    [string]$SacEpisodesPerEpoch = "",
    [string]$SacMaxSteps = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# ── MANDATORY PRE-RUN BACKUP ─────────────────────────────────────────────────
# Always snapshot previous results before starting a new pipeline run.
# This ensures every iteration is recoverable regardless of outcome.
Write-Host "[PRE-RUN] Taking backup of previous results..."
& "$PSScriptRoot\clean_pre_run.ps1" -ErrorAction Stop
Write-Host "[PRE-RUN] Backup complete. Workspace is clean."
Write-Host ""
# ─────────────────────────────────────────────────────────────────────────────

$logDir = Join-Path $root "output\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$runName = if ($All) { "all" } elseif ($Step -ne "") { $Step } else { "train" }
$stdoutLog = Join-Path $logDir ("pipeline_{0}_{1}.log" -f $runName, $ts)
$transcriptLog = Join-Path $logDir ("transcript_{0}_{1}.log" -f $runName, $ts)
$envLog = Join-Path $logDir ("env_{0}_{1}.txt" -f $runName, $ts)

if ($EpochOverride -ne "") { $env:IBS_EPOCH_OVERRIDE = $EpochOverride }
if ($TrainCap -ne "") { $env:IBS_MAX_TRAIN_SAMPLES = $TrainCap }
if ($ValCap -ne "") { $env:IBS_MAX_VAL_SAMPLES = $ValCap }
if ($EvalCap -ne "") { $env:IBS_MAX_EVAL_SAMPLES = $EvalCap }
if ($SacEpochs -ne "") { $env:IBS_SAC_EPOCHS = $SacEpochs }
if ($SacEpisodesPerEpoch -ne "") { $env:IBS_SAC_EPISODES_PER_EPOCH = $SacEpisodesPerEpoch }
if ($SacMaxSteps -ne "") { $env:IBS_SAC_MAX_STEPS = $SacMaxSteps }

# Default full pipeline to CPU-only to avoid inefficient GPU use.
if (-not $env:IBS_FORCE_CPU) { $env:IBS_FORCE_CPU = "1" }
if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "-1" }

# Keep preprocessing budget aligned with requested run profile.
if ($TrainCap -ne "") {
    if (-not $env:THU_MAX_SAMPLES) { $env:THU_MAX_SAMPLES = $TrainCap }
    if (-not $env:MENDELEY_MAX_SAMPLES) { $env:MENDELEY_MAX_SAMPLES = $TrainCap }
    if (-not $env:DAWN_MAX_SAMPLES) { $env:DAWN_MAX_SAMPLES = $TrainCap }
    if (-not $env:BDD_MAX_SAMPLES) { $env:BDD_MAX_SAMPLES = $TrainCap }
    if (-not $env:KITTI_MAX_SAMPLES) { $env:KITTI_MAX_SAMPLES = $TrainCap }
}

@(
    "Timestamp=$ts",
    "RunName=$runName",
    "IBS_EPOCH_OVERRIDE=$($env:IBS_EPOCH_OVERRIDE)",
    "IBS_MAX_TRAIN_SAMPLES=$($env:IBS_MAX_TRAIN_SAMPLES)",
    "IBS_MAX_VAL_SAMPLES=$($env:IBS_MAX_VAL_SAMPLES)",
    "IBS_MAX_EVAL_SAMPLES=$($env:IBS_MAX_EVAL_SAMPLES)",
    "IBS_SAC_EPOCHS=$($env:IBS_SAC_EPOCHS)",
    "IBS_SAC_EPISODES_PER_EPOCH=$($env:IBS_SAC_EPISODES_PER_EPOCH)",
    "IBS_SAC_MAX_STEPS=$($env:IBS_SAC_MAX_STEPS)",
    "IBS_FORCE_CPU=$($env:IBS_FORCE_CPU)",
    "CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)",
    "THU_MAX_SAMPLES=$($env:THU_MAX_SAMPLES)",
    "MENDELEY_MAX_SAMPLES=$($env:MENDELEY_MAX_SAMPLES)",
    "DAWN_MAX_SAMPLES=$($env:DAWN_MAX_SAMPLES)",
    "BDD_MAX_SAMPLES=$($env:BDD_MAX_SAMPLES)",
    "KITTI_MAX_SAMPLES=$($env:KITTI_MAX_SAMPLES)",
    "CWD=$((Get-Location).Path)"
) | Set-Content -Path $envLog -Encoding UTF8

$pythonExe = Join-Path $root ".venv-1\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

if ($All) {
    $cliArgs = @("main.py", "--all")
} elseif ($Step -ne "") {
    $cliArgs = @("main.py", "--step", $Step)
} else {
    $cliArgs = @("main.py", "--step", "train")
}

Write-Host "Logs:"
Write-Host "  Output: $stdoutLog"
Write-Host "  Transcript: $transcriptLog"
Write-Host "  Env: $envLog"

Start-Transcript -Path $transcriptLog -Force | Out-Null
try {
    $escapedArgs = $cliArgs | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }
    $argString = ($escapedArgs -join ' ')
    $cmdLine = '"' + $pythonExe + '" ' + $argString + ' 2>&1'
    cmd /c $cmdLine | Tee-Object -FilePath $stdoutLog
}
finally {
    Stop-Transcript | Out-Null
}
