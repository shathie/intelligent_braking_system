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
    "CWD=$((Get-Location).Path)"
) | Set-Content -Path $envLog -Encoding UTF8

$pythonExe = Join-Path $root ".venv-1\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

if ($All) {
    $args = @("main.py", "--all")
} elseif ($Step -ne "") {
    $args = @("main.py", "--step", $Step)
} else {
    $args = @("main.py", "--step", "train")
}

Write-Host "Logs:"
Write-Host "  Output: $stdoutLog"
Write-Host "  Transcript: $transcriptLog"
Write-Host "  Env: $envLog"

Start-Transcript -Path $transcriptLog -Force | Out-Null
try {
    & $pythonExe @args 2>&1 | Tee-Object -FilePath $stdoutLog
}
finally {
    Stop-Transcript | Out-Null
}
