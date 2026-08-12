param(
    [string]$EpochOverride = "25",
    [string]$TrainCap = "512",
    [string]$ValCap = "256",
    [string]$EvalCap = "256",
    [string]$SacEpochs = "",
    [string]$SacEpisodesPerEpoch = "",
    [string]$SacMaxSteps = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[1/2] Running pre-run cleanup (auto-backup enabled)..."
& (Join-Path $PSScriptRoot "clean_pre_run.ps1")

Write-Host "[2/2] Running full pipeline with continuous logs..."
$logRunner = Join-Path $PSScriptRoot "run_pipeline_with_logs.ps1"

$params = @{
    All = $true
    EpochOverride = $EpochOverride
    TrainCap = $TrainCap
    ValCap = $ValCap
    EvalCap = $EvalCap
}

if ($SacEpochs -ne "") { $params["SacEpochs"] = $SacEpochs }
if ($SacEpisodesPerEpoch -ne "") { $params["SacEpisodesPerEpoch"] = $SacEpisodesPerEpoch }
if ($SacMaxSteps -ne "") { $params["SacMaxSteps"] = $SacMaxSteps }

& $logRunner @params
