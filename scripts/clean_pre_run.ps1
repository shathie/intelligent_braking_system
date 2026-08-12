param(
    [switch]$IncludeModels = $true,
    [switch]$IncludeGeneratedReports = $true,
    [switch]$SkipAutoBackup = $false
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Clear-DirectoryContents {
    param([string]$Path)
    if (Test-Path $Path) {
        Get-ChildItem -Path $Path -Force | ForEach-Object {
            Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            if (Test-Path $_.FullName) {
                Write-Warning "Skipped (locked): $($_.FullName)"
            }
        }
        Write-Output "Cleared: $Path"
    }
}

function Remove-IfExists {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -Path $Path -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path $Path) {
            Write-Warning "Skipped (locked): $Path"
        } else {
            Write-Output "Removed: $Path"
        }
    }
}

Write-Output "Starting pre-run cleanup in: $root"

# Keep backups intact by design.
if (-not (Test-Path "output/backups")) {
    New-Item -ItemType Directory -Path "output/backups" -Force | Out-Null
}

# Safety-first: automatically snapshot previous run artifacts before cleanup.
if (-not $SkipAutoBackup) {
    $ts = Get-Date -Format 'yyyyMMdd_HHmmss'
    $backupRoot = Join-Path "output/backups" $ts
    New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

    $backupItems = @(
        "output/reports",
        "output/metrics",
        "output/models",
        "output/logs",
        "output/plots",
        "output/results"
    )

    foreach ($item in $backupItems) {
        if (Test-Path $item) {
            $name = Split-Path $item -Leaf
            Copy-Item -Path $item -Destination (Join-Path $backupRoot $name) -Recurse -Force
        }
    }

    $manifest = @(
        "backup_timestamp=$ts",
        "workspace=$root",
        "created_at=$((Get-Date).ToString('s'))",
        "auto_backup=true",
        "included_items=$($backupItems -join ', ')"
    )
    $manifest | Set-Content -Path (Join-Path $backupRoot "backup_manifest.txt")

    Write-Output "Auto-backup created: $backupRoot"
}

# Always clean run-generated intermediate directories.
Clear-DirectoryContents "output/logs"
Clear-DirectoryContents "output/metrics"
Clear-DirectoryContents "output/plots"
Clear-DirectoryContents "output/results"

# Remove model artifacts from previous runs (optional, enabled by default).
if ($IncludeModels) {
    Clear-DirectoryContents "output/models"
}

# Remove generated HTML/PNG reports but keep authored report drafts.
if ($IncludeGeneratedReports -and (Test-Path "output/reports")) {
    $generatedPatterns = @(
        "output/reports/*.html",
        "output/reports/*.png"
    )

    foreach ($pattern in $generatedPatterns) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -Path $_.FullName -Force
            Write-Output "Removed: $($_.FullName)"
        }
    }
}

# Remove Python cache folders from source tree only (exclude venv and backups).
$venvPath = (Resolve-Path ".venv-1" -ErrorAction SilentlyContinue)
$backupPath = (Resolve-Path "output/backups" -ErrorAction SilentlyContinue)

Get-ChildItem -Path . -Directory -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -eq "__pycache__" -and
        (-not $venvPath -or -not $_.FullName.StartsWith($venvPath.Path, [System.StringComparison]::OrdinalIgnoreCase)) -and
        (-not $backupPath -or -not $_.FullName.StartsWith($backupPath.Path, [System.StringComparison]::OrdinalIgnoreCase))
    } |
    ForEach-Object {
        Remove-Item -Path $_.FullName -Recurse -Force
        Write-Output "Removed: $($_.FullName)"
    }

Write-Output "Pre-run cleanup completed. Backups preserved in output/backups/."
