<#!
.SYNOPSIS
Creates a small, safe-to-share supervisor ZIP for the Stage 1-3 prototype.

.DESCRIPTION
The archive intentionally excludes raw data, virtual environments, trained model
files, and licensed SMPL assets.  Upload a trained model separately only when a
recipient needs to run the dashboard.
#>

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$dist = Join-Path $root 'dist'
$stage = Join-Path $dist "supervisor_share_$stamp"
$zip = "$stage.zip"

New-Item -ItemType Directory -Force -Path $stage | Out-Null

$files = @(
    'README.md', 'SUPERVISOR_HANDOFF.md', 'WEBSITE_REVIEW.md', 'research-question-card.md',
    'SMPL_SETUP.md', 'requirements.txt', 'app.py', 'stage3.py',
    'diabetes_risk.py', 'Dockerfile', 'docker-compose.yml',
    'digital_twin_full_pipeline.ipynb'
)
foreach ($file in $files) {
    $source = Join-Path $root $file
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination $stage }
}

foreach ($folder in @('templates', 'static', 'src')) {
    $source = Join-Path $root $folder
    if (Test-Path -LiteralPath $source) {
        $target = Join-Path $stage $folder
        Get-ChildItem -LiteralPath $source -File -Recurse |
            Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' } |
            ForEach-Object {
                $relative = $_.FullName.Substring($source.Length).TrimStart('\\')
                $destination = Join-Path $target $relative
                New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
                Copy-Item -LiteralPath $_.FullName -Destination $destination
            }
    }
}

# Share small visual and metric evidence, but never package .joblib model files.
foreach ($artifactFolder in @('artifacts_notebook')) {
    $source = Join-Path $root $artifactFolder
    if (Test-Path -LiteralPath $source) {
        $target = Join-Path $stage $artifactFolder
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Get-ChildItem -LiteralPath $source -File |
            Where-Object { $_.Extension -in '.json', '.csv', '.png' } |
            Copy-Item -Destination $target
    }
}

Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -Force
Remove-Item -LiteralPath $stage -Recurse -Force
Write-Host "Supervisor package created: $zip"
Write-Host 'Upload the ZIP to your university OneDrive/Google Drive and send the private link to your supervisor.'
