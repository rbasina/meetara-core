#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Extract compressed vectorstore backup.

.DESCRIPTION
    Extracts a compressed vectorstore backup file and restores it to the vectorstore directory.

.PARAMETER ZipFile
    Path to the compressed vectorstore backup file (.zip).

.PARAMETER Destination
    Destination directory. Defaults to current directory (extracts to ./vectorstore).

.EXAMPLE
    .\extract_vectorstore.ps1 -ZipFile "vectorstore_backup_20251122.zip"
    Extracts to ./vectorstore directory

.EXAMPLE
    .\extract_vectorstore.ps1 -ZipFile "backups/vectorstore_backup_20251122.zip" -Destination "."
    Extracts to current directory
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ZipFile,
    
    [string]$Destination = "."
)

# Check if zip file exists
if (-not (Test-Path $ZipFile)) {
    Write-Host "❌ Error: File not found: $ZipFile" -ForegroundColor Red
    exit 1
}

# Get file info
$fileInfo = Get-Item $ZipFile
$fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
$fileSizeGB = [math]::Round($fileInfo.Length / 1GB, 2)

Write-Host "📦 Extracting vectorstore backup..." -ForegroundColor Cyan
Write-Host "   File: $ZipFile" -ForegroundColor Gray
Write-Host "   Size: $fileSizeMB MB ($fileSizeGB GB)" -ForegroundColor Gray

# Check if vectorstore already exists
if (Test-Path "vectorstore") {
    Write-Host "" -ForegroundColor Yellow
    Write-Host "⚠️  Warning: vectorstore directory already exists!" -ForegroundColor Yellow
    $response = Read-Host "   Do you want to replace it? (yes/no)"
    if ($response -ne "yes") {
        Write-Host "   Extraction cancelled." -ForegroundColor Yellow
        exit 0
    }
    Write-Host "   Removing existing vectorstore..." -ForegroundColor Yellow
    Remove-Item -Path "vectorstore" -Recurse -Force
}

# Extract
$startTime = Get-Date
try {
    Expand-Archive -Path $ZipFile -DestinationPath $Destination -Force
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    # Verify extraction
    if (Test-Path "vectorstore") {
        $extractedSize = (Get-ChildItem vectorstore -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $extractedSizeMB = [math]::Round($extractedSize / 1MB, 2)
        $extractedSizeGB = [math]::Round($extractedSize / 1GB, 2)
        
        Write-Host "" -ForegroundColor Green
        Write-Host "✅ Extraction complete!" -ForegroundColor Green
        Write-Host "   Extracted to: vectorstore/" -ForegroundColor White
        Write-Host "   Extracted size: $extractedSizeMB MB ($extractedSizeGB GB)" -ForegroundColor White
        Write-Host "   Time taken: $([math]::Round($duration, 1)) seconds" -ForegroundColor Gray
        Write-Host ""
        Write-Host "🎉 Vectorstore restored successfully!" -ForegroundColor Green
        Write-Host "   You can now start the Meetara Core application." -ForegroundColor Gray
    } else {
        Write-Host "⚠️  Warning: vectorstore directory not found after extraction." -ForegroundColor Yellow
        Write-Host "   Check the zip file structure." -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error extracting vectorstore: $_" -ForegroundColor Red
    exit 1
}

