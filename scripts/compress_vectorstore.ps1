#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Compress vectorstore directory for backup and sharing.

.DESCRIPTION
    Creates a compressed backup of the vectorstore directory with date stamp.
    Useful for sharing with team members or creating backups.

.PARAMETER OutputPath
    Directory where the compressed file will be saved. Defaults to current directory.

.PARAMETER CompressionLevel
    Compression level: Fastest, Optimal (default), or NoCompression.

.EXAMPLE
    .\compress_vectorstore.ps1
    Creates vectorstore_backup_20251122_143022.zip in current directory

.EXAMPLE
    .\compress_vectorstore.ps1 -OutputPath "backups\" -CompressionLevel Optimal
    Creates compressed backup in backups directory
#>

param(
    [string]$OutputPath = ".",
    [ValidateSet("Fastest", "Optimal", "NoCompression")]
    [string]$CompressionLevel = "Optimal"
)

# Check if vectorstore exists
if (-not (Test-Path "vectorstore")) {
    Write-Host "❌ Error: vectorstore directory not found!" -ForegroundColor Red
    Write-Host "   Make sure you're running this from the project root directory." -ForegroundColor Yellow
    exit 1
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    Write-Host "📁 Created output directory: $OutputPath" -ForegroundColor Green
}

# Get original size
$originalSize = (Get-ChildItem vectorstore -Recurse -File | Measure-Object -Property Length -Sum).Sum
$originalSizeMB = [math]::Round($originalSize / 1MB, 2)
$originalSizeGB = [math]::Round($originalSize / 1GB, 2)

Write-Host "🗜️  Compressing vectorstore..." -ForegroundColor Cyan
Write-Host "   Original size: $originalSizeMB MB ($originalSizeGB GB)" -ForegroundColor Gray

# Generate filename with timestamp
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$zipFile = Join-Path $OutputPath "vectorstore_backup_$timestamp.zip"

# Compress
$startTime = Get-Date
try {
    Compress-Archive -Path vectorstore -DestinationPath $zipFile -CompressionLevel $CompressionLevel -Force
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    # Get compressed size
    $compressedSize = (Get-Item $zipFile).Length
    $compressedSizeMB = [math]::Round($compressedSize / 1MB, 2)
    $compressedSizeGB = [math]::Round($compressedSize / 1GB, 2)
    $reduction = [math]::Round((1 - ($compressedSize / $originalSize)) * 100, 1)
    
    Write-Host "" -ForegroundColor Green
    Write-Host "✅ Compression complete!" -ForegroundColor Green
    Write-Host "   Compressed file: $zipFile" -ForegroundColor White
    Write-Host "   Compressed size: $compressedSizeMB MB ($compressedSizeGB GB)" -ForegroundColor White
    Write-Host "   Size reduction: $reduction%" -ForegroundColor White
    Write-Host "   Time taken: $([math]::Round($duration, 1)) seconds" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📤 Ready to share!" -ForegroundColor Cyan
    Write-Host "   Upload to: Google Drive, OneDrive, Dropbox, or network drive" -ForegroundColor Gray
    Write-Host "   Extract with: Expand-Archive -Path `"$zipFile`" -DestinationPath ." -ForegroundColor Gray
    
} catch {
    Write-Host "❌ Error compressing vectorstore: $_" -ForegroundColor Red
    exit 1
}

