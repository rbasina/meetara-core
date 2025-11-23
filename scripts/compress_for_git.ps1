#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Compress vectorstore and images directories for git push.

.DESCRIPTION
    Creates compressed zip files of vectorstore and images directories
    that can be pushed to git (using Git LFS for large files).

.PARAMETER CompressionLevel
    Compression level: Fastest, Optimal (default), or NoCompression.

.EXAMPLE
    .\scripts\compress_for_git.ps1
    Creates vectorstore.zip and images.zip in project root
#>

param(
    [ValidateSet("Fastest", "Optimal", "NoCompression")]
    [string]$CompressionLevel = "Optimal"
)

Write-Host "🗜️  Compressing directories for git..." -ForegroundColor Cyan
Write-Host ""

# Check if directories exist
$directories = @("vectorstore", "images")
$missing = @()

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        $missing += $dir
    }
}

if ($missing.Count -gt 0) {
    Write-Host "⚠️  Warning: These directories are missing:" -ForegroundColor Yellow
    foreach ($dir in $missing) {
        Write-Host "   - $dir" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Compress vectorstore
if (Test-Path "vectorstore") {
    Write-Host "📦 Compressing vectorstore..." -ForegroundColor Cyan
    
    $originalSize = (Get-ChildItem vectorstore -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $originalSizeMB = [math]::Round($originalSize / 1MB, 2)
    $originalSizeGB = [math]::Round($originalSize / 1GB, 2)
    
    Write-Host "   Original size: $originalSizeMB MB ($originalSizeGB GB)" -ForegroundColor Gray
    
    $startTime = Get-Date
    try {
        Compress-Archive -Path vectorstore -DestinationPath "vectorstore.zip" -CompressionLevel $CompressionLevel -Force
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        $compressedSize = (Get-Item "vectorstore.zip").Length
        $compressedSizeMB = [math]::Round($compressedSize / 1MB, 2)
        $compressedSizeGB = [math]::Round($compressedSize / 1GB, 2)
        $reduction = [math]::Round((1 - ($compressedSize / $originalSize)) * 100, 1)
        
        Write-Host "   ✅ Compressed: $compressedSizeMB MB ($compressedSizeGB GB)" -ForegroundColor Green
        Write-Host "   📉 Reduction: $reduction%" -ForegroundColor Green
        Write-Host "   ⏱️  Time: $([math]::Round($duration, 1))s" -ForegroundColor Gray
        Write-Host ""
    } catch {
        Write-Host "   ❌ Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  vectorstore directory not found - skipping" -ForegroundColor Yellow
    Write-Host ""
}

# Compress images
if (Test-Path "images") {
    Write-Host "📦 Compressing images..." -ForegroundColor Cyan
    
    $originalSize = (Get-ChildItem images -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    $originalSizeMB = [math]::Round($originalSize / 1MB, 2)
    $originalSizeGB = [math]::Round($originalSize / 1GB, 2)
    
    Write-Host "   Original size: $originalSizeMB MB ($originalSizeGB GB)" -ForegroundColor Gray
    
    $startTime = Get-Date
    try {
        Compress-Archive -Path images -DestinationPath "images.zip" -CompressionLevel $CompressionLevel -Force
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        $compressedSize = (Get-Item "images.zip").Length
        $compressedSizeMB = [math]::Round($compressedSize / 1MB, 2)
        $compressedSizeGB = [math]::Round($compressedSize / 1GB, 2)
        $reduction = [math]::Round((1 - ($compressedSize / $originalSize)) * 100, 1)
        
        Write-Host "   ✅ Compressed: $compressedSizeMB MB ($compressedSizeGB GB)" -ForegroundColor Green
        Write-Host "   📉 Reduction: $reduction%" -ForegroundColor Green
        Write-Host "   ⏱️  Time: $([math]::Round($duration, 1))s" -ForegroundColor Gray
        Write-Host ""
    } catch {
        Write-Host "   ❌ Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  images directory not found - skipping" -ForegroundColor Yellow
    Write-Host ""
}

# Summary
Write-Host "📊 Summary:" -ForegroundColor Cyan
if (Test-Path "vectorstore.zip") {
    $vsSize = [math]::Round((Get-Item "vectorstore.zip").Length / 1GB, 2)
    Write-Host "   ✅ vectorstore.zip: $vsSize GB" -ForegroundColor Green
}
if (Test-Path "images.zip") {
    $imgSize = [math]::Round((Get-Item "images.zip").Length / 1GB, 2)
    Write-Host "   ✅ images.zip: $imgSize GB" -ForegroundColor Green
}

$totalSize = 0
if (Test-Path "vectorstore.zip") { $totalSize += (Get-Item "vectorstore.zip").Length }
if (Test-Path "images.zip") { $totalSize += (Get-Item "images.zip").Length }

if ($totalSize -gt 0) {
    $totalGB = [math]::Round($totalSize / 1GB, 2)
    Write-Host ""
    Write-Host "📦 Total compressed size: $totalGB GB" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Files larger than 100MB require Git LFS!" -ForegroundColor Yellow
    Write-Host "   Setup Git LFS: git lfs install" -ForegroundColor Yellow
    Write-Host "   Track zip files: git lfs track '*.zip'" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "✅ Ready to commit and push!" -ForegroundColor Green
}

