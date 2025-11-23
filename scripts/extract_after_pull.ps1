#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Extract compressed directories after git pull.

.DESCRIPTION
    Extracts vectorstore.zip and images.zip files after pulling from git.
    Skips extraction if directories already exist (unless -Force is used).

.PARAMETER Force
    Force extraction even if directories already exist.

.EXAMPLE
    .\scripts\extract_after_pull.ps1
    Extracts vectorstore.zip and images.zip if they exist

.EXAMPLE
    .\scripts\extract_after_pull.ps1 -Force
    Force extraction, replacing existing directories
#>

param(
    [switch]$Force
)

Write-Host "📦 Extracting compressed directories..." -ForegroundColor Cyan
Write-Host ""

# Extract vectorstore.zip
if (Test-Path "vectorstore.zip") {
    if (Test-Path "vectorstore" -and -not $Force) {
        Write-Host "⚠️  vectorstore directory already exists" -ForegroundColor Yellow
        Write-Host "   Skipping extraction. Use -Force to replace." -ForegroundColor Gray
    } else {
        Write-Host "📦 Extracting vectorstore.zip..." -ForegroundColor Cyan
        
        $zipSize = [math]::Round((Get-Item "vectorstore.zip").Length / 1GB, 2)
        Write-Host "   Zip size: $zipSize GB" -ForegroundColor Gray
        
        if (Test-Path "vectorstore" -and $Force) {
            Write-Host "   Removing existing vectorstore..." -ForegroundColor Yellow
            Remove-Item -Path "vectorstore" -Recurse -Force
        }
        
        $startTime = Get-Date
        try {
            Expand-Archive -Path "vectorstore.zip" -DestinationPath "." -Force
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            
            if (Test-Path "vectorstore") {
                $extractedSize = (Get-ChildItem vectorstore -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
                $extractedSizeGB = [math]::Round($extractedSize / 1GB, 2)
                
                Write-Host "   ✅ Extracted: $extractedSizeGB GB" -ForegroundColor Green
                Write-Host "   ⏱️  Time: $([math]::Round($duration, 1))s" -ForegroundColor Gray
            }
        } catch {
            Write-Host "   ❌ Error: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
} else {
    Write-Host "ℹ️  vectorstore.zip not found - skipping" -ForegroundColor Gray
    Write-Host ""
}

# Extract images.zip
if (Test-Path "images.zip") {
    if (Test-Path "images" -and -not $Force) {
        Write-Host "⚠️  images directory already exists" -ForegroundColor Yellow
        Write-Host "   Skipping extraction. Use -Force to replace." -ForegroundColor Gray
    } else {
        Write-Host "📦 Extracting images.zip..." -ForegroundColor Cyan
        
        $zipSize = [math]::Round((Get-Item "images.zip").Length / 1GB, 2)
        Write-Host "   Zip size: $zipSize GB" -ForegroundColor Gray
        
        if (Test-Path "images" -and $Force) {
            Write-Host "   Removing existing images..." -ForegroundColor Yellow
            Remove-Item -Path "images" -Recurse -Force
        }
        
        $startTime = Get-Date
        try {
            Expand-Archive -Path "images.zip" -DestinationPath "." -Force
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            
            if (Test-Path "images") {
                $extractedSize = (Get-ChildItem images -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
                $extractedSizeGB = [math]::Round($extractedSize / 1GB, 2)
                
                Write-Host "   ✅ Extracted: $extractedSizeGB GB" -ForegroundColor Green
                Write-Host "   ⏱️  Time: $([math]::Round($duration, 1))s" -ForegroundColor Gray
            }
        } catch {
            Write-Host "   ❌ Error: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
} else {
    Write-Host "ℹ️  images.zip not found - skipping" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "✅ Extraction complete!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Tip: You can delete the .zip files after extraction to save space:" -ForegroundColor Cyan
Write-Host "   Remove-Item vectorstore.zip, images.zip" -ForegroundColor Gray

