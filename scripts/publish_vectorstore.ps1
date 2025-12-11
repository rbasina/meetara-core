# PowerShell script to publish vectorstore to Hugging Face Hub
# Usage: .\scripts\publish_vectorstore.ps1 -Domain "general_health" -RepoId "meetara-lab/vectorstore-general_health"
# 
# Token will be automatically used from:
#   1. -Token parameter (if provided)
#   2. HF_TOKEN environment variable
#   3. Hugging Face login cache (~/.huggingface/token from 'huggingface-cli login')

param(
    [Parameter(Mandatory=$true)]
    [string]$Domain,
    
    [Parameter(Mandatory=$true)]
    [string]$RepoId,
    
    [Parameter(Mandatory=$false)]
    [string]$Token,
    
    [Parameter(Mandatory=$false)]
    [switch]$Private,
    
    [Parameter(Mandatory=$false)]
    [string]$CommitMessage
)

# Load .env file if it exists
$envPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)\s*=\s*(.*)\s*$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($key -and $value) {
                Set-Item -Path "env:$key" -Value $value
            }
        }
    }
    Write-Host "✅ Loaded environment variables from .env file" -ForegroundColor Green
}

# Get token from parameter, environment variable, or use cached token from HF login
# Priority: 1) Parameter 2) .env file 3) Environment variable 4) HF cache
if (-not $Token) {
    $Token = $env:HF_TOKEN
}

# If still no token, try to get from HF cache (from huggingface-cli login)
if (-not $Token) {
    $hfTokenPath = "$env:USERPROFILE\.huggingface\token"
    if (Test-Path $hfTokenPath) {
        try {
            $Token = Get-Content $hfTokenPath -Raw | ForEach-Object { $_.Trim() }
            Write-Host "✅ Using token from Hugging Face login cache" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Could not read token from cache" -ForegroundColor Yellow
        }
    }
}

# If no token found, the API will try to use cached token automatically
# (huggingface_hub library handles this)

# Build request body
$body = @{
    domain = $Domain
    repo_id = $RepoId
    private = $Private.IsPresent
} | ConvertTo-Json

if ($CommitMessage) {
    $bodyObj = $body | ConvertFrom-Json
    $bodyObj.commit_message = $CommitMessage
    $body = $bodyObj | ConvertTo-Json
}

# Make API request
try {
    Write-Host "🚀 Publishing domain '$Domain' to Hugging Face Hub..." -ForegroundColor Cyan
    Write-Host "   Repository: $RepoId" -ForegroundColor Gray
    
    # Build headers - only add Authorization if token is available
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    if ($Token) {
        $headers["Authorization"] = "Bearer $Token"
        Write-Host "   Using provided token" -ForegroundColor Gray
    } else {
        Write-Host "   Using token from Hugging Face login (cached)" -ForegroundColor Gray
    }
    
    $response = Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/publish" `
        -Method POST `
        -Headers $headers `
        -Body $body
    
    if ($response.success) {
        Write-Host "✅ Successfully published!" -ForegroundColor Green
        Write-Host "   Repository URL: $($response.repo_url)" -ForegroundColor Cyan
        Write-Host "   Domain: $($response.domain)" -ForegroundColor Gray
        if ($response.stats) {
            Write-Host "   Chunks: $($response.stats.chunk_count)" -ForegroundColor Gray
            Write-Host "   Documents: $($response.stats.document_count)" -ForegroundColor Gray
        }
    } else {
        Write-Host "❌ Failed to publish: $($response.message)" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    if ($_.ErrorDetails.Message) {
        Write-Host "   Details: $($_.ErrorDetails.Message)" -ForegroundColor Yellow
    }
    exit 1
}

