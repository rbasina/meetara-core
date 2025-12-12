# 🔧 Git Setup Guide

Guide for setting up git with compressed data files using Git LFS.

---

## 📋 Prerequisites

### Install Git LFS

**Windows:**
```powershell
# Download from: https://git-lfs.github.com/
# Or use Chocolatey:
choco install git-lfs

# Or use winget:
winget install GitHub.GitLFS
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install git-lfs

# Fedora
sudo dnf install git-lfs

# macOS
brew install git-lfs
```

**Verify Installation:**
```bash
git lfs version
```

---

## 🚀 Initial Setup

### Step 1: Initialize Git LFS

```bash
# Initialize Git LFS in your repository
git lfs install
```

### Step 2: Track Zip Files

```bash
# Track all zip files via Git LFS
git lfs track "*.zip"

# Or specifically track our compressed files
git lfs track "vectorstore.zip"
git lfs track "images.zip"
```

This creates/updates `.gitattributes` file.

### Step 3: Compress Directories

```powershell
# Compress vectorstore and images for git
.\scripts\compress_for_git.ps1
```

This creates:
- `vectorstore.zip` (~1.1-1.4 GB)
- `images.zip` (~2.5-3.0 GB)

### Step 4: Add and Commit

```bash
# Add .gitattributes (if new)
git add .gitattributes

# Add compressed files (Git LFS will handle them)
git add vectorstore.zip images.zip

# Commit
git commit -m "Add compressed vectorstore and images via Git LFS"

# Push (Git LFS files will be uploaded separately)
git push origin main
```

---

## 📥 After Pulling from Git

### Step 1: Pull Repository

```bash
git pull origin main
```

Git LFS will automatically download the zip files.

### Step 2: Extract Compressed Files

```powershell
# Extract vectorstore.zip and images.zip
.\scripts\extract_after_pull.ps1
```

This extracts:
- `vectorstore.zip` → `vectorstore/` directory
- `images.zip` → `images/` directory

### Step 3: Verify Extraction

```bash
# Check if directories exist
ls vectorstore/
ls images/
```

---

## 🔄 Workflow

### When Adding New Documents

1. **Upload documents** via API or scripts
2. **Compress updated directories**:
   ```powershell
   .\scripts\compress_for_git.ps1
   ```
3. **Commit and push**:
   ```bash
   git add vectorstore.zip images.zip
   git commit -m "Update vectorstore and images"
   git push origin main
   ```

### When Pulling Updates

1. **Pull repository**:
   ```bash
   git pull origin main
   ```
2. **Extract compressed files**:
   ```powershell
   .\scripts\extract_after_pull.ps1
   ```

---

## ⚠️ Important Notes

### File Size Limits

- **GitHub**: 100 MB per file (hard limit)
- **Git LFS**: No hard limit, but large files use bandwidth
- **Our files**: 
  - `vectorstore.zip`: ~1.1-1.4 GB (requires Git LFS)
  - `images.zip`: ~2.5-3.0 GB (requires Git LFS)

### Git LFS Storage

- **GitHub**: 1 GB free, then $5/month per 50 GB
- **Self-hosted**: Unlimited (if hosting your own Git server)
- **Alternative**: Use external storage + gitignore (see DATA_MANAGEMENT.md)

### Best Practices

1. ✅ **Use Git LFS** for files >100 MB
2. ✅ **Compress before committing** - Reduces size
3. ✅ **Extract after pulling** - Use provided scripts
4. ✅ **Don't commit uncompressed directories** - They're in `.gitignore`
5. ✅ **Update compressed files** only when directories change significantly

---

## 🛠️ Troubleshooting

### Git LFS Not Working

**Problem**: Zip files not tracked by Git LFS

**Solution**:
```bash
# Reinstall Git LFS hooks
git lfs install

# Verify tracking
git lfs track

# Re-add files
git add .gitattributes
git add vectorstore.zip images.zip
```

### Large File Push Fails

**Problem**: Push fails with "file too large"

**Solution**:
```bash
# Ensure Git LFS is installed
git lfs install

# Verify files are tracked
git lfs ls-files

# If files aren't tracked, add them:
git lfs track "*.zip"
git add .gitattributes
git add vectorstore.zip images.zip
```

### Extraction Fails

**Problem**: Zip files corrupted or incomplete

**Solution**:
```bash
# Re-download via Git LFS
git lfs pull

# Verify file integrity
# Check file sizes match expected values
```

---

## 📊 Size Estimates

### Compressed Sizes

| Directory | Original | Compressed | Reduction |
|-----------|----------|------------|-----------|
| `vectorstore/` | 1.52 GB | ~1.1-1.4 GB | 10-30% |
| `images/` | 3.61 GB | ~2.5-3.0 GB | 15-30% |
| **Total** | **5.13 GB** | **~3.6-4.4 GB** | **~15-30%** |

### Git LFS Storage

- **Free tier**: 1 GB/month
- **Paid**: $5/month per 50 GB
- **Our needs**: ~4 GB (may need paid plan)

---

## 🔄 Alternative: External Storage

If Git LFS storage is a concern, consider:

1. **External Storage** (Recommended for large teams):
   - Store zip files on Google Drive, OneDrive, etc.
   - Share download links in README
   - Extract manually after cloning

2. **Git LFS** (Recommended for small teams):
   - Convenient automatic download
   - Integrated with git workflow
   - May require paid plan for large files

See [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) for external storage options.

---

## 📝 Quick Reference

### Setup (First Time)
```bash
git lfs install
git lfs track "*.zip"
.\scripts\compress_for_git.ps1
git add .gitattributes vectorstore.zip images.zip
git commit -m "Add compressed data files"
git push origin main
```

### After Pull
```bash
git pull origin main
.\scripts\extract_after_pull.ps1
```

### Update After Changes
```bash
.\scripts\compress_for_git.ps1
git add vectorstore.zip images.zip
git commit -m "Update compressed data"
git push origin main
```

---

**Last Updated**: November 2025

