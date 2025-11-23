#!/usr/bin/env python3
"""
Python Version Checker for Meetara Core

This script checks the current Python version and provides
recommendations for optimal compatibility.
"""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check and report Python version compatibility."""
    print("=" * 60)
    print("🐍 PYTHON VERSION CHECKER")
    print("=" * 60)
    
    version = sys.version_info
    current_version = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Current Python version: {current_version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    
    # Version compatibility check
    if version.major == 3 and version.minor == 12:
        print("\n✅ PERFECT! Python 3.12 detected")
        print("   This is the recommended version for Meetara Core")
        print("   All features will work optimally")
        
    elif version.major == 3 and version.minor >= 10:
        print(f"\n⚠️  COMPATIBLE: Python {current_version}")
        print("   This version will work but Python 3.12 is recommended")
        print("   Consider upgrading for optimal performance")
        
    else:
        print(f"\n❌ INCOMPATIBLE: Python {current_version}")
        print("   Python 3.10+ is required")
        print("   Please upgrade to Python 3.12 for best results")
        return False
    
    return True


def find_python_versions():
    """Find available Python versions on the system."""
    print("\n🔍 Checking for available Python versions...")
    
    possible_versions = [
        "python3.12", "python312", "python-3.12",
        "python3.11", "python311", "python-3.11", 
        "python3.10", "python310", "python-3.10",
        "python3", "python"
    ]
    
    found_versions = []
    
    for version in possible_versions:
        try:
            result = subprocess.run([version, "--version"], 
                                 capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                version_output = result.stdout.strip()
                found_versions.append((version, version_output))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if found_versions:
        print("Available Python versions:")
        for name, version_info in found_versions:
            print(f"   {name}: {version_info}")
    else:
        print("No additional Python versions found")
    
    return found_versions


def get_installation_instructions():
    """Provide installation instructions for Python 3.12."""
    system = platform.system()
    
    print("\n📋 Installation Instructions for Python 3.12:")
    
    if system == "Windows":
        print("Windows:")
        print("1. Download from: https://www.python.org/downloads/")
        print("2. Run installer and check 'Add to PATH'")
        print("3. Verify: python --version")
        
    elif system == "Darwin":  # macOS
        print("macOS:")
        print("1. Using Homebrew: brew install python@3.12")
        print("2. Or download from: https://www.python.org/downloads/")
        print("3. Verify: python3.12 --version")
        
    else:  # Linux
        print("Linux:")
        print("Ubuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install python3.12 python3.12-venv")
        print("")
        print("CentOS/RHEL:")
        print("  sudo yum install python3.12")
        print("")
        print("Or use pyenv:")
        print("  pyenv install 3.12.0")
        print("  pyenv global 3.12.0")


def main():
    """Main function."""
    is_compatible = check_python_version()
    
    if not is_compatible:
        print("\n❌ Python version incompatible!")
        get_installation_instructions()
        sys.exit(1)
    
    found_versions = find_python_versions()
    
    # Check if Python 3.12 is available
    python_312_available = any("3.12" in version_info for _, version_info in found_versions)
    
    if not python_312_available and sys.version_info.minor != 12:
        print("\n💡 Python 3.12 not found on system")
        print("   Current version will work but 3.12 is recommended")
        get_installation_instructions()
    
    print("\n" + "=" * 60)
    if sys.version_info.minor == 12:
        print("🎉 Ready to proceed with Python 3.12!")
    else:
        print("⚠️  Can proceed but Python 3.12 recommended")
    print("=" * 60)


if __name__ == "__main__":
    main() 