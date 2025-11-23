#!/usr/bin/env python3
"""
Meetara Core Environment Setup Script

This script automates the setup of the meetara-core environment,
including virtual environment creation, dependency installation,
and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform


class MeetaraSetup:
    """Setup automation for Meetara Core."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.venv_name = "meetara-env"
        self.requirements_file = self.project_root / "requirements.txt"
        self.env_example = self.project_root / "env.example"
        self.env_file = self.project_root / ".env"
        
    def print_banner(self):
        """Print setup banner."""
        print("=" * 60)
        print("🚀 MEETARA CORE SETUP")
        print("=" * 60)
        print("Setting up enhanced RAG assistant with:")
        print("✅ 100+ domains for human assistance")
        print("✅ Telugu, Tamil, Bengali + 12 other languages")
        print("✅ Emotion-aware AI responses")
        print("✅ Offline-first processing")
        print("=" * 60)
    
    def check_python_version(self):
        """Check Python version compatibility."""
        print("🔍 Checking Python version...")
        version = sys.version_info
        
        # Check for Python 3.12 specifically
        if version.major == 3 and version.minor == 12:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Perfect! (3.12 recommended)")
        elif version.major == 3 and version.minor >= 10:
            print(f"⚠️  Python {version.major}.{version.minor}.{version.micro} - Compatible but 3.12 recommended")
            print("   💡 Consider upgrading to Python 3.12 for optimal performance")
        else:
            print("❌ Error: Python 3.10+ required (3.12 recommended)")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            print("   Please install Python 3.12 for best compatibility")
            sys.exit(1)
    
    def create_virtual_environment(self):
        """Create virtual environment."""
        print("\n🔧 Creating virtual environment...")
        venv_path = self.project_root / self.venv_name
        
        if venv_path.exists():
            print(f"⚠️  Virtual environment '{self.venv_name}' already exists")
            response = input("   Do you want to recreate it? (y/N): ").lower()
            if response == 'y':
                shutil.rmtree(venv_path)
            else:
                print("   Using existing virtual environment")
                return str(venv_path)
        
        # Try to use Python 3.12 specifically
        python_executable = self._find_python_312()
        
        try:
            subprocess.run([python_executable, "-m", "venv", self.venv_name], 
                         cwd=self.project_root, check=True)
            print(f"✅ Virtual environment created with {python_executable}: {venv_path}")
            return str(venv_path)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating virtual environment: {e}")
            print("💡 Trying with current Python executable...")
            try:
                subprocess.run([sys.executable, "-m", "venv", self.venv_name], 
                             cwd=self.project_root, check=True)
                print(f"✅ Virtual environment created with fallback: {venv_path}")
                return str(venv_path)
            except subprocess.CalledProcessError as e2:
                print(f"❌ Error creating virtual environment with fallback: {e2}")
                sys.exit(1)
    
    def _find_python_312(self):
        """Find Python 3.12 executable."""
        possible_names = [
            "python3.12",
            "python312", 
            "python-3.12",
            "python"
        ]
        
        for name in possible_names:
            try:
                result = subprocess.run([name, "--version"], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_output = result.stdout.strip()
                    if "Python 3.12" in version_output:
                        print(f"   Found Python 3.12: {name}")
                        return name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        print("   Python 3.12 not found, using current Python")
        return sys.executable
    
    def get_activate_script(self, venv_path):
        """Get the appropriate activation script for the platform."""
        if platform.system() == "Windows":
            return venv_path / "Scripts" / "activate.bat"
        else:
            return venv_path / "bin" / "activate"
    
    def install_dependencies(self, venv_path):
        """Install Python dependencies."""
        print("\n📦 Installing dependencies...")
        
        # Convert venv_path to Path object if it's a string
        venv_path = Path(venv_path)
        
        # Get pip path
        if platform.system() == "Windows":
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        try:
            # Upgrade pip using the recommended approach
            python_path = venv_path / "Scripts" / "python.exe" if platform.system() == "Windows" else venv_path / "bin" / "python"
            subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            print("✅ Pip upgraded")
            
            # Install requirements
            subprocess.run([str(pip_path), "install", "-r", str(self.requirements_file)], check=True)
            print("✅ Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            print("💡 Try installing manually:")
            print(f"   {venv_path}/Scripts/activate  # Windows")
            print(f"   source {venv_path}/bin/activate  # Linux/Mac")
            print(f"   pip install -r requirements.txt")
            sys.exit(1)
    
    def setup_environment_file(self):
        """Setup environment configuration."""
        print("\n⚙️  Setting up environment configuration...")
        
        if not self.env_example.exists():
            print("❌ Error: env.example file not found")
            sys.exit(1)
        
        if self.env_file.exists():
            print("⚠️  .env file already exists")
            response = input("   Do you want to overwrite it? (y/N): ").lower()
            if response != 'y':
                print("   Keeping existing .env file")
                return
        
        try:
            shutil.copy(self.env_example, self.env_file)
            print("✅ Environment file created: .env")
            print("💡 Edit .env file to customize settings")
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            sys.exit(1)
    
    def create_directories(self):
        """Create necessary directories."""
        print("\n📁 Creating directories...")
        
        directories = [
            "vectorstore",
            "models/emotion",
            "logs",
            "data/uploads",
            "data/exports"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def run_tests(self, venv_path):
        """Run basic tests to verify installation."""
        print("\n🧪 Running basic tests...")
        
        # Convert venv_path to Path object if it's a string
        venv_path = Path(venv_path)
        
        # Get python path
        if platform.system() == "Windows":
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            python_path = venv_path / "bin" / "python"
        
        try:
            # Test imports
            test_script = """
import sys
sys.path.append('.')

# Test core imports
try:
    from app.core.config import settings
    from app.core.logger import setup_logging
    from app.rag.domain_retrievers import DomainRetriever
    from app.agent.tools.translation_tool import TranslationTool
    from app.core.domain_mapper import DomainMapper
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test domain mapper
try:
    domains = DomainMapper.get_all_domains()
    print(f"✅ Domain mapper working - {len(domains)} domains available")
except Exception as e:
    print(f"❌ Domain mapper error: {e}")
    sys.exit(1)

print("✅ All tests passed!")
"""
            
            result = subprocess.run([str(python_path), "-c", test_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("❌ Tests failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
        
        return True
    
    def print_next_steps(self, venv_path):
        """Print next steps for the user."""
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        # Convert venv_path to Path object if it's a string
        venv_path = Path(venv_path)
        activate_script = self.get_activate_script(venv_path)
        
        print("\n📋 Next Steps:")
        print("1. Activate virtual environment:")
        if platform.system() == "Windows":
            print(f"   {activate_script}")
        else:
            print(f"   source {activate_script}")
        
        print("\n2. Start the server:")
        print("   python main.py")
        
        print("\n3. Test the API:")
        print("   curl http://localhost:8000/health")
        
        print("\n4. View API documentation:")
        print("   http://localhost:8000/docs")
        
        print("\n5. Test chat endpoint:")
        print("   curl -X POST http://localhost:8000/api/chat/ \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"query\": \"Hello in Telugu\", \"topic\": \"general_health\", \"lang\": \"te\"}'")
        
        print("\n📚 Documentation:")
        print("   - README.md - Complete setup and usage guide")
        print("   - http://localhost:8000/docs - Interactive API docs")
        
        print("\n🔧 Configuration:")
        print("   - Edit .env file to customize settings")
        print("   - Add documents to vectorstore/ for RAG capabilities")
        print("   - Configure emotion models in models/emotion/")
        
        print("\n" + "=" * 60)
    
    def run(self):
        """Run the complete setup process."""
        self.print_banner()
        self.check_python_version()
        venv_path = self.create_virtual_environment()
        self.install_dependencies(venv_path)
        self.setup_environment_file()
        self.create_directories()
        
        if self.run_tests(venv_path):
            self.print_next_steps(venv_path)
        else:
            print("\n❌ Setup completed with test failures.")
            print("💡 Check the error messages above and try manual installation.")


if __name__ == "__main__":
    setup = MeetaraSetup()
    setup.run() 