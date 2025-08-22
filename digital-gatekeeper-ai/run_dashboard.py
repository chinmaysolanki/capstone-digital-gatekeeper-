#!/usr/bin/env python3
"""
Digital Gatekeeper AI - Dashboard Runner
Simple script to start the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🖥️  Digital Gatekeeper AI - Dashboard")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("dashboard/app.py").exists():
        print("❌ Error: Please run this script from the digital-gatekeeper-ai directory")
        print("   cd digital-gatekeeper-ai")
        print("   python3 run_dashboard.py")
        return 1
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found. Please run setup first:")
        print("   python3 -m venv .venv")
        print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        return 1
    
    # Check if API server is running
    try:
        import requests
        response = requests.get("http://localhost:8088/", timeout=2)
        if response.status_code == 200:
            print("✅ API Server is running on port 8088")
        else:
            print("⚠️  API Server responded with status:", response.status_code)
    except:
        print("⚠️  API Server not accessible. Make sure it's running:")
        print("   python3 run_api_server.py")
        print()
    
    print("\n🚀 Starting Dashboard...")
    print("🌐 Web interface will open in your browser")
    print("📊 Real-time security event monitoring")
    print("🚨 CRITICAL threat alerts with enhanced styling")
    print("📸 Event snapshots and threat analysis")
    print("\n💡 Dashboard will open at: http://localhost:8501")
    print("=" * 50)
    
    # Start dashboard
    try:
        # Use virtual environment Python
        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv_path / "Scripts" / "python.exe"  # Windows
        
        # Use virtual environment streamlit
        venv_streamlit = venv_path / "bin" / "streamlit"
        if not venv_streamlit.exists():
            venv_streamlit = venv_path / "Scripts" / "streamlit.exe"  # Windows
        
        cmd = [
            str(venv_streamlit),
            "run",
            "dashboard/app.py",
            "--server.port", "8501",
            "--server.headless", "false"  # Show in browser
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print()
        
        # Run dashboard
        result = subprocess.run(cmd)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
