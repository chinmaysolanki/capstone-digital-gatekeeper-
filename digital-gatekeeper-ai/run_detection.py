#!/usr/bin/env python3
"""
Digital Gatekeeper AI - Detection Model Runner
Simple script to start the detection model
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🛡️ Digital Gatekeeper AI - Detection Model")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pi_integration/run_atm_guard.py").exists():
        print("❌ Error: Please run this script from the digital-gatekeeper-ai directory")
        print("   cd digital-gatekeeper-ai")
        print("   python3 run_detection.py")
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
    
    print("\n🚀 Starting Detection Model...")
    print("📷 Camera will open for real-time monitoring")
    print("🎯 Detecting: weapons, tools, face coverings, crowding")
    print("🔊 Audio monitoring: sharp-cutter sounds")
    print("📡 Alerts will be sent to API server")
    print("\n💡 Press ESC to stop detection")
    print("=" * 50)
    
    # Start detection model
    try:
        # Use virtual environment Python
        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv_path / "Scripts" / "python.exe"  # Windows
        
        cmd = [
            str(venv_python),
            "pi_integration/run_atm_guard.py",
            "--imgsz", "480",
            "--device", "cpu",  # Change to "mps" on Apple Silicon if needed
            "--audio-monitor"  # Enable audio detection
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print()
        
        # Run detection model
        result = subprocess.run(cmd)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error running detection: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
