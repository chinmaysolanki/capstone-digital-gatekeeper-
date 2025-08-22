#!/usr/bin/env python3
"""
Digital Gatekeeper AI - API Server Runner
Starts the FastAPI alert server for receiving security events
"""

import uvicorn
import sys
from pathlib import Path

# Add the alert_system directory to Python path
sys.path.append(str(Path(__file__).parent / "alert_system"))

try:
    from api import app
    print("✅ Alert API server starting...")
    print("📡 Endpoints:")
    print("   - POST /alert     - Receive security alerts")
    print("   - GET  /events    - Retrieve stored events")
    print("   - GET  /docs      - Interactive API documentation")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8088,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Error importing API: {e}")
    print("💡 Make sure alert_system/api.py exists")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
