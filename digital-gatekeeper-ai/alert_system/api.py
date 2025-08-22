from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sqlite3
import time
import os
import base64
from pathlib import Path

# Create database and images directories
DB_PATH = "alert_system/events.db"
IMAGES_DIR = "alert_system/snaps"
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Digital Gatekeeper AI - Alert API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def _db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)

def _init():
    """Initialize database tables"""
    with _db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            level TEXT,
            reasons TEXT,
            img_path TEXT,
            fps REAL,
            imgsz INTEGER,
            device TEXT
        )""")

# Initialize database
_init()

@app.post("/alert")
def alert(
    level: str = Form(...),
    reasons: str = Form(""),
    fps: Optional[float] = Form(None),
    imgsz: Optional[int] = Form(None),
    device: Optional[str] = Form(None),
    snapshot_b64: Optional[str] = Form(None)
):
    """Receive security alert with optional snapshot"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    img_path = None
    
    if snapshot_b64:
        try:
            raw = base64.b64decode(snapshot_b64)
            img_path = os.path.join(IMAGES_DIR, f"{ts.replace(' ','_').replace(':','-')}.jpg")
            with open(img_path, "wb") as f:
                f.write(raw)
        except Exception as e:
            print(f"⚠️ Failed to save snapshot: {e}")
    
    with _db() as c:
        c.execute(
            "INSERT INTO events(ts,level,reasons,img_path,fps,imgsz,device) VALUES(?,?,?,?,?,?,?)",
            (ts, level, reasons, img_path, fps, imgsz, device)
        )
    
    return {"ok": True, "timestamp": ts, "level": level}

@app.get("/events")
def events(since: Optional[str] = None, limit: int = 200):
    """Retrieve stored events"""
    q = "SELECT id,ts,level,reasons,img_path,fps,imgsz,device FROM events"
    args = []
    
    if since:
        q += " WHERE ts >= ?"
        args.append(since)
    
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    
    with _db() as c:
        rows = c.execute(q, args).fetchall()
    
    keys = ["id", "ts", "level", "reasons", "img_path", "fps", "imgsz", "device"]
    return [dict(zip(keys, r)) for r in rows]

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "name": "Digital Gatekeeper AI - Alert API",
        "version": "1.0.0",
        "endpoints": {
            "POST /alert": "Receive security alerts",
            "GET /events": "Retrieve stored events",
            "GET /docs": "Interactive API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8088)
