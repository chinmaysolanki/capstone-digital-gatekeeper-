# Digital Gatekeeper AI

End-to-end, real-time security monitoring with YOLOv8 vision and audio heuristics. Detects weapons/tools (gun, knife, rod, etc.), face coverings (mask/helmet), crowding, and sharp‑cutter sounds; triggers alerts via a FastAPI service and logs snapshots for review. Optimized for edge (Raspberry Pi) and desktop.

## Features
- Vision (YOLOv8): person, weapons/tools, PPE/face coverings; crowding.
- Audio: sharp‑cutter heuristic using microphone high‑frequency energy.
- Threat grading: LOW → MEDIUM → HIGH → CRITICAL with cooldowns.
- Alerts API (FastAPI): stores events in SQLite and snapshots on disk.
- Dashboard (Streamlit): live and historical incident views.
- Fallbacks: auto-switch to `yolov8n.pt` and heuristic PPE if custom weights missing.

## Repository Layout
```
digital-gatekeeper-ai/
  pi_integration/run_atm_guard.py   # vision+audio detection runner
  run_api_server.py                 # FastAPI alert server (port 8088)
  requirements.txt                  # dependencies
  yolov8n.pt                        # fallback general YOLOv8n model
```

## Quickstart (macOS/Linux)
```bash
# 1) Clone and enter
git clone https://github.com/chinmaysolanki/capstone-digital-gatekeeper-.git
cd capstone-digital-gatekeeper-/digital-gatekeeper-ai

# 2) Create venv and install
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
pip install -r requirements.txt

# 3) Start Alert API (background)
python3 run_api_server.py | cat &

# 4) Run detector (webcam). Use --device mps on Apple Silicon
python3 pi_integration/run_atm_guard.py --imgsz 480 --device cpu
# Optional audio monitoring:
# python3 pi_integration/run_atm_guard.py --imgsz 480 --device mps --audio-monitor
```

Check recent events:
```bash
curl -s http://localhost:8088/events | sed -E 's/},{/},\n{/g' | head -20
```

## CLI Highlights (detector)
```bash
# Device and size
--device cpu|cuda|mps      # inference device
--imgsz 480                # inference image size

# Thresholds
--conf-face 0.45           # face/helmet model confidence
--conf-obj  0.45           # object/weapon model confidence

# Weights
--face-weights PATH        # custom PPE/face-cover weights
--obj-weights  PATH        # custom object/weapon weights

# Threat mapping & audio
--threat-keywords gun,knife,rod,weapon,hammer,axe,blade,cutter
--audio-monitor            # enable sharp-cutter audio
--audio-highfreq-hz 4000   # cutoff for high-frequency energy
--audio-highfreq-ratio 0.55
--audio-min-blocks 3       # consecutive blocks to trigger
--audio-threat-level HIGH|CRITICAL
```

## Custom Models (optional)
- Place custom PPE model at `runs/detect/helmet_detection/weights/best.pt`.
- Place custom objects model at `runs/detect/atm_objects_v1/weights/best.pt`.
- If absent, the system falls back to `yolov8n.pt` and a PPE heuristic.

## Troubleshooting
- Grant camera and microphone permissions on first run.
- If `mps` errors on macOS, switch to `--device cpu`.
- If alerts don’t appear, ensure the API is running on port 8088.
- Network issues: try a different network or disable VPN/proxy temporarily.

## License
MIT (or institution policy). Update as required.
