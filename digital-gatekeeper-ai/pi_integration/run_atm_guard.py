"""
This module provides functionality for monitoring audio and detecting threats.
It includes an AudioSharpCutterMonitor for background audio analysis and
a grade_threat function to assess the overall threat level based on
object and face predictions.
"""

from ultralytics import YOLO
from pathlib import Path
import threading
import numpy as np
import time

# Optional audio library for sharp-cutter sound detection
try:
    import sounddevice as sd  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except Exception:
    sd = None  # type: ignore
    _SOUNDDEVICE_AVAILABLE = False

FPS_SMOOTH = 10

class AudioSharpCutterMonitor:
    """Background audio monitor that flags high-frequency, high-energy events.

    Heuristic: compute the ratio of spectral energy above a cutoff frequency
    to total energy; trigger when the ratio exceeds a threshold for N blocks.
    """

    def __init__(self, sample_rate: int = 16000, block_ms: int = 200,
                 highfreq_hz: int = 4000, ratio_thresh: float = 0.55,
                 min_consecutive: int = 3, active_window_s: float = 3.0):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.block_size = max(256, int(sample_rate * block_ms / 1000))
        self.highfreq_hz = highfreq_hz
        self.ratio_thresh = ratio_thresh
        self.min_consecutive = max(1, int(min_consecutive))
        self.active_window_s = float(active_window_s)

        self._stream = None
        self._consecutive = 0
        self._last_trigger_ts = 0.0

    def _callback(self, indata, frames, time_info, status):
        try:
            # Convert to mono float32 numpy array
            if indata.ndim == 2:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata

            # Apply window
            window = np.hanning(len(mono))
            x = mono.astype(np.float32) * window
            # FFT power spectrum
            fft = np.fft.rfft(x)
            power = np.abs(fft) ** 2

            # Frequency bins
            freqs = np.fft.rfftfreq(len(x), d=1.0 / self.sample_rate)
            total_energy = np.sum(power) + 1e-8
            high_mask = freqs >= float(self.highfreq_hz)
            high_energy = float(np.sum(power[high_mask]))
            ratio = high_energy / total_energy

            if ratio >= self.ratio_thresh:
                self._consecutive += 1
            else:
                self._consecutive = 0

            if self._consecutive >= self.min_consecutive:
                self._last_trigger_ts = time.time()
        except Exception:
            # Fail silently to avoid breaking main loop
            pass

    def start(self):
        if not _SOUNDDEVICE_AVAILABLE:
            return False
        if self._stream is not None:
            return True
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._callback,
        )
        self._stream.start()
        return True

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
        self._stream = None

    def is_active(self) -> bool:
        # Consider active if there was a trigger within the active window
        return (time.time() - float(self._last_trigger_ts)) <= self.active_window_s

def grade_threat(face_preds, obj_preds, threat_keywords=None, audio_active=False, audio_level="CRITICAL"):
    """
    face_preds/obj_preds: list of (box, class_name)
    threat_keywords: list of lowercase substrings that indicate a weapon/tool
    audio_active: whether sharp-cutter audio is active
    audio_level: threat level to assign for audio-only events
    """
    if threat_keywords is None:
        threat_keywords = ["gun", "knife", "rod", "weapon", "hammer", "axe", "blade", "cutter", "machete", "sword", "crowbar", "pipe", "baton", "stick", "wrench", "screwdriver"]

    matched_threats = []
    for b, c in obj_preds:
        c_lower = c.lower()
        if not in_roi(b.xyxy[0]):
            continue
        for kw in threat_keywords:
            if kw in c_lower:
                matched_threats.append(c)
                break

    has_weapon = len(matched_threats) > 0
    has_rod    = any("rod" in c.lower() for c in matched_threats)
    persons    = sum(1 for b,c in obj_preds if c=="person" and in_roi(b.xyxy[0]))
    face_cov   = any(c in ("face_covered","helmet","mask") and in_roi(b.xyxy[0]) for b,c in face_preds)

    reasons = []
    level = "LOW"

    # Primary escalation rules
    if has_weapon:
        level, reasons = "CRITICAL", ["weapon/tool detected: " + ", ".join(sorted(set(matched_threats)))[:120]]
    elif face_cov:
        level, reasons = "CRITICAL", ["face covered", "identity concealment", "critical security threat"]
    elif face_cov and has_rod:
        level, reasons = "CRITICAL", ["face covered + weapon", "security breach"]
    elif has_rod:
        level, reasons = "MEDIUM", ["rod detected"]

    if persons >= 3 and level != "CRITICAL":
        if level == "LOW":
            level = "MEDIUM"
        reasons.append("crowding")

    # Audio-only escalation
    if audio_active:
        if audio_level.upper() == "CRITICAL" and level != "CRITICAL":
            level = "CRITICAL"
        elif audio_level.upper() == "HIGH" and level not in ("HIGH", "CRITICAL"):
            level = "HIGH"
        if "sharp-cutter sound detected" not in reasons:
            reasons.append("sharp-cutter sound detected")

    return level, reasons

# === Restored imports and constants for full script functionality ===
import argparse
import csv
import statistics
from collections import deque
from datetime import datetime
import os
import base64
import cv2
import requests

# Paths (adjust if different)
FACE_COVER_WEIGHTS = "runs/detect/helmet_detection/weights/best.pt"   # trained helmet/mask model
OBJECT_WEIGHTS     = "runs/detect/atm_objects_v1/weights/best.pt"

CONF_FACE = 0.45
CONF_OBJ  = 0.45
IOU_NMS   = 0.5
IMG_SIZE = 480       # CPU-friendly inference size
ROI = None           # e.g., ROI = (x, y, w, h)
ALERT_COOLDOWN = 10  # seconds


def preprocess(frame):
    # Low-light equalization
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def _skin_ratio_hsv(bgr_region):
    """Estimate skin pixel ratio in a BGR image region using HSV heuristic.
    Returns a float in [0,1].
    """
    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    # Generic skin range in HSV (heuristic; may vary by lighting/subjects)
    # H: 0-25 or 160-180, S: 30-200, V: 50-255
    lower1 = (0, 30, 50)
    upper1 = (25, 200, 255)
    lower2 = (160, 30, 50)
    upper2 = (180, 200, 255)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    skin_pixels = int(cv2.countNonZero(mask))
    total_pixels = int(bgr_region.shape[0] * bgr_region.shape[1])
    if total_pixels <= 0:
        return 0.0
    return float(skin_pixels) / float(total_pixels)


def in_roi(xyxy):
    if ROI is None:
        return True
    x1,y1,x2,y2 = map(int, xyxy)
    rx,ry,rw,rh = ROI
    return (x1>=rx and y1>=ry and x2<=rx+rw and y2<=ry+rh)


def draw_boxes(frame, preds, color):
    for b,c in preds:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{c} {b.conf[0]:.2f}",(x1,max(20,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2,cv2.LINE_AA)


def to_preds(result, names, conf_thresh):
    out=[]
    for b in result.boxes:
        if float(b.conf[0]) < conf_thresh:
            continue
        cname = names[int(b.cls[0])]
        out.append((b, cname))
    return out


def _make_synthetic_box(x1, y1, x2, y2, conf=0.5):
    """Create a synthetic YOLO-like box object with required fields for drawing code."""
    class Box:
        def __init__(self, xyxy, conf):
            import numpy as np
            self.xyxy = [np.array(xyxy, dtype=float)]
            self.conf = [float(conf)]
    return Box([x1, y1, x2, y2], conf)


def send_alert(level, reasons, frame_bgr, alert_url="http://localhost:8088/alert", 
               fps=None, imgsz=None, device=None):
    """
    Send security alert to API endpoint with frame snapshot
    
    Args:
        level: Alert severity level (LOW, MEDIUM, HIGH, CRITICAL)
        reasons: List of reasons for the alert
        frame_bgr: OpenCV frame in BGR format
        alert_url: API endpoint URL for alerts
        fps: Current FPS of detection system
        imgsz: Image size used for detection
        device: Device used for inference
    """
    try:
        # JPEG-encode the frame
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # 85% quality for good compression
        success, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
        
        if not success:
            print("⚠️ Failed to encode frame for alert")
            return
        
        # Base64 encode the JPEG data
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare form data
        form_data = {
            'level': level,
            'reasons': ';'.join(reasons) if isinstance(reasons, list) else str(reasons),
            'snapshot_b64': img_b64
        }
        
        # Add optional performance metrics
        if fps is not None:
            form_data['fps'] = fps
        if imgsz is not None:
            form_data['imgsz'] = imgsz
        if device is not None:
            form_data['device'] = device
        
        # Send POST request to alert API
        response = requests.post(alert_url, data=form_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print(f"✅ Alert sent: {level} - {';'.join(reasons) if isinstance(reasons, list) else reasons}")
            else:
                print(f"⚠️ Alert API rejected: {result}")
        else:
            print(f"⚠️ Alert API error: HTTP {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("⚠️ Alert timeout - API server may be slow")
    except requests.exceptions.ConnectionError:
        print("⚠️ Alert connection failed - check if API server is running")
    except Exception as e:
        print(f"⚠️ Alert error: {e}")
    
    # Note: We don't crash on alert failures to keep detection running


def _ensure_csv_header(path, header_cols):
    """Ensure CSV file exists with proper header"""
    file_exists = os.path.exists(path)
    file_is_empty = not file_exists or os.path.getsize(path) == 0
    
    if file_is_empty:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header_cols)
        return True  # Header was written
    return False  # Header already exists


def _append_csv_row(path, row_dict, header_cols):
    """Append a row to CSV file"""
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_cols)
        writer.writerow(row_dict)


def _percentile_95(values):
    """Calculate 95th percentile from sorted list (no numpy dependency)"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(0.95 * (len(sorted_vals) - 1))
    return sorted_vals[index]


def run_realtime(args):
    """Run real-time ATM Guard monitoring"""
    print(f"🛡️ ATM Guard Starting...")
    print(f"📐 Inference size: {args.imgsz}px (smaller = faster on CPU)")
    print(f"🎯 Face confidence: {args.conf_face}, Object confidence: {args.conf_obj}")
    print(f"💻 Device: {args.device}")
    if args.fps_log:
        print(f"📊 FPS logging: {args.fps_log}")
    if args.no_display:
        print(f"🖥️  Display: Disabled (headless mode)")
    print("=" * 60)
    
    # Resolve weights from args or defaults
    face_weights_path = Path(args.face_weights or FACE_COVER_WEIGHTS)
    obj_weights_path = Path(args.obj_weights or OBJECT_WEIGHTS)

    use_face_heuristic = False

    # Load object model (always required)
    obj_model = YOLO(str(obj_weights_path if obj_weights_path.exists() else 'yolov8n.pt'))
    obj_model.to(args.device)

    # Load face-cover model if available; otherwise enable heuristic
    face_model = None
    if face_weights_path.exists():
        face_model = YOLO(str(face_weights_path))
        face_model.to(args.device)
    else:
        use_face_heuristic = True
        print("⚠️  Face-cover weights not found; using heuristic fallback for covered-face detection")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_alert = 0.0
    fps_hist = deque(maxlen=FPS_SMOOTH)
    prev = time.time()
    
    # Initialize CSV logging if requested
    csv_file = None
    csv_writer = None
    if args.fps_log:
        # Setup CSV for real-time logging
        header_written = _ensure_csv_header(args.fps_log, ['timestamp', 'mode', 'img_size', 'device', 'fps'])
        if header_written:
            print(f"📝 Created CSV log with header: {args.fps_log}")
        
        csv_file = open(args.fps_log, 'a', newline='')
        csv_writer = csv.writer(csv_file)

    # Optional audio monitor
    audio_monitor = None
    if getattr(args, 'audio_monitor', False):
        if not _SOUNDDEVICE_AVAILABLE:
            print("⚠️  Audio monitoring requested but 'sounddevice' is not installed. Install with: pip install sounddevice")
        else:
            audio_monitor = AudioSharpCutterMonitor(
                sample_rate=args.audio_rate,
                block_ms=args.audio_block_ms,
                highfreq_hz=args.audio_highfreq_hz,
                ratio_thresh=args.audio_highfreq_ratio,
                min_consecutive=args.audio_min_blocks,
                active_window_s=args.audio_active_window,
            )
            if audio_monitor.start():
                print("🔊 Audio sharp-cutter monitor: ENABLED")
            else:
                print("⚠️  Failed to start audio monitor")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = preprocess(frame)
            src = frame if ROI is None else frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]

            # Object detection
            obj_res  = obj_model.predict(src, imgsz=args.imgsz, conf=args.conf_obj, iou=IOU_NMS, device=args.device, verbose=False)[0]
            obj_preds  = to_preds(obj_res, obj_model.model.names, args.conf_obj)

            # Face-cover detection
            if not use_face_heuristic and face_model is not None:
                face_res = face_model.predict(src, imgsz=args.imgsz, conf=args.conf_face, iou=IOU_NMS, device=args.device, verbose=False)[0]
                face_preds = to_preds(face_res, face_model.model.names, args.conf_face)
            else:
                # Heuristic: infer covered face from person upper-body region skin ratio
                face_preds = []
                for b,c in obj_preds:
                    if c != 'person':
                        continue
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    # Upper portion where face likely resides
                    h = max(0, y2 - y1)
                    face_region_h = max(args.min_face_px, int(0.25 * h))
                    y2_face = y1 + face_region_h
                    y2_face = min(y2_face, y2)
                    if y2_face <= y1 or (x2 - x1) <= 0:
                        continue
                    roi = src[y1:y2_face, x1:x2]
                    ratio = _skin_ratio_hsv(roi)
                    if ratio < args.skin_thresh:
                        # Low skin ratio => likely covered
                        face_preds.append((_make_synthetic_box(x1, y1, x2, y2_face, conf=0.6), 'face_covered'))

            # Compute FPS before alert usage
            now = time.time()
            dt = now - prev; prev = now
            fps_hist.append(1.0/max(dt,1e-6))
            fps = sum(fps_hist)/len(fps_hist)

            audio_active = bool(audio_monitor.is_active()) if audio_monitor else False
            level, why = grade_threat(
                face_preds,
                obj_preds,
                threat_keywords=args.threat_keywords,
                audio_active=audio_active,
                audio_level=args.audio_threat_level,
            )

            # Drawing and display (skip if no_display is True)
            if not args.no_display:
                draw_boxes(src, face_preds, (0,255,0))
                draw_boxes(src, obj_preds,  (0,128,255))
                cv2.putText(frame,f"THREAT: {level} | {', '.join(why) if why else '-'}",
                            (12,30), cv2.FONT_HERSHEY_DUPLEX,0.8,
                            (0,0,255) if level in ("HIGH","CRITICAL") else (0,255,0),2)
                cv2.putText(frame,f"{fps:.1f} FPS",(12,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                cv2.imshow("ATM Guard", frame)

            if level in ("HIGH","CRITICAL") and now - last_alert > ALERT_COOLDOWN:
                send_alert(level, why, frame, args.alert_url, fps, args.imgsz, args.device)
                last_alert = now

            if not args.no_display:
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if audio_monitor:
            audio_monitor.stop()
        
        # Close CSV file if it was opened
        if csv_file:
            csv_file.close()
            print(f"📝 FPS log saved to: {args.fps_log}")


def run_benchmark(args):
    """Run automated FPS benchmark across multiple image sizes"""
    print(f"🏁 ATM Guard - Auto Benchmark Mode")
    print(f"💻 Device: {args.device}")
    print(f"📊 Sizes: {args.bench_sizes}")
    print(f"🔥 Warmup frames: {args.bench_warmup}")
    print(f"📏 Measurement frames: {args.bench_frames}")
    if args.fps_log:
        print(f"📝 Results log: {args.fps_log}")
    if args.no_display:
        print(f"🖥️  Display: Disabled (headless mode)")
    print("=" * 60)
    
    # Parse benchmark sizes
    try:
        sizes = [int(s.strip()) for s in args.bench_sizes.split(',')]
    except ValueError as e:
        print(f"❌ Invalid bench-sizes format: {e}")
        return 1
    
    # Setup CSV logging if requested
    if args.fps_log:
        header_cols = ['timestamp', 'mode', 'imgsz', 'device', 'frames', 'mean', 'median', 'p95', 'min', 'max']
        header_written = _ensure_csv_header(args.fps_log, header_cols)
        if header_written:
            print(f"📝 Created benchmark CSV with header: {args.fps_log}")
    
    # Load models once
    print("🔄 Loading models...")
    face_model = YOLO(FACE_COVER_WEIGHTS)
    obj_model = YOLO(OBJECT_WEIGHTS)
    face_model.to(args.device)
    obj_model.to(args.device)
    print("✅ Models loaded")
    
    print(f"\n📊 Benchmark Results:")
    print(f"{'Size':<6} {'Mean':<8} {'Median':<8} {'P95':<8} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    
    for size in sizes:
        # Create fresh camera capture for each size
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            # Warmup phase
            for _ in range(args.bench_warmup):
                ok, frame = cap.read()
                if not ok:
                    print(f"❌ Camera read failed during warmup for size {size}")
                    cap.release()
                    continue
                
                frame = preprocess(frame)
                src = frame if ROI is None else frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                
                # Run inference (warmup, don't measure)
                _ = face_model.predict(src, imgsz=size, conf=args.conf_face, iou=IOU_NMS, device=args.device, verbose=False)[0]
                _ = obj_model.predict(src, imgsz=size, conf=args.conf_obj, iou=IOU_NMS, device=args.device, verbose=False)[0]
            
            # Measurement phase
            fps_measurements = []
            for _ in range(args.bench_frames):
                ok, frame = cap.read()
                if not ok:
                    print(f"❌ Camera read failed during measurement for size {size}")
                    break
                
                frame = preprocess(frame)
                src = frame if ROI is None else frame[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
                
                # Time the inference
                t0 = time.time()
                face_res = face_model.predict(src, imgsz=size, conf=args.conf_face, iou=IOU_NMS, device=args.device, verbose=False)[0]
                obj_res = obj_model.predict(src, imgsz=size, conf=args.conf_obj, iou=IOU_NMS, device=args.device, verbose=False)[0]
                dt = time.time() - t0
                
                fps = 1.0 / max(dt, 1e-6)
                fps_measurements.append(fps)
                
                # Optional processing for complete pipeline test
                if not args.no_display:
                    face_preds = to_preds(face_res, face_model.model.names, args.conf_face)
                    obj_preds = to_preds(obj_res, obj_model.model.names, args.conf_obj)
                    _ = grade_threat(face_preds, obj_preds)
            
            # Calculate statistics
            if fps_measurements:
                fps_mean = statistics.mean(fps_measurements)
                fps_median = statistics.median(fps_measurements)
                fps_p95 = _percentile_95(fps_measurements)
                fps_min = min(fps_measurements)
                fps_max = max(fps_measurements)
                
                # Print results
                print(f"{size:<6} {fps_mean:<8.1f} {fps_median:<8.1f} {fps_p95:<8.1f} {fps_min:<8.1f} {fps_max:<8.1f}")
                
                # Log to CSV if requested
                if args.fps_log:
                    row_dict = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'mode': 'auto-bench',
                        'imgsz': size,
                        'device': args.device,
                        'frames': len(fps_measurements),
                        'mean': f"{fps_mean:.1f}",
                        'median': f"{fps_median:.1f}",
                        'p95': f"{fps_p95:.1f}",
                        'min': f"{fps_min:.1f}",
                        'max': f"{fps_max:.1f}"
                    }
                    header_cols = ['timestamp', 'mode', 'imgsz', 'device', 'frames', 'mean', 'median', 'p95', 'min', 'max']
                    _append_csv_row(args.fps_log, row_dict, header_cols)
            else:
                print(f"{size:<6} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")
        
        finally:
            cap.release()
    
    print("-" * 50)
    print("🎉 Benchmark completed!")
    if args.fps_log:
        print(f"📝 Results saved to: {args.fps_log}")
        print(f"📈 Plot results: python tools/plot_bench.py {args.fps_log}")
    
    return 0


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ATM Guard - Digital Gatekeeper AI')
    parser.add_argument('--imgsz', type=int, default=480,
                       help='Inference image size in pixels')
    parser.add_argument('--conf-face', type=float, default=CONF_FACE,
                       help=f'Face detection confidence threshold (default: {CONF_FACE})')
    parser.add_argument('--conf-obj', type=float, default=CONF_OBJ,
                       help=f'Object detection confidence threshold (default: {CONF_OBJ})')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference (cpu, cuda, mps, etc.)')
    parser.add_argument('--fps-log', type=str, default=None,
                       help='Optional CSV file to log FPS results over time')
    parser.add_argument('--auto-bench', action='store_true',
                       help='Run automated FPS benchmark and exit')
    parser.add_argument('--bench-sizes', type=str, default='320,480,640,800,1024',
                       help='Comma list of imgsz values for benchmark')
    parser.add_argument('--bench-frames', type=int, default=300,
                       help='Frames to measure per size (after warmup)')
    parser.add_argument('--bench-warmup', type=int, default=60,
                       help='Warmup frames per size')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable imshow/drawing for headless benchmark')
    parser.add_argument('--alert-url', type=str, default='http://localhost:8088/alert',
                       help='Alert API endpoint URL')
    parser.add_argument('--face-weights', type=str, default=None,
                       help='Path to face-cover/helmet/mask weights (.pt). If missing, heuristic fallback is used')
    parser.add_argument('--obj-weights', type=str, default=None,
                       help='Path to object/weapon/person weights (.pt). Defaults to yolov8n.pt if missing')
    parser.add_argument('--skin-thresh', type=float, default=0.08,
                       help='Skin ratio threshold for covered-face heuristic (lower = more likely covered)')
    parser.add_argument('--min-face-px', type=int, default=40,
                       help='Minimum face region height in pixels for heuristic')
    parser.add_argument('--threat-keywords', type=lambda s: [x.strip().lower() for x in s.split(',') if x.strip()],
                       default=["gun","knife","rod","weapon","hammer","axe","blade","cutter"],
                       help='Comma-separated list of substrings to treat as weapon/tool threats')
    # Audio monitoring options
    parser.add_argument('--audio-monitor', action='store_true',
                       help='Enable audio-based sharp-cutter detection (requires sounddevice)')
    parser.add_argument('--audio-rate', type=int, default=16000,
                       help='Audio sample rate for monitoring')
    parser.add_argument('--audio-block-ms', type=int, default=200,
                       help='Audio analysis block size in milliseconds')
    parser.add_argument('--audio-highfreq-hz', type=int, default=4000,
                       help='High-frequency cutoff in Hz for energy ratio')
    parser.add_argument('--audio-highfreq-ratio', type=float, default=0.55,
                       help='Threshold for high-frequency energy ratio to trigger')
    parser.add_argument('--audio-min-blocks', type=int, default=3,
                       help='Consecutive blocks over threshold required to trigger')
    parser.add_argument('--audio-active-window', type=float, default=3.0,
                       help='Seconds after trigger to consider audio threat active')
    parser.add_argument('--audio-threat-level', type=str, choices=['HIGH','CRITICAL'], default='CRITICAL',
                       help='Threat level applied when audio trigger occurs')

    args = parser.parse_args()
    
    # Choose mode
    if args.auto_bench:
        return run_benchmark(args)
    else:
        run_realtime(args)
        return 0


if __name__ == "__main__":
    exit(main())
