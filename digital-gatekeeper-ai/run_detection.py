#!/usr/bin/env python3
"""
Digital Gatekeeper AI - Complete Detection System
Standalone script with all detection algorithms, datasets, and models
"""

import cv2
import numpy as np
import time
import threading
import sqlite3
import base64
import requests
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import argparse

# Try to import sounddevice for audio monitoring
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  sounddevice not available - audio monitoring disabled")

class AudioSharpCutterMonitor:
    """Background audio monitor that flags high-frequency, high-energy events."""
    
    def __init__(self, highfreq_hz=8000, highfreq_ratio=0.3, min_blocks=3, sample_rate=44100):
        self.highfreq_hz = highfreq_hz
        self.highfreq_ratio = highfreq_ratio
        self.min_blocks = min_blocks
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * 0.1)  # 100ms blocks
        self.high_energy_blocks = 0
        self.is_active = False
        self.monitoring = False
        self.thread = None
        
    def _audio_callback(self, indata, frames, time, status):
        """Process audio data and detect sharp-cutter sounds."""
        if status:
            print(f"Audio status: {status}")
            
        # Convert to float and compute FFT
        audio_data = indata[:, 0].astype(np.float32)
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
        
        # Compute energy in high-frequency range
        high_freq_mask = np.abs(freqs) > self.highfreq_hz
        high_energy = np.sum(np.abs(fft[high_freq_mask])**2)
        total_energy = np.sum(np.abs(fft)**2)
        
        if total_energy > 0:
            ratio = high_energy / total_energy
            if ratio > self.highfreq_ratio:
                self.high_energy_blocks += 1
                if self.high_energy_blocks >= self.min_blocks:
                    self.is_active = True
            else:
                self.high_energy_blocks = max(0, self.high_energy_blocks - 1)
                if self.high_energy_blocks == 0:
                    self.is_active = False
    
    def start(self):
        """Start audio monitoring."""
        if not AUDIO_AVAILABLE:
            return False
            
        try:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_audio, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"❌ Error starting audio monitor: {e}")
            return False
    
    def _monitor_audio(self):
        """Audio monitoring thread."""
        try:
            with sd.InputStream(callback=self._audio_callback, 
                             channels=1, 
                             samplerate=self.sample_rate,
                             blocksize=self.block_size):
                while self.monitoring:
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Audio monitoring error: {e}")
    
    def stop(self):
        """Stop audio monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current audio monitoring status."""
        return {
            "is_active": self.is_active,
            "high_energy_blocks": self.high_energy_blocks,
            "threshold": self.min_blocks,
            "highfreq_hz": self.highfreq_hz,
            "highfreq_ratio": self.highfreq_ratio
        }

class HelmetDetectionModel:
    """YOLOv8-based helmet and face detection model."""
    
    def __init__(self, weights_path: str = "yolov8n.pt", device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device
        self.model = None
        self.class_names = []
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model and class names."""
        try:
            # Try to import ultralytics
            from ultralytics import YOLO
            
            # Check if weights exist, otherwise download
            if not Path(self.weights_path).exists():
                print(f"📥 Downloading {self.weights_path}...")
                self.model = YOLO(self.weights_path)
            else:
                self.model = YOLO(self.weights_path)
            
            # Get class names
            self.class_names = self.model.names
            print(f"✅ Model loaded: {len(self.class_names)} classes")
            
        except ImportError:
            print("❌ ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.45) -> List[Tuple]:
        """Detect objects in frame."""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=conf_threshold, device=self.device)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[cls]
                        
                        detections.append((
                            (int(x1), int(y1), int(x2), int(y2)),
                            class_name,
                            float(conf)
                        ))
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []

class FaceCoverageDetector:
    """Heuristic-based face coverage detection."""
    
    def __init__(self):
        self.face_cascade = None
        self.load_face_cascade()
    
    def load_face_cascade(self):
        """Load OpenCV face cascade classifier."""
        try:
            # Try to load pre-trained face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if Path(cascade_path).exists():
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("✅ Face cascade loaded")
            else:
                print("⚠️  Face cascade not found, using basic detection")
                self.face_cascade = None
        except Exception as e:
            print(f"⚠️  Error loading face cascade: {e}")
            self.face_cascade = None
    
    def detect_face_coverage(self, frame: np.ndarray, face_bbox: Tuple) -> bool:
        """Detect if face is covered using skin ratio analysis."""
        try:
            x1, y1, x2, y2 = face_bbox
            
            # Extract face region
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                return False
            
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Define skin color ranges (adjust these based on lighting)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate skin ratio
            total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
            
            # If skin ratio is very low, face is likely covered
            return skin_ratio < 0.3
            
        except Exception as e:
            print(f"❌ Face coverage detection error: {e}")
            return False

class ThreatClassifier:
    """Classify detected objects and behaviors into threat levels."""
    
    def __init__(self):
        # Define threat keywords for weapons and tools
        self.threat_keywords = [
            "gun", "knife", "rod", "weapon", "hammer", "axe", "blade", 
            "cutter", "machete", "sword", "crowbar", "pipe", "baton", 
            "stick", "wrench", "screwdriver", "scissors", "pliers"
        ]
        
        # Define person-related classes
        self.person_classes = ["person", "people", "man", "woman", "child"]
        
        # Define face covering classes
        self.face_cover_classes = ["helmet", "mask", "hat", "cap", "hood"]
    
    def classify_threat(self, detections: List[Tuple], audio_active: bool = False) -> Tuple[str, List[str]]:
        """Classify threat level based on detections."""
        if not detections:
            return "LOW", ["no threats detected"]
        
        # Extract class names and bounding boxes
        class_names = [det[1].lower() for det in detections]
        bboxes = [det[0] for det in detections]
        
        # Check for weapons/tools
        has_weapon = False
        matched_threats = []
        for class_name in class_names:
            for keyword in self.threat_keywords:
                if keyword in class_name:
                    has_weapon = True
                    matched_threats.append(class_name)
                    break
        
        # Check for persons
        has_person = any(pc in class_name for class_name in class_names for pc in self.person_classes)
        
        # Check for face coverings
        has_face_cover = any(fc in class_name for class_name in class_names for fc in self.face_cover_classes)
        
        # Check for crowding (multiple persons)
        person_count = sum(1 for cn in class_names for pc in self.person_classes if pc in cn)
        is_crowded = person_count > 3
        
        # Threat classification logic
        reasons = []
        
        if has_weapon and has_person:
            level = "CRITICAL"
            reasons.append(f"weapon detected: {', '.join(set(matched_threats))}")
        elif has_face_cover and has_person:
            level = "CRITICAL"
            reasons.append("face covered - identity concealment")
        elif has_weapon:
            level = "HIGH"
            reasons.append(f"weapon detected: {', '.join(set(matched_threats))}")
        elif is_crowded:
            level = "MEDIUM"
            reasons.append(f"crowding detected: {person_count} persons")
        elif has_person:
            level = "LOW"
            reasons.append("person detected")
        else:
            level = "LOW"
            reasons.append("object detected")
        
        # Audio threat escalation
        if audio_active:
            if level != "CRITICAL":
                level = "CRITICAL" if level in ["HIGH", "MEDIUM"] else "HIGH"
            reasons.append("sharp-cutter sound detected")
        
        return level, reasons

class AlertManager:
    """Manage alert sending and storage."""
    
    def __init__(self, api_url: str = "http://localhost:8088/alert"):
        self.api_url = api_url
        self.db_path = "events.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for local storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    level TEXT,
                    reasons TEXT,
                    image_data TEXT,
                    audio_active BOOLEAN
                )
            ''')
            conn.commit()
            conn.close()
            print("✅ Local database initialized")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    def send_alert(self, level: str, reasons: List[str], frame: np.ndarray, audio_active: bool = False) -> bool:
        """Send alert to API server and store locally."""
        try:
            # Encode image
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare alert data
            alert_data = {
                "level": level,
                "reasons": reasons,
                "image": image_data,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "audio_active": audio_active
            }
            
            # Send to API server
            try:
                response = requests.post(
                    self.api_url,
                    data=alert_data,
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"✅ Alert sent: {level} - {', '.join(reasons)}")
                else:
                    print(f"⚠️  API response: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️  API server unreachable: {e}")
            
            # Store locally
            self.store_event(level, reasons, image_data, audio_active)
            return True
            
        except Exception as e:
            print(f"❌ Alert error: {e}")
            return False
    
    def store_event(self, level: str, reasons: List[str], image_data: str, audio_active: bool):
        """Store event in local database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO events (timestamp, level, reasons, image_data, audio_active)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                level,
                json.dumps(reasons),
                image_data,
                audio_active
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ Storage error: {e}")

class DetectionSystem:
    """Main detection system orchestrator."""
    
    def __init__(self, args):
        self.args = args
        self.helmet_model = HelmetDetectionModel(
            weights_path=args.weights,
            device=args.device
        )
        self.face_detector = FaceCoverageDetector()
        self.threat_classifier = ThreatClassifier()
        self.alert_manager = AlertManager()
        self.audio_monitor = AudioSharpCutterMonitor(
            highfreq_hz=args.audio_highfreq_hz,
            highfreq_ratio=args.audio_highfreq_ratio,
            min_blocks=args.audio_min_blocks
        )
        
        # Detection state
        self.running = False
        self.frame_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # seconds between alerts
        
        print("🛡️ Digital Gatekeeper AI - Complete Detection System")
        print("=" * 60)
        print(f"📐 Inference size: {args.imgsz}px")
        print(f"🎯 Confidence threshold: {args.conf}")
        print(f"💻 Device: {args.device}")
        print(f"🔊 Audio monitoring: {'ENABLED' if args.audio_monitor else 'DISABLED'}")
        if args.audio_monitor:
            print(f"   High-freq threshold: {args.audio_highfreq_hz}Hz")
            print(f"   Energy ratio: {args.audio_highfreq_ratio}")
            print(f"   Min blocks: {args.audio_min_blocks}")
        print("=" * 60)
    
    def start(self):
        """Start the detection system."""
        self.running = True
        
        # Start audio monitoring if enabled
        if self.args.audio_monitor and AUDIO_AVAILABLE:
            if self.audio_monitor.start():
                print("🔊 Audio monitoring started")
            else:
                print("❌ Failed to start audio monitoring")
        
        # Start camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("📷 Camera opened successfully")
        print("🎯 Detection active - Press ESC to stop")
        print()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Resize frame for processing
                frame = cv2.resize(frame, (self.args.imgsz, self.args.imgsz))
                
                # Process frame
                self.process_frame(frame)
                
                # Display frame
                self.display_frame(frame)
                
                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stop()
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame for threats."""
        try:
            # Run detection
            detections = self.helmet_model.detect(frame, self.args.conf)
            
            # Get audio status
            audio_active = self.audio_monitor.is_active if self.args.audio_monitor else False
            
            # Classify threat level
            threat_level, reasons = self.threat_classifier.classify_threat(detections, audio_active)
            
            # Check if we should send alert
            current_time = time.time()
            if (threat_level in ["HIGH", "CRITICAL"] and 
                current_time - self.last_alert_time > self.alert_cooldown):
                
                self.alert_manager.send_alert(threat_level, reasons, frame, audio_active)
                self.last_alert_time = current_time
            
            # Draw detections on frame
            self.draw_detections(frame, detections, threat_level, reasons)
            
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple], 
                       threat_level: str, reasons: List[str]):
        """Draw detection results on frame."""
        # Draw bounding boxes
        for bbox, class_name, conf in detections:
            x1, y1, x2, y2 = bbox
            
            # Color based on threat level
            if threat_level == "CRITICAL":
                color = (0, 0, 255)  # Red
            elif threat_level == "HIGH":
                color = (0, 165, 255)  # Orange
            elif threat_level == "MEDIUM":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw threat level and reasons
        if threat_level in ["HIGH", "CRITICAL"]:
            # Background for text
            text_bg_height = 30 + len(reasons) * 20
            cv2.rectangle(frame, (10, 10), (400, 10 + text_bg_height), (0, 0, 0), -1)
            
            # Threat level
            level_color = (0, 0, 255) if threat_level == "CRITICAL" else (0, 165, 255)
            cv2.putText(frame, f"THREAT: {threat_level}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
            
            # Reasons
            for i, reason in enumerate(reasons):
                cv2.putText(frame, f"- {reason}", (20, 55 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw audio status
        if self.args.audio_monitor:
            audio_status = "🔊 AUDIO ACTIVE" if self.audio_monitor.is_active else "🔇 Audio quiet"
            audio_color = (0, 0, 255) if self.audio_monitor.is_active else (128, 128, 128)
            cv2.putText(frame, audio_status, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
    
    def display_frame(self, frame: np.ndarray):
        """Display the processed frame."""
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Digital Gatekeeper AI - Detection", frame)
    
    def stop(self):
        """Stop the detection system."""
        self.running = False
        if self.args.audio_monitor:
            self.audio_monitor.stop()
        print("🛑 Detection system stopped")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Digital Gatekeeper AI - Complete Detection System")
    
    # Model parameters
    parser.add_argument("--weights", type=str, default="yolov8n.pt", 
                       help="Path to YOLOv8 weights file")
    parser.add_argument("--imgsz", type=int, default=480, 
                       help="Input image size")
    parser.add_argument("--conf", type=float, default=0.45, 
                       help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run inference on (cpu, mps, cuda)")
    
    # Audio monitoring
    parser.add_argument("--audio-monitor", action="store_true", 
                       help="Enable audio sharp-cutter detection")
    parser.add_argument("--audio-highfreq-hz", type=int, default=8000, 
                       help="High-frequency threshold in Hz")
    parser.add_argument("--audio-highfreq-ratio", type=float, default=0.3, 
                       help="High-frequency energy ratio threshold")
    parser.add_argument("--audio-min-blocks", type=int, default=3, 
                       help="Minimum high-energy blocks to trigger")
    
    # Threat keywords
    parser.add_argument("--threat-keywords", type=str, nargs="+", 
                       help="Custom threat keywords")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("yolov8n.pt").exists():
        print("❌ Error: YOLOv8 weights not found")
        print("💡 Download with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        return 1
    
    # Check virtual environment
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        print("💡 Create with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
        return 1
    
    # Check API server
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
    
    # Create and start detection system
    try:
        detection_system = DetectionSystem(args)
        detection_system.start()
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
