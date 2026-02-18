"""
Fabric Defect Detection — Jetson Orin Nano Edition
===================================================
Real-time YOLO defect detection with Hikvision GigE camera.
Saves ONLY frames with defects. Displays live feed (if display available).

All configuration is via environment variables (set in docker-compose.yml)
or falls back to defaults below.
"""

import torch
from ultralytics import YOLO
import time
from collections import deque
import os
import sys
import numpy as np
import cv2
from ctypes import *
from datetime import datetime
import threading
import logging
from flask import Flask, Response, render_template_string, jsonify

# =========================
# GPIO / RELAY CONTROL
# =========================
RELAY_ENABLED = os.environ.get("RELAY_ENABLED", "true").lower() in ("true", "1", "yes")
RELAY_PIN = int(os.environ.get("RELAY_PIN", "7"))              # BOARD pin 7  — Relay output
TRIGGER_PIN = int(os.environ.get("TRIGGER_PIN", "33"))          # BOARD pin 33 — Trigger input
RELAY_ON_DURATION = float(os.environ.get("RELAY_ON_DURATION", "2.0"))  # seconds relay stays ON
TRIGGER_POLL_INTERVAL = 0.1  # seconds between GPIO input polls

GPIO = None
if RELAY_ENABLED:
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        # Pin 7: Relay output (HIGH = relay OFF, LOW = relay ON)
        GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)
        # Pin 33: Trigger input (pull-down, idle LOW, detect HIGH to start detection)
        GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        print(f"\u2713 GPIO initialized")
        print(f"  Pin {RELAY_PIN}: Relay OUTPUT (HIGH=OFF, LOW=ON)")
        print(f"  Pin {TRIGGER_PIN}: Trigger INPUT (pull-down, idle LOW, detect HIGH)")
        print(f"  Flow: Detect → defect found → relay ON {RELAY_ON_DURATION}s + wait Pin {TRIGGER_PIN} HIGH → resume")
    except ImportError:
        print("WARNING: Jetson.GPIO not available — relay control disabled")
        print("  Install with: pip3 install Jetson.GPIO")
        RELAY_ENABLED = False
    except Exception as e:
        print(f"WARNING: GPIO init failed: {e} — relay control disabled")
        RELAY_ENABLED = False

# Relay status for UI
relay_status = "DETECTING"  # DETECTING, DEFECT_STOP, RELAY_ON, WAITING_RESUME
relay_status_lock = threading.Lock()

def set_relay_status(status):
    global relay_status
    with relay_status_lock:
        relay_status = status

def get_relay_status():
    with relay_status_lock:
        return relay_status

def pulse_relay():
    """Turn relay ON for RELAY_ON_DURATION seconds, then turn OFF (runs in background thread)"""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    set_relay_status("RELAY_ON")
    if RELAY_ENABLED and GPIO:
        GPIO.output(RELAY_PIN, GPIO.LOW)  # LOW = relay ON
        print(f"  [{ts}] Pin {RELAY_PIN} → LOW (Relay ON)")
    print(f"  [{ts}] \u26a0 RELAY ON for {RELAY_ON_DURATION}s...")
    time.sleep(RELAY_ON_DURATION)
    if RELAY_ENABLED and GPIO:
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # HIGH = relay OFF
    ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  [{ts2}] Pin {RELAY_PIN} → HIGH (Relay OFF)")
    print(f"  [{ts2}] \u2713 RELAY OFF")

def wait_for_resume_signal():
    """Block until Pin 33 receives a sustained HIGH signal to resume detection"""
    set_relay_status("WAITING_RESUME")
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    DEBOUNCE_COUNT = 5       # Must read HIGH this many times in a row
    DEBOUNCE_DELAY = 0.1     # Delay between each debounce read (total = 0.5s sustained HIGH)
    if RELAY_ENABLED and GPIO:
        initial = 'HIGH' if GPIO.input(TRIGGER_PIN) else 'LOW'
        print(f"  [{ts}] Waiting for resume signal on Pin {TRIGGER_PIN} (HIGH)...")
        print(f"  [{ts}] Pin {TRIGGER_PIN} current state: {initial}")
        if initial == 'HIGH':
            print(f"  [{ts}] Pin {TRIGGER_PIN} already HIGH — waiting for LOW first...")
            while GPIO.input(TRIGGER_PIN) == GPIO.HIGH:
                time.sleep(TRIGGER_POLL_INTERVAL)
            ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{ts2}] Pin {TRIGGER_PIN} is now LOW — ready to detect resume signal")
            time.sleep(0.5)  # Settle time

        while True:
            pin_state = GPIO.input(TRIGGER_PIN)
            if pin_state == GPIO.HIGH:
                # Debounce: require DEBOUNCE_COUNT consecutive HIGH reads
                high_count = 0
                for i in range(DEBOUNCE_COUNT):
                    time.sleep(DEBOUNCE_DELAY)
                    if GPIO.input(TRIGGER_PIN) == GPIO.HIGH:
                        high_count += 1
                    else:
                        break
                if high_count == DEBOUNCE_COUNT:
                    ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"  [{ts2}] Pin {TRIGGER_PIN} confirmed HIGH — resume signal received!")
                    return
                else:
                    ts2 = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"  [{ts2}] Pin {TRIGGER_PIN} false trigger ({high_count}/{DEBOUNCE_COUNT} HIGH reads) — ignoring")
            time.sleep(TRIGGER_POLL_INTERVAL)
    else:
        # No GPIO — just log and return immediately (continuous detection)
        print(f"  [{ts}] No GPIO — skipping resume wait")

# =========================
# MVS SDK IMPORT
# =========================
MVS_SDK_PATH = os.environ.get("MVCAM_COMMON_RUNENV", "/opt/MVS")
# MVS v3.0.1 uses aarch64 subdirectory
sys.path.append(os.path.join(MVS_SDK_PATH, "Samples", "aarch64", "Python", "MvImport"))

from MvCameraControl_class import *
from CameraParams_header import *

# =========================
# CONFIGURATION (from env vars or defaults)
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/best.engine")
TARGET_CAMERA_IP = os.environ.get("CAMERA_IP", "169.254.147.1")

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.20"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.20"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", "416"))

TARGET_FPS = int(os.environ.get("TARGET_FPS", "15"))
DISPLAY_INTERVAL = 0.033  # ~30 FPS display update

# Camera Settings
EXPOSURE_TIME = float(os.environ.get("EXPOSURE_TIME", "410.0"))
GAIN = float(os.environ.get("GAIN", "20.0"))

# Headless mode (no display window)
HEADLESS = os.environ.get("HEADLESS", "false").lower() in ("true", "1", "yes")

# Web streaming
WEB_PORT = int(os.environ.get("WEB_PORT", "3000"))
ENABLE_WEB = os.environ.get("ENABLE_WEB", "true").lower() in ("true", "1", "yes")

# Shared frame and stats for web streaming
latest_frame = None
frame_lock = threading.Lock()
live_stats = {
    "fps": 0.0,
    "frame_count": 0,
    "current_defects": 0,
    "total_defects": 0,
    "defect_frames_saved": 0,
    "uptime": 0,
    "model": "",
    "camera_ip": "",
    "relay_enabled": RELAY_ENABLED,
    "relay_status": "DETECTING",
    "relay_pin": RELAY_PIN,
    "trigger_pin": TRIGGER_PIN
}
stats_lock = threading.Lock()

# Results folder
RESULTS_FOLDER = "/app/results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Auto-detect display availability
HAS_DISPLAY = False
if not HEADLESS:
    try:
        display_env = os.environ.get("DISPLAY", "")
        if display_env:
            # Try to open and close a test window
            HAS_DISPLAY = True
            print(f"Display detected: {display_env}")
        else:
            print("No DISPLAY environment variable — running headless")
    except Exception:
        print("Display not available — running headless")

print("=" * 70)
print("  FABRIC DEFECT DETECTION — Jetson Orin Nano")
print("=" * 70)
print(f"  Model:         {MODEL_PATH}")
print(f"  Camera IP:     {TARGET_CAMERA_IP}")
print(f"  Confidence:    {CONF_THRESHOLD}")
print(f"  IOU:           {IOU_THRESHOLD}")
print(f"  Image Size:    {IMG_SIZE}")
print(f"  Target FPS:    {TARGET_FPS}")
print(f"  Exposure:      {EXPOSURE_TIME} µs")
print(f"  Gain:          {GAIN}")
print(f"  Display:       {'ON' if HAS_DISPLAY else 'HEADLESS'}")
print(f"  Web Stream:    {'http://0.0.0.0:' + str(WEB_PORT) if ENABLE_WEB else 'OFF'}")
print(f"  Relay:         {'Pin ' + str(RELAY_PIN) + ' (ENABLED)' if RELAY_ENABLED else 'DISABLED'}")
print(f"  Trigger Pin:   {'Pin ' + str(TRIGGER_PIN) + ' (resume after defect)' if RELAY_ENABLED else 'N/A'}")
print(f"  Relay Duration:{RELAY_ON_DURATION}s")
print(f"  Results:       {RESULTS_FOLDER}")
print("=" * 70)

# =========================
# WEB STREAMING SERVER
# =========================
web_app = Flask(__name__)

# Suppress noisy Flask /api/stats request logs
class StatsFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if '/api/stats' in msg:
            return False
        return True

log = logging.getLogger('werkzeug')
log.addFilter(StatsFilter())

WEB_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fabric Defect Detection - Live</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a1a; color: #eee; font-family: 'Segoe UI', Arial, sans-serif; }
        .header { background: linear-gradient(135deg, #1a1a3e, #0d0d2b); padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #00d4ff; }
        .header h1 { color: #00d4ff; font-size: 22px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-dot.live { background: #00ff88; animation: pulse 1.5s infinite; }
        .status-dot.stopped { background: #ff4444; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
        .container { display: flex; flex-wrap: wrap; gap: 15px; padding: 15px; height: calc(100vh - 70px); }
        .video-panel { flex: 1; min-width: 60%; display: flex; flex-direction: column; }
        .video-panel img { width: 100%; height: auto; max-height: calc(100vh - 120px); object-fit: contain; border-radius: 8px; border: 1px solid #222; background: #000; }
        .stats-panel { min-width: 280px; max-width: 350px; display: flex; flex-direction: column; gap: 12px; }
        .stat-card { background: linear-gradient(135deg, #1a1a3e, #12122a); border: 1px solid #2a2a5a; border-radius: 12px; padding: 18px; text-align: center; }
        .stat-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
        .stat-card .value { font-size: 36px; font-weight: bold; }
        .fps .value { color: #00d4ff; }
        .current .value { color: #ff6b6b; }
        .total .value { color: #ffa500; }
        .saved .value { color: #00ff88; }
        .frames .value { color: #bb86fc; font-size: 28px; }
        .uptime .value { color: #aaa; font-size: 24px; }
        .defect-alert { display: none; background: rgba(255,50,50,0.15); border: 1px solid #ff4444; border-radius: 12px; padding: 12px; text-align: center; }
        .defect-alert.active { display: block; animation: flash 1s infinite; }
        @keyframes flash { 0%,100% { background: rgba(255,50,50,0.15); } 50% { background: rgba(255,50,50,0.35); } }
        .defect-alert .alert-text { color: #ff4444; font-size: 18px; font-weight: bold; }
        .machine-card { border-radius: 12px; padding: 18px; text-align: center; }
        .machine-card.running { background: rgba(0,255,136,0.08); border: 2px solid #00ff88; }
        .machine-card.stopped { background: rgba(255,68,68,0.12); border: 2px solid #ff4444; animation: flash 1s infinite; }
        .machine-status { font-size: 18px; font-weight: bold; margin-bottom: 14px; }
        .machine-card.running .machine-status { color: #00ff88; }
        .machine-card.stopped .machine-status { color: #ff4444; }
        .btn-row { display: flex; gap: 10px; }
        .ctrl-btn { flex: 1; padding: 14px; font-size: 16px; font-weight: bold; color: #fff; border: none; border-radius: 10px; cursor: pointer; transition: transform 0.1s, opacity 0.2s; letter-spacing: 1px; }
        .ctrl-btn:hover { transform: scale(1.03); }
        .ctrl-btn:active { transform: scale(0.97); }
        .ctrl-btn:disabled { opacity: 0.35; cursor: not-allowed; transform: none; }
        .btn-start { background: linear-gradient(135deg, #00aa55, #00ff88); color: #003d1a; }
        .btn-stop  { background: linear-gradient(135deg, #cc2222, #ff4444); }
        @media (max-width: 800px) {
            .container { flex-direction: column; }
            .stats-panel { max-width: 100%; flex-direction: row; flex-wrap: wrap; }
            .stat-card { flex: 1; min-width: 120px; }
            .machine-card { min-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="status-dot live" id="statusDot"></span> Fabric Defect Detection</h1>
        <span style="color:#666;font-size:13px;">Jetson Orin Nano | YOLO + CUDA</span>
    </div>
    <div class="container">
        <div class="video-panel">
            <img src="/video_feed" alt="Live Stream">
        </div>
        <div class="stats-panel">
            <!-- Machine Control Card -->
            <div class="machine-card running" id="machineCard">
                <div class="machine-status" id="machineStatus">MACHINE RUNNING</div>
            </div>
            <!-- Defect Alert -->
            <div class="defect-alert" id="defectAlert">
                <div class="alert-text">DEFECT DETECTED &mdash; Relay Pulsed</div>
            </div>
            <!-- Relay Status Card -->
            <div class="stat-card" id="relayCard" style="border-color:#ffaa00;">
                <div class="label">STATUS</div>
                <div class="value" id="relayStatus" style="font-size:18px;color:#00ff88;">DETECTING</div>
                <div style="margin-top:8px;font-size:12px;color:#666;">
                    <span>Relay: Pin <span id="relayPin">7</span></span> &nbsp;|&nbsp;
                    <span>Trigger: Pin <span id="triggerPin">33</span></span>
                </div>
            </div>
            <!-- Stats Cards -->
            <div class="stat-card fps">
                <div class="label">FPS</div>
                <div class="value" id="fps">--</div>
            </div>
            <div class="stat-card current">
                <div class="label">Current Defects</div>
                <div class="value" id="current_defects">0</div>
            </div>
            <div class="stat-card total">
                <div class="label">Total Defects</div>
                <div class="value" id="total_defects">0</div>
            </div>
            <div class="stat-card saved">
                <div class="label">Frames Saved</div>
                <div class="value" id="saved">0</div>
            </div>
            <div class="stat-card frames">
                <div class="label">Total Frames</div>
                <div class="value" id="frame_count">0</div>
            </div>
            <div class="stat-card uptime">
                <div class="label">Uptime</div>
                <div class="value" id="uptime">00:00</div>
            </div>
        </div>
    </div>
    <script>
        function updateUI() {
            fetch('/api/stats').then(r=>r.json()).then(data => {
                document.getElementById('fps').textContent = data.fps.toFixed(1);
                document.getElementById('current_defects').textContent = data.current_defects;
                document.getElementById('total_defects').textContent = data.total_defects;
                document.getElementById('saved').textContent = data.defect_frames_saved;
                document.getElementById('frame_count').textContent = data.frame_count;
                const s = Math.floor(data.uptime);
                const m = Math.floor(s/60);
                const h = Math.floor(m/60);
                document.getElementById('uptime').textContent = h>0 ? h+'h '+m%60+'m' : m+'m '+s%60+'s';

                const card  = document.getElementById('machineCard');
                const mstat = document.getElementById('machineStatus');
                const dot   = document.getElementById('statusDot');
                const alert = document.getElementById('defectAlert');

                // Update machine status based on relay state
                var rs = document.getElementById('relayStatus');
                var rc = document.getElementById('relayCard');
                if (data.relay_status) {
                    if (data.relay_status === 'RELAY_ON') {
                        rs.textContent = 'RELAY ON';
                        rs.style.color = '#ff4444';
                        rc.style.borderColor = '#ff4444';
                        card.className = 'machine-card stopped';
                        mstat.textContent = 'RELAY ON (' + data.relay_on_duration + 's)';
                        dot.className = 'status-dot stopped';
                        alert.classList.add('active');
                    } else if (data.relay_status === 'DEFECT_STOP' || data.relay_status === 'WAITING_RESUME') {
                        rs.textContent = 'WAITING RESUME (Pin ' + data.trigger_pin + ')';
                        rs.style.color = '#ffaa00';
                        rc.style.borderColor = '#ffaa00';
                        card.className = 'machine-card stopped';
                        mstat.textContent = 'DEFECT — WAITING RESUME';
                        dot.className = 'status-dot stopped';
                        alert.classList.add('active');
                    } else {
                        rs.textContent = 'DETECTING';
                        rs.style.color = '#00ff88';
                        rc.style.borderColor = '#00ff88';
                        card.className = 'machine-card running';
                        mstat.textContent = 'DETECTING';
                        dot.className = 'status-dot live';
                        if (data.current_defects > 0) {
                            alert.classList.add('active');
                        } else {
                            alert.classList.remove('active');
                        }
                    }
                }
                if (data.relay_pin) document.getElementById('relayPin').textContent = data.relay_pin;
                if (data.trigger_pin) document.getElementById('triggerPin').textContent = data.trigger_pin;
            }).catch(()=>{});
        }
        setInterval(updateUI, 500);
        updateUI();
    </script>
</body>
</html>
"""

def generate_mjpeg():
    """Generator for MJPEG stream"""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.05)
                continue
            ret, jpeg = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS max

@web_app.route('/')
def index():
    return render_template_string(WEB_PAGE)

@web_app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@web_app.route('/api/stats')
def api_stats():
    with stats_lock:
        return jsonify(live_stats)

def start_web_server():
    """Start Flask in a background thread"""
    web_app.run(host='0.0.0.0', port=WEB_PORT, threaded=True, use_reloader=False)

if ENABLE_WEB:
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    print(f"\n>>> Web stream available at: http://<jetson-ip>:{WEB_PORT}")
    print(f">>> On this device: http://localhost:{WEB_PORT}\n")

# =========================
# HELPER FUNCTIONS
# =========================
def to_hex_str(num):
    """Convert error code to hex string"""
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

# =========================
# MODEL INITIALIZATION
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

if device == "cpu":
    print("WARNING: CUDA not available! Inference will be very slow.")
    print("Make sure docker is run with --runtime=nvidia")

# Determine if model is TensorRT engine
is_engine = MODEL_PATH.endswith(".engine")

# Fallback: if .engine not found, try .pt
if not os.path.exists(MODEL_PATH):
    alt_path = MODEL_PATH.replace(".engine", ".pt")
    if os.path.exists(alt_path):
        print(f"TensorRT engine not found. Falling back to PyTorch model: {alt_path}")
        MODEL_PATH = alt_path
        is_engine = False
    else:
        print(f"ERROR: Model not found at {MODEL_PATH} or {alt_path}")
        print("Please place your model in the ./model/ directory")
        exit(1)

print(f"Loading model: {MODEL_PATH} ({'TensorRT' if is_engine else 'PyTorch'})")
model = YOLO(MODEL_PATH)

if not is_engine:
    model.to(device)

# Warm up
print("Warming up model...")
dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for _ in range(5):
    _ = model.predict(dummy, verbose=False, imgsz=IMG_SIZE)
print("Model warmed up!\n")

# =========================
# CAMERA INITIALIZATION
# =========================
print("Initializing MVS Camera SDK...")

MvCamera.MV_CC_Initialize()

deviceList = MV_CC_DEVICE_INFO_LIST()
n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)

if ret != 0:
    print(f"ERROR: Enum devices failed! ret = 0x{to_hex_str(ret)}")
    MvCamera.MV_CC_Finalize()
    exit(1)

if deviceList.nDeviceNum == 0:
    print("ERROR: No camera devices found!")
    print(f"\nPlease verify:")
    print(f"  1. Camera at {TARGET_CAMERA_IP} is powered on")
    print(f"  2. Camera is connected to the Jetson via Ethernet")
    print(f"  3. Jetson network interface is on same subnet (169.254.x.x)")
    print(f"  4. Container is running with --network=host --privileged")
    MvCamera.MV_CC_Finalize()
    exit(1)

print(f"Found {deviceList.nDeviceNum} camera device(s)")

# Find the target camera by IP
target_device_index = -1
for i in range(deviceList.nDeviceNum):
    mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
    
    if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
        nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
        nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
        nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
        nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
        camera_ip = f"{nip1}.{nip2}.{nip3}.{nip4}"
        
        print(f"  [{i}] GigE Camera at IP: {camera_ip}")
        
        if camera_ip == TARGET_CAMERA_IP:
            target_device_index = i
            print(f"  ✓ Target camera matched!")

if target_device_index < 0:
    print(f"\nERROR: Camera with IP {TARGET_CAMERA_IP} not found!")
    MvCamera.MV_CC_Finalize()
    exit(1)

# Create camera handle
cam = MvCamera()
stDeviceList = cast(deviceList.pDeviceInfo[target_device_index], POINTER(MV_CC_DEVICE_INFO)).contents

ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"ERROR: Create handle failed! ret = 0x{to_hex_str(ret)}")
    MvCamera.MV_CC_Finalize()
    exit(1)

# Open device
ret = cam.MV_CC_OpenDevice()
if ret != 0:
    print(f"ERROR: Open device failed! ret = 0x{to_hex_str(ret)}")
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    exit(1)

print("✓ Camera opened successfully!")

# Set optimal packet size for GigE camera
nPacketSize = cam.MV_CC_GetOptimalPacketSize()
if int(nPacketSize) > 0:
    ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
    if ret != 0:
        print(f"  Warning: Set packet size failed! ret = 0x{to_hex_str(ret)}")
else:
    print(f"  Warning: Get optimal packet size failed!")

# Set trigger mode to off (continuous acquisition)
ret = cam.MV_CC_SetEnumValue("TriggerMode", 0)

# Set camera frame rate
ret = cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
if ret == 0:
    ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(TARGET_FPS))
    if ret == 0:
        print(f"✓ Camera frame rate set to {TARGET_FPS} FPS")

# Set exposure time
ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)
if ret == 0:
    ret = cam.MV_CC_SetFloatValue("ExposureTime", float(EXPOSURE_TIME))
    if ret == 0:
        print(f"✓ Exposure time set to {EXPOSURE_TIME} µs")

# Set gain
ret = cam.MV_CC_SetEnumValue("GainAuto", 0)
if ret == 0:
    ret = cam.MV_CC_SetFloatValue("Gain", float(GAIN))
    if ret == 0:
        print(f"✓ Gain set to {GAIN}")

# Start grabbing
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print(f"ERROR: Start grabbing failed! ret = 0x{to_hex_str(ret)}")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    exit(1)

print("✓ Camera started grabbing!")
print()
print(f"Saving ONLY defect frames to: {RESULTS_FOLDER}")
if HAS_DISPLAY:
    print("Press 'q' in the camera window to stop")
else:
    print("Press Ctrl+C to stop")
print("-" * 70)

# =========================
# LOGGING AND STATS
# =========================
defect_log = []
fps_queue = deque(maxlen=30)
frame_count = 0
defect_frame_count = 0
start_time = time.time()
last_display_time = 0

# Buffer for image data
stOutFrame = MV_FRAME_OUT()
memset(byref(stOutFrame), 0, sizeof(stOutFrame))

# Create display window if display available
if HAS_DISPLAY:
    cv2.namedWindow("Fabric Defect Detection - Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fabric Defect Detection - Live Feed", 1280, 720)

# =========================
# MAIN LOOP
# =========================
# Flow: Detect continuously → defect found → relay ON 2s (background) + wait Pin 33 HIGH → resume detection
try:
    while True:
        set_relay_status("DETECTING")
        loop_start = time.time()

        # Get image from camera
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            continue

        # Convert image data to numpy array
        frame_info = stOutFrame.stFrameInfo
        
        frame = np.ctypeslib.as_array(
            cast(stOutFrame.pBufAddr, POINTER(c_ubyte)),
            shape=(frame_info.nFrameLen,)
        ).copy()
        
        # Reshape based on pixel format
        if frame_info.enPixelType == PixelType_Gvsp_Mono8:
            frame = frame.reshape((frame_info.nHeight, frame_info.nWidth))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame_info.enPixelType == PixelType_Gvsp_RGB8_Packed:
            frame = frame.reshape((frame_info.nHeight, frame_info.nWidth, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame_info.enPixelType in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8, 
                                         PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
            frame = frame.reshape((frame_info.nHeight, frame_info.nWidth))
            frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
        else:
            frame = frame.reshape((frame_info.nHeight, frame_info.nWidth, -1))
            if frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Release image buffer immediately
        cam.MV_CC_FreeImageBuffer(stOutFrame)

        frame_count += 1
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Run YOLO detection
        predict_kwargs = {
            "conf": CONF_THRESHOLD,
            "iou": IOU_THRESHOLD,
            "imgsz": IMG_SIZE,
            "verbose": False,
            "augment": False,
            "max_det": 50,
            "agnostic_nms": True,
        }

        # Only pass device/half for PyTorch models (not TensorRT engines)
        if not is_engine:
            predict_kwargs["device"] = device
            predict_kwargs["half"] = True

        results = model.predict(frame, **predict_kwargs)[0]

        boxes = results.boxes
        defect_count = len(boxes)

        # Create display frame
        display_frame = frame.copy()

        # Process detections
        if defect_count > 0:
            defect_frame_count += 1
            time_str = time.strftime("%H:%M:%S")
            
            print(f"\n{'='*60}")
            print(f"  DEFECT DETECTED | Frame #{frame_count} | {time_str}")
            print(f"  Defects in frame: {defect_count}")
            print(f"{'='*60}")
            
            for idx, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                print(f"  [{idx+1}] Class: {class_name} | Confidence: {conf:.2f} | BBox: ({x1},{y1})-({x2},{y2})")

                defect_log.append({
                    "time": time_str,
                    "frame": frame_count,
                    "class": class_name,
                    "confidence": f"{conf:.2f}",
                    "bbox": f"({x1},{y1})-({x2},{y2})"
                })

                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # ── DEFECT ACTIONS (two parallel branches) ──
            # Branch 1: Relay ON for 2 seconds then OFF (background thread)
            print(f"  >> PULSING RELAY — ON for {RELAY_ON_DURATION}s (background)...")
            relay_thread = threading.Thread(target=pulse_relay, daemon=True)
            relay_thread.start()

            # Update web stream with defect info
            if ENABLE_WEB:
                with frame_lock:
                    latest_frame = display_frame.copy()
                with stats_lock:
                    live_stats["relay_status"] = "DEFECT_STOP"
                    live_stats["current_defects"] = defect_count

            # Branch 2: Wait for HIGH signal on Pin 33 to resume detection
            print(f"  >> WAITING for HIGH signal on Pin {TRIGGER_PIN} to resume detection...")
            wait_for_resume_signal()
            ts_resume = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"  [{ts_resume}] Resume signal received — detection RESUMED!")
            set_relay_status("DETECTING")

        # Calculate FPS
        loop_time = time.time() - loop_start
        actual_fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_queue.append(actual_fps)
        avg_fps = sum(fps_queue) / len(fps_queue)

        # Display (if available)
        # Add stats overlay (for both display and web stream)
        current_time = time.time()
        stats = [
            f"FPS: {avg_fps:.1f}",
            f"Frame: {frame_count}",
            f"Defects: {defect_count}",
            f"Total Defects: {len(defect_log)}",
            f"Saved: {defect_frame_count}"
        ]
        y = 30
        for text in stats:
            ts, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (10, y - 25), (20 + ts[0], y + 5), (0, 0, 0), -1)
            color = (0, 255, 0) if defect_count > 0 else (255, 255, 255)
            cv2.putText(display_frame, text, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 35

        if HAS_DISPLAY:
            if current_time - last_display_time >= DISPLAY_INTERVAL:
                cv2.imshow("Fabric Defect Detection - Live Feed", display_frame)
                last_display_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting (pressed 'q')...")
                break

        # Update web stream frame and stats
        if ENABLE_WEB:
            with frame_lock:
                latest_frame = display_frame.copy()
            with stats_lock:
                live_stats["fps"] = avg_fps
                live_stats["frame_count"] = frame_count
                live_stats["current_defects"] = defect_count
                live_stats["total_defects"] = len(defect_log)
                live_stats["defect_frames_saved"] = defect_frame_count
                live_stats["uptime"] = time.time() - start_time
                live_stats["model"] = MODEL_PATH
                live_stats["camera_ip"] = TARGET_CAMERA_IP
                live_stats["relay_enabled"] = RELAY_ENABLED
                live_stats["relay_status"] = get_relay_status()
                live_stats["relay_pin"] = RELAY_PIN
                live_stats["trigger_pin"] = TRIGGER_PIN
                live_stats["relay_on_duration"] = RELAY_ON_DURATION

        # Console status
        if frame_count % 30 == 0:
            print(f"Frame {frame_count:06d} | FPS: {avg_fps:.1f} | "
                  f"Defects: {defect_count} | Saved: {defect_frame_count}")

except KeyboardInterrupt:
    print("\nInterrupted by user (Ctrl+C)")

# =========================
# CLEANUP
# =========================
print("\nCleaning up...")
if RELAY_ENABLED and GPIO:
    try:
        GPIO.output(RELAY_PIN, GPIO.HIGH)  # Relay OFF
    except Exception:
        pass
    GPIO.cleanup()
    print("\u2713 GPIO cleaned up (relay OFF)")
if HAS_DISPLAY:
    cv2.destroyAllWindows()
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
MvCamera.MV_CC_Finalize()

total_time = time.time() - start_time

print()
print("=" * 70)
print("  DETECTION SUMMARY")
print("=" * 70)
print(f"  Total Frames Processed:    {frame_count}")
print(f"  Defect Frames Saved:       {defect_frame_count}")
print(f"  Clean Frames (not saved):  {frame_count - defect_frame_count}")
print(f"  Total Defects Detected:    {len(defect_log)}")
print(f"  Runtime:                   {total_time:.2f} seconds")
print(f"  Average FPS:               {frame_count / total_time:.2f}" if total_time > 0 else "  Average FPS: N/A")
print(f"  Save Location:             {RESULTS_FOLDER}")
print("=" * 70)

# Save defect log to CSV
if defect_log:
    import csv
    log_path = os.path.join(RESULTS_FOLDER, f"defect_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=defect_log[0].keys())
        writer.writeheader()
        writer.writerows(defect_log)
    print(f"  Defect log: {log_path}")
    print("=" * 70)
