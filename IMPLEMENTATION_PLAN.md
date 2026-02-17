# 🚀 Jetson Orin Nano — Docker Deployment Plan
## Fabric Defect Detection with Hikvision GigE Camera

---

## 📋 Your Setup

| Item | Detail |
|------|--------|
| **Device** | NVIDIA Jetson Orin Nano |
| **JetPack** | 6.1 (Ubuntu 22.04, CUDA 12.x, TensorRT 10.x) |
| **Architecture** | ARM64 / aarch64 |
| **GPU** | 1024 CUDA cores (Ampere) |
| **RAM** | 8 GB |
| **Camera** | Hikvision GigE industrial camera (MVS SDK) |
| **Model** | YOLO (best.pt → best.engine via TensorRT) |

---

## 📁 Project Structure (All Files Created ✅)

```
jetson-docker/
├── IMPLEMENTATION_PLAN.md        ← This file
├── Dockerfile                    ← Docker image (JetPack 6 base + MVS SDK)
├── docker-compose.yml            ← Run with GPU + camera + display + editable code
├── app/                          ← YOUR CODE (mounted as volume — editable!)
│   ├── defect_detection.py       ← Main detection script (Linux/Jetson version)
│   └── export_tensorrt.py        ← Convert .pt → .engine (run once)
├── model/
│   ├── README.md                 ← Place best.pt here
│   └── best.pt                   ← (you copy this)
├── mvs_sdk/
│   ├── README.md                 ← Download instructions
│   └── *.deb                     ← (you download MVS SDK ARM64 .deb)
├── results/                      ← Defect frames saved here (auto-created)
└── scripts/
    ├── setup_jetson.sh           ← One-time Jetson setup
    ├── build.sh                  ← Build Docker image
    └── run.sh                    ← Run the container
```

### 🔑 Key Design: Editable App Code
The `app/` folder is **volume-mounted**, NOT baked into the Docker image.
- Edit files on the Jetson: `nano ~/jetson-docker/app/defect_detection.py`
- Restart the container to pick up changes: `sudo docker-compose restart`
- **No rebuild needed** when you edit Python code!

---

## ✅ Step-by-Step: What To Do Now

### STEP 1: Run the Setup Script on Jetson (one-time)
```bash
# SSH into your Jetson Orin Nano
ssh <user>@<jetson-ip>

# Create project directory
mkdir -p ~/jetson-docker

# Transfer files from your Windows PC (run on Windows):
# Open PowerShell on Windows and run:
scp -r C:\SNIX\LPT_INTERN\training\jetson-docker\* <user>@<jetson-ip>:~/jetson-docker/

# Back on Jetson:
cd ~/jetson-docker
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

### STEP 2: Download Hikvision MVS SDK (.deb for ARM64)
```
1. Go to: https://www.hikrobotics.com/en/machinevision/service/download
2. Find: "Machine Vision Software MVS" → Linux ARM / aarch64 version
3. Download the .deb package
4. Copy it to the Jetson:
   scp MVS-*.deb <user>@<jetson-ip>:~/jetson-docker/mvs_sdk/
```

### STEP 3: Copy Your Trained Model
```bash
# From Windows PowerShell:
scp "C:\SNIX\LPT_INTERN\training\training_results_20260108_002532\train\weights\best.pt" ^
    <user>@<jetson-ip>:~/jetson-docker/model/best.pt
```

### STEP 4: Build the Docker Image (on Jetson)
```bash
cd ~/jetson-docker
sudo docker-compose build
# ⏱️ First build: 10-20 minutes (downloads ~5GB base image)
```

### STEP 5: Export TensorRT Model (one-time, on Jetson)
```bash
# This converts best.pt → best.engine (3-5x faster inference!)
sudo docker-compose run --rm defect-detector python /app/export_tensorrt.py
# ⏱️ Takes 5-15 minutes
```

### STEP 6: Connect Camera & Run!
```bash
# Connect Hikvision camera via Ethernet to Jetson
# Make sure Jetson ethernet is on same subnet (169.254.x.x)

# If monitor is connected:
./scripts/run.sh

# Or manually:
xhost +local:docker
sudo docker-compose up
```

---

## 🖥️ Display Options

| Mode | How | What You See |
|------|-----|-------------|
| **Monitor on Jetson** | `DISPLAY=:0` (default) | Live OpenCV window with bounding boxes |
| **Headless (SSH)** | `HEADLESS=true` in docker-compose.yml | Console output only, defect frames saved |
| **Remote SSH + X11** | `ssh -X user@jetson` | Forward display to your PC |

---

## ✏️ How to Edit the Program

Since `app/` is mounted as a Docker volume:

```bash
# Option 1: Edit directly on Jetson
nano ~/jetson-docker/app/defect_detection.py

# Option 2: Edit with VS Code Remote SSH
# Install "Remote - SSH" extension in VS Code
# Connect to jetson, open ~/jetson-docker/app/

# After editing, restart:
sudo docker-compose restart

# Or stop and re-run:
sudo docker-compose down
sudo docker-compose up
```

### What You Can Change Without Rebuilding:
- ✅ `defect_detection.py` — all Python logic
- ✅ Confidence/IOU thresholds (also via env vars)
- ✅ Camera settings (exposure, gain, FPS)
- ✅ Save logic, display logic, etc.

### What Requires Rebuild (`docker-compose build`):
- ❌ Changing the Dockerfile (adding system packages)
- ❌ Updating the MVS SDK version

---

## 📊 Expected Performance (Jetson Orin Nano)

| Configuration | Expected FPS |
|---------------|-------------|
| `.pt` model, IMG_SIZE=416 | 10-15 FPS |
| `.engine` (TensorRT), IMG_SIZE=416 | **25-35 FPS** |
| `.engine` (TensorRT), IMG_SIZE=320 | **35-45 FPS** |

> The Orin Nano is **much** faster than the original Jetson Nano (1024 vs 128 CUDA cores!)

---

## ⚠️ Important Notes

1. **GigE Camera requires `--network=host`** — already configured in docker-compose
2. **`--privileged`** — required for MVS SDK raw socket access
3. **TensorRT .engine is NOT portable** — must be built ON the Jetson
4. **Camera subnet** — Jetson ethernet must be `169.254.x.x` to match camera
5. **Model fallback** — if `best.engine` isn't found, it auto-falls back to `best.pt`
6. **Auto headless** — if no display is detected, display code is skipped automatically

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `No camera devices found` | Check ethernet cable, verify subnet (`ip addr`), ping camera |
| `CUDA not available` | Make sure `--runtime=nvidia` is used. Check: `sudo docker run --rm --runtime=nvidia nvidia/cuda:12.0-base nvidia-smi` |
| `Display not working` | Run `xhost +local:docker` on Jetson before starting container |
| `Out of memory` | Reduce `IMG_SIZE` to 320, close other apps, verify swap is enabled |
| `Model not found` | Check that `best.pt` is in `~/jetson-docker/model/` |
| `MVS SDK error` | Verify the `.deb` is for aarch64/ARM64, not x86 |
