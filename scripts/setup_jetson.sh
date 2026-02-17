#!/bin/bash
# ============================================================
# One-Time Jetson Orin Nano Setup Script
# Run this ONCE after flashing JetPack 6.1
# ============================================================

set -e

echo "=================================================="
echo "  Jetson Orin Nano — One-Time Setup"
echo "=================================================="

# 1. Verify JetPack version
echo ""
echo "[1/7] Checking JetPack version..."
cat /etc/nv_tegra_release 2>/dev/null || echo "  (nv_tegra_release not found, checking dpkg)"
dpkg -l | grep nvidia-jetpack || echo "  Run: sudo apt install nvidia-jetpack"

# 2. Verify CUDA
echo ""
echo "[2/7] Checking CUDA..."
nvcc --version 2>/dev/null || echo "  WARNING: nvcc not found. CUDA may not be installed."

# 3. Verify Docker
echo ""
echo "[3/7] Checking Docker..."
docker --version 2>/dev/null || {
    echo "  Docker not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
}

# 4. Verify nvidia-container-toolkit
echo ""
echo "[4/7] Checking NVIDIA container toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "  Installing nvidia-container-toolkit..."
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
else
    echo "  ✓ nvidia-container-toolkit is installed"
fi

# 5. Add user to docker group (no sudo needed for docker commands)
echo ""
echo "[5/7] Adding user to docker group..."
sudo usermod -aG docker $USER
echo "  ✓ Added $USER to docker group (logout/login to take effect)"

# 6. Increase swap (recommended for model export)
echo ""
echo "[6/7] Checking swap..."
free -h | grep -i swap
SWAP_SIZE=$(free -b | grep -i swap | awk '{print $2}')
if [ "$SWAP_SIZE" -lt "4000000000" ]; then
    echo "  Adding 8GB swap file..."
    sudo fallocate -l 8G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "  ✓ 8GB swap added"
else
    echo "  ✓ Swap is sufficient"
fi

# 7. Set max performance mode
echo ""
echo "[7/7] Setting max performance mode..."
sudo nvpmodel -m 0 2>/dev/null || echo "  (nvpmodel not available)"
sudo jetson_clocks 2>/dev/null || echo "  (jetson_clocks not available)"

echo ""
echo "=================================================="
echo "  ✓ Setup complete!"
echo ""
echo "  NEXT STEPS:"
echo "  1. Logout and login (for docker group)"
echo "  2. Download MVS SDK from hikrobotics.com"
echo "     - Get the ARM64/aarch64 .deb package"
echo "     - Place it in: jetson-docker/mvs_sdk/"
echo "  3. Copy your best.pt model to: jetson-docker/model/"
echo "  4. Run: cd jetson-docker && sudo docker-compose build"
echo "=================================================="
