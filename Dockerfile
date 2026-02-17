# ============================================================
# Fabric Defect Detection — Jetson Orin Nano (JetPack 6.1)
# Base: Ultralytics YOLO with JetPack 6 (CUDA 12.x + TensorRT)
# ============================================================

FROM ultralytics/ultralytics:latest-jetson-jetpack6

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MVCAM_COMMON_RUNENV=/opt/MVS
ENV MVCAM_SDK_PATH=/opt/MVS
ENV LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    libglib2.0-0 \
    libgomp1 \
    net-tools \
    iputils-ping \
    ethtool \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# --- Install Hikvision MVS SDK (ARM64) ---
# Copy the MVS SDK .deb package (downloaded from hikrobotics.com)
# If you have a .tar.gz instead, adjust accordingly
COPY mvs_sdk/*.deb /tmp/mvs_sdk.deb
RUN dpkg -i /tmp/mvs_sdk.deb || apt-get install -f -y \
    && rm /tmp/mvs_sdk.deb

# Fix MVS SDK library path — Python wrapper expects /opt/MVS/aarch64/ but libs are in /opt/MVS/lib/aarch64/
RUN ln -sf /opt/MVS/lib/aarch64 /opt/MVS/aarch64

# Verify MVS SDK installation
RUN ls -la /opt/MVS/lib/aarch64/libMvCameraControl.so && \
    ls -la /opt/MVS/aarch64/libMvCameraControl.so && \
    ls -la /opt/MVS/Samples/aarch64/Python/MvImport/

# --- Setup working directory ---
WORKDIR /app

# Install Flask for web streaming and Jetson.GPIO for relay control
RUN pip3 install --no-cache-dir flask Jetson.GPIO

# Create directories (actual code comes via volume mount)
RUN mkdir -p /app/model /app/results

# --- Default command ---
# App code is mounted at /app via volume — NOT baked into image
CMD ["python", "/app/defect_detection.py"]
