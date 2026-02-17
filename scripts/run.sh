#!/bin/bash
# ============================================================
# Run the defect detector container
# ============================================================

# Allow Docker to access the display (for cv2.imshow)
xhost +local:docker 2>/dev/null || true

echo "Starting Fabric Defect Detector..."
echo "  - App code mounted from ./app (editable!)"
echo "  - Model from ./model"
echo "  - Results saved to ./results"
echo ""

# Run with docker-compose
sudo docker-compose up

# To run in background:
# sudo docker-compose up -d
# sudo docker-compose logs -f
