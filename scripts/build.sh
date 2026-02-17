#!/bin/bash
# Build the Docker image
echo "Building fabric-defect-detector image..."
sudo docker-compose build
echo ""
echo "✓ Build complete!"
echo "Run './scripts/run.sh' to start the detector"
