#!/bin/bash

# RadarsMonitor startup script
# This script starts both the frontend (Bun) and backend (Python) services

set -e  # Exit on error

PROJECT_DIR="/home/ykv3/RadarsMonitor"
BUN_PATH="/home/ykv3/.bun/bin/bun"
cd "$PROJECT_DIR" || exit 1

# Start Python radars manager
echo "Starting Python backend..."
python src/be/radars/radars_manager.py &> python.log &

# Start Bun frontend in production mode
echo "Starting Bun frontend..."
$BUN_PATH run prod &> bun.log &

echo "Starting rtsp-to-web..."
docker run --rm --name rtsp-to-web \
  -p 8083:8083 \
  -p 8443:8443/udp \
  -p 50000-50010:50000-50010/udp \
  -v "$PWD/config.json:/config/config.json" \
  ghcr.io/deepch/rtsptoweb:latest &> rtsp-to-web.log &

# Wait for all background processes
wait


