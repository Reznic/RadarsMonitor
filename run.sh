#!/bin/bash

# RadarsMonitor startup script
# This script starts both the frontend (Bun) and backend (Python) services
set -e  # Exit on error

export DISPLAY=:0
export XAUTHORITY=/home/ideon/.Xauthority

sleep 5

#rotate screen inverted
/usr/bin/xrandr --output DP-1 --rotate inverted
/usr/bin/xinput set-prop 'WaveShare WS170120' "Coordinate Transformation Matrix" -1 0 1 0 -1 1 0 0 1

PROJECT_DIR="/home/ideon/RadarsMonitor"
BUN_PATH="/home/ideon/.bun/bin/bun"
cd "$PROJECT_DIR" || exit 1

# Start Python radars manager
echo "Starting Python backend..."
python src/be/radars/radars_manager.py &> python.log &

# Start Bun frontend in production mode
echo "Starting Bun frontend..."
$BUN_PATH run prod &> bun.log &


# Start MediaMTX for camera streaming (use sudo on Linux only)
echo "Starting MediaMTX..."
DOCKER_CMD="docker"
#if [ "$(uname -s)" = "Linux" ]; then
#  DOCKER_CMD="sudo docker"
#fi

# Stop existing mediamtx container if running
$DOCKER_CMD stop mediamtx 2>/dev/null || true
$DOCKER_CMD rm mediamtx 2>/dev/null || true

$DOCKER_CMD run -d --name mediamtx \
  -p 8554:8554 \
  -p 8889:8889 \
  -p 8189:8189/udp \
  -p 9997:9997 \
  -v "$PWD/mediamtx.yml:/mediamtx.yml" \
  bluenviron/mediamtx:latest \
  &> mediamtx.log &
 


# Wait for all background processes
wait


