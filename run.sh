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



# Wait for all background processes
wait


