#!/bin/bash

# RadarsMonitor startup script
# This script starts both the frontend (Bun) and backend (Python) services

set -e  # Exit on error

PROJECT_DIR="/home/ideon/RadarsMonitor_2"
cd "$PROJECT_DIR" || exit 1

# Start Bun frontend in production mode
bun run prod &

# Start Python radars manager
python src/be/radars/radars_manager.py &

# Wait for all background processes
wait
