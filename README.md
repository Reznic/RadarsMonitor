# Simple Radar

A beautiful radar visualization web application that displays real-time tracking data from CSV files.

## Project Structure

```
simple-radar/
├── fe/                      # Frontend
│   ├── index.html          # Main HTML page
│   ├── style.css           # Styling
│   └── scripts/
│       ├── config.js       # Configuration constants
│       ├── radar.js        # Radar rendering logic
│       ├── network.js      # API calls and polling
│       └── main.js         # Application initialization
│
├── be/                      # Backend
│   ├── server.js           # Main server file
│   ├── config.js           # Server configuration
│   ├── data/
│   │   └── csvHandler.js   # CSV data reader and playback
│   └── routes/
│       ├── health.js       # Health check endpoint
│       └── radar.js        # Radar data endpoint
│
├── trks.csv                # Source data: tracking information
└── package.json            # Project metadata
```

## Features

- **CSV Data Integration**: Reads and plays back real tracking data from `trks.csv`
- **Beautiful Radar Display**: Circular radar with cartesian coordinate mapping
- **Real-time Playback**: Automatically advances through CSV timestamps every 500ms
- **Multi-Target Tracking**: Displays multiple tracked objects simultaneously
- **Health Monitoring**: System status HUD in top-right corner
- **Dot Tooltips**: Each radar dot displays track ID, range, and velocity
- **Track History**: Maintains position history for smooth trail effects
- **Auto-cleanup**: Clears dots when server becomes unavailable
- **Modern Design**: Dark theme with cyan and red neon accents

## How to Run

### Initial Setup

First time setup (creates Python venv and installs dependencies):

```bash
bun install
bun run python:setup
```

### Development Mode (default)

```bash
bun run dev
# or
bun start
```

This starts both the Python backend (port 1337) and frontend (port 8001) servers with color-coded output.

Then open your browser to: **http://localhost:8001**

Press `Ctrl+C` to stop both servers.

### Production Mode

```bash
bun run prod
# or
bun src/run.ts --prod
```

### Alternative: Run Separately

If you prefer to run servers in separate terminals:

**Terminal 1 - Backend (Python):**

```bash
bun run be
# or
./venv/bin/python src/radar_tracks_server.py
```

**Terminal 2 - Frontend:**

```bash
bun run fe
```

> **Note**: You cannot open `fe/index.html` directly in the browser due to CORS restrictions with ES6 modules. You must use the frontend server.

## API Endpoints

- `GET /tracks` - Returns current track data for all radars
- `GET /radars_status` - Returns status for all radars
- `POST /radar/on` - Turn a radar on (requires `radar_id` in body)
- `POST /radar/off` - Turn a radar off (requires `radar_id` in body)

### Tracks Response Format

```json
{
  "radar1": {
    "track_id": 1,
    "azimuth": 45.0,
    "range": 25.5
  }
}
```

### Radar Status Response Format

```json
{
  "radar1": {
    "is_active": true,
    "orientation_angle": 70.0
  }
}
```

## CSV Data Format

The `trks.csv` file contains tracking information with the following key fields:

- **track_id**: Unique identifier for each tracked object
- **timestamp**: Time in seconds
- **x, y**: Cartesian coordinates (meters) from radar origin
- **doppler**: Doppler velocity (mapped to speed)
- **range**: Distance from radar (meters)
- **class**: Object classification (e.g., "Human")

## Configuration

### Frontend (`fe/scripts/config.js`)

- `API_BASE`: Backend server URL
- `HEALTH_CHECK_INTERVAL`: Health check polling interval (ms)
- `RADAR_CHECK_INTERVAL`: Radar data polling interval (ms)
- `SERVER_TIMEOUT`: Time before clearing dots when server is offline (ms)
- `MAX_DOTS`: Maximum number of dots to display

## Technologies

- **Frontend**: TypeScript, HTML5 Canvas, CSS3
- **Backend**: Python Flask server
- **Runtime**: Bun (for frontend and dev scripts), Python 3.8+ (for backend)
- **No build process required** for frontend
