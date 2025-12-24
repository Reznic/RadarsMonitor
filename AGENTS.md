# RadarsMonitor

Real-time multi-radar tracking visualization system with a Python backend for radar signal processing and a TypeScript/Canvas frontend for interactive web-based display.

## Quick Start

```bash
bun install                    # Install JS dependencies
bun run python:setup           # Create venv and install Python deps

bun run fe                     # Frontend dev server (http://localhost:8001)
bun run be                     # Backend API server (http://localhost:1337)
bun src/run.ts                 # Run full stack together
bun src/run.ts --prod          # Production mode with optimized assets
```

## Common Commands

| Command | Description |
|---------|-------------|
| `bun run fe` | Start frontend dev server |
| `bun run be` | Start Python backend API |
| `bun test` | Run frontend tests |
| `tsc --noEmit` | TypeScript type checking |
| `biome check .` | Lint and format check |
| `biome check --write .` | Auto-fix lint/format issues |

## Project Structure

```
src/
├── fe/                         # Frontend (TypeScript/Canvas)
│   ├── index.html             # Main HTML with HUD, canvas, debug menu
│   ├── style.css              # Dark theme styling
│   ├── serve.ts               # Bun dev server
│   └── scripts/
│       ├── main.ts            # Entry point, render loop
│       ├── radar.ts           # Canvas rendering engine
│       ├── network.ts         # API polling, state management
│       ├── config.ts          # Configuration constants
│       └── debugMenu.ts       # Debug UI controls
│
├── be/radars/                  # Backend (Python/Flask)
│   ├── radar_tracks_server.py # REST API (/tracks, /radars_status)
│   ├── radars_manager.py      # Radar lifecycle orchestration
│   ├── radar.py               # Individual radar instance
│   ├── tracker_process.py     # Multi-threaded data acquisition
│   ├── radar_frame_parser.py  # Binary TLV frame parsing
│   ├── demo.py                # Simulation mode for development
│   └── tracker_algo/          # 3D tracking algorithms
│
├── adapter_node/               # Hardware adapter interface
│   ├── adapter_node_server.py # Device discovery server
│   └── radar_device.py        # Low-level hardware control
│
├── types.ts                    # Shared TypeScript type definitions
└── run.ts                      # Full-stack orchestrator
```

## Architecture

**Data Flow:**
1. Radar hardware → TCP stream → `RadarNodeClient`
2. `TrackerProcess` → 3D tracker algorithm → tracked targets
3. `RadarTracksServer` exposes `/tracks` and `/radars_status` endpoints
4. Frontend polls backend every 50ms, renders at 60fps via `requestAnimationFrame`

**Key APIs:**
- `GET /tracks` - Current track data for all radars
- `GET /radars_status` - Radar status and orientation angles
- `POST /radar/on` / `POST /radar/off` - Activate/deactivate radars

**Configuration** (`src/fe/scripts/config.ts`):
- `API_BASE`: `http://localhost:1337`
- `RADAR_CHECK_INTERVAL`: 50ms polling
- `MAX_DOTS`: 200 trail history per track

## Code Style

- **TypeScript**: 2-space indent, camelCase variables, PascalCase types, ES modules
- **Python**: 4-space indent, PEP 8 naming
- **CSS**: BEM-inspired class names (`.sensor-grid`, `.debug-menu`)
- Run `biome check --write .` before committing

## Key Types

```typescript
interface TrackData {
  track_id: number;
  azimuth: number;
  range: number;
  class_name: string;
}

interface RadarStatus {
  is_active: boolean;
  orientation_angle: number;
}
```

## Testing

- `bun test` for frontend unit tests
- Manual verification: run full stack and verify HUD + radar plotting
- Type check with `tsc --noEmit` before commits

## Camera Streaming (MediaMTX)

The project uses MediaMTX for RTSP to WebRTC streaming.

**Start MediaMTX:**
```bash
docker run -d --name mediamtx \
  -p 8554:8554 -p 8889:8889 -p 8189:8189/udp -p 9997:9997 \
  -v "$PWD/mediamtx.yml:/mediamtx.yml" \
  bluenviron/mediamtx:latest
```

**Key files:**
- `mediamtx.yml` - Camera RTSP sources and streaming config
- `src/fe/src/stream/webrtc.ts` - WebRTC WHEP client
- `src/fe/src/config.ts` - Stream URLs (`STREAM_BASE_URL`)

**WebRTC URL format:** `http://localhost:8889/{streamId}_{channel}/whep`
- Channel 0 = day, Channel 1 = night
- Example: `http://localhost:8889/cam7_0/whep`

## Notes

- Demo mode (`demo.py`) provides simulated radar data without hardware
- Frontend uses offscreen canvas for performance (static radar base cached)
- Track history maintains 200-dot trails with fade effects
- Never commit `.env` or credentials
- Honor polling intervals to avoid overwhelming the backend
