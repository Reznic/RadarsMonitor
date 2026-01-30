# Demo Mode Refactoring

## Overview
Refactor `demo.py` to use `RadarsManager` with real radar serials from `radar_azimuth_mapping.json` instead of hardcoded fake radar IDs.

## Current State
- `demo.py` creates hardcoded radars (radar1, radar2, radar3, radar4)
- Uses `RadarTracksServer` directly
- Doesn't integrate with RadarsManager

## Goals
- Use real radar serials from configuration
- Integrate with RadarsManager architecture
- Maintain realistic moving target simulation
- Simplify demo mode setup

## Design

### Data Source
- Read `radar_azimuth_mapping.json` to get radar IDs and orientations
- Use existing radar configuration (supports both formats: float or dict with azimuth/x/y)

### Architecture
```
RadarsManager.__init__()
  └─> loads radar_azimuth_mapping.json
  └─> RadarTracksServer.start_server()

demo.py:
  └─> reads radar_azimuth_mapping.json
  └─> creates RadarsManager (gets server automatically)
  └─> for each radar_id:
        - update_radar_status() with orientation
        - update_radar_data() with simulated tracks
```

### Simulation Behavior
- Each radar shows 1 moving track
- Track parameters vary per radar (phase offset, range pattern)
- 200 samples per cycle, 100ms intervals
- Classes rotate: human, car, truck
- Azimuth oscillates ±5° around base angle
- Range decreases from start to end over the cycle

### Implementation Changes
1. Load radar IDs from `radar_azimuth_mapping.json`
2. Create `RadarsManager` instance (includes server)
3. Generate simulation parameters per radar
4. Loop: update status and track data for all radars
5. Handle KeyboardInterrupt gracefully

## Benefits
- Tests with production radar IDs
- Validates RadarsManager integration
- Matches real deployment configuration
- Easier to maintain (no hardcoded IDs)
