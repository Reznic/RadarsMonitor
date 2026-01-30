from radars_manager import RadarsManager
import math
import time
import json
import os


def load_radar_config(config_file="radar_azimuth_mapping.json"):
    """Load radar configuration from JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading radar config: {e}")
        return {}


if __name__ == "__main__":
    # Load radar configuration
    radar_config = load_radar_config()

    if not radar_config:
        print("No radar configuration found. Exiting.")
        exit(1)

    print(f"Loaded {len(radar_config)} radars from configuration")

    # Create RadarsManager (includes RadarTracksServer)
    manager = RadarsManager()
    server = manager.radar_tracks_server

    # Define simulation parameters for each radar
    radars = []
    class_names = ["human", "car", "truck"]

    for idx, (radar_id, config) in enumerate(radar_config.items()):
        # Extract azimuth from config (supports both dict and float formats)
        if isinstance(config, dict):
            orientation = config.get("azimuth", 0.0)
        else:
            orientation = config

        radar_params = {
            "radar_id": radar_id,
            "orientation": orientation,
            "base_azimuth": (orientation + 45 + idx * 30) % 360,  # Spread targets around
            "range_start": 50.0 - idx * 0.5,  # Vary starting range
            "range_end": 5.0 + idx * 2,  # Vary ending range
            "phase_offset": idx * 12,  # Phase offset for azimuth oscillation
            "class_name": class_names[idx % len(class_names)],  # Rotate through classes
        }
        radars.append(radar_params)
        print(f"  {radar_id}: orientation={orientation}°, class={radar_params['class_name']}")

    # Seed status entries so the frontend sees all radars
    for radar in radars:
        server.update_radar_status(
            radar_id=radar["radar_id"],
            is_active=True,
            orientation_angle=radar["orientation"],
        )

    samples = 200

    # Precompute range steps
    for radar in radars:
        radar["range_step"] = (radar["range_start"] - radar["range_end"]) / samples

    print(f"\nStarting simulation with {samples} samples per cycle")
    print("Press Ctrl+C to stop\n")

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"Cycle {cycle}:")

            for i in range(samples):
                for idx, radar in enumerate(radars, start=1):
                    current_range = radar["range_start"] - (radar["range_step"] * i)

                    # Small azimuth oscillation with different phase per radar
                    azimuth_variation = 5 * math.sin((i + radar["phase_offset"]) * 2 * math.pi / 50)
                    current_azimuth = (radar["base_azimuth"] + azimuth_variation) % 360

                    # Track IDs are stable per radar so the UI can show multiple targets
                    server.update_radar_data(
                        radar_id=radar["radar_id"],
                        track_id=idx,
                        azimuth=current_azimuth,
                        range_meters=max(radar["range_end"], current_range),
                        class_name=radar["class_name"],
                    )

                # Sleep for a short time to simulate real-time updates
                time.sleep(0.1)  # Update every 100 ms

            print(f"  Cycle {cycle} completed ({samples} samples)")

    except KeyboardInterrupt:
        print("\n\nShutting down demo mode...")
        manager.stop()
        print("Demo stopped")
