from radar_tracks_server import RadarTracksServer
import math
import time


if __name__ == "__main__":
    # Create server instance
    server = RadarTracksServer(port=1337)

    # Start the server
    server.start_server()

    # Define dummy radars with distinct behavior patterns
    radars = [
        {
            "radar_id": "radar1",
            "base_azimuth": 45.0,
            "orientation": 70.0,
            "range_start": 50.0,
            "range_end": 5.0,
            "phase_offset": 0,
        },
        {
            "radar_id": "radar2",
            "base_azimuth": 135.0,
            "orientation": 190.0,
            "range_start": 47.0,
            "range_end": 15.0,
            "phase_offset": 12,
        },
        {
            "radar_id": "radar3",
            "base_azimuth": 225.0,
            "orientation": 260.0,
            "range_start": 46.0,
            "range_end": 10.0,
            "phase_offset": 24,
        },
        {
            "radar_id": "radar4",
            "base_azimuth": 315.0,
            "orientation": 340.0,
            "range_start": 45.0,
            "range_end": 20.0,
            "phase_offset": 36,
        },
    ]

    # Seed status entries so the frontend sees all radars; manual control happens via requests
    for radar in radars:
        server.update_radar_status(
            radar_id=radar["radar_id"],
            is_active=False,
            orientation_angle=radar["orientation"],
        )

    samples = 200

    # Precompute range steps (status is controlled manually via frontend requests)
    for radar in radars:
        radar["range_step"] = (radar["range_start"] - radar["range_end"]) / samples

    try:
        while True:
            for i in range(samples):
                print(f"Sample {i + 1}/{samples}:")
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
                        class_name="human",
                    )

                    print(
                        f"  {radar['radar_id']}: Track={idx}, "
                        f"Azimuth={current_azimuth:.1f}°, Range={current_range:.1f}m"
                    )

                # Sleep for a short time to simulate real-time updates
                time.sleep(0.1)  # Update every 100 ms

            print("Simulation completed")

    except KeyboardInterrupt:
        server.stop_server()
