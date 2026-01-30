from radars_manager import RadarsManager
import math
import time
import json
import os
import logging
from typing import Optional, Dict, Any, Union
from logging_setup import configure_logging

logger = logging.getLogger(__name__)


def load_radar_config(config_file="radar_azimuth_mapping.json"):
    """Load radar configuration from JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Error loading radar config")
        return {}

class DemoRadar:
    """Lightweight radar stub for demo mode.

    The real manager expects entries in `manager.radars` so the REST endpoints
    can validate radar IDs. In demo mode we don't talk to hardware, so this
    class implements the minimum surface area used by RadarTracksServer.
    """

    def __init__(self, radar_id: str, azimuth: Optional[float] = None):
        self.radar_id = radar_id
        self.azimuth = azimuth

    def configure(self, *args, **kwargs) -> bool:
        return True

    def send_command(self, *args, **kwargs) -> str:
        return "ok"

    def get_data_reception_health(self) -> bool:
        return True

    def get_tracks(self):
        return {}

    def stop(self) -> None:
        return None


def _normalize_mapping_for_manager(
    radar_config: Dict[str, Any],
) -> Dict[str, Union[Dict[str, Any], float]]:
    """Normalize config into RadarsManager.radars_azimuth_mapping format.

    Supports both:
    - {"RADAR_ID": 123.4}
    - {"RADAR_ID": {"azimuth": 123.4, "x": ..., "y": ..., ...}}
    """
    normalized = {}
    for radar_id, config in radar_config.items():
        if isinstance(config, dict):
            azimuth = float(config.get("azimuth", 0.0))
            normalized[radar_id] = {**config, "azimuth": azimuth}
        else:
            normalized[radar_id] = float(config)
    return normalized


if __name__ == "__main__":
    configure_logging()
    # Load radar configuration
    radar_config = load_radar_config()

    if not radar_config:
        logger.error("No radar configuration found. Exiting.")
        exit(1)

    logger.info("Loaded %s radars from configuration", len(radar_config))

    # Create RadarsManager (includes RadarTracksServer)
    manager = RadarsManager()
    # Ensure the manager has the demo radars + mapping so /radar/on|off validation works.
    manager.radars_azimuth_mapping = _normalize_mapping_for_manager(radar_config)
    with manager._radars_lock:
        for radar_id, mapping in manager.radars_azimuth_mapping.items():
            azimuth = mapping.get("azimuth") if isinstance(mapping, dict) else mapping
            manager.radars[radar_id] = DemoRadar(radar_id=radar_id, azimuth=azimuth)
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
        logger.info("Radar %s: orientation=%s°, class=%s", radar_id, orientation, radar_params["class_name"])

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

    logger.info("Starting simulation with %s samples per cycle", samples)
    logger.info("Press Ctrl+C to stop")

    try:
        cycle = 0
        while True:
            cycle += 1
            logger.info("Cycle %s", cycle)

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

            logger.info("Cycle %s completed (%s samples)", cycle, samples)

    except KeyboardInterrupt:
        logger.info("Shutting down demo mode...")
        manager.stop()
        logger.info("Demo stopped")
