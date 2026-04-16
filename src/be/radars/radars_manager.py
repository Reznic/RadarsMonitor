from typing import List, Dict, Callable, Optional, TypedDict, Union
import os
import re
import json
import threading
import csv
import logging
from urllib.parse import urlparse
from datetime import datetime
from radar_tracks_server import RadarTracksServer
from radar import Radar, RadarConfiguration
from logging_setup import configure_logging
from CameraSnapshot import CameraSnapshotManager
from ds1307_rtc import DS1307RTC, find_ds1307_bus

try:
    from flask import Flask, request, jsonify
    flask_available = True
except ImportError:
    flask_available = False


class RadarMapping(TypedDict, total=False):
    """Radar mapping structure with azimuth, x, and y coordinates"""
    azimuth: float
    x: float
    y: float


class RadarsManager:
    """Register new radars on the network, and manage their lifecycle"""
    RADAR_AZIMUTH_MAPPING_FILE = "radar_azimuth_mapping.json"
    BOOT_SERVER_PORT = 9090
    RADAR_SERVER_PORT = 1337
    CONFIG_RETRY_COUNT = 3
    CONFIG_DELAY = 0
    UI_TRACKS_UPDATE_FILE = "Track_Logs/trks.csv"
    RADARS_FREQ_MARGIN = 0.25  # 250 MHz   to prevent interference between radars
    INIT_FREQ = 60  # GHz
    CAMERA_STREAMS_CONFIG_FILE = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "config.json"
    )
    FE_CAMERA_CONFIG_TS = os.path.join(
        os.path.dirname(__file__), "..", "..", "fe", "src", "config.ts"
    )
    CAMERA_SNAPSHOT_INTERVAL_WITH_TRACKS_SECONDS = 0.5  # 2 Hz when tracks exist (default mode)
    CAMERA_SNAPSHOT_INTERVAL_DETECTION_MODE_SECONDS = 1.0  # 1 Hz all cameras (detection mode)
    CAMERA_USERNAME = "admin"
    CAMERA_PASSWORD = "password"
    SNAPSHOT_MODES = frozenset(("default", "detection"))

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._init_rtc_time_sync()
        self.snapshot_mode = self._read_snapshot_mode_from_config()
        self._logger.info(
            "Camera snapshot mode: %s (from config radars_manager.snapshot_mode)",
            self.snapshot_mode,
        )
        self.radars: Dict[str, Radar] = {}
        self.next_radar_freq = self.INIT_FREQ
        self.radar_config = RadarConfiguration()
        # Dictionary mapping radar_id to RadarMapping (azimuth, x, y) or just float (azimuth) for backward compatibility
        self.radars_azimuth_mapping: Dict[str, Union[RadarMapping, float]] = {}
        self._radars_lock = threading.Lock()
        
        # CSV file handling - open once and keep it open
        self._csv_file = None
        self._csv_writer = None
        self._csv_file_lock = threading.Lock()
        self._csv_write_count = 0
        self._csv_file_path = None
        self._setup_csv_file()
        self._camera_snapshot_managers: List[CameraSnapshotManager] = []
        self._stream_to_radar_serial: Dict[str, str] = self._load_stream_to_radar_from_fe_config()
        self._start_camera_snapshots()
        
        self._load_azimuth_mapping(self.RADAR_AZIMUTH_MAPPING_FILE)
        
        # Start boot server to listen for radar node servers
        self._start_boot_server(self.BOOT_SERVER_PORT)
        
        # Start radar tracks server
        self.radar_tracks_server = RadarTracksServer(port=self.RADAR_SERVER_PORT, radars_manager=self)
        self.radar_tracks_server.start_server()

    def _init_rtc_time_sync(self) -> None:
        """Create DS1307 object and let it apply sync policy on init."""
        try:
            bus = find_ds1307_bus()
            if bus is None:
                self._logger.warning("DS1307 not detected on any I2C bus; skipping RTC/system time sync.")
                return
            # side-effect: will sync RTC<->system according to cutoff year
            DS1307RTC(bus=bus, sync_on_init=True, sync_cutoff_year=2026)
            self._logger.info("RTC/system time sync attempted using DS1307 on bus=%s", bus)
        except Exception:
            self._logger.exception("Failed during RTC/system time sync on startup")

    def _read_snapshot_mode_from_config(self) -> str:
        """Load ``radars_manager.snapshot_mode`` from the top-level config.json."""
        try:
            with open(self.CAMERA_STREAMS_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            section = data.get("radars_manager") or {}
            mode = section.get("snapshot_mode", "default")
            if not isinstance(mode, str):
                self._logger.warning(
                    "radars_manager.snapshot_mode must be a string; got %r, using default",
                    mode,
                )
                return "default"
            mode = mode.strip().lower()
            if mode not in self.SNAPSHOT_MODES:
                self._logger.warning(
                    "Invalid radars_manager.snapshot_mode %r; use default or detection. Using default.",
                    section.get("snapshot_mode"),
                )
                return "default"
            return mode
        except FileNotFoundError:
            self._logger.warning(
                "Config not found at %s; snapshot_mode=default",
                self.CAMERA_STREAMS_CONFIG_FILE,
            )
            return "default"
        except Exception:
            self._logger.exception(
                "Could not read snapshot_mode from %s; using default",
                self.CAMERA_STREAMS_CONFIG_FILE,
            )
            return "default"

    def _load_stream_to_radar_from_fe_config(self) -> Dict[str, str]:
        """Parse ``streamId`` → ``radarSerial`` from ``src/fe/src/config.ts`` (CAMERAS array)."""
        mapping: Dict[str, str] = {}
        try:
            with open(self.FE_CAMERA_CONFIG_TS, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            self._logger.warning(
                "FE camera config not found at %s; radar-linked snapshots disabled in default mode.",
                self.FE_CAMERA_CONFIG_TS,
            )
            return mapping
        except Exception:
            self._logger.exception("Could not read FE camera config %s", self.FE_CAMERA_CONFIG_TS)
            return mapping

        pattern = re.compile(
            r"""\{\s*
                id:\s*(\d+),\s*
                name:\s*"([^"]+)",\s*
                streamId:\s*"([^"]+)",\s*
                radarSerial:\s*"([^"]+)"\s*
                \}""",
            re.VERBOSE,
        )
        for m in pattern.finditer(text):
            stream_id, radar_serial = m.group(3), m.group(4)
            mapping[stream_id] = radar_serial
        self._logger.info(
            "Loaded %s camera→radar mappings from %s",
            len(mapping),
            self.FE_CAMERA_CONFIG_TS,
        )
        return mapping

    @staticmethod
    def _radar_ids_match(a: Optional[str], b: Optional[str]) -> bool:
        if not a or not b:
            return False
        return a.strip().upper() == b.strip().upper()

    def register_radar(self, radar_id: str, host: str, http_port: int, tcp_port: int) -> None:
        """Register a new radar or update an existing one"""
        # Get azimuth from mapping (handle both old format: float, and new format: dict with azimuth)
        mapping = self.radars_azimuth_mapping.get(radar_id)
        if isinstance(mapping, dict):
            azimuth = mapping.get('azimuth')
        else:
            azimuth = mapping  # Backward compatibility: just a float
        
        with self._radars_lock:
            if radar_id in self.radars:
                self._logger.info("Radar %s already registered; updating instance", radar_id)
                # Stop old radar
                self.radars[radar_id].stop()
            else:
                self._logger.info("New radar registered: %s from %s:%s", radar_id, host, http_port)
            
            # Create new radar instance
            radar = Radar(
                radar_id=radar_id,
                host=host,
                http_port=http_port,
                tcp_port=tcp_port,
                radar_config=RadarConfiguration(chirp_start_freq=self.next_radar_freq),
                azimuth=azimuth,
                on_tracked_targets_callback=self._on_tracked_targets_callback,
                on_detections_callback=self._on_detections_callback,
            )
            self.radars[radar_id] = radar
            self.next_radar_freq += self.RADARS_FREQ_MARGIN
        # Schedule radar start 
        threading.Thread(target=radar.start, name=f"StartRadar-{radar_id}", daemon=True).start()

    def unregister_radar(self, radar_id: str) -> bool:
        """Unregister a radar and stop its processes."""
        with self._radars_lock:
            radar = self.radars.pop(radar_id, None)
        if radar:
            try:
                radar.stop()
                self._logger.info("Radar %s unregistered and stopped", radar_id)
            except Exception as e:
                self._logger.exception("Error stopping radar %s", radar_id)
            return True
        self._logger.warning("Radar %s not found to unregister", radar_id)
        return False
    
    def _setup_csv_file(self) -> None:
        """Setup CSV file for writing tracks - open once and keep it open with timestamp in filename"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(self.UI_TRACKS_UPDATE_FILE)[0]  # Remove .csv extension
            csv_filename = f"{base_name}_{timestamp}.csv"
            csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
            
            # Open in append mode with buffering for better performance
            self._csv_file = open(csv_path, 'a', newline='', buffering=8192)
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_file_path = csv_path  # Store the path for reference
            self._logger.info("CSV file opened for writing: %s", csv_path)
        except Exception as e:
            self._logger.exception("Error opening CSV file")
            self._csv_file = None
            self._csv_writer = None
            self._csv_file_path = None
    
    def _load_azimuth_mapping(self, mapping_file: Optional[str] = None) -> None:
        """Load radar azimuth angle mapping from a JSON file
        
        Supports two formats:
        1. Old format: {"radar_id": azimuth_float}
        2. New format: {"radar_id": {"azimuth": float, "x": float, "y": float}}
        """
        file_path = mapping_file or self.RADAR_AZIMUTH_MAPPING_FILE
        loaded_mapping = {}
        
        # Try to load from be/data directory first
        try:
            data_dir_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(data_dir_path):
                with open(data_dir_path, 'r') as f:
                    loaded_mapping = json.load(f)
                self.radars_azimuth_mapping = loaded_mapping
                self._logger.info("Loaded radar mapping from %s", data_dir_path)
                return
        except Exception as e:
            self._logger.warning("Could not load azimuth mapping from %s: %s", data_dir_path, e)
        
        # Fallback to current directory or provided path
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    loaded_mapping = json.load(f)
                self.radars_azimuth_mapping = loaded_mapping
                self._logger.info("Loaded radar mapping from %s", file_path)
            else:
                self._logger.warning("Azimuth mapping file not found: %s", file_path)
        except Exception as e:
            self._logger.warning("Could not load azimuth mapping from %s: %s", file_path, e)

    def _extract_day_camera_configs(self) -> List[Dict[str, str]]:
        """Load day-channel camera definitions from config.json."""
        camera_configs: List[Dict[str, str]] = []
        try:
            with open(self.CAMERA_STREAMS_CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            streams = config_data.get("streams", {})
            for stream_key, stream_data in streams.items():
                channels = stream_data.get("channels", {})
                day_channel = channels.get("0", {})
                day_url = day_channel.get("url")
                if not day_url:
                    continue
                parsed_url = urlparse(day_url)
                if not parsed_url.hostname:
                    continue
                camera_name = stream_data.get("name", stream_key)
                camera_configs.append({
                    "stream_key": stream_key,
                    "camera_name": camera_name,
                    "camera_ip": parsed_url.hostname,
                    "day_url": day_url,
                })
        except Exception:
            self._logger.exception("Failed to load camera configs from %s", self.CAMERA_STREAMS_CONFIG_FILE)
        return camera_configs

    def _start_camera_snapshots(self) -> None:
        """Create per-camera snapshot managers. Sampling depends on ``snapshot_mode``."""
        camera_configs = self._extract_day_camera_configs()
        for camera_cfg in camera_configs:
            try:
                stream_key = camera_cfg["stream_key"]
                radar_serial = self._stream_to_radar_serial.get(stream_key)
                if self.snapshot_mode == "default" and not radar_serial:
                    self._logger.warning(
                        "No radarSerial in FE config for stream %s (%s); "
                        "skipping snapshot manager (default mode is radar-scoped).",
                        stream_key,
                        camera_cfg["camera_name"],
                    )
                    continue

                interval = self.CAMERA_SNAPSHOT_INTERVAL_DETECTION_MODE_SECONDS
                manager = CameraSnapshotManager(
                    camera_id=camera_cfg["camera_name"],
                    ip=camera_cfg["camera_ip"],
                    username=self.CAMERA_USERNAME,
                    password=self.CAMERA_PASSWORD,
                    interval=interval,
                    day_rtsp_url=camera_cfg.get("day_url"),
                    radar_serial=radar_serial,
                )
                if self.snapshot_mode == "detection":
                    manager.start_sampling()
                self._camera_snapshot_managers.append(manager)
            except Exception:
                self._logger.exception(
                    "Failed to start camera snapshot manager for %s (%s)",
                    camera_cfg["camera_name"],
                    camera_cfg["camera_ip"],
                )
        self._logger.info(
            "Initialized %s camera snapshot manager(s) (mode=%s)",
            len(self._camera_snapshot_managers),
            self.snapshot_mode,
        )
    
    def _start_boot_server(self, port: int) -> None:
        """Start a Flask server to receive boot messages from radar node servers"""
        if not flask_available:
            raise RuntimeError("Flask is required for boot server: pip install flask")
        
        app = Flask(__name__)
        
        @app.route('/ping', methods=['GET'])
        def ping():
            """Health check endpoint for adapter nodes to verify manager is ready"""
            return jsonify({"status": "ok"})
        
        @app.route('/boot', methods=['POST'])
        def boot():
            """Handle boot message from radar node server"""
            try:
                data = request.json
                host = request.remote_addr
                radar_serial = data.get('radar_serial')
                http_port = data.get('http_port')
                tcp_port = data.get('tcp_port')
                
                if not radar_serial or not http_port or not tcp_port:
                    self._logger.warning("Boot message missing required fields: %s", data)
                    return jsonify({"error": "Missing radar_serial, http_port or tcp_port"}), 400
                
                self.register_radar(radar_serial, host, http_port, tcp_port)
                
                return jsonify({
                    "status": "registered",
                    "radar_id": radar_serial
                })
                    
            except Exception as e:
                self._logger.exception("Error handling boot message")
                return jsonify({"error": str(e)}), 500

        @app.route('/lost_radar', methods=['POST'])
        def lost_radar():
            """Handle lost radar notification to unregister the radar."""
            try:
                data = request.json or {}
                radar_serial = data.get('radar_serial')
                if not radar_serial:
                    return jsonify({"error": "Missing radar_serial"}), 400

                success = self.unregister_radar(radar_serial)
                if success:
                    return jsonify({"status": "unregistered", "radar_id": radar_serial})
                return jsonify({"error": "Radar not found"}), 404
            except Exception as e:
                self._logger.exception("Error handling lost_radar message")
                return jsonify({"error": str(e)}), 500
        
        @app.after_request
        def after_request(resp):
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return resp
        
        # Disable Werkzeug HTTP request logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        # Run Flask server in a separate thread
        self.nodes_boot_server = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port, use_reloader=False),
            name="RadarsManagerBootServer", daemon=True
        )
        self.nodes_boot_server.start()
        self._logger.info("RadarsManager boot server started on port %s", port)

    def stop_boot_server(self) -> None:
        self.nodes_boot_server.join()
    
    def get_radar_azimuth(self, radar_id: str) -> Optional[float]:
        """Get the azimuth angle for a specific radar by its ID"""
        mapping = self.radars_azimuth_mapping.get(radar_id)
        if isinstance(mapping, dict):
            return mapping.get('azimuth')
        return mapping  # Backward compatibility: just a float
    
    def get_radar_x(self, radar_id: str) -> Optional[float]:
        """Get the x coordinate for a specific radar by its ID"""
        mapping = self.radars_azimuth_mapping.get(radar_id)
        if isinstance(mapping, dict):
            return mapping.get('x')
        return None  # Old format doesn't have x
    
    def get_radar_y(self, radar_id: str) -> Optional[float]:
        """Get the y coordinate for a specific radar by its ID"""
        mapping = self.radars_azimuth_mapping.get(radar_id)
        if isinstance(mapping, dict):
            return mapping.get('y')
        return None  # Old format doesn't have y
    
    def get_radar_mapping(self, radar_id: str) -> Optional[RadarMapping]:
        """Get the complete mapping (azimuth, x, y) for a specific radar by its ID"""
        mapping = self.radars_azimuth_mapping.get(radar_id)
        if isinstance(mapping, dict):
            return mapping
        elif mapping is not None:
            # Convert old format to new format
            return {"azimuth": mapping}
        return None

    def health(self) -> Dict[str, bool]:
        return {radar_id: radar.health() for radar_id, radar in self.radars.items()}

    def configure_radar(self, radar_id: str) -> bool:
        """Configure a specific radar"""
        if radar_id not in self.radars:
            self._logger.warning("Radar %s not found", radar_id)
            return False
        return self.radars[radar_id].configure(self.radar_config, self.CONFIG_RETRY_COUNT, self.CONFIG_DELAY)

    def send_command(self, radar_id: str, command: str) -> str:
        """Send a command to a specific radar"""
        return self.radars[radar_id].send_command(command)

    def broadcast_command(self, command: str) -> Dict[str, str]:
        """Broadcast a command to all radars"""
        results: Dict[str, str] = {}
        for radar_id, radar in self.radars.items():
            try:
                results[radar_id] = radar.send_command(command)
            except Exception as e:
                results[radar_id] = f"error: {e}"
        return results

    def start_all_events(self, on_event: Callable[[str, str, Dict], None]) -> None:
        """Start events for all radars"""
        for radar_id, radar in self.radars.items():
            radar.client.start_events(lambda event, data, rid=radar_id: on_event(rid, event, data))

    def stop(self) -> None:
        """Stop all radars and close CSV file"""
        for camera_snapshot_manager in self._camera_snapshot_managers:
            try:
                camera_snapshot_manager.stop_sampling()
            except Exception:
                self._logger.exception("Error stopping camera snapshot manager")
        self._camera_snapshot_managers.clear()

        for radar in self.radars.values():
            radar.stop()
        self.radars.clear()
        
        # Close CSV file
        if self._csv_file:
            try:
                with self._csv_file_lock:
                    self._csv_file.flush()
                    self._csv_file.close()
                    self._logger.info("CSV file closed")
            except Exception as e:
                self._logger.exception("Error closing CSV file")
            finally:
                self._csv_file = None
                self._csv_writer = None

    def join(self) -> None:
        self.nodes_boot_server.join()

    def _on_tracked_targets_callback(self, radar_id, tracks):
        if self.snapshot_mode == "detection":
            return

        for m in self._camera_snapshot_managers:
            if not self._radar_ids_match(m.radar_serial, radar_id):
                continue
            try:
                if tracks:
                    if not m.is_running:
                        m.start_sampling()
                    m.set_interval(self.CAMERA_SNAPSHOT_INTERVAL_WITH_TRACKS_SECONDS)
                elif m.is_running:
                    m.stop_sampling()
            except Exception:
                self._logger.exception(
                    "Failed to update snapshot state for %s (radar %s)",
                    m.camera_id,
                    radar_id,
                )

        return
        #****Disabled - not needed for now***
        
        
        #if tracks:
        #    classified_tracks = [track for track in tracks if track.target_class and track.target_class != 'n']
        #    if len(classified_tracks) > 0:
        #        tracks_csv = convert_tracks_to_csv(tracks, radar_id, self)
        #        if tracks_csv and self._csv_writer:
        #            try:
        #                with self._csv_file_lock:
        #                    self._csv_writer.writerows(tracks_csv)
        #                    self._csv_write_count += 1
        #                    # Flush periodically (every 50 writes) instead of every time for better performance
        #                    if self._csv_write_count % 50 == 0:
        #                        self._csv_file.flush()
        #            except Exception as e:
        #                self._logger.exception("Error writing to CSV file")


    def _on_detections_callback(self, radar_id, detections, frame_number):
        # detection mode: all cameras snapshot at 1 Hz via background loops only
        return


def convert_tracks_to_csv(
    tracks,
    radar_id: Optional[str] = None,
    radars_manager: Optional["RadarsManager"] = None,
) -> List[str]:
    if not tracks:
        return []
    
    # Convert tracks to UI format
    ui_tracks = []
    
    for track in tracks:
        # Skip unclassified tracks
        if not hasattr(track, 'target_class') or track.target_class is None or track.target_class == 'n':
            continue
        
        try:
            class_map = {'c': 'Car', 'h': 'Human', 't': 'Truck', 'n': 'None'}
            class_name = class_map.get(track.target_class, 'None')
            azimuth = 0.0
            if radars_manager and radar_id and radar_id in radars_manager.radars:
                azimuth = radars_manager.radars[radar_id].azimuth or 0.0
            track_csv_data = [radar_id,
                              azimuth,
                              f"{track.id}",
                              track.last_assoc_timestamp,
                              0,
                              0,
                              0,
                              f"{track.range_val:.2f}",
                              f"{track.get_avg_doppler():.2f}",
                              f"{track.median_az:.2f}",
                              0, 0,0,0, 0, 0, 0, 0, 0, 0, 0, 0,
                              class_name]
            ui_tracks.append(track_csv_data)
        except Exception as e:
            logging.getLogger(__name__).exception("Error converting track to UI format")
            continue
    return ui_tracks


def main():
    configure_logging()
    radars_manager = None
    try:
        radars_manager = RadarsManager()
        radars_manager.join()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down...")
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal error")

    finally:
        if radars_manager:
            radars_manager.stop()

if __name__ == "__main__":
    main()
