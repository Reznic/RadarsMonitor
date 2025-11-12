from typing import List, Dict, Callable, Optional
import os
import json
import threading
import csv
from radar_tracks_server import RadarTracksServer
from radar import Radar, RadarConfiguration

try:
    from flask import Flask, request, jsonify
    flask_available = True
except ImportError:
    flask_available = False


class RadarsManager:
    """Register new radars on the network, and manage their lifecycle"""
    RADAR_AZIMUTH_MAPPING_FILE = "radar_azimuth_mapping.json"
    BOOT_SERVER_PORT = 9090
    RADAR_SERVER_PORT = 1337
    CONFIG_RETRY_COUNT = 3
    UI_TRACKS_UPDATE_FILE = "trks.csv"
    RADARS_FREQ_MARGIN = 0.25  # 250 MHz   to prevent interference between radars
    INIT_FREQ = 60  # GHz
    
    def __init__(self):
        self.radars: Dict[str, Radar] = {}
        self.next_radar_freq = self.INIT_FREQ
        self.radar_config = RadarConfiguration()
        self.radars_azimuth_mapping: Dict[str, float] = {}
        self._radars_lock = threading.Lock()
        
        self._load_azimuth_mapping(self.RADAR_AZIMUTH_MAPPING_FILE)
        
        # Start boot server to listen for radar node servers
        self._start_boot_server(self.BOOT_SERVER_PORT)
        
        # Start radar tracks server
        self.radar_tracks_server = RadarTracksServer(port=self.RADAR_SERVER_PORT, radars_manager=self)
        self.radar_tracks_server.start_server()

    def register_radar(self, radar_id: str, host: str, http_port: int, tcp_port: int) -> None:
        """Register a new radar or update an existing one"""
        azimuth = self.radars_azimuth_mapping.get(radar_id)
        
        with self._radars_lock:
            if radar_id in self.radars:
                print(f"Radar {radar_id} already registered! updating radar")
                # Stop old radar
                self.radars[radar_id].stop()
            else:
                print(f"New radar registered: {radar_id} from {host}:{http_port}")
            
            # Create new radar instance
            radar = Radar(
                radar_id=radar_id,
                host=host,
                http_port=http_port,
                tcp_port=tcp_port,
                radar_config=RadarConfiguration(chirp_start_freq=self.next_radar_freq),
                azimuth=azimuth
            )
            self.radars[radar_id] = radar
            self.next_radar_freq += self.RADARS_FREQ_MARGIN
        # Schedule radar start 
        threading.Thread(target=radar.start, name=f"StartRadar-{radar_id}", daemon=True).start()
    
    def _load_azimuth_mapping(self, mapping_file: Optional[str] = None) -> None:
        """Load radar azimuth angle mapping from a JSON file"""
        file_path = mapping_file or self.RADAR_AZIMUTH_MAPPING_FILE
        
        # Try to load from be/data directory first
        try:
            data_dir_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(data_dir_path):
                with open(data_dir_path, 'r') as f:
                    self.radars_azimuth_mapping = json.load(f)
                return
        except Exception as e:
            print(f"Warning: Could not load azimuth mapping from {data_dir_path}: {e}")
        
        # Fallback to current directory or provided path
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.radars_azimuth_mapping = json.load(f)
            else:
                print(f"Warning: Azimuth mapping file not found: {file_path}")
        except Exception as e:
            print(f"Warning: Could not load azimuth mapping from {file_path}: {e}")
    
    def _start_boot_server(self, port: int) -> None:
        """Start a Flask server to receive boot messages from radar node servers"""
        if not flask_available:
            raise RuntimeError("Flask is required for boot server: pip install flask")
        
        app = Flask(__name__)
        
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
                    print("Error: Missing radar_serial, http_port or tcp_port")
                    return jsonify({"error": "Missing radar_serial, http_port or tcp_port"}), 400
                
                self.register_radar(radar_serial, host, http_port, tcp_port)
                
                return jsonify({
                    "status": "registered",
                    "radar_id": radar_serial
                })
                    
            except Exception as e:
                print(f"Error handling boot message: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.after_request
        def after_request(resp):
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return resp
        
        # Run Flask server in a separate thread
        self.nodes_boot_server = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port, use_reloader=False),
            name="RadarsManagerBootServer", daemon=True
        )
        self.nodes_boot_server.start()
        print(f"RadarsManager boot server started on port {port}")

    def stop_boot_server(self) -> None:
        self.nodes_boot_server.join()
    
    def get_radar_azimuth(self, radar_id: str) -> Optional[float]:
        """Get the azimuth angle for a specific radar by its ID"""
        return self.radars_azimuth_mapping.get(radar_id)

    def health(self) -> Dict[str, bool]:
        return {radar_id: radar.health() for radar_id, radar in self.radars.items()}

    def configure_radar(self, radar_id: str) -> bool:
        """Configure a specific radar"""
        if radar_id not in self.radars:
            print(f"Radar {radar_id} not found")
            return False
        return self.radars[radar_id].configure(self.radar_config, self.CONFIG_RETRY_COUNT)

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
        """Stop all radars"""
        for radar in self.radars.values():
            radar.stop()
        self.radars.clear()

    def join(self) -> None:
        self.nodes_boot_server.join()

    def _on_tracked_targets_callback(self, radar_id, tracks):
        if tracks:
            classified_tracks = [track for track in tracks if track.target_class and track.target_class != 'n']
            if len(classified_tracks) > 0:
                self._update_radar_ui_with_tracks(radar_id, classified_tracks)


def convert_tracks_to_csv(tracks, radar_id: Optional[str] = None) -> List[str]:
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
            track_csv_data = [f"{track.id}",
                              track.last_assoc_timestamp,
                              0,
                              0,
                              0,
                              f"{track.range_val:.2f}",
                              f"{track.get_avg_doppler():.2f}",
                              f"{track.x:.2f}",
                              f"{track.y:.2f}",
                              0, 0,0,0, 0, 0, 0, 0, 0, 0, 0, 0,
                              class_name]
            ui_tracks.append(track_csv_data)
        except Exception as e:
            print(f"Error converting track to UI format: {e}")
            continue
    return ui_tracks


def main():
    radars_manager = None
    try:
        radars_manager = RadarsManager()
        radars_manager.join()
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")

    finally:
        if radars_manager:
            radars_manager.stop()

if __name__ == "__main__":
    main()