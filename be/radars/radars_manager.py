from radar_node_client import RadarNodeClient
from tracker_process import TrackerProcess
from typing import List, Dict, Callable, Optional
import os
import json
import threading

try:
    from flask import Flask, request, jsonify
    flask_available = True
except ImportError:
    flask_available = False


class RadarsManager:
    RADAR_AZIMUTH_MAPPING_FILE = "radar_azimuth_mapping.json"
    BOOT_SERVER_PORT = 9090
    
    def __init__(self, on_tracked_targets: Callable[[str, list], None]) -> None:
        self.on_tracked_targets = on_tracked_targets
        self.clients: Dict[str, RadarNodeClient] = {}
        self.radar_config = RadarConfiguration()
        self.radars_azimuth_mapping: Dict[str, float] = {}
        self.tracker_processes: Dict[str, TrackerProcess] = {}
        self._clients_lock = threading.Lock()
        
        self._load_azimuth_mapping(self.RADAR_AZIMUTH_MAPPING_FILE)
        self._start_boot_server(self.BOOT_SERVER_PORT)

    def register_radar(self, radar_id: str, host: str, http_port: int, tcp_port: int) -> None:
        # Create client with the provided ports
        radar_client = RadarNodeClient(radar_id, host, config_port=http_port, data_port=tcp_port)
        
        with self._clients_lock:
            if radar_id in self.clients:
                print(f"Radar {radar_id} already registered! updating client")
                # Stop old tracker process if it exists
                if radar_id in self.tracker_processes:
                    self.tracker_processes[radar_id].stop()
            else:
                print(f"New radar registered: {radar_id} from {host}:{http_port}")
            self.clients[radar_id] = radar_client
        
        # Start tracker process for this radar
        # Calculate frame_period from fps (default is 20 fps = 0.05 seconds)
        frame_period = 1.0 / self.radar_config.fps if self.radar_config.fps > 0 else 0.05
        tracker_process = TrackerProcess(
            radar_id=radar_id,
            host=host,
            tcp_port=tcp_port,
            frame_period=frame_period,
            on_tracked_targets=self.on_tracked_targets
        )
        tracker_process.start()
        self.tracker_processes[radar_id] = tracker_process
        
        # Schedule radar configuration 
        threading.Thread(target=self.configure_radar, args=(radar_id,), name=f"ConfigureRadar-{radar_id}", daemon=True).start()
    
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
        return {radar_id: client.health() for radar_id, client in self.clients.items()}

    def configure_radar(self, radar_id: str) -> bool:
        config = self.radar_config.get_config()
        response = self.clients[radar_id].send_command(config)
        if response == '"sensorStart"':
            print(f"radar {radar_id} configured successfully")
        else:
            print(f"error configuring radar {radar_id}: {response}")
            return False
            #Todo: notify radar config failed
        return True

    def send_command(self, radar_id: str, command: str) -> str:
        return self.clients[radar_id].send_command(command)

    def broadcast_command(self, command: str) -> Dict[str, str]:
        results: Dict[str, str] = {}
        for radar_id, client in self.clients.items():
            try:
                results[radar_id] = client.send_command(command)
            except Exception as e:
                results[radar_id] = f"error: {e}"
        return results

    def start_all_events(self, on_event: Callable[[str, str, Dict], None]) -> None:
        for radar_id, client in self.clients.items():
            client.start_events(lambda event, data, rid=radar_id: on_event(rid, event, data))

    def stop(self) -> None:
        # Stop all tracker processes
        for tracker_process in self.tracker_processes.values():
            tracker_process.stop()
        self.tracker_processes.clear()
        
        # Stop all clients
        for client in self.clients.values():
            client.stop()
        self.clients.clear()
        # Todo: implement this

    def join(self) -> None:
        self.nodes_boot_server.join()
        # Todo: implement this


class RadarConfiguration:
    RADAR_CONFIG_TEMPLATE = "./radar_config_template.cfg"

    def __init__(self, fps=20, 
                       chirp_start_freq=60, 
                       chirp_slope=18, 
                       cfar_range_threshold=6, 
                       cfar_doppler_threshold=6, 
                       cfar_min_range_fov=0, 
                       cfar_max_range_fov=85, 
                       cfar_min_doppler=-11, 
                       cfar_max_doppler=-0.2):
        self.fps = fps
        self.frame_period_ms = int(1000 / fps)
        self.chirp_start_freq = chirp_start_freq
        self.chirp_slope = chirp_slope
        self.cfar_range_threshold = cfar_range_threshold
        self.cfar_doppler_threshold = cfar_doppler_threshold
        self.cfar_min_range_fov = cfar_min_range_fov
        self.cfar_max_range_fov = cfar_max_range_fov
        self.cfar_min_doppler = cfar_min_doppler
        self.cfar_max_doppler = cfar_max_doppler

    def get_config(self) -> str:
        # Load the template file
        try:
            # Try to load from package resources first
            template_path = os.path.join(os.path.dirname(__file__), "radar_config_template.cfg")
            with open(template_path, 'r') as f:
                lines = f.readlines()
        except Exception:
            # Fallback to the hardcoded path
            with open(self.RADAR_CONFIG_TEMPLATE, 'r') as f:
                lines = f.readlines()
        
        # Replace template variables with actual values
        config_lines = []
        for line in lines:
            if line.startswith("%") or line.strip() == "":
                continue
            # Replace all template variables in the line
            line = line.format(
                chirp_start_freq=self.chirp_start_freq,
                chirp_slope=self.chirp_slope,
                frame_period_ms=self.frame_period_ms,
                cfar_range_threshold=self.cfar_range_threshold,
                cfar_doppler_threshold=self.cfar_doppler_threshold,
                cfar_min_range_fov=self.cfar_min_range_fov,
                cfar_max_range_fov=self.cfar_max_range_fov,
                cfar_min_doppler=self.cfar_min_doppler,
                cfar_max_doppler=self.cfar_max_doppler
            )
            config_lines.append(line.strip())

        return "\n".join(config_lines)


def on_tracked_targets(radar_id, tracks):
    if tracks and len(tracks) > 0:
        print(f"radar {radar_id} tracks: {[(track.target_class, track.range_val, track.get_avg_doppler()) for track in tracks]}")


def main():
    radars_manager = None
    try:
        radars_manager = RadarsManager(on_tracked_targets=on_tracked_targets)
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