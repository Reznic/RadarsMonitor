from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, TypedDict
import threading
import logging
import logging.handlers
import os
import json
from datetime import datetime

class TrackData(TypedDict):
    track_id: int
    azimuth: float
    range: float

class RadarStatus(TypedDict):
    is_active: bool
    orientation_angle: float
    x: float
    y: float

class RadarTracksServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 1337, radars_manager=None):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        self.host = host
        self.port = port
        # Dictionary to store radar data: radar_id -> TrackData
        self.radar_tracks: Dict[str, TrackData] = {}
        # Dictionary to store radar status: radar_id -> RadarStatus
        self.radar_status: Dict[str, RadarStatus] = {}
        # Reference to RadarsManager instance
        self.radars_manager = radars_manager
        
        # Setup logging to file with rotation (99MB max size)
        self._setup_logging()
        
        # Register routes
        self.app.route('/tracks', methods=['GET'])(self.get_tracks)
        self.app.route('/radars_status', methods=['GET'])(self.get_radars_status)
        self.app.route('/radar/on', methods=['POST'])(self.turn_radar_on)
        self.app.route('/radar/off', methods=['POST'])(self.turn_radar_off)
        
        # Create a thread for the server
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
    
    def _setup_logging(self):
        """Setup rotating file logger for radar data"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Log file path
        log_file = os.path.join(log_dir, 'radar_data.log')
        
        # Create logger
        self.data_logger = logging.getLogger('radar_data')
        self.data_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.data_logger.handlers.clear()
        
        # Create rotating file handler (99MB max, keep 10 backup files)
        max_bytes = 99 * 1024 * 1024  # 99MB
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=10,
            encoding='utf-8'
        )
        
        # Set format: timestamp, type, data
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        self.data_logger.addHandler(handler)
        self.data_logger.propagate = False  # Don't propagate to root logger
    
    def update_radar_data(self, radar_id: str, track_id: int, azimuth: float, range_meters: float) -> None:
        """
        Update the radar data for a specific radar ID
        
        Args:
            radar_id: Unique identifier for the radar
            track_id: Track ID (positive number)
            azimuth: Angle in degrees (0-359)
            range_meters: Range in meters (positive number)
        """
        # Validate inputs
        if not isinstance(track_id, int) or track_id <= 0:
            raise ValueError("Track ID must be a positive integer")
        
        if not 0 <= azimuth <= 359:
            raise ValueError("Azimuth must be between 0 and 359 degrees")
            
        if range_meters <= 0:
            raise ValueError("Range must be positive")
            
        self.radar_tracks[radar_id] = {
            "track_id": track_id,
            "azimuth": azimuth,
            "range": range_meters
        }
    
    def get_tracks(self):
        """Route handler for /tracks endpoint"""
        all_tracks = {}
        if self.radars_manager:
            # Get tracks from all radars
            with self.radars_manager._radars_lock:
                for radar_id, radar in self.radars_manager.radars.items():
                    if radar and hasattr(radar, 'get_tracks'):
                        radar_tracks = radar.get_tracks()
                        if radar_tracks:
                            all_tracks.update(radar_tracks)
        
        # Log tracks to file only if there is data
        if all_tracks:
            try:
                log_data = {
                    "type": "tracks",
                    "data": all_tracks,
                    "timestamp": datetime.now().isoformat()
                }
                # Format JSON with indentation for readability
                formatted_json = json.dumps(log_data, indent=2)
                self.data_logger.info(f"TRACKS:\n{formatted_json}")
            except Exception as e:
                self.data_logger.error(f"Error logging tracks: {e}")
        
        return jsonify(all_tracks)
    
    def start_server(self):
        """Start the server in a separate thread"""
        self.server_thread.start()
    
    def run_server(self):
        """Run the Flask server"""
        # Disable Werkzeug HTTP request logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.run(host=self.host, port=self.port)
    
    def update_radar_status(self, radar_id: str, is_active: bool, orientation_angle: float) -> None:
        stop_cmd = "sensorStop" 
        if self.radars_manager:
           # Get tracks from all radars
           with self.radars_manager._radars_lock:
               if radar_id in self.radars_manager.radars:
                if is_active:
                    self.radars_manager.radars[radar_id].configure(self.radars_manager.CONFIG_RETRY_COUNT, delay=0)
                    print(f"Radar {radar_id} status updated to ON")
                else:
                   self.radars_manager.radars[radar_id].send_command(stop_cmd)
                   print(f"Radar {radar_id} status updated to OFF")
                    
    
    def get_radars_status(self):
        """Route handler for /radars_status endpoint"""
        all_status: Dict[str, RadarStatus] = {}
        
        if self.radars_manager:
            with self.radars_manager._radars_lock:
                for radar_id, radar in self.radars_manager.radars.items():
                    if radar and hasattr(radar, 'get_tracks'):
                        # Get radar health status (returns bool)
                        is_active = radar.get_data_reception_health()
                        
                        # Create RadarStatus structure
                        radar_status: RadarStatus = {
                            "is_active": bool(is_active),
                            "orientation_angle": getattr(radar, 'azimuth', 0.0) or 0.0,
                            "x": self.radars_manager.get_radar_mapping(radar_id)['x'],
                            "y": self.radars_manager.get_radar_mapping(radar_id)['y']
                            }
                        
                        # Add to all_status dictionary
                        all_status[radar_id] = radar_status
                        
                        # Debug print
                        #print(f"Radar {radar_id} status: {radar_status}")
        
        # Log status to file (always log, even if empty)
        try:
            log_data = {
                "type": "status",
                "data": all_status,
                "timestamp": datetime.now().isoformat()
            }
            # Format JSON with indentation for readability
            formatted_json = json.dumps(log_data, indent=2)
            self.data_logger.info(f"STATUS:\n{formatted_json}")
        except Exception as e:
            self.data_logger.error(f"Error logging status: {e}")
        
        return jsonify(all_status)
    
    def turn_radar_on(self):
        """Route handler for /radar/off POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        
        # If radar exists in status, update it; if not, return error
        if radar_id in self.radars_manager.radars:
            self.update_radar_status(radar_id, True, 0.0)
            return jsonify({"message": f"Radar {radar_id} turned on"})
        else:
            return jsonify({"error": f"Radar {radar_id} not found"}), 404
        
    def turn_radar_off(self):
        """Route handler for /radar/off POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        
        # If radar exists in status, update it; if not, return error
        if radar_id in self.radars_manager.radars:
            self.update_radar_status(radar_id, False, 0.0)
            return jsonify({
                "message": f"Radar {radar_id} turned off"
            })
        else:
            return jsonify({"error": f"Radar {radar_id} not found"}), 404
    
    def stop_server(self):
        """Stop the server (for cleanup)"""
        # Implement shutdown logic if needed
        pass

## Example usage:
#if __name__ == "__main__":
#    import time
#    import math
#    
#    # Create server instance
#    server = RadarTracksServer(port=1337)
#    
#    # Start the server
#    server.start_server()
#    
#    # Initialize radar status
#    server.update_radar_status(
#        radar_id="radar1",
#        is_active=True,
#        orientation_angle=45.0
#    )
#    
#    # Simulate a track moving closer
#    initial_range = 100.0  # Start at 100m
#    final_range = 5.0     # End at 5m
#    samples = 200
#    
#    # Calculate range decrease per step
#    range_step = (initial_range - final_range) / samples
#    
#    # Simulate small azimuth oscillation using sine wave
#    base_azimuth = 45.0
#    
#    try:
#        # Generate 200 samples
#        for i in range(samples):
#            # Calculate current range
#            current_range = initial_range - (range_step * i)
#            
#            # Add small sine wave variation to azimuth (-5 to +5 degrees)
#            azimuth_variation = 5 * math.sin(i * 2 * math.pi / 50)  # Complete oscillation every 50 samples
#            current_azimuth = base_azimuth + azimuth_variation
#            
#            # Ensure azimuth stays in 0-359 range
#            current_azimuth = current_azimuth % 360
#            
#            # Update track data
#            server.update_radar_data(
#                radar_id="radar1",
#                track_id=1,
#                azimuth=current_azimuth,
#                range_meters=current_range
#            )
#            
#            # Simulate radar status changes
#            # Radar orientation follows a slower oscillation pattern
#            orientation_angle = 70
#            
#            # Simulate radar going active/inactive every 50 samples
#            is_active = (i % 50) < 40  # Active for 40 samples, inactive for 10 samples
#            
#            # Update radar status
#            server.update_radar_status(
#                radar_id="radar1",
#                is_active=is_active,
#                orientation_angle=orientation_angle   # Keep angle in 0-359 range
#            )
#            
#            # Add a second radar with different pattern
#            second_orientation = 190 
#            server.update_radar_status(
#                radar_id="radar2",
#                is_active=not is_active,  # Opposite active status of radar1
#                orientation_angle=second_orientation 
#            )
#            
#            # Sleep for a short time to simulate real-time updates
#            time.sleep(3)  # Update every 3 seconds
#            
#            # Print update (optional)
#            print(f"Sample {i+1}/200:")
#            print(f"  Radar1: Active={is_active}, Orientation={orientation_angle:.1f}°")
#            print(f"  Radar2: Active={not is_active}, Orientation={second_orientation:.1f}°")
#            
#        print("Simulation completed")
#        
#        # Keep server running
#        while True:
#            time.sleep(1)
#            
#    except KeyboardInterrupt:
#        server.stop_server()