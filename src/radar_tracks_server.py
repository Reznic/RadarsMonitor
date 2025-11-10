from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, TypedDict
import threading

class TrackData(TypedDict):
    track_id: int
    azimuth: float
    range: float

class RadarStatus(TypedDict):
    is_active: bool
    orientation_angle: float

class RadarTracksServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 1337):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        self.host = host
        self.port = port
        # Dictionary to store radar data: radar_id -> TrackData
        self.radar_tracks: Dict[str, TrackData] = {}
        # Dictionary to store radar status: radar_id -> RadarStatus
        self.radar_status: Dict[str, RadarStatus] = {}
        
        # Register routes
        self.app.route('/tracks', methods=['GET'])(self.get_tracks)
        self.app.route('/radars_status', methods=['GET'])(self.get_radars_status)
        self.app.route('/radar/on', methods=['POST'])(self.turn_radar_on)
        self.app.route('/radar/off', methods=['POST'])(self.turn_radar_off)
        
        # Create a thread for the server
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
    
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
        return jsonify(self.radar_tracks)
    
    def start_server(self):
        """Start the server in a separate thread"""
        self.server_thread.start()
    
    def run_server(self):
        """Run the Flask server"""
        self.app.run(host=self.host, port=self.port)
    
    def update_radar_status(self, radar_id: str, is_active: bool, orientation_angle: float) -> None:
        """
        Update the status for a specific radar
        
        Args:
            radar_id: Unique identifier for the radar
            is_active: Boolean indicating if radar is active
            orientation_angle: Current orientation angle in degrees (0-359)
        """
        if not 0 <= orientation_angle <= 359:
            raise ValueError("Orientation angle must be between 0 and 359 degrees")
            
        self.radar_status[radar_id] = {
            "is_active": is_active,
            "orientation_angle": orientation_angle
        }
    
    def get_radars_status(self):
        """Route handler for /radars_status endpoint"""
        return jsonify(self.radar_status)
    
    def turn_radar_on(self):
        """Route handler for /radar/on POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        
        # If radar exists in status, update it; if not, create new entry
        if radar_id in self.radar_status:
            current_angle = self.radar_status[radar_id]['orientation_angle']
            self.update_radar_status(radar_id, True, current_angle)
        else:
            # Default orientation angle if radar is new
            self.update_radar_status(radar_id, True, 0.0)
            
        return jsonify({
            "message": f"Radar {radar_id} turned on",
            "status": self.radar_status[radar_id]
        })
    
    def turn_radar_off(self):
        """Route handler for /radar/off POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        
        # If radar exists in status, update it; if not, return error
        if radar_id in self.radar_status:
            current_angle = self.radar_status[radar_id]['orientation_angle']
            self.update_radar_status(radar_id, False, current_angle)
            return jsonify({
                "message": f"Radar {radar_id} turned off",
                "status": self.radar_status[radar_id]
            })
        else:
            return jsonify({
                "error": f"Radar {radar_id} not found"
            }), 404
    
    def stop_server(self):
        """Stop the server (for cleanup)"""
        # Implement shutdown logic if needed
        pass

# Example usage:
if __name__ == "__main__":
    import time
    import math
    import os

    # Print environment
    environment = os.environ.get('ENVIRONMENT', 'unknown')
    print(f"Running in {environment.upper()} environment")

    # Create server instance
    server = RadarTracksServer(port=1337)
    
    # Start the server
    server.start_server()
    
    # Initialize eight radars evenly spaced around the circle
    num_radars = 8
    sector_size = 360 / num_radars
    radar_configs = []


    server.update_radar_status(
        radar_id="radar0",
        is_active=True,
        orientation_angle=150
    )
    
    for idx in range(num_radars):
        radar_id = f"radar{idx + 1}"
        # Aim each radar at the center of its sector (e.g., 45°, 135° for four radars)
        base_orientation = (idx * sector_size + sector_size / 2) % 360
        radar_configs.append({"id": radar_id, "base_orientation": base_orientation})
        server.update_radar_status(
            radar_id=radar_id,
            is_active=True,
            orientation_angle=base_orientation
        )

    # Simulate a track moving closer
    initial_range = 45.0  # Start at 100m
    final_range = 5.0     # End at 5m
    samples = 100
    
    # Calculate range decrease per step
    range_step = (initial_range - final_range) / samples
    
    try:
        # Endless simulation loop
        while True:
            print("Starting new simulation cycle...")

            # Generate samples for this cycle
            for i in range(samples):
                for radar_index, config in enumerate(radar_configs):
                    # Offset each radar so they do not move in lockstep
                    phase_offset = radar_index * 10
                    current_sample = (i + phase_offset) % samples

                    # Calculate current range with wrap-around so each radar sweeps repeatedly
                    current_range = initial_range - (range_step * current_sample)
                    current_range = max(final_range, current_range)

                    # Azimuth oscillates gently around each radar's base orientation
                    azimuth_variation = 5 * math.sin(
                        (i + radar_index * 12) * 2 * math.pi / 50
                    )
                    current_azimuth = (config["base_orientation"] + azimuth_variation) % 360

                    # Update track data for this radar
                    server.update_radar_data(
                        radar_id=config["id"],
                        track_id=radar_index + 1,
                        azimuth=current_azimuth,
                        range_meters=current_range
                    )

                # Sleep for a short time to simulate real-time updates
                time.sleep(0.1)  # Update every 0.1 seconds

                # Print update (optional)
                print(f"Sample {i + 1}/{samples}:")

            print("Simulation cycle completed. Restarting...")

    except KeyboardInterrupt:
        server.stop_server()
