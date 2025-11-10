from flask import Flask, jsonify
from typing import Dict, List, TypedDict
import threading

class TrackData(TypedDict):
    track_id: int
    azimuth: float
    range: float

class RadarStatus(TypedDict):
    is_active: bool
    orientation_angle: float

class RadarTracksServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5000, radars_manager=None):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        # Dictionary to store radar data: radar_id -> TrackData
        self.radar_tracks: Dict[str, TrackData] = {}
        # Dictionary to store radar status: radar_id -> RadarStatus
        self.radar_status: Dict[str, RadarStatus] = {}
        # Reference to RadarsManager instance
        self.radars_manager = radars_manager
        
        # Register routes
        self.app.route('/tracks', methods=['GET'])(self.get_tracks)
        self.app.route('/radars_status', methods=['GET'])(self.get_radars_status)
        
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
        all_tracks = {}
        
        if self.radars_manager:
            # Get tracks from all radars
            with self.radars_manager._radars_lock:
                for radar_id, radar in self.radars_manager.radars.items():
                    if radar and hasattr(radar, 'get_tracks'):
                        radar_tracks = radar.get_tracks()
                        if radar_tracks:
                            all_tracks.update(radar_tracks)
        
        return jsonify(all_tracks)
    
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
        all_status = {}
        
        if self.radars_manager:
            # Get tracks from all radars
            with self.radars_manager._radars_lock:
                for radar_id, radar in self.radars_manager.radars.items():
                    if radar and hasattr(radar, 'get_tracks'):
                        radar_status = radar.get_and_reset_health()  
                        if radar_status:
                            all_status.update(radar_status)
        return jsonify(all_status)
    
    def stop_server(self):
        """Stop the server (for cleanup)"""
        # Implement shutdown logic if needed
        pass

# Example usage:
if __name__ == "__main__":
    import time
    import math
    
    # Create server instance
    server = RadarTracksServer(port=5000)
    
    # Start the server
    server.start_server()
    
    # Initialize radar status
    server.update_radar_status(
        radar_id="radar1",
        is_active=True,
        orientation_angle=45.0
    )
    
    # Simulate a track moving closer
    initial_range = 100.0  # Start at 100m
    final_range = 5.0     # End at 5m
    samples = 200
    
    # Calculate range decrease per step
    range_step = (initial_range - final_range) / samples
    
    # Simulate small azimuth oscillation using sine wave
    base_azimuth = 45.0
    
    try:
        # Generate 200 samples
        for i in range(samples):
            # Calculate current range
            current_range = initial_range - (range_step * i)
            
            # Add small sine wave variation to azimuth (-5 to +5 degrees)
            azimuth_variation = 5 * math.sin(i * 2 * math.pi / 50)  # Complete oscillation every 50 samples
            current_azimuth = base_azimuth + azimuth_variation
            
            # Ensure azimuth stays in 0-359 range
            current_azimuth = current_azimuth % 360
            
            # Update track data
            server.update_radar_data(
                radar_id="radar1",
                track_id=1,
                azimuth=current_azimuth,
                range_meters=current_range
            )
            
            # Simulate radar status changes
            # Radar orientation follows a slower oscillation pattern
            orientation_angle = 70
            
            # Simulate radar going active/inactive every 50 samples
            is_active = (i % 50) < 40  # Active for 40 samples, inactive for 10 samples
            
            # Update radar status
            server.update_radar_status(
                radar_id="radar1",
                is_active=is_active,
                orientation_angle=orientation_angle   # Keep angle in 0-359 range
            )
            
            # Add a second radar with different pattern
            second_orientation = 190 
            server.update_radar_status(
                radar_id="radar2",
                is_active=not is_active,  # Opposite active status of radar1
                orientation_angle=second_orientation 
            )
            
            # Sleep for a short time to simulate real-time updates
            time.sleep(3)  # Update every 3 seconds
            
            # Print update (optional)
            print(f"Sample {i+1}/200:")
            print(f"  Radar1: Active={is_active}, Orientation={orientation_angle:.1f}°")
            print(f"  Radar2: Active={not is_active}, Orientation={second_orientation:.1f}°")
            
        print("Simulation completed")
        
        # Keep server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        server.stop_server()