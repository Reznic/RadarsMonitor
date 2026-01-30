from flask import Flask, jsonify, request
from flask_cors import CORS
from typing import Dict, TypedDict, Optional, List, Tuple
import threading
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TrackData(TypedDict):
    track_id: int
    azimuth: float
    range: float
    class_name: str

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
        # GPIO controller for MOSFET control
        self.gpio_controller: Optional["JetsonGPIOController"] = None
        try:
            from jetson_gpio_controller import JetsonGPIOController
            self.gpio_controller = JetsonGPIOController()
        except Exception as e:
            logger.warning("Failed to initialize GPIO controller; running without GPIO control: %s", e)
        
        # Register routes
        self.app.route('/tracks', methods=['GET'])(self.get_tracks)
        self.app.route('/radars_status', methods=['GET'])(self.get_radars_status)
        self.app.route('/radar/on', methods=['POST'])(self.turn_radar_on)
        self.app.route('/radar/off', methods=['POST'])(self.turn_radar_off)
        self.app.route('/radar/all/on', methods=['POST'])(self.turn_all_radars_on)
        self.app.route('/radar/all/off', methods=['POST'])(self.turn_all_radars_off)
        
        # Create a thread for the server
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)

    def _get_existing_orientation(self, radar_id: str) -> float:
        """Return the last known orientation for a radar, falling back to mapping/instance."""
        status = self.radar_status.get(radar_id)
        if status and "orientation_angle" in status:
            return status["orientation_angle"]

        if self.radars_manager:
            mapping = self.radars_manager.get_radar_mapping(radar_id)
            if mapping:
                azimuth = mapping.get("azimuth")
                if azimuth is not None:
                    return azimuth
            radar = self.radars_manager.radars.get(radar_id)
            if radar and hasattr(radar, "azimuth") and radar.azimuth is not None:
                return radar.azimuth

        return 0.0

    def _get_existing_xy(self, radar_id: str) -> Tuple[float, float]:
        status = self.radar_status.get(radar_id)
        if status:
            return float(status.get("x", 0.0)), float(status.get("y", 0.0))
        return 0.0, 0.0

    def _get_all_radar_ids(self) -> List[str]:
        radar_ids = set(self.radar_status.keys())
        if self.radars_manager:
            with self.radars_manager._radars_lock:
                radar_ids.update(self.radars_manager.radars.keys())
        return list(radar_ids)
    
    def update_radar_data(
        self,
        radar_id: str,
        track_id: int,
        azimuth: float,
        range_meters: float,
        class_name: str = "unknown",
    ) -> None:
        """
        Update the radar data for a specific radar ID
        
        Args:
            radar_id: Unique identifier for the radar
            track_id: Track ID (positive number)
            azimuth: Angle in degrees (0-359)
            range_meters: Range in meters (positive number)
            class_name: Target classification label (e.g., human/car/truck)
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
            "range": range_meters,
            "class_name": class_name,
        }
    
    def get_tracks(self):
        """Route handler for /tracks endpoint"""
        # Start with any tracks already cached (e.g., demo mode without manager)
        all_tracks = dict(self.radar_tracks)
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
                logger.info("TRACKS:\n%s", formatted_json)
            except Exception as e:
                logger.exception("Error logging tracks")
            
            # Turn on MOSFET when tracks are detected
            if self.gpio_controller:
                try:
                    self.gpio_controller.start_warning_alarm()
                except Exception as e:
                    logger.exception("Failed to turn MOSFET ON")
        else:
            # Turn off MOSFET when no tracks are detected
            if self.gpio_controller:
                try:
                    if self.gpio_controller.is_alarm_active():
                        self.gpio_controller.stop_warning_alarm()
                except Exception as e:
                    logger.exception("Failed to turn MOSFET OFF")
        
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
    
    def update_radar_status(
        self,
        radar_id: str,
        is_active: bool,
        orientation_angle: float,
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        """Update the cached radar status and optionally control the radar via manager."""
        self.radar_status[radar_id] = {
            "is_active": bool(is_active),
            "orientation_angle": orientation_angle,
            "x": x,
            "y": y,
        }

        stop_cmd = "sensorStop"
        if self.radars_manager:
            with self.radars_manager._radars_lock:
                radar = self.radars_manager.radars.get(radar_id)
                if not radar:
                    return
                if is_active:
                    radar.configure(self.radars_manager.CONFIG_RETRY_COUNT, delay=0)
                    logger.info("Radar %s status updated to ON", radar_id)
                else:
                    radar.send_command(stop_cmd)
                    logger.info("Radar %s status updated to OFF", radar_id)
                    
    
    def get_radars_status(self):
        """Route handler for /radars_status endpoint"""
        all_status: Dict[str, RadarStatus] = dict(self.radar_status)
        
        if self.radars_manager:
            with self.radars_manager._radars_lock:
                for radar_id, radar in self.radars_manager.radars.items():
                    if radar and hasattr(radar, 'get_tracks'):
                        cached_status = self.radar_status.get(radar_id)

                        # Get radar health status (returns bool)
                        is_active = cached_status.get("is_active") if cached_status else radar.get_data_reception_health()
                        
                        # Get mapping once and cache it
                        mapping = self.radars_manager.get_radar_mapping(radar_id)
                        orientation_angle = (
                            cached_status.get("orientation_angle")
                            if cached_status and cached_status.get("orientation_angle") is not None
                            else getattr(radar, 'azimuth', 0.0) or 0.0
                        )
                        
                        # Create RadarStatus structure
                        radar_status: RadarStatus = {
                            "is_active": bool(is_active),
                            "orientation_angle": orientation_angle,
                            "x": mapping.get("x", 0.0) if mapping else 0.0,
                            "y": mapping.get("y", 0.0) if mapping else 0.0
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
            logger.info("STATUS:\n%s", formatted_json)
        except Exception as e:
            logger.exception("Error logging status")
        
        return jsonify(all_status)
    
    def turn_radar_on(self):
        """Route handler for /radar/off POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        orientation_angle = self._get_existing_orientation(radar_id)
        
        # If a manager exists, validate the radar id; otherwise allow the cached status to be toggled in demo mode
        if self.radars_manager:
            if radar_id in self.radars_manager.radars:
                self.update_radar_status(radar_id, True, orientation_angle)
                return jsonify({"message": f"Radar {radar_id} turned on"})
            return jsonify({"error": f"Radar {radar_id} not found"}), 404

        self.update_radar_status(radar_id, True, orientation_angle)
        return jsonify({"message": f"Radar {radar_id} turned on (cached)"})
        
    def turn_radar_off(self):
        """Route handler for /radar/off POST endpoint"""
        data = request.get_json()
        if not data or 'radar_id' not in data:
            return jsonify({"error": "radar_id is required"}), 400
            
        radar_id = data['radar_id']
        orientation_angle = self._get_existing_orientation(radar_id)
        
        # If a manager exists, validate the radar id; otherwise allow the cached status to be toggled in demo mode
        if self.radars_manager:
            if radar_id in self.radars_manager.radars:
                self.update_radar_status(radar_id, False, orientation_angle)
                return jsonify({
                    "message": f"Radar {radar_id} turned off"
                })
            return jsonify({"error": f"Radar {radar_id} not found"}), 404

        self.update_radar_status(radar_id, False, orientation_angle)
        return jsonify({"message": f"Radar {radar_id} turned off (cached)"})

    def turn_all_radars_on(self):
        """Route handler for /radar/all/on POST endpoint"""
        radar_ids = self._get_all_radar_ids()
        if not radar_ids:
            return jsonify({"message": "No radars available", "updated": []})

        updated: List[str] = []
        for radar_id in radar_ids:
            orientation_angle = self._get_existing_orientation(radar_id)
            x, y = self._get_existing_xy(radar_id)
            self.update_radar_status(radar_id, True, orientation_angle, x=x, y=y)
            updated.append(radar_id)

        return jsonify({"message": "All radars turned on", "updated": updated})

    def turn_all_radars_off(self):
        """Route handler for /radar/all/off POST endpoint"""
        radar_ids = self._get_all_radar_ids()
        if not radar_ids:
            return jsonify({"message": "No radars available", "updated": []})

        updated: List[str] = []
        for radar_id in radar_ids:
            orientation_angle = self._get_existing_orientation(radar_id)
            x, y = self._get_existing_xy(radar_id)
            self.update_radar_status(radar_id, False, orientation_angle, x=x, y=y)
            updated.append(radar_id)

        return jsonify({"message": "All radars turned off", "updated": updated})
    
    def stop_server(self):
        """Stop the server (for cleanup)"""
        # Cleanup GPIO controller
        if self.gpio_controller:
            try:
                self.gpio_controller.cleanup()
            except Exception as e:
                logger.exception("Error cleaning up GPIO controller")
