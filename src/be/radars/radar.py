import os
import math
import time
import numpy as np
import logging
from typing import Dict, Optional, Callable, TypedDict
from radar_node_client import RadarNodeClient
from tracker_process import TrackerProcess

logger = logging.getLogger(__name__)

class TrackData(TypedDict):
    track_id: int
    azimuth: float
    range: float
    
class Radar:
    """Represents a single radar with its adapter client, tracker, and configuration"""
    CONFIG_RETRY_COUNT = 3
    
    def __init__(self, radar_id: str, host: str, http_port: int, tcp_port: int, 
                 radar_config: 'RadarConfiguration', azimuth: Optional[float] = None,
                 on_tracked_targets_callback: Optional[Callable] = None,
                 on_detections_callback: Optional[Callable] = None ):
        self.radar_id = radar_id
        self.config = radar_config
        self.host = host  # ip address of the radar node server
        self.http_port = http_port  # http port of the radar node server
        self.tcp_port = tcp_port  # tcp port for detections data stream
        self.on_tracked_targets_callback = on_tracked_targets_callback
        self.on_detections_callback = on_detections_callback
        # installation azimuth angle of the radar
        if azimuth is None:
            self.azimuth = None
            self.rotation_matrix = None
        else:
            self.set_azimuth(azimuth)

        self.client = RadarNodeClient(radar_id, host, config_port=http_port, data_port=tcp_port)
        self.tracker_process = TrackerProcess(
            radar_id=self.radar_id,
            host=self.host,
            tcp_port=self.tcp_port,
            frame_period=1.0 / self.config.fps,
            on_tracked_targets=self.on_tracks_detect,
            on_detections=self.on_detections
        )

        # Dictionary to store radar data: radar_id -> TrackData
        self.radar_tracks: Dict[str, TrackData] = {}
        
    def start(self) -> bool:
        if self.configure(self.CONFIG_RETRY_COUNT):
            self.tracker_process.start()
            return True
        else:
            return False

    def set_azimuth(self, azimuth: float) -> None:
        self.azimuth = azimuth
        azimuth_rad = math.radians(azimuth)
        self.rotation_matrix = np.array([
            [math.cos(azimuth_rad), -math.sin(azimuth_rad)],
            [math.sin(azimuth_rad),  math.cos(azimuth_rad)]
        ])
    
    def configure(self, retries: int, delay: int = 70) -> bool:
        """Configure the radar with retry logic"""
        config = self.config.get_config()
        #wait for 70 seconds
        time.sleep(delay)
        logger.info("Sending sensor start for %s", self.radar_id)
        for attempt in range(1, retries + 1):
            try:
                response = self.client.send_command(config)
                if response == '"sensorStart"'  or "Done" in response:
                    if attempt > 1:
                        logger.info("Radar %s configured successfully on attempt %s", self.radar_id, attempt)
                    else:
                        logger.info("Radar %s configured successfully", self.radar_id)
                    return True
                else:
                    logger.warning(
                        "Error configuring radar %s (attempt %s/%s): %s",
                        self.radar_id,
                        attempt,
                        retries,
                        response,
                    )
                    if attempt < retries:
                        logger.info("Retrying configuration for radar %s...", self.radar_id)
                        time.sleep(10)
            except Exception as e:
                logger.exception(
                    "Exception configuring radar %s (attempt %s/%s)",
                    self.radar_id,
                    attempt,
                    retries,
                )
                if attempt < retries:
                    logger.info("Retrying configuration for radar %s...", self.radar_id)
        
        # All retries failed
        logger.error("Failed to configure radar %s after %s attempts", self.radar_id, retries)
        return False
    
    def send_command(self, command: str) -> str:
        """Send a command to the radar"""
        return self.client.send_command(command)
    
    def health(self) -> bool:
        """Check if the radar is healthy"""
        return self.client.health()
    
    def get_data_reception_health(self) -> bool:
        """
        Get the health status based on whether the tracker process is receiving data.
        
        Returns:
            bool: True if data was received on TCP socket since last check, False otherwise
        """
        if self.tracker_process:
            return self.tracker_process.get_and_reset_health()
        return False
    def rotate_track(self, track) -> None:
        """
        Rotate a position (x, y) using this radar's rotation matrix.
        Uses numpy matrix multiplication for efficient computation.
        
        Args:
            track: The track to rotate
        """
        x, y, _ = track.get_position()
        if self.rotation_matrix is None:
            track.x, track.y = x, y
        
        else:
            position_vector = np.array([x, y])
            rotated_vector = self.rotation_matrix @ position_vector
            track.x, track.y = rotated_vector[0], rotated_vector[1]
    
    def stop(self) -> None:
        """Stop the radar's tracker process and client"""
        if self.tracker_process:
            self.tracker_process.stop()
        if self.client:
            self.client.stop()

    def on_tracks_detect(self, radar_id, tracks):
        tracks = tracks or []
        # Defensive: some tracker implementations store tracks as list[list[Track]].
        # If we accidentally receive that shape, unwrap the first element.
        if tracks and isinstance(tracks[0], list):
            tracks = tracks[0]
        classified_tracks = [
            track for track in tracks
            if track.target_class and track.target_class != 'n'
        ]
        if classified_tracks:
            class_map = {'c': 'car', 'h': 'human', 't': 'truck', 'n': 'none'}
            for track in classified_tracks:
                azimuth_base = self.azimuth if self.azimuth is not None else 0.0
                self.radar_tracks[radar_id] = {
                    "track_id": track.id,
                    "azimuth": -track.median_az + azimuth_base,
                    "range": track.range_val,
                    "class_name": class_map.get(track.target_class, 'unknown'),
                }
        else:
            self.radar_tracks.pop(radar_id, None)

        if self.on_tracked_targets_callback:
            self.on_tracked_targets_callback(radar_id, classified_tracks)

    def on_detections(self, radar_id, detections, frame_number):
        if self.on_detections_callback:
            self.on_detections_callback(radar_id, detections, frame_number)
    
    def get_tracks(self) -> Dict[str, TrackData]:
        """
        Get a copy of the radar tracks dictionary and clear the original.
        
        Returns:
            Dict[str, TrackData]: Copy of the tracks before clearing
        """
        tracks_copy = self.radar_tracks.copy()
        self.radar_tracks.clear()
        return tracks_copy

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
