import socket
import threading
from typing import Optional, Callable
import traceback

from tracker_algo.tracker import TrackerManager_3D
from radar_frame_parser import RadarFrameParser


class TrackerProcess:
    """
    A tracker process that listens on the TCP data port for a radar
    and passes data bytes to the tracker algo frame parser.
    """
    
    def __init__(self, radar_id: str, host: str, tcp_port: int, frame_period: float,
                 on_tracked_targets: Optional[Callable[[str, list], None]]):
        """
        Initialize a TrackerProcess.
        
        Args:
            radar_id: Unique identifier for the radar
            host: Host address to connect to
            tcp_port: TCP port number for data stream
            frame_period: Frame period in seconds (default: 0.05 for 20fps)
            on_detections: Optional callback function(radar_id, detections, frame_number)
        """
        self.radar_id = radar_id
        self.host = host
        self.tcp_port = tcp_port
        self.frame_period = frame_period
        self.on_tracked_targets = on_tracked_targets
        
        self._data_buffer = bytearray()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self.frame_parser = RadarFrameParser(radar_id, frame_period, doppler_threshold=0.1, range_threshold=0.1)
        self.tracker = TrackerManager_3D()

    def _process_frame(self, data_buffer: bytearray) -> bool:
        detections, frame_number = self.frame_parser.parse_frame_from_buffer(data_buffer)
                            
        if detections is None:
            return False  # No complete frame yet

        frame_time = frame_number * self.frame_period
        
        # Process the detections
        self.tracker.update(detections, 0, frame_time)
        if len(self.tracker.tracks) > 0:
            self.on_tracked_targets(self.radar_id, self.tracker.tracks[0])

        return True
    
    def start(self) -> None:
        """Start the tracker process in a separate thread"""
        if self._running:
            print(f"TrackerProcess for radar {self.radar_id} is already running")
            return
        
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"TrackerProcess-{self.radar_id}",
            daemon=True
        )
        self._thread.start()
        print(f"TrackerProcess started for radar {self.radar_id} on {self.host}:{self.tcp_port}")
    
    def stop(self) -> None:
        """Stop the tracker process"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print(f"TrackerProcess stopped for radar {self.radar_id}")
    
    def _run(self) -> None:
        """Main loop that connects to TCP port and processes frames"""
        while not self._stop_event.is_set():
            try:
                with socket.create_connection((self.host, self.tcp_port), timeout=5) as radar_data_socket:
                    radar_data_socket.settimeout(1.0)
                    
                    while not self._stop_event.is_set():
                        try:
                            # Receive data chunks
                            chunk = radar_data_socket.recv(4096)
                            self._process_frame(chunk)
                        
                        except socket.timeout:
                            continue
                        except Exception as e:
                            print(f"Error receiving data for radar {self.radar_id}: {e}")
                            print(traceback.format_exc())
                            break
                            
            except socket.timeout:
                print(f"Connection timeout for radar {self.radar_id}, retrying...")
                continue
            except Exception as e:
                print(f"Connection error for radar {self.radar_id}: {e}, retrying...")
                if not self._stop_event.is_set():
                    self._stop_event.wait(1)  # Wait before retrying
    
    def is_running(self) -> bool:
        """Check if the tracker process is running"""
        return self._running and self._thread is not None and self._thread.is_alive()

