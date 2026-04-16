import socket
import threading
import csv
import io
import os
import time
from typing import Optional, Callable
import traceback

from tracker_algo.tracker import TrackerManager_3D
from radar_frame_parser import RadarFrameParser


class TrackerProcess:
    """
    A tracker process that listens on the TCP data port for a radar
    and passes data bytes to the tracker algo frame parser.
    """

    MAX_CSV_FILE_SIZE_BYTES = 200 * 1024 * 1024
    CSV_HEADER = [
        "system_time",
        "radar_id",
        "frame_number",
        "x",
        "y",
        "z",
        "doppler",
        "snr",
        "range",
        "noise",
        "update_time_us",
    ]
    
    def __init__(self, radar_id: str, host: str, tcp_port: int, frame_period: float,
                 on_tracked_targets: Optional[Callable[[str, list], None]],
                 on_detections: Optional[Callable[[str, list, int], None]] = None):
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
        self.on_detections = on_detections
        
        self._data_buffer = bytearray()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self.frame_parser = RadarFrameParser(radar_id, frame_period, doppler_threshold=0.1, range_threshold=0.1)
        self.tracker = TrackerManager_3D()
        
        # Health flag: True when data is received on TCP socket, reset to False after each GET request
        self._data_received_flag = False
        self._health_lock = threading.Lock()
        self._csv_session_timestamp = int(time.time())
        self._csv_file_index = 0
        self._csv_file = None
        self._csv_current_size = 0
        self._csv_directory = "tracker_log"
        os.makedirs(self._csv_directory, exist_ok=True)

    def _extract_detection_value(self, detection, key: str):
        if isinstance(detection, dict):
            return detection.get(key, "")
        return getattr(detection, key, "")

    def _open_next_csv_file(self) -> None:
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()

        self._csv_file_index += 1
        filename = f"{self._csv_session_timestamp}_{self._csv_file_index}.csv"
        filepath = os.path.join(self._csv_directory, filename)
        self._csv_file = open(filepath, "w", newline="", encoding="utf-8")

        header_line = io.StringIO()
        csv.writer(header_line).writerow(self.CSV_HEADER)
        serialized_header = header_line.getvalue()
        self._csv_file.write(serialized_header)
        self._csv_file.flush()
        self._csv_current_size = len(serialized_header.encode("utf-8"))

    def _write_detections_to_csv(
        self, detections: list, frame_number: int, update_time_us: int
    ) -> None:
        if self._csv_file is None or self._csv_file.closed:
            self._open_next_csv_file()

        for detection in detections:
            row = [
                int(time.time()),
                self.radar_id,
                frame_number,
                self._extract_detection_value(detection, "x"),
                self._extract_detection_value(detection, "y"),
                self._extract_detection_value(detection, "z"),
                self._extract_detection_value(detection, "doppler"),
                self._extract_detection_value(detection, "snr"),
                self._extract_detection_value(detection, "range"),
                self._extract_detection_value(detection, "noise"),
                update_time_us,
            ]

            row_line = io.StringIO()
            csv.writer(row_line).writerow(row)
            serialized_row = row_line.getvalue()
            row_size = len(serialized_row.encode("utf-8"))

            if self._csv_current_size + row_size > self.MAX_CSV_FILE_SIZE_BYTES:
                self._open_next_csv_file()

            self._csv_file.write(serialized_row)
            self._csv_current_size += row_size

        self._csv_file.flush()

    def _process_frame(self, data_buffer: bytearray) -> bool:
        detections, frame_number = self.frame_parser.parse_frame_from_buffer(data_buffer)
                            
        if detections is None:
            return False  # No complete frame yet

        if self.on_detections:
            self.on_detections(self.radar_id, detections, frame_number)

        frame_time = frame_number * self.frame_period

        t0 = time.perf_counter_ns()
        self.tracker.update(detections, 0, frame_time)
        update_time_us = (time.perf_counter_ns() - t0) // 1000

        self._write_detections_to_csv(detections, frame_number, update_time_us)
        # Process the detections
        # tracker.tracks is a nested structure; historically we passed tracks[0]
        # (the list of active tracks for this frame). Keep that behavior, but
        # still call back with [] when empty so consumers can stop sampling.
        if self.on_tracked_targets:
            tracks_for_cb = self.tracker.tracks[0] if len(self.tracker.tracks) > 0 else []
            self.on_tracked_targets(self.radar_id, tracks_for_cb)
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
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
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
                            if chunk:
                                # Set health flag to True when data is received (before passing to frame parser)
                                with self._health_lock:
                                    self._data_received_flag = True
                                self._process_frame(chunk)
                        
                        except socket.timeout:
                            print(f"Socket timeout while receiving data for radar {self.radar_id}, continuing...")
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
    def get_and_reset_health(self) -> bool:
        """
        Get the health status (whether data was received) and reset the flag to False.
        
        Returns:
            bool: True if data was received since last check, False otherwise
        """
        with self._health_lock:
            is_healthy = self._data_received_flag
            self._data_received_flag = False  # Reset after each GET request
        return is_healthy

