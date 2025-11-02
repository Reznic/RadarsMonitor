import socket
import threading
import struct
from typing import Optional, Callable
from datetime import datetime
import traceback

from tracker_algo.tracker import TrackerManager_3D


# Magic word for frame detection
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAX_DATA_BUFFER = 100000


def parse_frame_header(byte_data):
    """Parse the frame header from byte data"""
    header_format = 'Q8I'
    header_size = struct.calcsize(header_format)
    header = struct.unpack(header_format, byte_data[:header_size])
    return {
        'magic': header[0],
        'version': header[1],
        'total_packet_len': header[2],
        'platform': header[3],
        'frame_number': header[4],
        'cpu_cycles': header[5],
        'num_detected_obj': header[6],
        'num_tlvs': header[7],
        'sub_frame_number': header[8],
        'header_length': header_size
    }


def parse_detections(tlv1_payload, tlv7_payload, num_points, frame_num, frame_period, radar_id, doppler_threshold=0.1, range_threshold=0.1):
    """Parse detections from TLV payloads"""
    detections = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    for i in range(num_points):
        try:
            p_offset = i * 16
            x, y, z, doppler = struct.unpack('<ffff', tlv1_payload[p_offset:p_offset+16])
        except Exception:
            print(f'detection lost')
            continue
        
        s_offset = i * 4
        # skip snr data if it not exists
        try:
            snr, noise = struct.unpack('<HH', tlv7_payload[s_offset:s_offset+4])
        except Exception:
            snr = -1
            noise = 0
        
        range_val = (x**2 + y**2 + z**2)**0.5
        if abs(doppler) < doppler_threshold or range_val < range_threshold:
            continue  # Skip static detections

        detections.append({
            'curr_timestamp': current_time,
            'timestamp': frame_period * frame_num,
            'radar_id': radar_id,
            'frame_number': frame_num,
            'x': x,
            'y': y,
            'z': z,
            'range': range_val,
            'doppler': doppler,
            'snr': snr,
            'noise': noise
        })
    return detections


def parse_frame_from_buffer(data_buffer: bytearray, frame_period: float, radar_id: str):
    """
    Parse a frame from the data buffer.
    Returns (detections, frame_number) or (None, None) if no complete frame found.
    """
    magic_idx = data_buffer.find(MAGIC_WORD)
    
    if magic_idx == -1 or len(data_buffer) < magic_idx + 40:
        return None, None
    
    header = parse_frame_header(data_buffer[magic_idx:])
    offset = magic_idx + header['header_length']
    
    # Check if we have the complete packet
    if magic_idx + header['total_packet_len'] > len(data_buffer):
        return None, None
    
    detections = []
    tlv1_payload = None
    tlv7_payload = None

    # Parse TLVs
    for _ in range(header['num_tlvs']):
        if offset + 8 > len(data_buffer):
            break
        tlv_type, tlv_len = struct.unpack('<II', data_buffer[offset:offset + 8])
        offset = offset + 8
        tlv_data = data_buffer[offset:offset + tlv_len]
        offset = offset + tlv_len

        if tlv_type == 1:
            tlv1_payload = tlv_data
        elif tlv_type == 7:
            tlv7_payload = tlv_data
    
    # Parse detections
    if tlv1_payload and tlv7_payload:
        detections = parse_detections(tlv1_payload, tlv7_payload, header['num_detected_obj'],
                                     header['frame_number'], frame_period, radar_id)
    elif tlv1_payload:
        detections = parse_detections(tlv1_payload, None, header['num_detected_obj'],
                                    header['frame_number'], frame_period, radar_id)
    
    # Remove processed data from buffer (everything up to offset)
    del data_buffer[:offset]
    
    # Clear buffer if it gets too large
    if len(data_buffer) > MAX_DATA_BUFFER:
        print(f'##########buffer_deleted!!!######## for radar {radar_id}')
        data_buffer.clear()
    
    return detections, header['frame_number']


class TrackerProcess:
    """
    A tracker process that listens on the TCP data port for a radar
    and passes data bytes to the tracker algo frame parser.
    """
    
    def __init__(self, radar_id: str, host: str, tcp_port: int, frame_period: float,
                 on_tracked_targets: Optional[Callable[[str, list], None]] = None):
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
        self.tracker = TrackerManager_3D()
    
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
                    print(f"TrackerProcess connected to {self.host}:{self.tcp_port} for radar {self.radar_id}")
                    
                    while not self._stop_event.is_set():
                        try:
                            # Receive data chunks
                            chunk = radar_data_socket.recv(4096)
                            
                            # Add to buffer
                            self._data_buffer.extend(chunk)
                            
                            # Try to parse frames from buffer (may contain multiple frames)
                            while True:
                                detections, frame_number = parse_frame_from_buffer(
                                    self._data_buffer, 
                                    self.frame_period, 
                                    self.radar_id
                                )
                                
                                if detections is None:
                                    break  # No complete frame yet
                                
                                print(f"{len(detections)} detections")
                                # Process the detections
                                self.tracker.update(detections, 0, frame_number*self.frame_period)
                                if self.on_tracked_targets and len(self.tracker.tracks) > 0:
                                    self.on_tracked_targets(self.radar_id, self.tracker.tracks[0])
                        
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

