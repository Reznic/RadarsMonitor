import struct
from datetime import datetime

class RadarFrameParser:
    # Magic word for frame detection
    MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
    HEADER_FORMAT = 'Q8I'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(self, radar_id: str, frame_period: float, doppler_threshold: float, range_threshold: float):
        self.radar_id = radar_id
        self.frame_period = frame_period
        self.doppler_threshold = doppler_threshold
        self.range_threshold = range_threshold
        self.data_buffer = bytearray()

    def parse_frame_header(self, byte_data):
        """Parse the frame header from byte data"""
        header = struct.unpack(self.HEADER_FORMAT, byte_data[:self.HEADER_SIZE])
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
            'header_length': self.HEADER_SIZE
        }

    def parse_detections(self, tlv1_payload, tlv7_payload, num_points, frame_num):
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
            if abs(doppler) < self.doppler_threshold or abs(range_val) < self.range_threshold:
                continue  # Skip static detections

            detections.append({
                'curr_timestamp': current_time,
                'timestamp': self.frame_period * frame_num,
                'radar_id': self.radar_id,
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

    def parse_frame_from_buffer(self, buffer: bytearray):
        """
        Parse a frame from the data buffer.
        Returns (detections, frame_number) or (None, None) if no complete frame found.
        """
        self.data_buffer.extend(buffer)
        magic_idx = self.data_buffer.find(self.MAGIC_WORD)
        
        if magic_idx == -1 or len(self.data_buffer) < magic_idx + 40:
            return None, None
        
        header = self.parse_frame_header(self.data_buffer[magic_idx:])
        offset = magic_idx + header['header_length']
        
        # Check if we have the complete packet
        if magic_idx + header['total_packet_len'] > len(self.data_buffer):
            return None, None
        
        detections = []
        tlv1_payload = None
        tlv7_payload = None

        # Parse TLVs
        for _ in range(header['num_tlvs']):
            if offset + 8 > len(self.data_buffer):
                break
            tlv_type, tlv_len = struct.unpack('<II', self.data_buffer[offset:offset + 8])
            offset = offset + 8
            tlv_data = self.data_buffer[offset:offset + tlv_len]
            offset = offset + tlv_len

            if tlv_type == 1:
                tlv1_payload = tlv_data
            elif tlv_type == 7:
                tlv7_payload = tlv_data
        
        # Parse detections
        if tlv1_payload and tlv7_payload:
            detections = self.parse_detections(tlv1_payload, tlv7_payload, header['num_detected_obj'],
                                        header['frame_number'])
        elif tlv1_payload:
            detections = self.parse_detections(tlv1_payload, None, header['num_detected_obj'],
                                        header['frame_number'])
        
        # Remove processed data from buffer (everything up to offset)
        self.data_buffer = self.data_buffer[offset:]
        
        return detections, header['frame_number']

