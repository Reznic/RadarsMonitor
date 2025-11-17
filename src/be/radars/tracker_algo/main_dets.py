import serial
import time
import struct
import numpy as np
import matplotlib.pyplot as plt
from tracker import *
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import cv2
import shutil

record_results = True
record_only_mode = True

# ---------------------- init write results to CSV ----------------------
if record_results:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder_name = timestamp
    path2records_folder = Path(f"records_human_detection/{exp_folder_name}")

    os.makedirs(path2records_folder, exist_ok=True)
    os.path.join(path2records_folder,"det.csv")
    detections_file = open(os.path.join(path2records_folder,"det.csv"), mode='w', newline='')
    # tracks_file = open(f"records\\tracks_{timestamp}.csv", mode='w', newline='')

    detections_writer = csv.writer(detections_file)
    # tracks_writer = csv.writer(tracks_file)

    # Write headers
    detections_writer.writerow(["curr_timestamp","timestamp", "Radar id", "Frame number", "x", "y", "z", "doppler", "snr","noise", "range"])
    detections_file.flush()
    # detections_writer.writerow(["timestamp", "range", "azimuth", "elevation", "doppler"])

    # tracks_writer.writerow(["timestamp", "id", "x", "y", "vx", "vy"])

    # Create video filename with timestamp
    camera_frames_folder_name = "camera_frames"
    camera_frames_folder_path = os.path.join(path2records_folder, camera_frames_folder_name)
    os.makedirs(Path(camera_frames_folder_path), exist_ok=True)


    # Setup video capture and writer
    camera = cv2.VideoCapture('/dev/video2')  # 0 = default camera
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ----------------------  Config ----------------------
# CFG_FILE =  r"C:\Users\admin\PycharmProjects\shula\.venv\dopp_20_range_100_last.cfg"
# CFG_FILE =  r"C:\Users\admin\PycharmProjects\shula\.venv\dopp_20_range_100_last_0_1.cfg"
# CFG_FILE =  r"C:\Users\admin\PycharmProjects\shula\.venv\dopp_20_range_100_last_0_1_144chirps_10cfar_4noise.cfg"

# CFG_FILE =  Path("./profile_2025_05_08T07_14_46_480.cfg")
# CFG_FILE =  Path("./profile_2025_05_08T07_14_46_480_20fps.cfg")
# CFG_FILE =  Path("./profile_2025_05_08T07_14_46_480_20fps_diff_freq.cfg")
CFG_FILE =  Path("./profile_humans.cfg")
# CFG_FILE =  Path("./profile_humans_for_tests.cfg")


# deploiment mode
# CFG_FILE =  Path("./deploiment_mode/profile_2025_09_01T06_05_23_904.cfg")
# CFG_FILE =  Path("./deploiment_mode/profile_2025_09_02T07_14_02_302_20m.cfg")
# CFG_FILE =  Path("./deploiment_mode/profile_20m.cfg")
# CFG_FILE =  Path("./profile_20m_high_dopp_res_5Hz_modified.cfg")


# CFG_FILE =  Path("./profile_2025_05_08T07_14_46_480_2D.cfg")

if record_results:
    # copy config file to record folder
    shutil.copy(CFG_FILE, path2records_folder)

if os.name == 'nt':
    # using the booster
    # CONFIG_PORT = "COM8"
    # DATA_PORT = "COM7"
    # using the j5 direct to the sensor
    CONFIG_PORT = "COM9"
    DATA_PORT = "COM10"
else:
    # CONFIG_PORT = "/dev/ttyACM0"
    # DATA_PORT = "/dev/ttyACM1"
    # CONFIG_PORT = "/dev/ttyUSB0"
    # DATA_PORT = "/dev/ttyUSB1"
    # CONFIG_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A58E4-if00-port0"
    # DATA_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A58E4-if01-port0"
    CONFIG_PORT_rdr0 =  "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A5BCC-if00-port0"
    DATA_PORT_rdr0 =    "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A5BCC-if01-port0"
    CONFIG_PORT_rdr1 =  "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A58E4-if00-port0"
    DATA_PORT_rdr1 =    "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_016A58E4-if01-port0"
    CONFIG_PORTs = [CONFIG_PORT_rdr0, CONFIG_PORT_rdr1]
    DATA_PORTs = [DATA_PORT_rdr0, DATA_PORT_rdr1]
    
BAUDRATE_CONFIG = 115200
BAUDRATE_DATA = 921600
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
DATA_BUFFER_0 = b'' # Buffer to store incomplete frame data - radar 0
DATA_BUFFER_1 = b'' # Buffer to store incomplete frame data - radar 1
DATA_BUFFER = [DATA_BUFFER_0, DATA_BUFFER_1]
MAX_DATA_BUFFER = 100000

# Tracker params
# show_only_Trucks = True
show_only_Trucks = False

show_only_classified = True
# show_only_classified = False

thr_num_assoc4class_car = 4


dist_from_road = 15 # meters
# ---------------------- Serial Setup ----------------------
def send_config(config_file, ser_config):
    freqs = []
    frame_period = None
    with open(config_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() and not line.startswith('%'):
            if line.startswith('profileCfg'):
                # calc bandwidth and change the start freq of every radar to be different
                line_parts = line.split()
                slope = float(line_parts[8])
                ramp_end_time = float(line_parts[5])
                bandwidth = int(np.ceil(slope * ramp_end_time + 100))
            if line.startswith('sensorStart'):
                continue
            for i_rdr in range(len(ser_config)):
                if line.startswith('profileCfg'):
                    freq_i_rdr = float(line_parts[2]) + (bandwidth * i_rdr)/1000
                    if len(freqs) < i_rdr + 1:
                        freqs.append(freq_i_rdr)
                    else:
                        freqs[i_rdr] = freq_i_rdr
                    line_parts[2] = str(freq_i_rdr)
                    line = ' '.join(line_parts)

                ser_config[i_rdr].write((line.strip() + '\n').encode())
            if line.startswith('frameCfg'):
                frame_period = float(line.split()[5])
            time.sleep(0.01)
    return frame_period/1000, freqs

def connect_serial():
    ser_config = []
    ser_data = []
    for i_rdr in range(len(CONFIG_PORT)):
        try:
            ser_config.append(serial.Serial(CONFIG_PORT[i_rdr], BAUDRATE_CONFIG, timeout=0.5))
            ser_data.append(serial.Serial(DATA_PORT[i_rdr], BAUDRATE_DATA, timeout=0.5))
        except:
            pass
    return ser_config, ser_data

def stop_radar(ser_config):
    """
    Stop all Radars
    """
    for i_rdr in range(len(ser_config)):
        ser_config[i_rdr].write(('sensorStop'+ '\n').encode())
    time.sleep(0.03)

def start_radar(ser_config):
    """
    Start all Radars
    """
    for i_rdr in range(len(ser_config)):
        ser_config[i_rdr].write(('sensorStart' + '\n').encode())
    time.sleep(0.03)
# ---------------------- Header Parsing ----------------------
def parse_frame_header(byte_data):
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

# ---------------------- TLV Parsing ----------------------
def parse_detections(tlv1_payload, tlv7_payload, num_points, frame_num, frame_period, radar_id, doppler_threshold=0.1 , range_threshold=0.1):
    detections = []
    current_time = timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    for i in range(num_points):
        try:
            p_offset = i * 16
            x, y, z, doppler = struct.unpack('<ffff', tlv1_payload[p_offset:p_offset+16])
        except:
            print(f'detection lost')
            continue
        s_offset = i * 4
        # skip snr data if it not exists
        try:
            snr, noise = struct.unpack('<HH', tlv7_payload[s_offset:s_offset+4])
        except:
            snr = -1
        range_val = np.sqrt(x**2+y**2+z**2)
        if abs(doppler) < doppler_threshold or range_val<range_threshold:
            continue  # Skip static detections

        detections.append({
            'curr_timestamp': current_time,
            'timestamp': frame_period*frame_num,
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

# ---------------------- Frame Reader ----------------------
def read_frame(ser_data, frame_period, i_rdr):
    global DATA_BUFFER
    global MAX_DATA_BUFFER
    buffer = ser_data.read(ser_data.in_waiting)
    DATA_BUFFER[i_rdr] += buffer
    # print('data buffer length - ',len(DATA_BUFFER))
    magic_idx = DATA_BUFFER[i_rdr].find(MAGIC_WORD)

    if magic_idx == -1 or len(DATA_BUFFER[i_rdr]) < magic_idx + 40:
        return None , None
    # print('found magic ^^^^^^^^^^^^^')
    header = parse_frame_header(DATA_BUFFER[i_rdr][magic_idx:])
    # print(f'packet len: {header["total_packet_len"]} frame num: {header["frame_number"]} sub frame: {header["sub_frame_number"]}')
    offset = magic_idx + header['header_length']
    if magic_idx + header['total_packet_len'] > len(DATA_BUFFER[i_rdr]):
        return None , None
    detections = []
    point_cloud_detections = []
    tlv1_payload = None
    tlv7_payload = None

    for _ in range(header['num_tlvs']):
        if offset + 8 > len(DATA_BUFFER[i_rdr]):
            break
        tlv_type, tlv_len = struct.unpack('<II', DATA_BUFFER[i_rdr][offset:offset + 8])
        offset = offset + 8
        tlv_data = DATA_BUFFER[i_rdr][offset : offset + tlv_len]
        offset = offset + tlv_len

        if tlv_type == 1:
            tlv1_payload = tlv_data
        elif tlv_type == 7:
            tlv7_payload = tlv_data
    DATA_BUFFER[i_rdr] = DATA_BUFFER[i_rdr][offset:]

    if tlv1_payload and tlv7_payload:
        detections = parse_detections(tlv1_payload, tlv7_payload, header['num_detected_obj'],header['frame_number'], frame_period, i_rdr)
    if tlv1_payload and not tlv7_payload:
        detections = parse_detections(tlv1_payload, None, header['num_detected_obj'],
                                      header['frame_number'],frame_period)
    if len(DATA_BUFFER[i_rdr]) > MAX_DATA_BUFFER:
        print('##########buffer_deleted!!!########')
        DATA_BUFFER[i_rdr] = b''
    return detections , header['frame_number']

# ---------------------- plot tracks ----------------------

def init_xy_plot():
    plt.ion()
    fig_main, ax_main = plt.subplots()
    # radar pos
    ax_main.scatter(0, 0, c='red', s=100, marker='x')  # Radar origin

    # dets scatter
    det_sc = ax_main.scatter([], [], s=10, c='gray', alpha=0.4, label='Detections')

    # track scatter
    trk_sc = ax_main.scatter([], [], c='blue', s=80, marker='o')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('2D Top-Down Track Map')
    plt.grid(True)
    plt.axis('auto')
    plt.xlim(-30, 30)
    plt.ylim(0, 100)
    return fig_main, ax_main, det_sc, trk_sc

def plot_tracks_xy(
    tracks,
    detections=None,
    show_id=True,
    min_assoc2show=10,
    dict_class={'n': 'None', 'c': 'Car', 'h': 'Human', 't': 'Truck'}
):
    plt.clf()

    # --- Plot detections if provided ---
    if detections:
        det_xs = [d['x'] for d in detections if d['y'] > 0]
        det_ys = [((d['y'])**2 + (d['z'])**2)**0.5 for d in detections if d['y'] > 0]
        plt.scatter(det_xs, det_ys, c='gray', s=10, alpha=0.4, label='Detections')

    # --- Plot tracks ---
    for t in tracks:
        t_class = classify_tgt4plot(t)
        if show_only_classified:
            if show_only_Trucks:
                if t_class != 't':
                    continue
            else:
                if t_class == 'n':
                    continue
        if t.assoc_dets > min_assoc2show:
            x, y, z = t.get_position()
            if y > 0:
                t2a = round(t.t2a, 2)
                plt.scatter(x, (y**2 + z**2)**0.5, c='blue', s=80, marker='o')
                if show_id:
                    plt.text(
                        x + 0.1,
                        (y**2 + z**2)**0.5 + 0.1,
                        f'ID:{t.id} class:{dict_class[t_class]} t2a:{t2a}',
                        color='black',
                        fontsize=9
                    )

    plt.scatter(0, 0, c='red', s=100, marker='x')  # Radar origin
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('2D Top-Down Track Map')
    plt.grid(True)
    plt.axis('auto')
    plt.xlim(-30, 30)
    plt.ylim(0, 100)
    plt.pause(0.01)

def save_freqs_to_file(freqs, filename="freqs.txt"):
    if freqs:  # only write if list is not empty
        with open(filename, "w") as f:
            for num in freqs:
                f.write(f"{num}\n")

# ---------------------- Main 3D----------------------
def main_3D():
    last_vid_frame_time = time.time()
    print("Connecting to radar...")
    ser_config, ser_data = connect_serial()
    # stop the radar
    stop_radar(ser_config)
    print("Sending config...")
    nub_rdrs = len(ser_config)
    frame_period, freqs = send_config(CFG_FILE, ser_config)
    save_freqs_to_file(freqs, filename=os.path.join(path2records_folder, "freqs.txt"))

    start_radar(ser_config)

    print("Reading frames...")
    if not record_only_mode:
        plt.ion()
    tracker = TrackerManager_3D(num_rdrs=nub_rdrs)
    ## video capture
    ret, frame = camera.read()
    if ret:
        last_vid_frame_time = time.time()
        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # up to milliseconds
        # Put timestamp text on the frame
        cv2.putText(frame, timestamp_now, (10, 30),  # position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # font
                    0.7,  # scale
                    (0, 255, 0),  # color (green)
                    2,  # thickness
                    cv2.LINE_AA)
        # Write the frame with overlay to video
        cv2.imwrite(f"{os.path.join(camera_frames_folder_path,timestamp_now)}.jpg", frame)
    #######
    max_proc_time = 0
    first_det_time = 0
    try:
        while True:
            start = time.perf_counter()
            current_time = time.time()
            if first_det_time == 0:
                first_det_time = time.time()

            ## video capture
            if current_time - last_vid_frame_time > frame_period/3:
                ret, frame = camera.read()
                if ret:
                    last_vid_frame_time = time.time()
                    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # up to milliseconds
                    # Put timestamp text on the frame
                    cv2.putText(frame, timestamp_now, (10, 30),  # position (x, y)
                                cv2.FONT_HERSHEY_SIMPLEX,  # font
                                0.7,  # scale
                                (0, 255, 0),  # color (green)
                                2,  # thickness
                                cv2.LINE_AA)
                    # Write the frame with overlay to video
                    cv2.imwrite(f"{os.path.join(camera_frames_folder_path, timestamp_now)}.jpg", frame)
            ####### end video capture
            for i_rdr in range(len(ser_data)):
                detections, frame_number = read_frame(ser_data[i_rdr], frame_period, i_rdr)
                if detections:
                    tracks = tracker.update(detections, i_rdr, frame_number*frame_period)
                    if not record_only_mode:
                        plot_tracks_xy(tracks,detections, min_assoc2show=5)


                    print('frame num - ',frame_number)
                    for d in detections:
                        if record_results:
                            if 'x' in d:
                                detections_writer.writerow([ d['curr_timestamp'],
                                                             d['timestamp'],
                                                             d['radar_id'],
                                                             d['frame_number'],
                                                             round(d['x'],5),
                                                             round(d['y'],5),
                                                             round(d['z'],5),
                                                             round(d['doppler'],3),
                                                             d['snr'],
                                                             d['noise'],
                                                             round(d['range'],3)
                                ])
                        else:
                            print(d)
                    detections_file.flush()
                    end = time.perf_counter()
                    cur_proc_time = end-start
                    if cur_proc_time > max_proc_time:
                        max_proc_time = cur_proc_time
                    print(f"Runtime: {cur_proc_time:.4f} seconds | {len(detections)} dets | radar_id: {i_rdr}")
                else:

                    if frame_number:
                        if len(tracker.tracks)>0:
                            tracks = tracker.update(detections, i_rdr, frame_number*frame_period)
                            if not record_only_mode:
                                plot_tracks_xy(tracks, min_assoc2show=5)
            time.sleep(0.005)

    except KeyboardInterrupt:
        # stop the radar
        stop_radar(ser_config)
        print("\nStopped by user.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        print(f"max proc time : {max_proc_time}")
        stop_radar(ser_config)
        for i_rdr in range(len(ser_config)):
            ser_config[i_rdr].close()
            ser_data[i_rdr].close()
        detections_file.close()

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_3D()
