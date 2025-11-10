from collections import defaultdict
import matplotlib.pyplot as plt
from tracker import *  # import your classes
import csv
import os
import mplcursors
import pandas as pd
from plot_scripts.plot_range_vs_time_from_csv import plot_range_vs_time_from_csv ,extract_experiment_name, get_latest_detection_csv, plot_3d_detections_with_tracks
from plot_scripts.get_records import get_latest_entry
from datetime import datetime, timedelta
from pathlib import Path
from find_closest_image import find_closest_image
import re

fps = 10
frame_period = 1/fps
RT_xy_plot = False
# RT_xy_plot = True
range_doppler_plot = not RT_xy_plot

show_only_classified = True
# show_only_classified = False

# show_only_Trucks = True
show_only_Trucks = False

thr_num_assoc4class_car = 4
thr_num_assoc4class_human = 1

# show_xyz = True
show_xyz = False

# show_only_trk_id = [21,10,7,13,15,20,27,25,28]
show_only_trk_id = []
dict_class = {'n': 'None', 'c': 'Car', 'h': 'Human', 't': 'Truck'}
min_assoc2plot = 2
dist_from_road = 7.5  # meters

def get_fps(folder_path):
    frame_period = None
    config_file = None
    default_fps = 10
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".cfg"):
            config_file = file_name
            break
    if not config_file:
        return default_fps
    with open(os.path.join(folder_path, config_file), 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() and not line.startswith('%'):
            if line.startswith('frameCfg'):
                frame_period = float(line.split()[5])
            time.sleep(0.01)
    return 1/(frame_period/1000)

def plot_tracks_xy_frame(tracks, ax, timestamp, detections=None, show_road=True):
    ax.clear()

    # --- Plot detections if provided ---
    if detections:
        det_xs = [d['x'] for d in detections if d['y'] > 0]
        det_ys = [((d['y'])**2 + (d['z'])**2)**0.5 for d in detections if d['y'] > 0]
        ax.scatter(det_xs, det_ys, c='gray', s=10, alpha=0.4, label='Detections')
    # --- Plot tracks ---
    for t in tracks:
        if not len(show_only_trk_id) ==0:
            if not t.id in show_only_trk_id:
                continue
        t_class = classify_tgt4plot(t)
        if show_only_classified:
            if show_only_Trucks:
                if t_class != 't':
                    continue
            else:
                if t_class == 'n':
                    continue
        # if t.last_doppler >=2:
        #     continue
        t2a = []
        x, y, z= t.get_position4plot()

        t2a = round(float(t.t2a), 2)

        ax.scatter(x, (y**2 + z**2)**0.5, c='blue', s=80, marker='o')
        ax.text(x + 0.2, (y**2 + z**2)**0.5 + 0.2, f'ID:{t.id} class:{dict_class[t_class]}  t2a: {t2a}', color='black', fontsize=9)
    ax.scatter(0, 0, c='red', s=100, marker='x')  # Radar position
    if show_road:
        # Define x values
        x_road = np.linspace(-100, 100, 100)
        d = 10
        theta = 18 * np.pi / 180

        road_width = 8
        r_w = road_width / (2 * np.cos(theta))
        n = (d) * np.cos(theta) / np.sin(theta)

        n1 = (d + r_w) * np.cos(theta) / np.sin(theta)
        n2 = (d - r_w) * np.cos(theta) / np.sin(theta)

        m = - np.cos(theta) / np.sin(theta)
        # Define the equation of the line, e.g., y = 2x + 1
        y_road = m * x_road + n

        y_road1 = m * x_road + n1
        y_road2 = m * x_road + n2
        # Plot the line
        ax.plot(x_road, y_road)
        ax.plot(x_road, y_road1)
        ax.plot(x_road, y_road2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 100)
    ax.set_title(f'2D Track Map — Time: {timestamp:.2f}s')
    ax.grid(True)
    ax.set_aspect('equal')

class dets:
    def __init__(self,detections):
        self.df_dets = pd.DataFrame(detections)
    def print_dets(self):
        print(self.df_dets)

class trks:
    def __init__(self,tracks):
        self.tracks = tracks
        self.load_data()
    def load_data(self):
        ids = []
        ages = []
        num_assoc = []
        ranges = []
        dopplers = []
        x = []
        y = []
        z = []
        z_var = []
        missed = []
        was_associated = []
        avg_dop = []
        doppler_history = []
        for t in self.tracks:
            ids.append(t.id)
            ages.append(t.age)
            num_assoc.append(t.assoc_dets)
            ranges.append(t.last_range)
            dopplers.append(t.last_doppler)
            x.append(round(t.kf.x[0][0],2))
            y.append(round(t.kf.x[1][0],2))
            if isinstance(t,ExtendedKalmanTrack_3D):
                z.append(round(t.kf.x[2][0],2))
            else:
                z.append(0)
            z_var.append(t.get_z_variance())
            missed.append(t.missed)
            was_associated.append(t.was_associated)
            doppler_history.append(t.doppler_history)
            avg_dop.append(t.get_avg_doppler())
        dict = {'ID': ids,
                'Age': ages,
                'Num assoc': num_assoc,
                'Range': ranges,
                'Doppler': dopplers,
                'Avg doppler': avg_dop,
                'x': x ,
                'y': y,
                'z': z,
                'z var': z_var,
                'Missed': missed,
                'Was assoc': was_associated,
                'Doppler History': doppler_history
                }
        self.df_trks = pd.DataFrame(dict)

    def print_trks(self):
        with pd.option_context('display.max_rows', None,
                          'display.max_columns', None,
                          'display.precision', 2,
                          ):
            print(self.df_trks)

# --------   playback 3D ------------
def run_tracker_3D_on_csv(csv_path, frame_period = 0.1, dist_threshold=1, min_track_length=5, show_ids=True, tracker_csv_path=None):
    # Create tracker output CSV
    debug_mode = True
    tracker_writer = None
    tracker_rec_cols = ['track_id', 'timestamp', 'Radar id', 'time_from_start','num assoc',
                                 'range', 'doppler', 'x', 'y', 'z','vx','vy','vz','calc_dopp','is assoc',
                                 'assoc_timestamp', 'assoc_x', 'assoc_y', 'assoc_z','assoc_dopp','z var','class']
    if os.path.isdir(csv_path):
        tracker_csv_path = os.path.join(Path(csv_path),"trks.csv")
        filename = csv_path.name
        datetime_str = filename[:15]
        folder_time = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        pics_folder = os.path.join(csv_path, 'camera_frames')

        impact_pics_folder = os.path.join(csv_path, 'impact_pics')
        os.makedirs(impact_pics_folder, exist_ok=True)

        offset_path = os.path.join(csv_path, 'offset_time.txt')
        offset_seconds = 0
        if os.path.isfile(offset_path):
            with open(offset_path, "r") as f:
                content = f.read().strip()  # read file and remove whitespace
                try:
                    offset_seconds = float(content)  # convert to float (handles negative and decimal)
                    print(f"Offset (in seconds): {offset_seconds}")
                except ValueError:
                    print("The file does not contain a valid number.")
            start_datetime = folder_time + timedelta(seconds=offset_seconds)
        else:
            start_datetime = folder_time

        fps = get_fps(csv_path)
        frame_period = 1/fps
        csv_path = os.path.join(csv_path, 'det.csv')

    if tracker_csv_path:
        os.makedirs(os.path.dirname(tracker_csv_path), exist_ok=True)
        tracker_file = open(tracker_csv_path, 'w', newline='')
        tracker_writer = csv.writer(tracker_file)
        tracker_writer.writerow(tracker_rec_cols)

    # Step 1: Load detections grouped by timestamp
    frames = defaultdict(list)
    all_raw_times = []
    all_raw_ranges = []
    all_raw_dopp = []
    num_rdrs = 0
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        first_row = True
        for row in reader:
            try:
                timestamp = float(row['Frame number'])* frame_period

                if first_row == True:
                    t0_timestamp = timestamp
                    first_row = False
                    t0_curr_time = row.get('curr_timestamp')
                    if t0_curr_time is not None:
                        start_datetime = datetime.strptime(t0_curr_time,"%Y-%m-%d %H:%M:%S.%f") - timedelta(seconds=t0_timestamp) + timedelta(seconds=offset_seconds)
                rng = float(row['range'])
                dopp = float(row['doppler'])

                # If cur_timestamp exists, parse to time string
                cur_time_str = None
                if 'curr_timestamp' in row and row['curr_timestamp'].strip():
                    try:
                        dt_obj = datetime.strptime(row['curr_timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                        cur_time_str = dt_obj.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
                    except ValueError:
                        pass
                rdr_id = int(row['Radar id'])
                num_rdrs = max(num_rdrs, rdr_id +1)
                if num_rdrs > len(all_raw_times):
                    all_raw_times.append([])
                    all_raw_ranges.append([])
                    all_raw_dopp.append([])
                all_raw_times[rdr_id].append(timestamp)
                all_raw_ranges[rdr_id].append(rng)
                all_raw_dopp[rdr_id].append(dopp)
                det = {
                    'timestamp': timestamp,
                    'time' : round(timestamp - t0_timestamp,2),
                    'Frame_number': int(row['Frame number']),
                    'radar_id' : rdr_id,
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'doppler': float(row['doppler']),
                    'snr': round(10*np.log10(float(row['snr'])),2),
                    'range': rng,
                    'cur_time_str': cur_time_str  # <-- Add here
                }
                frames[timestamp].append(det)
            except:
                continue
    #####
    gap_threshold = frame_period * 2

    # Original detection timestamps
    sorted_times = sorted(frames.keys())

    # New list with missing times filled
    filled_times = []

    for i in range(len(sorted_times) - 1):
        current_time = sorted_times[i]
        next_time = sorted_times[i + 1]

        filled_times.append(current_time)

        gap = next_time - current_time
        if gap > gap_threshold:
            # Insert intermediate timestamps in steps of frame_period
            t = current_time + frame_period
            while t < next_time - 1e-6:
                filled_times.append(round(t, 6))  # rounding for floating point safety
                t += frame_period

    # Don't forget the last timestamp
    filled_times.append(sorted_times[-1])
    ####
    sorted_times = sorted(frames.keys())
    t0 = min(min(all_raw_times))
    relative_raw_times = [[] for _ in range(num_rdrs)]
    for i_rdr in range(num_rdrs):
        relative_raw_times[i_rdr] = [t - t0 for t in all_raw_times[i_rdr]]

    # Step 2: Initialize tracker
    tracker = TrackerManager_3D(num_rdrs=num_rdrs, dist_threshold=dist_threshold)
    track_history_range = [defaultdict(list) for _ in range(num_rdrs)]
    track_history_doppler = [defaultdict(list) for _ in range(num_rdrs)]
    track_hist_position = [defaultdict(list) for _ in range(num_rdrs)]
    track_history_xy = [defaultdict(list) for _ in range(num_rdrs)]

    if RT_xy_plot:
        fig, ax_xy = plt.subplots(figsize=(6, 6))
        previous_timestamp = filled_times[0] - 0.01
    for timestamp in filled_times:
        detections = frames[timestamp]
        batch_dets = dets(detections)
        # if detections:
        #     if detections[0].get('cur_time_str') is not None:
        #         print(detections[0].get('cur_time_str'))
        #     else:
        #         print(timedelta(seconds=detections[0].get('timestamp')) + start_datetime)
        for i_rdr in range(num_rdrs):
            det_per_rdr = [det for det in detections if det.get('radar_id') == i_rdr]
            tracks = tracker.update(det_per_rdr,i_rdr ,timestamp, debug_mode=debug_mode)

        for i_rdr in range(num_rdrs):
            trks_df = trks(tracks[i_rdr])
            if RT_xy_plot:
                plot_tracks_xy_frame(tracks, ax_xy, timestamp, detections)

                dt = timestamp - previous_timestamp
                plt.pause(dt)
                previous_timestamp = timestamp
                # show_reported_trks(tracker.report_tracks,tracks )

            for t in tracks[i_rdr]:

                t_class = t.target_class
                if show_only_classified:
                    if show_only_Trucks:
                        if t_class != 't':
                            continue
                    else:
                        if t_class == 'n':
                            continue
                if t.assoc_dets < min_assoc2plot:
                    continue
                x, y, z= t.get_position4plot()
                # rng = (x**2 + y**2+ z**2)**0.5
                rng = t.get_filtered_range()
                # doppler = t.last_doppler
                doppler = t.get_avg_doppler()
                # doppler = t.get_true_vel()

                time_from_start = timestamp - t0

                assoc_ts = t.last_assoc['timestamp'] if t.was_associated else ''
                assoc_x = t.last_assoc['x'] if t.was_associated else ''
                assoc_y = t.last_assoc['y'] if t.was_associated else ''
                assoc_z = t.last_assoc['z'] if t.was_associated else ''
                assoc_dopp = t.last_assoc['dopp'] if t.was_associated else ''



                if range_doppler_plot:
                    track_history_range[i_rdr][t.id].append((timestamp, rng, t_class))
                    track_history_doppler[i_rdr][t.id].append((timestamp, doppler, t_class))
                    track_hist_position[i_rdr][t.id].append((x, y, z))

                if tracker_writer:
                    tracker_writer.writerow([
                        t.id,
                        round(timestamp, 3),
                        t.rdr_id,
                        round(time_from_start, 3),
                        round(t.assoc_dets),
                        round(rng, 1),
                        round(doppler, 1),
                        round(x, 1),
                        round(y, 1),
                        round(z, 1),
                        round(float(t.kf.x[3].item()), 1),
                        round(float(t.kf.x[4].item()), 1),
                        round(float(t.kf.x[5].item()), 1),
                        round((t.kf.x[3].item() * x + t.kf.x[4].item() * y + t.kf.x[5].item() * z) / rng,1),
                        round(t.was_associated),
                        round(assoc_ts, 3) if assoc_ts else '',
                        round(assoc_x, 1) if assoc_x != '' else '',
                        round(assoc_y, 1) if assoc_y != '' else '',
                        round(assoc_z, 1) if assoc_z != '' else '',
                        round(assoc_dopp, 1) if assoc_dopp != '' else '',
                        round(t.get_z_variance(),2),
                        dict_class[t_class]
                    ])
    if tracker_writer:
        tracker_file.close()
    # Step 3: Plot
    if range_doppler_plot:
        # Build figure: 2 rows (Range, Doppler) × num_rdrs columns (one per radar)
        fig, axes = plt.subplots(
            2, num_rdrs, figsize=(6 * max(1, num_rdrs), 8), sharex='col'
        )

        # Make axes consistently 2×N even when N=1
        try:
            axes_2d = axes if axes.ndim == 2 else axes.reshape(2, 1)
        except AttributeError:
            # Matplotlib returns a numpy ndarray; this is just a safety fallback.
            axes_2d = axes

        ax_range_cols = axes_2d[0]  # array/list of length num_rdrs
        ax_dopp_cols = axes_2d[1]

        # Accumulate lines for post-plot datatips
        all_range_lines = []
        all_dopp_lines = []

        for r in range(num_rdrs):
            axR = ax_range_cols[r]
            axD = ax_dopp_cols[r]

            # ---------- Raw detections (Range vs Time) ----------
            try:
                rel_t_r = relative_raw_times[r]
                raw_rng = all_raw_ranges[r]
                sc_dets = axR.scatter(rel_t_r, raw_rng, s=5, color='lightgray', alpha=0.4, label="Raw Detections")
            except Exception:
                sc_dets = None  # in case any per-radar raw arrays are missing

            # Tooltip for raw detections -> show original frame time (from frames[..])
            if sc_dets is not None:
                cursor_dets = mplcursors.cursor(sc_dets, hover=True)

                @cursor_dets.connect("add")
                def on_add(sel, r=r):
                    det_idx = sel.index
                    det_time = None

                    # Assumed: you also have per-radar all_raw_times with frame indices:
                    frame_idx = all_raw_times[r][det_idx]  # <-- if you use a different index source, swap here
                    # Original pattern: frames[frame_idx][0].get('cur_time_str', None)
                    f0 = frames[frame_idx][0] if isinstance(frames[frame_idx], (list, tuple)) else frames[frame_idx]
                    det_time = f0.get('cur_time_str', None)

                    if det_time:
                        sel.annotation.set_text(f"Time: {det_time}")

            # Titles / labels / grids
            axR.set_title(f"Radar {r} — Track Range vs. Time")
            axR.set_ylabel("Range (m)")
            axR.grid(True)
            axR.set_ylim(0, 80)

            # ---------- Plot per-track Range history (with optional XYZ tooltip) ----------
            for tid in track_history_range[r]:
                if len(show_only_trk_id) != 0 and tid not in show_only_trk_id:
                    continue
                if len(track_history_range[r][tid]) < min_track_length:
                    continue

                times, ranges, t_class = zip(*track_history_range[r][tid])
                rel_times = [t - t0 for t in times]
                class_name = dict_class.get(t_class[-1], t_class[-1])

                lineR, = axR.plot(rel_times, ranges, label=f'Track {tid} \nclass: {class_name}')
                all_range_lines.append(lineR)

                if show_ids:
                    axR.text(rel_times[-1], ranges[-1] + 1.5,
                             f'ID {tid} class: {class_name}', fontsize=9)


                # Invisible scatter for XYZ hover (if requested)
                sc_hidden = axR.scatter(rel_times, ranges, s=0.1, alpha=0.0)
                if show_xyz:
                    xys = track_hist_position[r][tid]
                    cursor_xyz = mplcursors.cursor(sc_hidden, hover=True)
                    cursor_xyz.connect(
                        "add",
                        lambda sel, tid=tid, xys=xys, class_name=class_name:
                        sel.annotation.set_text(
                            f"ID {tid}\nclass: {class_name}\n"
                            f"x: {xys[sel.index][0]:.1f}, y: {xys[sel.index][1]:.1f}, z: {xys[sel.index][2]:.1f}"
                        )
                    )

            # ---------- Raw detections (Doppler vs Time) ----------
            try:
                rel_t_r = relative_raw_times[r]
                raw_dop = all_raw_dopp[r]
                axD.scatter(rel_t_r, raw_dop, s=5, color='lightgray', alpha=0.4, label="Raw Doppler")
            except Exception:
                pass

            # ---------- Plot per-track Doppler history ----------
            for tid in track_history_doppler[r]:
                if len(show_only_trk_id) != 0 and tid not in show_only_trk_id:
                    continue
                if len(track_history_doppler[r][tid]) < min_track_length:
                    continue

                times, dopplers, t_class = zip(*track_history_doppler[r][tid])
                rel_times = [t - t0 for t in times]

                lineD, = axD.plot(rel_times, dopplers, label=f'Track {tid}')
                all_dopp_lines.append(lineD)

                if show_ids:
                    axD.text(rel_times[-1], dopplers[-1] + 0.5, f'ID {tid}', fontsize=9)



            axD.set_title(f"Radar {r} — Track Doppler vs. Time")
            axD.set_ylabel("Doppler (m/s)")
            axD.set_xlabel("Time (s)")
            axD.grid(True)
            axD.set_ylim(-10, 0)

        # Global title and layout
        exp_name = extract_experiment_name(csv_path)
        if exp_name:
            fig.suptitle(f"{exp_name}", fontsize=14)

        # Add datatips for line labels (only when not showing per-point XYZ)
        if not show_xyz and len(all_range_lines) > 0:
            cursorR = mplcursors.cursor(all_range_lines, hover=True)
            cursorR.connect("add", lambda sel: sel.annotation.set_text(f"{sel.artist.get_label()}"))

        if len(all_dopp_lines) > 0:
            cursorD = mplcursors.cursor(all_dopp_lines, hover=True)
            cursorD.connect("add", lambda sel: sel.annotation.set_text(f"{sel.artist.get_label()}"))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save
        if exp_name:
            safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', exp_name.strip())
        else:
            safe_name = "experiment"

        save_path = os.path.join("figures", f"{safe_name}_r{num_rdrs}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")

        plt.show()
        ###################################################################################
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        #
        # # Range vs Time
        # ax1.scatter(relative_raw_times, all_raw_ranges, s=5, color='lightgray', alpha=0.4, label="Raw Detections")
        #
        # sc_dets = ax1.scatter(relative_raw_times, all_raw_ranges, s=5, color='lightgray', alpha=0.4)
        #
        # cursor_dets = mplcursors.cursor(sc_dets, hover=True)
        #
        # @cursor_dets.connect("add")
        # def on_add(sel):
        #     det_idx = sel.index
        #     det_time = frames[all_raw_times[det_idx]][0].get('cur_time_str', None)
        #     if det_time:
        #         sel.annotation.set_text(
        #             f"Time: {det_time}"
        #         )
        #
        # for tid in track_history_range:
        #     if not len(show_only_trk_id) == 0:
        #         if not tid in show_only_trk_id:
        #             continue
        #     if len(track_history_range[tid]) < min_track_length:
        #         continue
        #     times, ranges, t_class = zip(*track_history_range[tid])
        #     xys = track_hist_position[tid]
        #
        #     rel_times = [t - t0 for t in times]
        #     ax1.plot(rel_times, ranges, label=f'Track {tid} \nclass: {dict_class[t_class[-1]]}')
        #     if show_ids:
        #         ax1.text(rel_times[-1], ranges[-1] + 1.5, f'ID {tid} class: {dict_class[t_class[-1]]}', fontsize=9)
        #     # Highlight first truck classification
        #     if 't' in t_class:
        #         for idx, cls in enumerate(t_class):
        #             if cls == 't':
        #                 ax1.plot(rel_times[idx], ranges[idx], 'r*', markersize=10, label='First truck' if 'First truck' not in ax1.get_legend_handles_labels()[1] else "")
        #                 break  # only first instance
        #
        #     # Plot scatter (invisible, only for cursor interaction)
        #     sc = ax1.scatter(rel_times, ranges, s=0.1, alpha=0.0)
        #
        #     # show xyz position for each track
        #     if show_xyz:
        #         cursor = mplcursors.cursor(sc, hover=True)
        #         cursor.connect("add", lambda sel, tid=tid, xys=xys:
        #         sel.annotation.set_text(
        #             f"ID {tid}\nclass: {dict_class[t_class[-1]]}\nx: {xys[sel.index][0]:.1f}, y: {xys[sel.index][1]:.1f}, z: {xys[sel.index][2]:.1f}"
        #         ))
        #
        # ax1.set_ylabel("Range (m)")
        # ax1.set_title("Track Range vs. Time")
        #
        # exp_name = extract_experiment_name(csv_path)
        # if exp_name:
        #     fig.suptitle(f"{exp_name}", fontsize=14)
        #
        # ax1.grid(True)
        # # ax1.legend()
        #
        # # Doppler vs Time (with ID labels)
        # ax2.scatter(relative_raw_times, all_raw_dopp, s=5, color='lightgray', alpha=0.4, label="Raw Doppler")
        #
        # for tid in track_history_doppler:
        #     if not len(show_only_trk_id) == 0:
        #         if not tid in show_only_trk_id:
        #             continue
        #     if len(track_history_doppler[tid]) < min_track_length:
        #         continue
        #     times, dopplers, t_class= zip(*track_history_doppler[tid])
        #     rel_times = [t - t0 for t in times]
        #     ax2.plot(rel_times, dopplers, label=f'Track {tid}')
        #     if show_ids:
        #         ax2.text(rel_times[-1], dopplers[-1] + 0.5, f'ID {tid}', fontsize=9)
        #
        #     # Highlight first truck classification in doppler plot
        #     if 't' in t_class:
        #         for idx, cls in enumerate(t_class):
        #             if cls == 't':
        #                 ax2.plot(rel_times[idx], dopplers[idx], 'r*', markersize=10,
        #                          label='First truck (doppler)' if 'First truck (doppler)' not in ax2.get_legend_handles_labels()[1] else "")
        #                 break
        #
        # ax2.set_ylabel("Doppler (m/s)")
        # ax2.set_xlabel("Time (s)")
        # ax2.set_title("Track Doppler vs. Time")
        # ax2.grid(True)
        #
        # ax1.set_ylim(0, 80)
        # ax2.set_ylim(-10, 0)
        #
        # plt.tight_layout()
        #
        # # After plotting all lines, add datatips:
        # if not show_xyz:
        #     cursor1 = mplcursors.cursor(ax1.lines, hover=True)
        #     cursor1.connect("add", lambda sel: sel.annotation.set_text(
        #         f"{sel.artist.get_label()}"
        #     ))
        #
        #
        # cursor2 = mplcursors.cursor(ax2.lines, hover=True)
        # cursor2.connect("add", lambda sel: sel.annotation.set_text(
        #     f"{sel.artist.get_label()}"
        # ))
        # # Clean up filename (remove special characters, spaces to underscores)
        # safe_name = exp_name.replace(' ', '_')
        # save_path = os.path.join("figures", f"{safe_name}.png")
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #
        # plt.savefig(save_path, dpi=150)
        # print(f"Figure saved to: {save_path}")
        #
        # plt.show()
##########################################################################################




        # if os.path.isdir(pics_folder):
        #     for t_id, t_impact, t_dopp in tgt_arrive_times:
        #         target_info = f"ID: {t_id}   Doppler: {round(t_dopp,2)} m/s \nimpact_time: {t_impact}"
        #         path2impact_pic = os.path.join(impact_pics_folder, f"{t_id}.png")
        #         result = find_closest_image(pics_folder, t_impact, copy_to=path2impact_pic, text=target_info )
        #         path, ts = result
        #         print(f"expected impact time: {t_impact}   trk id: {t_id}    trk dopp: {round(t_dopp, 2)} m/s")
        #         pass






# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250508_164913_human_and_cars_aproching.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250508_165809_car.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250508_164333_human_walking.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_142817_car_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_171444_human_and_car.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_172359_human_walking_60m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_165532_human_and_car.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250521_145446_cars_and_humans_raanana.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250521_145708_cars_and_humans_raanana.csv"


# data from day tests -  22/05/2025
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_093816_1_man1_walk_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_093920_2_man1_walk_from_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_105007_3_man_walk_30_degree_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_105113_4_man_walk_30_degree_from_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_094309_5_man2_walk_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_094427_6_man2_walk_from_70m_to_sensor.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_100108_7_man_walk_zigzag_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_100233_8_man_walk_zigzag_from_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_100557_9_small_car_to_max_range_100m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_100657_10_small_car_from_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_100929_11_small_car_to_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_101726_11_take2_small_car_to_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_101001_12_2_small_cars_from_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_101757_12_take2_small_car_from_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_101944_13_large_car_to_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_102043_14_large_car_from_max_range.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_102619_15_tangent_walk_15m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_102846_16_tangent_walk_20m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_094826_17_2_walker_3m_apart_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_095039_18_2_walker_3m_apart_from_70_to_sensor.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_095441_19_2_walkers_5m_apart_to_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_095606_20_2_walkers_5m_apart_from_70m.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_113336_27_fast_cars_bridge.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_110342_29_large_car_to_max_range_30_deg.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_110424_30_large_car_from_max_range_30_deg_and_caterpiller.csv"

########## data from day tests -  17/06/2025 with TI sensor   #######################

# TI sensor
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_152233_1_Two_Trucks_high_cabin_20kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_152819_2_Taavura_track_outbound_0_deg_30kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_153052_3_Taavura_truck_inbound_0_deg_60kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_153331_4_Taavura_truck_outbound_30_deg_40kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_153543_5_car_and_Taavura_truck_inbound_30_deg_60kmh.csv"

# Radomatics sensor
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_154057_6_Taavura_truck_inbound_30_deg_60kmh.csv"

# TI sensor - 7m from the road
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_154808_7_car_and_Taavura_truck_inbound_10_deg_50_kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_155343_full_trailer_0_deg_30_kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_155343_Taavura_0_deg_60_kmh.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_160044_Taavura_truck.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_160250_cars_and_Taavura_Truck.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_160508_car_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_160622_truck_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_163156_two_close_cars_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_163410_two_close_cars_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_163626_two_close_cars_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_165011_full_trailer_inbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_165254_Truck_outbound.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_170017_Truck_outbound.csv"


##################  data from day tests - 24/06/2025 with TI sensor ##########################

# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_143303_inbound_Truck.csv" # good classification
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_143919.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_145331.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_145547_inbound_Truck.csv"  # no Truck classification
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_145858.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_145922.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_152306.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_152335.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_152426_inbound_Truck.csv" # good classification
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_152647.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_153227.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_153408_inbound_Truck.csv"   # late classification (30m)
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_153704.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_154629.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_154848_inbound_Truck.csv" # late classification,  mirror track in different direction
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_155017.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_155113.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_155844.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_160045.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_160354.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_160613.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_160717.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_163224.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_163324.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_163546.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_163718.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_163847.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_164259.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_164428_inbound_Truck.csv"  # late classification (35m)
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_164547.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_164730_inbound_Truck.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_165019_outbound_Truck.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_165238.csv" # no classification
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_165750.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250624_170157.csv"

###################################################################################################
###################    data from day tests 02/07/2025            ##################################
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_114435.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_114650.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_114906.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_115227.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_121417.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_121428.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_121452.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_121517.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_122319.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124543.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124616.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124659.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124730.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124811.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_124836.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_125003.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_125406.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_125541.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_130112.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_130157.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_130331.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_130431.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_130535.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_131050.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_131513.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_132616.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_132730.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_132857.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_133051.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_133330.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_134204.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_134408.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_134549.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_134812.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_134946.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_135126.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_140725.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141015.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141238.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141352.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141437.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141728.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141808.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_141959.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_142307.csv"
# path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250702_142711.csv"

##### data from day tests 24/07/2025  vehicles only ####################
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_113444.csv" # Duration: 00:50 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_122433.csv" # Duration: 02:12 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_122751.csv" # Duration: 00:59 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_123850.csv" # Duration: 00:31 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_124504.csv" # Duration: 01:17 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_124838.csv" # Duration: 00:29 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_125214.csv" # Duration: 01:28 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_125617.csv" # Duration: 05:17 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_130324.csv" # Duration: 03:43 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_131834.csv" # Duration: 01:59 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_132414.csv" # Duration: 13:57 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_133950.csv" # Duration: 05:34 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_134739.csv" # Duration: 16:32 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_140632.csv" # Duration: 00:33 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_141010_radom_50.csv" # Duration: 02:31 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_141339_radom_25.csv" # Duration: 03:12 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_142259.csv" # Duration: 00:36 (MM:SS)
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_142851_radom_25.csv" # Duration: 08:55 (MM:SS)
#
#
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250724_142851_radom_25_part.csv"

#### experiments day 06/08/2025 Raanana - derech ahaklaim ####

# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_164956.csv" # Duration: 00:11 - car low speed
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_165408.csv" # Duration: 04:13
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_170119.csv" # Duration: 00:56
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_170944.csv" # Duration: 00:48
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_171218.csv" # Duration: 01:37
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_171500.csv" # Duration: 00:51
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_171701.csv" # Duration: 01:08   no video
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_172348.csv" # Duration: 01:06     2 fast cars
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_172543.csv" # Duration: 01:07
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_172741.csv" # Duration: 01:20
# path2dets = r"C:\Users\orelm\Documents\projects\shula\.venv\records\detections_test_TI20250806_175314.csv" # Duration: 12:59


######### experiments day from 20/08/2025 Hadarim   near tel mond  ###########

# check bias in range
# path2dets = Path(r"../records/20250820_114939_calibration_30m/det.csv") # Duration: 00:23
# path2dets = Path(r"../records/20250820_115038_calibration_40m/det.csv") # Duration: 00:17
# path2dets = Path(r"../records/20250820_115118_calibration_70m/det.csv") # Duration: 00:16

# path2dets = Path(r"../records/20250820_122419/det.csv") # Duration: 00:22
# path2dets = Path(r"../records/20250820_123549/det.csv") # Duration: 01:00

# path2dets = Path(r"../records/20250820_124226_long_rec_15m") # Duration: 05:28
# path2dets = Path(r"../records/20250820_124803_long_rec_15m") # Duration: 03:59
# path2dets = Path(r"../records/20250820_125728_long_rec_15m") # Duration: 09:04

# path2dets = Path(r"../records/20250820_131233_long_rec_8m") # Duration: 08:03
# path2dets = Path(r"../records/20250820_132046_long_rec_8m") # Duration: 03:12   2 trucks and D9
# path2dets = Path(r"../records/20250820_133514_long_rec_8m_walking") # Duration: 08:34

# path2dets = Path(r"../records/20250820_135040_long_rec_5_50") # Duration: 07:05
# path2dets = Path(r"../records/20250820_140410_long_rec_5_50_20fps") # Duration: 03:54

######### experiments day from 27/08/2025  near topit stone  ###########################

# path2dets = Path(r"../records/20250827_105148") # Duration: 01:09   FPS: 10.0
# path2dets = Path(r"../records/20250827_105443") # Duration: 09:38   FPS: 10.0
# path2dets = Path(r"../records/20250827_110604") # Duration: 17:40   FPS: 10.0

# path2dets = Path(r"../records/20250827_112555") # Duration: 02:52   FPS: 20.0
# path2dets = Path(r"../records/20250827_112900") # Duration: 10:14   FPS: 20.0
# path2dets = Path(r"../records/20250827_113924") # Duration: 09:48   FPS: 20.0
# path2dets = Path(r"../records/20250827_114958") # Duration: 07:49   FPS: 20.0
# path2dets = Path(r"../records/20250827_121502") # Duration: 00:17   FPS: 20.0

# bush
# path2dets = Path(r"../records/20250827_122530") # Duration: 01:59   FPS: 20.0
# path2dets = Path(r"../records/20250827_122807_bush_20fps") # Duration: 00:26   FPS: 20.0
# path2dets = Path(r"../records/20250827_122936_bush_10fps") # Duration: 02:57   FPS: 10.0

# path2dets = Path(r"../records/20250827_124454") # Duration: 00:22   FPS: 20.0     no pictures
# path2dets = Path(r"../records/20250827_125323") # Duration: 00:48   FPS: 20.0   no pictures

# path2dets = Path(r"../records/20250827_125653_living_together") # Duration: 01:40   FPS: 20.0

##############################################################################

######################## deploiment mode experiment   ###################################

# path2dets = Path(r"20250908_124832")  # Duration: 00:30 | FPS: 20.00 | dets: 5882
# path2dets = Path(r"20250908_125722")  # Duration: 00:20 | FPS: 20.00 | dets: 2462
# path2dets = Path(r"20250908_125924")  # Duration: 00:05 | FPS: 20.00 | dets: 818

# path2dets = Path(r"20250908_130030")  # Duration: 00:23 | FPS: 20.00 | dets: 3010 # 2 radars same freq
# path2dets = Path(r"20250908_130309")  # Duration: 00:15 | FPS: 10.00 | dets: 2572 # 2 radars same freq
# path2dets = Path(r"20250908_130350")  # Duration: 00:29 | FPS: 10.00 | dets: 5448 # 2 radars same freq

# bag interference (1m)
# path2dets = Path(r"../records/20250908_131119")  # Duration: 00:15 | FPS: 10.00 | dets: 2504
# path2dets = Path(r"../records/20250908_131224")  # Duration: 00:24 | FPS: 20.00 | dets: 2231
# path2dets = Path(r"20250908_131342")  # Duration: 00:32 | FPS: 20.00 | dets: 3522

# radar on ground
# path2dets = Path(r"20250908_133248")  # Duration: 00:16 | FPS: 10.00 | dets: 1880
# path2dets = Path(r"20250908_133414")  # Duration: 00:10 | FPS: 20.00 | dets: 1198

# regular data
# path2dets = Path(r"../records/20250908_133535")  # Duration: 00:36 | FPS: 20.00 | dets: 4932
# path2dets = Path(r"../records/20250908_134001")  # Duration: 00:57 | FPS: 20.00 | dets: 10806

# different angle position
# path2dets = Path(r"20250908_134258")  # Duration: 00:34 | FPS: 20.00 | dets: 6505
# path2dets = Path(r"20250908_134408")  # Duration: 01:29 | FPS: 20.00 | dets: 17816

# open field
# path2dets = Path(r"../records/20250908_140606")  # Duration: 00:27 | FPS: 20.00 | dets: 1676
# path2dets = Path(r"../records/20250908_140651")  # Duration: 00:21 | FPS: 10.00 | dets: 1689

# path2dets = Path(r"../records/20250908_140736")  # Duration: 00:16 | FPS: 10.00 | dets: 1362

# path2dets = Path(r"../records/20250908_140816")  # Duration: 00:28 | FPS: 10.00 | dets: 2615
# path2dets = Path(r"../records/20250908_140855")  # Duration: 00:27 | FPS: 20.00 | dets: 1524

# open field on the ground
# path2dets = Path(r"20250908_141045")  # Duration: 00:25 | FPS: 20.00 | dets: 685
# path2dets = Path(r"20250908_141122")  # Duration: 00:31 | FPS: 10.00 | dets: 2653



#####   human detection first tests   #######

# path2dets = Path(r"../records/20250921_141421_human_walking_indoor")  # Duration: 00:36 | FPS: 20.00 | dets: 727
# path2dets = Path(r"../records/20250921_141946_human_walking")  # Duration: 01:10 | FPS: 20.00 | dets: 1006
# path2dets = Path(r"../records/20250921_142151_human_running")  # Duration: 00:50 | FPS: 20.00 | dets: 539

# 2 radars
# path2dets = Path(r"../records/20250921_185255")
# path2dets = Path(r"../records/20250923_135445")  # Duration: 00:49 | FPS: 20.00 | dets: 1862

######  experiment at nasa with Namer #######
# path2dets = Path(r"../records_human_detection/20250924_115727")  # Duration: 00:06 | FPS: 20.00 | dets: 180
# path2dets = Path(r"../records_human_detection/20250924_122142")  # Duration: 01:18 | FPS: 20.00 | dets: 2935
# path2dets = Path(r"../records_human_detection/20250925_111111_only_rdr_human_aproching")  # Duration: 00:40 | FPS: 20.00 | dets: 1893
# path2dets = Path(r"../records_human_detection/20250925_111626_only_rdr_human")  # Duration: 00:54 | FPS: 20.00 | dets: 1675
# path2dets = Path(r"../records_human_detection/20250925_115302_low_freq_with_meil")  # Duration: 02:49 | FPS: 20.00 | dets: 3768
# path2dets = Path(r"../records_human_detection/20250925_115655")  # Duration: 01:30 | FPS: 20.00 | dets: 125
# path2dets = Path(r"../records_human_detection/20250925_115837")  # Duration: 00:12 | FPS: 20.00 | dets: 57
# path2dets = Path(r"../records_human_detection/20250925_120003_high_freq_with_meil")  # Duration: 01:38 | FPS: 20.00 | dets: 1168
# path2dets = Path(r"../records_human_detection/20250925_130244")  # Duration: 00:03 | FPS: 20.00 | dets: 76
# path2dets = Path(r"../records_human_detection/20250925_132024_low_freq_with_meil_human_target")  # Duration: 01:13 | FPS: 20.00 | dets: 1721
# path2dets = Path(r"../records_human_detection/20250925_132742_high_freq_with_meil_human_target")  # Duration: 01:00 | FPS: 20.00 | dets: 1267
# path2dets = Path(r"../records_human_detection/20250925_134116_low_freq_with_meil_human_target_from_bushes")  # Duration: 01:20 | FPS: 20.00 | dets: 1932
# path2dets = Path(r"../records_human_detection/20250925_134500_low_freq_namer_on_no_meil_human_target")  # Duration: 01:00 | FPS: 20.00 | dets: 1608
# path2dets = Path(r"../records_human_detection/20250925_134845_low_freq_namer_on_with_meil_human_target")  # Duration: 00:44 | FPS: 20.00 | dets: 1415
# path2dets = Path(r"../records_human_detection/20250925_135549_low_freq_namer_on_with_meil_human_target")  # Duration: 02:55 | FPS: 20.00 | dets: 4040
# path2dets = Path(r"../records_human_detection/20250925_140507_rdr_against_meil")  # Duration: 00:57 | FPS: 20.00 | dets: 1660
# path2dets = Path(r"../records_human_detection/20250925_140618")  # Duration: 01:29 | FPS: 20.00 | dets: 205
# path2dets = Path(r"../records_human_detection/20250925_141118_exhaust_namer_on_with_meil_human_target")  # Duration: 00:47 | FPS: 20.00 | dets: 3033
# path2dets = Path(r"../records_human_detection/20250925_141236_exhast_high_RPM_with_meil_human_target")  # Duration: 01:34 | FPS: 20.00 | dets: 3076
# path2dets = Path(r"../records_human_detection/20250925_142118_testing_human_running_parking")  # Duration: 01:00 | FPS: 20.00 | dets: 2415
# path2dets = Path(r"../records_human_detection/20250925_142309_testing_parking_human_walking")  # Duration: 00:14 | FPS: 20.00 | dets: 395
# path2dets = Path(r"../records_human_detection/20250925_151126")  # Duration: 00:02 | FPS: 20.00 | dets: 51


# path2dets = Path(r"../records_human_detection/20250925_202637")  # Duration: 01:05 | FPS: 20.00 | dets: 1112
# path2dets = Path(r"../records_human_detection/20250928_131952")  # Duration: 01:27 | FPS: 20.00 | dets: 672
# path2dets = Path(r"../records_human_detection/20250928_132621")  # Duration: 01:22 | FPS: 20.00 | dets: 520
# path2dets = Path(r"../records_human_detection/20250928_133721")  # Duration: 01:25 | FPS: 20.00 | dets: 763
# path2dets = Path(r"../records_human_detection/20250928_134224")  # Duration: 00:21 | FPS: 20.00 | dets: 121

if "path2dets" not in globals():
    path2dets = get_latest_entry(Path(r"../records_human_detection"))

output_tracks_path = Path("playback_tracks/tracks_log_01.csv")
run_tracker_3D_on_csv(path2dets,frame_period=frame_period, dist_threshold = 3,min_track_length=2 ,tracker_csv_path=output_tracks_path)


# plot_range_vs_time_from_csv(path2dets)
#
# plot_3d_detections_with_tracks(
#     detection_csv=os.path.join(path2dets, "det.csv"),
#     track_csv=output_tracks_path,
#     color_by='timestamp'
# )