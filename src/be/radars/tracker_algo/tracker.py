from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
import time
from collections import deque
from scipy.spatial.distance import cdist
import math
# import winsound


# ---------------------- Extended Kalman Track 3D ----------------------

class ExtendedKalmanTrack_3D:
    def __init__(self, detection, track_id):
        self.id = track_id
        self.rdr_id = detection['radar_id']
        self.missed = 0
        self.last_assoc_timestamp = detection['timestamp']
        self.time_no_assoc = 0
        self.birth_time = detection['timestamp']
        self.age = 0
        self.last_doppler = detection['doppler']
        self.last_range = detection['range']
        self.assoc_dets = 1

        self.kf = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self._init_kf(dt=0.1)

        self.kf.x[:3] = np.array([[detection['x']], [detection['y']], [detection['z']]])
        unit_position = self.kf.x[:3]/self.last_range

        # self.kf.x[3:] = np.array([[0.0], [self.last_doppler/2], [0.0]])  # Initial velocity guess
        self.kf.x[3:] = unit_position * self.last_doppler

        self.kf.P = np.diag([15.0, 15.0, 15.0, 20.0, 20.0, 20.0])  # Initial uncertainty

        # range doppler kalman filter
        self.range_dopp_kf = ExtendedKalmanFilter(dim_x=2, dim_z=2)  # [range, doppler] ← [measured_range, measured_doppler]
        self._init_range_dopp_kf(dt=0.1)  # dt will be updated dynamically
        self.range_dopp_kf.x = np.array([[self.last_range], [self.last_doppler]])
        self.range_dopp_kf.P *= np.diag([3.0, 1.0])
        self.range_dopp_kf.Q = np.eye(2)  # Process noise

        ##

        self.z_mean = float(self.kf.x[2])      # running mean of z
        self.z_M2 = 0.0     # running sum of squared deviations of z dim
        self.z_count = 1

        self.range_val = (self.kf.x[0]**2 + self.kf.x[1]**2 + self.kf.x[2]**2)**0.5

        self.doppler_variance = 1.0
        self.was_associated = True
        self.dopp_thr4class_car = 3
        self.dopp_thr4class_human = 6
        self.target_class = 'n'  # None
        self.t2a = 0 # time to arrive
        self.reported = False
        self.last_assoc = {
            'timestamp': detection['timestamp'],
            'x': detection['x'],
            'y': detection['y'],
            'z': detection['z'],
            'dopp' : detection['doppler']
        }
        # Doppler buffer
        self.doppler_history = deque(maxlen=20)
        self.doppler_history.append(detection['doppler'])
        self.az_history = deque(maxlen=6)
        az = get_az_from_det(detection)
        self.az_history.append(az)
        self.median_az = az

        # car assoc is detection with doppler above 10m/s
        if np.abs(detection['doppler']) > self.dopp_thr4class_car:
            self.count_car_assoc = 1
        else:
            self.count_car_assoc = 0

        if np.abs(detection['doppler']) > self.dopp_thr4class_human:
            self.count_pass_dopp4human = 1
        else:
            self.count_pass_dopp4human = 0

    def _init_kf(self, dt, confidence_scale=1.0):
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0],  # x position and velocity model
            [0, 1, 0, 0, dt, 0],  # y position and velocity model
            [0, 0, 1, 0, 0, dt],  # z position and velocity model
            [0, 0, 0, 1, 0, 0],   # x velocity model
            [0, 0, 0, 0, 1, 0],   # y velocity model
            [0, 0, 0, 0, 0, 1]    # z velocity model
        ])
        if self.assoc_dets < 4:
            self.kf.R = np.diag([5.0, 5.0, 20.0])*confidence_scale  # Measurement noise (x, y, z)
            # self.kf.P = np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])*confidence_scale  # Initial uncertainty
            self.kf.Q = np.diag([3.0, 3.0, 3.0, 5.0, 5.0, 5.0])*confidence_scale  # x, y, z, vx, vy, vz process noise covariance
        else:
            self.kf.R = np.diag([20.0, 20.0, 40.0])*confidence_scale  # Measurement noise (x, y, z)
            # self.kf.P = np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])*confidence_scale  # Initial uncertainty
            self.kf.Q = np.diag([5.0, 5.0, 5.0, 10.0, 10.0, 10.0])*confidence_scale  # x, y, z, vx, vy, vz process noise covariance

    def _init_range_dopp_kf(self, dt):
        # State vector: [range, range_rate, doppler]
        self.range_dopp_kf.F = np.array([
            [1, dt],
            [0, 1]
        ])
        self.range_dopp_kf.H = np.array([
            [1, 0],  # Measuring range
            [0, 1]  # Measuring doppler
        ])
        self.range_dopp_kf.R = np.diag([5, 2.0])  # Measurement noise: range, doppler


    def hx(self, x):
        # Measurement function: projects state into measurement space
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = x.flatten()
        return np.array([pos_x, pos_y, pos_z]).reshape(-1, 1)

    def H_jacobian(self, x):
        # Jacobian of the measurement function
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
    def H_rd_jacobian(self, x):
        # Jacobian of the measurement function
        return np.array([
            [1, 0],
            [0, 1]
        ])
    def hx_rd(self, x):
        # Measurement function: projects state into measurement space
        return x

    def predict(self, current_timestamp, confidence_scale, dt):
        # dt = current_timestamp - self.last_timestamp

        dt = max(dt, 1e-3)
        # predict cartesian model
        self._init_kf(dt, confidence_scale)
        self.kf.predict()

         # predict range-doppler model
        self._init_range_dopp_kf(dt)
        self.range_dopp_kf.predict()

        if not self.was_associated:
            self.doppler_variance = min(self.doppler_variance * 1.2, 100.0)

        self.was_associated = False  # Reset for next frame
        # self.range_val = (self.kf.x[0] ** 2 + self.kf.x[1] ** 2 + self.kf.x[2] ** 2) ** 0.5
        self.range_val = self.get_filtered_range()
        # self.last_timestamp = current_timestamp

    def update_z_variance(self, z_val):
        self.z_mean = float(self.kf.x[2])
        self.z_count += 1
        delta = z_val - self.z_mean
        self.z_mean +=  delta / self.z_count
        delta2 = z_val - self.z_mean
        self.z_M2 += delta * delta2

    def update(self, detection):
        z = np.array([detection['x'], detection['y'] ,detection['z']]).reshape(-1, 1)

        z_val = detection.get('z', 0.0)
        self.update_z_variance(z_val)

        self.kf.update(z,
                       HJacobian=self.H_jacobian,
                       Hx=self.hx)

        z_rd = np.array([[detection['range']], [detection['doppler']]])
        self.range_dopp_kf.update(z_rd,HJacobian=self.H_rd_jacobian,
                       Hx=self.hx_rd)

        self.last_assoc = {
            'timestamp': detection['timestamp'],
            'x': detection['x'],
            'y': detection['y'],
            'z': detection['z'],
            'dopp': detection['doppler']
        }
        self.last_doppler = detection['doppler']
        self.doppler_history.append(detection['doppler'])

        az = get_az_from_det(detection)
        self.az_history.append(az)
        self.median_az = self.get_median_az()

        if all(self.dopp_thr4class_human > num for num in self.doppler_history):
            self.count_pass_dopp4human = 0
        self.last_doppler = detection['doppler']
        self.last_range = detection['range']
        self.last_assoc_timestamp = detection['timestamp']
        self.assoc_dets += 1
        self.was_associated = True
        self.time_no_assoc = 0
        self.missed = 0
        # self.range_val = (self.kf.x[0] ** 2 + self.kf.x[1] ** 2 + self.kf.x[2] ** 2) ** 0.5
        self.range_val = self.get_filtered_range()
        self.doppler_variance = max(1.0, self.doppler_variance * 0.5)
        self.age = detection['timestamp'] - self.birth_time
        # for car classification
        if np.abs(detection['doppler']) > self.dopp_thr4class_car:
            self.count_car_assoc += 1
        # for human classification
        if np.abs(detection['doppler']) > self.dopp_thr4class_human:
            self.count_pass_dopp4human += 1

    def get_position(self):
        return self.kf.x[:3].flatten()

    def get_position4plot(self):
        x, y ,z= self.kf.x[:3].flatten()
        return x, y, z

    def get_la_doppler(self):
        return self.last_doppler

    def get_z_variance(self):
        if self.z_count < 2:
            return 0.0
        return self.z_M2 / (self.z_count - 1)

    def classify_tgt(self, thr_num_assoc4class_human = 10):
        if self.target_class == 'n':
            if self.is_human_track(thr_num_assoc4class_human):
                self.target_class = 'h' # human track
            else:
                self.target_class = 'n'
    def get_filtered_range(self):
        return float(self.range_dopp_kf.x[0])

    def get_filtered_doppler(self):
        return float(self.range_dopp_kf.x[1])

    def get_true_vel(self):
        r = self.get_filtered_range()
        l=5
        if r > 1.5*l:
            vel = float(self.get_avg_doppler()*r/(r**2 - l**2)**0.5)
        else:
            vel = float(self.get_avg_doppler())
        return vel

    def is_car_track(self, thr_num_assoc4class_car=4):
        is_car = False
        if self.target_class == 'n' or self.target_class == 'h':
            if self.assoc_dets > 5:
                is_car = self.count_car_assoc > thr_num_assoc4class_car
        else:
            if self.target_class == 'c':
                is_car = True
        return is_car

    def is_human_track(self, thr_num_assoc4class_human):
        is_human = False
        if self.target_class == 'n':
            if self.assoc_dets > 15:
                is_human = self.count_pass_dopp4human < thr_num_assoc4class_human
        else:
            if self.target_class == 'h':
                is_human = True
        return is_human

    def get_avg_doppler(self):
        if not self.doppler_history:
            return 0.0
        return sum(self.doppler_history) / len(self.doppler_history)

    def get_median_az(self):
        np.nanmedian(self.az_history)

def get_az_from_det(det):
    az = math.degrees(math.atan2(det['y'], det['x']))
    az = max(-35, az)
    az = min(35, az)
    return az

# ---------------------- Tracker Manager 3D----------------------

class TrackerManager_3D:
    def __init__(self, num_rdrs=1 , dist_threshold=3,dopp_dist_threshold = 2, max_missed=20, max_range=80.0, max_speed=10.0):
        self.tracks: list[list[ExtendedKalmanTrack_3D]] = [[] for _ in range(num_rdrs)] # blank list for each radar
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.dopp_dist_threshold = dopp_dist_threshold
        self.max_missed = max_missed
        self.max_range = max_range
        self.max_speed = max_speed
        # self.duplicate_range_thresh = 2  # meters
        # self.duplicate_doppler_thresh = 4  # m/s
        self.report_tracks = []
        self.duplicate_range_thresh = 3  # meters
        self.duplicate_doppler_thresh = 1  # m/s
        self.create_rng2trk_thresh = 5  # m
        self.create_dop2trk_thresh = 8 # m/s
        self.create_thresh_min_range = 2 # m
        self.ignore_near_duplicates = False
        self.last_timestamp: list[float] = [float() for _ in range(num_rdrs)]
        self.thr_num_assoc4class_human = 10


    def _debug_capture_frame(self, detections, timestamp=None):
        self._last_frame_timestamp = timestamp
        # Keep a lightweight copy for printing
        self._last_frame_detections = [
            {
                'timestamp': d.get('timestamp'),
                'radar_id': d.get('radar_id'),
                'x': d.get('x'), 'y': d.get('y'), 'z': d.get('z'),
                'range': d.get('range'),
                'doppler': d.get('doppler'),
            }
            for d in detections
        ]
    def classify_tgts(self, i_rdr):
        for t in self.tracks[i_rdr]:
            t.classify_tgt(thr_num_assoc4class_human = self.thr_num_assoc4class_human)


    def debug_all(self, i_rdr):
        """Print id, range, doppler for every current track."""
        if not self.tracks[i_rdr]:
            print("[DEBUG] No tracks")
            return
        print(f"[DEBUG] Tracks ({len(self.tracks)})")
        print(" id |   range(m) | doppler(m/s) | class | assoc_dets | missed")
        print("----+------------+--------------+-------+------------+--------")
        for t in self.tracks[i_rdr]:
            rng = self._debug_track_range(t)
            dop = self._debug_track_doppler(t)
            tclass = getattr(t, 'target_class', 'n')
            print(
                f"{t.id:3d} | {rng:10.2f} | {dop:12.2f} | {tclass:>5} | {getattr(t, 'assoc_dets', 0):10d} | {getattr(t, 'missed', 0):6d}")

    def debug_id(self,i_rdr, track_id: int):
        """Print id, range, doppler (and xyz) for a specific track."""
        t = next((tr for tr in self.tracks[i_rdr] if tr.id == track_id), None)
        if t is None:
            print(f"[DEBUG] Track {track_id} not found")
            return
        rng = self._debug_track_range(t)
        dop = self._debug_track_doppler(t)
        pos = self._debug_track_pos(t)
        tclass = getattr(t, 'target_class', 'n')
        print("[DEBUG] Track detail")
        print(f" id={t.id} class={tclass} assoc_dets={getattr(t, 'assoc_dets', 0)} missed={getattr(t, 'missed', 0)}")
        print(f" x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}  |  range={rng:.2f} m  doppler={dop:.2f} m/s")

    def debug_dets(self, max_rows: int = 20):
        """Print the detections given to the most recent update() call."""
        dets = self._last_frame_detections
        if not dets:
            print("[DEBUG] No detections captured for last frame")
            return
        ts = self._last_frame_timestamp
        print(f"[DEBUG] Last frame detections (count={len(dets)})  timestamp={ts}")
        print("  # |   range(m) | doppler(m/s) |   x     y     z   |  SNR |")
        print("----+------------+--------------+--------------------+------+------")
        for i, d in enumerate(dets[:max_rows]):
            x = d.get('x')
            y = d.get('y')
            z = d.get('z')
            rng = d.get('range')
            dop = d.get('doppler')

            print(
                f"{i:3d} | {rng if rng is not None else float('nan'):10.2f} | {dop if dop is not None else float('nan'):12.2f} | "
                f"{(x if x is not None else float('nan')):5.1f} {(y if y is not None else float('nan')):5.1f} {(z if z is not None else float('nan')):5.1f} ")

        if len(dets) > max_rows:
            print(f"... ({len(dets) - max_rows} more)")

        # ---------- Small internal helpers ----------

    def _debug_track_pos(self, t):
        # Prefer a stable getter if you have one
        if hasattr(t, 'get_position4plot'):
            x, y, z = t.get_position4plot()
        elif hasattr(t, 'get_position'):
            x, y, z = t.get_position()
        else:
            # last state fallback: assumes EKF state [x,y,z,vx,vy,vz]^T
            x, y, z = float(t.kf.x[0]), float(t.kf.x[1]), float(t.kf.x[2])
        return (float(x), float(y), float(z))

    def _debug_track_range(self, t):
        # Prefer range/doppler Kalman if present
        if hasattr(t, 'get_filtered_range'):
            return float(t.get_filtered_range())
        # else compute from position
        x, y, z = self._debug_track_pos(t)
        return float((x ** 2 + y ** 2 + z ** 2) ** 0.5)

    def _debug_track_doppler(self, t):
        if hasattr(t, 'get_filtered_doppler'):
            return float(t.get_filtered_doppler())
        if hasattr(t, 'get_avg_doppler'):
            return float(t.get_avg_doppler())
        return float(getattr(t, 'last_doppler', 0.0))

    def update(self, detections, i_rdr, timestamp = None, debug_mode=False):
        if debug_mode:
            self._debug_capture_frame(detections, timestamp)
            # self.debug_all()

        current_time = timestamp if timestamp else time.time()
        # Step 1: Predict all tracks forward
        for t in self.tracks[i_rdr]:
            if t.target_class == 't':
                conf = 1
            else:
                conf = 1
            dt = current_time - self.last_timestamp[i_rdr]
            t.predict(current_time, conf, dt)
            t.time_no_assoc = current_time - t.last_assoc_timestamp


        if not detections:
            for t in self.tracks[i_rdr]:
                t.missed += 1
            self.last_timestamp[i_rdr] = current_time
            self.tracks[i_rdr] = self.filter_tracks(i_rdr)
            return self.tracks
        self.tracks[i_rdr] = self.filter_tracks_time_no_assoc(i_rdr)

        # Step 3: Compute distance matrix
        track_positions = [t.get_position()[0:3] for t in self.tracks[i_rdr]]
        track_la_dopplers = [[t.get_la_doppler()] for t in self.tracks[i_rdr]]
        det_positions = [(d['x'], d['y'], d['z']) for d in detections]
        det_dopplers = [[d['doppler']] for d in detections]

        matched_tracks = set()
        matched_detections = set()
        used_range_doppler = set()

        if track_positions:
            dists = cdist(track_positions, det_positions,'euclidean')
            dopp_dists = cdist(track_la_dopplers, det_dopplers,'euclidean')

            # Create sorted list of (track_idx, det_idx, dist)
            candidates = []
            for i in range(len(self.tracks[i_rdr])):
                for j in range(len(detections)):
                    candidates.append((i, j, dists[i][j],dopp_dists[i][j]))
            candidates.sort(key=lambda x: x[2])  # sort by distance

            # Step 4: Greedy exclusive association
            for i, j, dist, dopp_dist in candidates:
                if j in matched_detections:
                    continue
                if i in matched_tracks:
                    continue

                det = detections[j]
                rng = det['range']
                dop = det['doppler']
                rd_key = (round(rng, 2), round(dop, 2))  # Round to reduce floating-point noise

                if rd_key in used_range_doppler:
                    continue  # Skip this detection — already matched

                # Compute positions
                track_pos = track_positions[i]
                det_pos = det_positions[j]

                # Azimuth and elevation gating
                track_az, track_el = compute_azimuth_elevation(*track_pos)
                det_az, det_el = compute_azimuth_elevation(*det_pos)

                az_diff = abs(track_az - det_az)
                az_diff = min(az_diff, 360 - az_diff)  # wrap-around correction
                el_diff = abs(track_el - det_el)

                # Azimuth gating
                # if az_diff > 30:
                #     continue

                dt = current_time-self.last_timestamp[i_rdr]
                manuaver_dopp_thr = 3
                # if self.tracks[i_rdr][i].assoc_dets > 10:
                #     manuaver_dopp_thr = 7
                range_innov = rng - self.tracks[i_rdr][i].range_val
                predicted_range_innov = dt*self.tracks[i_rdr][i].get_avg_doppler()
                dist_thr = self.dist_threshold

                if (abs(range_innov) < dist_thr and
                    dopp_dist < self.dopp_dist_threshold and
                    abs(range_innov - predicted_range_innov) < manuaver_dopp_thr):

                    self.tracks[i_rdr][i].update(det)
                    matched_tracks.add(i)
                    matched_detections.add(j)
                    used_range_doppler.add(rd_key)


        # Step 5: Unmatched tracks → increment missed counter
        for i, t in enumerate(self.tracks[i_rdr]):
            if i not in matched_tracks:
                t.missed += 1
                t.time_no_assoc = current_time - t.last_assoc_timestamp
                t.age = current_time - t.birth_time

        # Step 6: Unmatched detections → create new tracks
        for j, det in enumerate(detections):
            if j in matched_detections:
                continue
            rng = det['range']
            dop = det['doppler']
            rd_key = (round(rng, 2), round(dop, 2))

            if dop < 0 and rng < self.create_thresh_min_range:
                continue

            if rd_key in used_range_doppler:
                continue  # already used, skip track creation

            # Skip near-duplicates (soft gating)
            too_close = False
            for ar, ad in used_range_doppler:
                if abs(rng - ar) < self.duplicate_range_thresh:
                    if abs(dop - ad) < self.duplicate_doppler_thresh:
                        too_close = True
                        break
            if too_close:
                continue  # Skip — redundant detection
            # skip dets close to tracks
            close2trk = False
            for t in self.tracks[i_rdr]:
                range_diff = abs(rng - t.get_filtered_range())
                dopp_diff = abs(dop - t.get_filtered_doppler())
                if t.target_class == 't':
                    truck_factor = 3
                else:
                    truck_factor = 1
                is_close_rng = range_diff < self.create_rng2trk_thresh * truck_factor
                is_close_dop = dopp_diff < self.create_dop2trk_thresh

                if is_close_rng and is_close_dop:
                    close2trk = True
            if close2trk:
                continue
            self.tracks[i_rdr].append(ExtendedKalmanTrack_3D(det, self.next_id))

            self.next_id += 1
            used_range_doppler.add(rd_key)

        # Step 7: Filter invalid tracks
        filtered_tracks = self.filter_tracks(i_rdr)

        self.tracks[i_rdr] = filtered_tracks

        # Step 8: classify targets
        self.classify_tgts(i_rdr)

        self.last_timestamp[i_rdr] = current_time

        return self.tracks

    def filter_tracks(self, i_rdr):
        filtered_tracks = []
        for t in self.tracks[i_rdr]:
            x, y, z = t.get_position()
            range_val = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            vx, vy, vz= t.kf.x[3], t.kf.x[4], t.kf.x[5]
            # speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            if t.missed > self.max_missed:
                continue
            if t.time_no_assoc > 2:
                continue
            if range_val > self.max_range:
                continue
            if y < -10:
                continue
            # if np.abs(t.get_avg_doppler()) < 0.1:
            #     continue
            doppler = (vx * x + vy * y + vz * z) / range_val
            if t.target_class == 'n' and not t.was_associated and abs((range_val - t.last_range) - doppler * t.time_no_assoc) > 10:
                continue

            filtered_tracks.append(t)
        return filtered_tracks

    def filter_tracks_time_no_assoc(self, i_rdr):
        filtered_tracks = []
        for t in self.tracks[i_rdr]:
            if t.time_no_assoc > 2:
                continue
            filtered_tracks.append(t)
        return filtered_tracks
# ---------------------- side functions ----------------------

def classify_tgt4plot(track, thr_num_assoc4class_car = 4, thr_num_assoc4class_human = 1):
    if track.target_class == 't':
        t_class = 't'
        return t_class
    t_class = 'n'  # none
    if track.is_car_track(thr_num_assoc4class_car):
        t_class = 'c'
    if track.is_human_track(thr_num_assoc4class_human):
        t_class = 'h'
    return t_class

def show_reported_trks(report_tracks,tracks,dict_class = {'n': 'None', 'c': 'Car', 'h': 'Human', 't': 'Truck'}):
    if len(report_tracks) > 0:
        for trk in tracks:
            if trk.id in report_tracks:
                print('ID- ', trk.id, '  ,Type- ', dict_class[trk.target_class], '  , time to impact - ',
                      round(trk.t2a, 2), 's,  Range -  ', round(trk.range_val[0], 2), 'm,  Doppler -  ', round(-trk.last_doppler,2),'m/s')

def compute_azimuth_elevation(x, y, z):
    azimuth = math.degrees(math.atan2(y, x))             # [-180, 180]
    elevation = math.degrees(math.atan2(z, math.hypot(x, y)))  # [-90, 90]
    return azimuth, elevation