import csv
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import glob
import os
import re
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
import math
from tracker import compute_azimuth_elevation
from matplotlib import animation
from collections import defaultdict
from pathlib import Path
import pandas as pd
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
from typing import Iterable, List, Sequence, Tuple, Union





def extract_experiment_name(csv_path):

    filename = os.path.basename(csv_path)
    match = re.search(r'\d{8}_\d{6}_(.+)\.csv', filename)
    if not match:
        return ''

    name = match.group(1).replace('_', ' ')

    # If name starts with a number followed by space and text, add "-"
    match_num_prefix = re.match(r'^(\d+)\s+(.*)', name)
    if match_num_prefix:
        number, rest = match_num_prefix.groups()
        return f"{number} - {rest}"

    return name


def get_latest_detection_csv(folder="detections"):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No detection CSV files found.")
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def plot_range_vs_time_from_csv(csv_path):
    timestamps = []
    ranges = []
    dopplers = []
    snrs = []
    x_vals = []
    y_vals = []
    z_vals = []


    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                # doppler_val = float(row['doppler'])
                # if doppler_val != 0:
                #     continue  # Skip rows where doppler is not 0

                timestamps.append(float(row['timestamp']))
                ranges.append(float(row['range']))
                dopplers.append(float(row['doppler']))
                snrs.append(10*np.log10(float(row['snr'])))
                x_vals.append(float(row['x']))
                y_vals.append(float(row['y']))
                z_vals.append(float(row['z']))

            except ValueError:
                continue  # Skip malformed lines

    if not timestamps:
        print("No data found.")
        return

    # Normalize time to start at 0
    t0 = timestamps[0]
    relative_times = [t - t0 for t in timestamps]

    # fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # plot range
    scatter = ax1.scatter(relative_times, ranges, s=5, alpha=0.6, c='blue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Range (m)")
    ax1.set_title("Range vs. Time Tooltips")
    ax1.grid(True)

    scatter_dopp = ax2.scatter(relative_times, dopplers, s=5, alpha=0.6, c='blue')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Doppler (m/s)")
    ax2.set_title("Doppler vs. Time with Tooltips")
    ax2.grid(True)
    # Add tooltips showing doppler
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        doppler = dopplers[index]
        relative_time = relative_times[index]
        r = ranges[index]
        cur_snr = snrs[index]
        x_val = x_vals[index]
        y_val = y_vals[index]
        z_val = z_vals[index]
        sel.annotation.set_text(f"Doppler: {doppler:.1f} m/s  \n  Time: {relative_time: .2f} s \n  Range: {r: .2f} s \n SNR: {cur_snr: .2f} dB \n X: {x_val: .2f} \n Y: {y_val: .2f}\n Z: {z_val: .2f}")

    plt.show()

def plot_z_vs_time_from_csv(csv_path):
    timestamps = []
    ranges = []
    dopplers = []
    snrs = []
    x_vals = []
    y_vals = []
    z_vals = []

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                timestamps.append(float(row['timestamp']))
                ranges.append(float(row['range']))
                dopplers.append(float(row['doppler']))
                snrs.append(10 * np.log10(float(row['snr'])))
                x_vals.append(float(row['x']))
                y_vals.append(float(row['y']))
                z_vals.append(float(row['z']))

            except ValueError:
                continue  # Skip malformed lines

    if not timestamps:
        print("No data found.")
        return

    # Normalize time to start at 0
    t0 = timestamps[0]
    relative_times = [t - t0 for t in timestamps]

    # fig, ax = plt.subplots()
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    # plot range
    scatter = ax1.scatter(relative_times, z_vals, s=5, alpha=0.6, c='blue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Alt (m)")
    ax1.set_title("Altitude vs. Time Tooltips")
    ax1.grid(True)

    # Add tooltips showing doppler
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        doppler = dopplers[index]
        relative_time = relative_times[index]
        r = ranges[index]
        cur_snr = snrs[index]
        x_val = x_vals[index]
        y_val = y_vals[index]
        z_val = z_vals[index]
        sel.annotation.set_text(
            f"Doppler: {doppler:.1f} m/s  \n  Time: {relative_time: .2f} s \n  Range: {r: .2f} s \n SNR: {cur_snr: .2f} dB \n X: {x_val: .2f} \n Y: {y_val: .2f}\n Z: {z_val: .2f}")

    plt.show()


def plot_3d_detections_from_csv(csv_path, color_by='doppler', show=True):
    xs, ys, zs = [], [], []
    color_vals = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row['x']))
                ys.append(float(row['y']))
                zs.append(float(row['z']))
                if color_by == 'snr':
                    color_vals.append(20 * np.log10(float(row.get(color_by, 0))))  # default 0 if missing
                else:
                    color_vals.append(float(row.get(color_by, 0)))  # default 0 if missing
            except:
                continue

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(xs, ys, zs, c=color_vals, cmap='viridis', s=5)
    # Mark the radar origin (0,0,0)
    ax.scatter(0, 0, 0, c='red', s=50, marker='x', label='Radar Origin')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Detections Colored by {color_by}")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_by)

    if show:
        plt.tight_layout()
        plt.show()

    return fig


def plot_3d_detections_with_tracks(detection_csv, track_csv=None, color_by='doppler', show=True):
    xs, ys, zs = [], [], []
    dopplers, snrs_db = [], []
    color_vals = []
    azs, els = [] , []

    # --- Load detections ---
    with open(detection_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xs.append(float(row['x']))
                ys.append(float(row['y']))
                zs.append(-float(row['z']))

                dop = float(row.get('doppler', 0))
                snr = float(row.get('snr', 0))
                snr_db = 10 * math.log10(snr) if snr > 0 else 0
                dopplers.append(dop)
                snrs_db.append(snr_db)

                det_az, det_el = compute_azimuth_elevation(float(row['x']),float(row['y']),float(row['y']))
                azs.append(det_az)
                els.append(det_el)

                # color_vals.append(float(row.get(color_by, 0)))  # fallback if field missing
                # Default coloring
                val = snr_db if color_by.lower() == 'snr' else dop
                color_vals.append(val)
            except:
                continue

    # --- Create plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(xs, ys, zs, c=color_vals, cmap='viridis', s=5, alpha=0.6)

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(
            f"x: {xs[i]:.2f}, y: {ys[i]:.2f}, z: {zs[i]:.2f}\n"
            f"doppler: {dopplers[i]:.2f} m/s\n"
            f"SNR: {snrs_db[i]:.1f} dB\n"
            f"Az: {azs[i]:.1f} deg\n"
            f"El: {els[i]:.1f} deg"
        )

    # Mark the radar origin (0,0,0)
    ax.scatter(0, 0, 0, c='red', s=50, marker='x', label='Radar Origin')

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_by)
    ax.set_title("3D Detections" + (f" + Tracks" if track_csv else ""))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # --- Load and plot tracks if provided ---
    if track_csv:
        tracks = defaultdict(list)

        with open(track_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row['track_id'])
                    x = float(row['x'])
                    y = float(row['y'])
                    z = -float(row.get('z', 0.0))  # allow fallback if z is missing
                    t_class = row.get('class', '').strip().lower()

                    tracks[tid].append((x, y, z, t_class))
                except:
                    continue

        # Plot each track as a line
        for tid, entries  in tracks.items():
            if len(entries) < 2:
                continue
            x_vals, y_vals, z_vals, classes = zip(*entries)
            ax.plot(x_vals, y_vals, z_vals, label=f"Track {tid}", linewidth=1.5)

            # Plot special point for first 't' classification
            for i, cls in enumerate(classes):
                if cls == 'truck':
                    ax.plot([x_vals[i]], [y_vals[i]], [z_vals[i]], 'b*', markersize=10,
                            label='First truck')
                    break
        ax.legend(fontsize=8, loc='upper right')

    if show:
        plt.tight_layout()
        plt.show()

    return fig


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms_doppler0(csv_path, az_bins=36, range_bins=30):
    """
    Reads a CSV file with columns x, y, z, range, doppler.
    Filters to rows where doppler == 0.
    Computes azimuth angles (atan2(x, y)) and plots histograms of azimuth and range.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file (must contain columns 'x', 'y', 'z', 'range', 'doppler').
    az_bins : int, optional
        Number of bins for the azimuth histogram (default: 36, i.e. 10° bins).
    range_bins : int, optional
        Number of bins for the range histogram (default: 30).

    Returns:
    --------
    None
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter doppler == 0
    df = df[df["doppler"] == 0]

    if df.empty:
        print("No detections with doppler == 0 found.")
        return

    # Compute azimuth relative to y-axis (atan2(x, y))
    azimuth = np.arctan2(df['x'], df['y'])
    azimuth_deg = np.degrees(azimuth)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Azimuth histogram (centered at 0°)
    axes[0].hist(azimuth_deg, bins=az_bins, edgecolor="black", alpha=0.7)
    axes[0].set_xlim(-180, 180)
    axes[0].set_xticks(np.arange(-180, 181, 45))
    axes[0].set_xlabel("Azimuth (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Azimuth Histogram (doppler = 0, atan2(x, y))")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Range histogram
    axes[1].hist(df["range"], bins=range_bins, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Range Histogram (doppler = 0)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_histograms_doppler0(csv_path, az_bins=36, range_bins=50, smooth_window=10):
    """
    Reads a CSV file with columns x, y, z, range, doppler.
    Filters to rows where doppler == 0.
    Computes azimuth angles (atan2(x, y)) and plots histograms of azimuth and range.
    Adds a moving average curve (10-meter window by default) on the range histogram.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file (must contain columns 'x', 'y', 'z', 'range', 'doppler').
    az_bins : int, optional
        Number of bins for the azimuth histogram (default: 36).
    range_bins : int, optional
        Number of bins for the range histogram (default: 30).
    smooth_window : float, optional
        Length of moving average window in meters (default: 10).

    Returns:
    --------
    None
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter doppler == 0
    df = df[df["doppler"] == 0]
    df = df[df["range"] < 120]
    df = df[df["range"] > 50]


    if df.empty:
        print("No detections with doppler == 0 found.")
        return

    # Compute azimuth relative to y-axis (atan2(x, y))
    azimuth = np.arctan2(df['x'], df['y'])
    azimuth_deg = np.degrees(azimuth)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    # --- Azimuth histogram ---
    axes[0].hist(azimuth_deg, bins=az_bins, edgecolor="black", alpha=0.7)
    axes[0].set_xlim(-180, 180)
    axes[0].set_xticks(np.arange(-180, 181, 45))
    axes[0].set_xlabel("Azimuth (degrees)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Azimuth Histogram (doppler = 0, atan2(x, y))")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # --- Range histogram ---
    counts, bin_edges, _ = axes[1].hist(df["range"], bins=range_bins, edgecolor="black", alpha=0.7)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute bin width in meters
    bin_width = bin_edges[1] - bin_edges[0]
    window_bins = max(1, int(round(smooth_window / bin_width)))

    # Moving average of counts
    smooth_counts = np.convolve(counts, np.ones(window_bins)/window_bins, mode="same")

    # Overlay the moving average line
    axes[1].plot(centers, smooth_counts, color="red", linewidth=2, label=f"{smooth_window} m moving average")

    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Range Histogram (doppler = 0)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_topview_heatmap(
    csv_path,
    doppler_filter=0,          # set to None to use all detections
    x_col="x",
    y_col="y",
    gridsize=100,              # hex cells across the x-axis (increase for finer detail)
    log=True,                  # log color scale helps when counts vary a lot
    range_m=None,              # if set (e.g., 50), fixes axes to [-range_m, +range_m] for both x and y
    cmap="viridis",            # any Matplotlib colormap name
    save_path=None             # e.g., "topview_heatmap.png" to save instead of just showing
):
    """
    Top view density heat map of detections (x vs y).

    Parameters
    ----------
    csv_path : str
        Path to the CSV with at least columns x, y (and optionally doppler).
    doppler_filter : int|float|None
        If not None, keep only rows where 'doppler' == doppler_filter. Default: 0.
    x_col, y_col : str
        Column names for coordinates. Defaults: 'x', 'y'.
    gridsize : int
        Number of hexagons across the x-axis (controls resolution).
    log : bool
        Use logarithmic color scaling on counts.
    range_m : float|None
        If provided, sets both axes to [-range_m, +range_m].
    cmap : str
        Matplotlib colormap.
    save_path : str|None
        If provided, saves the figure to this path.

    Returns
    -------
    None
    """
    # Load
    df = pd.read_csv(csv_path)

    # Optional doppler filtering
    if doppler_filter is not None and "doppler" in df.columns:
        df = df[df["doppler"] == doppler_filter]

    # Guard rails
    if df.empty:
        print("No detections to plot after filtering.")
        return

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must contain '{x_col}' and '{y_col}' columns.")

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Compute extent
    if range_m is not None:
        xlim = (-range_m, range_m)
        ylim = (-range_m, range_m)
    else:
        # Auto with small padding
        pad_x = (np.nanmax(x) - np.nanmin(x)) * 0.05 if np.nanmax(x) != np.nanmin(x) else 1.0
        pad_y = (np.nanmax(y) - np.nanmin(y)) * 0.05 if np.nanmax(y) != np.nanmin(y) else 1.0
        xlim = (np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
        ylim = (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(
        x, y,
        gridsize=gridsize,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        C=None, reduce_C_function=np.sum,  # counts per cell
        bins="log" if log else None,
        cmap=cmap
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Detections per cell" + (" (log scale)" if log else ""))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title_dopp = f" (doppler = {doppler_filter})" if doppler_filter is not None and "doppler" in df.columns else ""
    ax.set_title(f"Top View Heat Map of Detections{title_dopp}")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved heat map to {save_path}")
    else:
        plt.show()



def plot_topview_with_cluster_circles(
    csv_path,
    doppler_filter=0,           # set to None to skip filtering
    x_col="x",
    y_col="y",
    gridsize=20,
    log=True,
    range_m=None,               # e.g., 50 to clamp axes to [-50, 50]
    cmap="viridis",
    eps=3.0,                    # DBSCAN neighborhood radius in meters
    min_samples=50,             # minimum points to form a cluster
    radius_quantile=0.90,       # circle radius = this quantile of distances to centroid (0..1)
    circle_linewidth=2.0,
    circle_alpha=0.9,
    annotate=True
):
    """
    Plot top-view hexbin heat map and mark dense areas with circles using DBSCAN.

    Parameters
    ----------
    csv_path : str
        Path to CSV with columns x, y (and optionally doppler).
    doppler_filter : int|float|None
        If not None and 'doppler' exists, keep only rows with doppler == doppler_filter.
    gridsize : int
        Hexbin resolution across x-axis.
    log : bool
        Log color scale for hexbin.
    range_m : float|None
        If provided, square extent [-range_m, +range_m] on both axes.
    eps : float
        DBSCAN eps (meters).
    min_samples : int
        DBSCAN min_samples.
    radius_quantile : float
        Circle radius as given quantile (e.g., 0.90) of distances from centroid.
        Use 1.0 to enclose all points; smaller values give tighter circles.
    annotate : bool
        If True, annotate circle centers with cluster sizes.

    Returns
    -------
    clusters : list of dict
        Each dict has keys: 'label', 'center_x', 'center_y', 'radius', 'size'
    """
    # --- Load & filter
    df = pd.read_csv(csv_path)
    if doppler_filter is not None and "doppler" in df.columns:
        df = df[df["doppler"] == doppler_filter]
    df = df[np.abs(df["x"]) < 50]

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must contain '{x_col}' and '{y_col}' columns.")
    if df.empty:
        print("No detections to plot after filtering.")
        return []

    pts = df[[x_col, y_col]].to_numpy()

    # --- Compute extent
    if range_m is not None:
        xlim = (-range_m, range_m)
        ylim = (-range_m, range_m)
    else:
        x, y = pts[:, 0], pts[:, 1]
        pad_x = (np.nanmax(x) - np.nanmin(x)) * 0.05 if np.nanmax(x) != np.nanmin(x) else 1.0
        pad_y = (np.nanmax(y) - np.nanmin(y)) * 0.05 if np.nanmax(y) != np.nanmin(y) else 1.0
        xlim = (np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
        ylim = (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)

    # --- Plot base heat map
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(
        pts[:, 0], pts[:, 1],
        gridsize=gridsize,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        bins="log" if log else None,
        cmap=cmap
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Detections per cell" + (" (log scale)" if log else ""))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title_dopp = f" (doppler = {doppler_filter})" if doppler_filter is not None and "doppler" in df.columns else ""
    ax.set_title(f"Top View Heat Map with Cluster Circles{title_dopp}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Define x values
    x_road = np.linspace(-100, 100, 100)
    d = 10
    theta = 12*np.pi/180

    road_width = 8
    r_w = road_width/(2*np.cos(theta))
    n = (d)*np.cos(theta)/np.sin(theta)

    n1 = (d+r_w)*np.cos(theta)/np.sin(theta)
    n2 = (d-r_w)*np.cos(theta)/np.sin(theta)

    m = - np.cos(theta)/np.sin(theta)
    # Define the equation of the line, e.g., y = 2x + 1
    y_road = m * x_road + n

    y_road1 = m * x_road + n1
    y_road2 = m * x_road + n2

    # Plot the line
    ax.plot(x_road, y_road)
    ax.plot(x_road, y_road1)
    ax.plot(x_road, y_road2)


    # --- Cluster with DBSCAN
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn") from e

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique_labels = [lab for lab in np.unique(labels) if lab != -1]  # -1 = noise

    clusters = []
    for lab in unique_labels:
        mask = labels == lab
        cluster_pts = pts[mask]
        size = cluster_pts.shape[0]

        # centroid
        cx, cy = cluster_pts.mean(axis=0)

        # radius as quantile of radial distances to centroid
        dists = np.sqrt((cluster_pts[:, 0] - cx) ** 2 + (cluster_pts[:, 1] - cy) ** 2)
        q = np.clip(radius_quantile, 0.0, 1.0)
        radius = float(np.quantile(dists, q))

        # draw circle
        circ = Circle(
            (cx, cy),
            radius,
            fill=False,
            linewidth=circle_linewidth,
            alpha=circle_alpha
        )
        ax.add_patch(circ)

        # label cluster size
        if annotate:
            ax.text(cx, cy, str(size), ha="center", va="center", fontsize=10, fontweight="bold")

        clusters.append({
            "label": int(lab),
            "center_x": float(cx),
            "center_y": float(cy),
            "radius": radius,
            "size": int(size),
        })

    # Legend/info
    if len(clusters) == 0:
        ax.text(0.02, 0.98, "No clusters found", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    else:
        ax.text(0.02, 0.98, f"{len(clusters)} clusters", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    plt.show()

    return clusters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_topview_with_spectral_circles(
    csv_path,
    doppler_filter=0,           # set to None to skip filtering
    x_col="x",
    y_col="y",
    gridsize=100,
    log=True,
    range_m=None,               # e.g., 50 to clamp axes to [-50, 50]
    cmap="viridis",

    # --- Spectral clustering controls ---
    n_clusters=None,            # if None, auto-estimate via simple search over k=2..k_max
    k_max=8,                    # max clusters to consider when auto-estimating
    eps=2.0,                    # local length scale in meters for affinity
    use_neighbor_mask=True,     # sparsify affinity by keeping neighbors within ~2*eps
    gamma=None,                 # if set, use RBF gamma directly; else derived from eps

    # --- Circle rendering ---
    radius_quantile=0.90,       # circle radius = this quantile of distances to centroid
    circle_linewidth=2.0,
    circle_alpha=0.9,
    annotate=True
):
    # --- Load & filter
    df = pd.read_csv(csv_path)
    if doppler_filter is not None and "doppler" in df.columns:
        df = df[df["doppler"] == doppler_filter]

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must contain '{x_col}' and '{y_col}' columns.")
    if df.empty:
        print("No detections to plot after filtering.")
        return []

    pts = df[[x_col, y_col]].to_numpy()

    # --- Compute extent
    if range_m is not None:
        xlim = (-range_m, range_m)
        ylim = (-range_m, range_m)
    else:
        x, y = pts[:, 0], pts[:, 1]
        pad_x = (np.nanmax(x) - np.nanmin(x)) * 0.05 if np.nanmax(x) != np.nanmin(x) else 1.0
        pad_y = (np.nanmax(y) - np.nanmin(y)) * 0.05 if np.nanmax(y) != np.nanmin(y) else 1.0
        xlim = (np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
        ylim = (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)

    # --- Base heat map
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(
        pts[:, 0], pts[:, 1],
        gridsize=gridsize,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        bins="log" if log else None,
        cmap=cmap
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Detections per cell" + (" (log scale)" if log else ""))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title_dopp = f" (doppler = {doppler_filter})" if doppler_filter is not None and "doppler" in df.columns else ""
    ax.set_title(f"Top View Heat Map with Spectral Clusters{title_dopp}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.4)

    # --- Spectral Clustering bits
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.manifold import spectral_embedding
        from sklearn.metrics import pairwise_distances
        from sklearn.cluster import KMeans
        from scipy.sparse import csr_matrix, issparse
    except Exception as e:
        raise ImportError(
            "Needs scikit-learn and scipy. Install with:\n"
            "  pip install scikit-learn scipy"
        ) from e

    # Pairwise distances
    D = pairwise_distances(pts, metric="euclidean")

    # RBF gamma from eps if not provided
    if gamma is None:
        gamma = 1.0 / (2.0 * max(eps, 1e-6) ** 2)

    # RBF affinity
    A = np.exp(-gamma * (D ** 2))

    # Optional sparsification by distance threshold (~2*eps)
    if use_neighbor_mask:
        neighbor_radius = 2.0 * eps
        mask = (D <= neighbor_radius).astype(np.float32)
        A = A * mask
        np.fill_diagonal(A, 1.0)
        A = csr_matrix(A)

    # Ensure symmetry (important if you sparsify)
    if issparse(A):
        A = (A + A.T) * 0.5
    else:
        A = 0.5 * (A + A.T)

    # Helper to call spectral_embedding across sklearn versions
    def _spectral_embed(adj, k):
        try:
            # Newer sklearn: argument name is 'adjacency'
            return spectral_embedding(adjacency=adj, n_components=k, eigen_solver="arpack", random_state=0)
        except TypeError:
            # Older sklearn: argument name was 'affinity'
            return spectral_embedding(affinity=adj, n_components=k, eigen_solver="arpack", random_state=0)

    if n_clusters is None:
        # Auto-pick k by trying 2..k_max and choosing smallest KMeans inertia in embedding space
        k_candidates = list(range(2, min(k_max, len(pts) - 1) + 1))
        best_k, best_inertia, best_labels = None, np.inf, None
        for k in k_candidates:
            emb = _spectral_embed(A, k)
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels_k = km.fit_predict(emb)
            if km.inertia__ < best_inertia:
                best_inertia = km.inertia__
                best_k = k
                best_labels = labels_k
        labels = best_labels
        est_k = best_k if best_k is not None else 2
    else:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0
        )
        labels = sc.fit_predict(A)
        est_k = n_clusters

    unique_labels = [lab for lab in np.unique(labels) if lab >= 0]

    # --- Draw circles at centroids
    clusters = []
    for lab in unique_labels:
        cluster_pts = pts[labels == lab]
        size = cluster_pts.shape[0]
        if size == 0:
            continue

        cx, cy = cluster_pts.mean(axis=0)
        dists = np.sqrt((cluster_pts[:, 0] - cx) ** 2 + (cluster_pts[:, 1] - cy) ** 2)
        radius = float(np.quantile(dists, float(np.clip(radius_quantile, 0.0, 1.0))))

        circ = Circle((cx, cy), radius, fill=False, linewidth=circle_linewidth, alpha=circle_alpha)
        ax.add_patch(circ)

        if annotate:
            ax.text(cx, cy, f"{size}", ha="center", va="center", fontsize=10, fontweight="bold")

        clusters.append({
            "label": int(lab),
            "center_x": float(cx),
            "center_y": float(cy),
            "radius": radius,
            "size": int(size),
        })

    if len(clusters) == 0:
        ax.text(0.02, 0.98, "No clusters found", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    else:
        ax.text(0.02, 0.98, f"{len(clusters)} clusters (k={est_k})", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    plt.show()

    return clusters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rotated_slope(m, delta_deg):
    theta = np.arctan(m)  # current angle
    theta_new = theta + np.deg2rad(delta_deg)
    if np.isclose(np.cos(theta_new), 0.0):
        return np.inf  # vertical line
    return np.tan(theta_new)

def line_segment(m, y_cap, n=200):
    if np.isinf(m):
        x_vals = np.zeros(n)
        y_vals = np.linspace(0.0, y_cap, n)
    elif m == 0:
        x_vals = np.linspace(0.0, y_cap, n)
        y_vals = np.zeros(n)
    else:
        x_end_local = y_cap / m
        x0, x1 = min(0.0, x_end_local), max(0.0, x_end_local)
        x_vals = np.linspace(x0, x1, n)
        y_vals = m * x_vals
        mask = (y_vals >= 0) & (y_vals <= y_cap)
        x_vals, y_vals = x_vals[mask], y_vals[mask]
    return x_vals, y_vals

def plot_topview_scatter(
    csv_path,
    x_col="x",
    y_col="y",
    doppler_filter=None,       # e.g., 0 to keep only doppler==0; None = no filter
    color_by=None,             # None | "range" | "doppler" | any numeric column name
    cmap="viridis",
    point_size=8,
    alpha=0.6,
    range_m=None,              # e.g., 50 -> axes set to [-50, 50] in both x and y
    max_points=500_000,        # downsample if more than this many points
    random_state=0,            # for reproducible downsampling
    title_suffix="",            # extra text for the title
    rdr_side = 'right',
    dist_from_road = 8
):
    """
    Plot a simple top-view scatter of detections (x vs y).

    Parameters
    ----------
    csv_path : str
        Path to the CSV (must include x_col and y_col).
    doppler_filter : int|float|None
        If not None and 'doppler' exists, keep rows where doppler == doppler_filter.
    color_by : str|None
        Column to color points by (e.g., "range", "doppler"). Must be numeric.
    range_m : float|None
        If provided, fixes both axes to [-range_m, +range_m].
    max_points : int
        If the dataset is larger, a random subset of size max_points is plotted.
    """
    # Load
    df = pd.read_csv(csv_path)
    # Add range (radial distance)
    df["range"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    # Add azimuth (angle in XY-plane, in radians)
    df["azimuth"] = np.degrees(np.arctan2(df["y"], df["x"]))

    # Optional doppler filter
    if doppler_filter is not None and "doppler" in df.columns:
        df = df[df["doppler"] == doppler_filter]

    # Ensure required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must contain '{x_col}' and '{y_col}' columns.")

    # Optional coloring
    c = None
    if color_by is not None:
        if color_by not in df.columns:
            print(f"Warning: '{color_by}' not found; plotting without color mapping.")
        elif not np.issubdtype(df[color_by].dtype, np.number):
            print(f"Warning: '{color_by}' is not numeric; plotting without color mapping.")
        else:
            c = df[color_by].to_numpy()

    # Downsample if very large
    n = len(df)
    if n > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_points, replace=False)
        df = df.iloc[idx]
        if c is not None:
            c = c[idx]
        print(f"Downsampled {n:,} → {len(df):,} points for plotting.")

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Figure & axes
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8, 8))
    sc = ax1.scatter(x, y, s=point_size, c=c, cmap=cmap if c is not None else None, alpha=alpha, edgecolors="none")

    # Define x values
    d = dist_from_road
    theta = 18*np.pi/180
    # rdr_dir = 'left'
    rdr_dir = rdr_side
    if rdr_dir == 'right':
        sgn = -1
    else:
        sgn = 1
    road_width = 8
    r_w = road_width/(2*np.cos(theta))
    n = (d)*np.cos(theta)/np.sin(theta)

    n1 = (d+r_w)*np.cos(theta)/np.sin(theta)
    n2 = (d-r_w)*np.cos(theta)/np.sin(theta)

    m = - sgn*np.cos(theta)/np.sin(theta)

    x_start = -n/m
    x_end = (100 - n)/m
    min_x = min(x_start, x_end)
    max_x = max(x_start, x_end)
    x_road = np.linspace(min_x, max_x, 10)

    x1_start = -n1/m
    x1_end = (100 - n1)/m
    min_x1 = min(x1_start, x1_end)
    max_x1 = max(x1_start, x1_end)
    x1_road = np.linspace(min_x1, max_x1, 10)

    x2_start = -n2/m
    x2_end = (100 - n2)/m
    min_x2 = min(x2_start, x2_end)
    max_x2 = max(x2_start, x2_end)
    x2_road = np.linspace(min_x2, max_x2, 10)

    # Define the equation of the line, e.g., y = 2x + 1
    y_road = m * x_road + n

    y_road1 = m * x1_road + n1
    y_road2 = m * x2_road + n2

    # Plot the line
    ax1.plot(x_road, y_road)
    ax1.plot(x1_road, y_road1)
    ax1.plot(x2_road, y_road2)

    all_lines = []
    ranges = []
    for i_n1_dir in range(len(x_road)):
        x1 = x_road[i_n1_dir]
        y1 = y_road1[i_n1_dir]
        if y1<10 or y1>100:
            continue
        ax1.scatter(x1,y1,s=20,c="black")

        dir_line_n = 0
        dir_line_m = y1/x1
        all_lines.append(dir_line_m)
        ranges.append(np.sqrt(x1**2 + y1**2))
        if i_n1_dir == 8:
            x_start = 0
            x_end = 100 / dir_line_m
            min_x_up = min(x_start, x_end)
            max_x_up = max(x_start, x_end)
            x_az = np.linspace(min_x_up, max_x_up, 20)
            y_az = dir_line_m * x_az
            mask1_az = np.sqrt(y_az ** 2 + x_az ** 2) > np.sqrt(x1 ** 2 + y1 ** 2) - 3
            mask_all_az = mask1_az
            ax1.plot(x_az[mask_all_az],y_az[mask_all_az])

            beam_w_idx = np.linspace(-10, 10, 9)

            m_plus = rotated_slope(dir_line_m, +act_func(beam_w_idx[i_n1_dir],rdr_side))
            m_minus = rotated_slope(dir_line_m, -act_func(beam_w_idx[i_n1_dir],rdr_side))

            x_plus, y_plus = line_segment(m_plus, 100, 20)
            x_minus, y_minus = line_segment(m_minus, 100, 20)
            ax1.plot(x_plus, y_plus, c="red", label="+10°")
            ax1.plot(x_minus, y_minus, c="red", label="-10°")


    line_az = np.array([(np.degrees(np.arctan(m)) + 180) % 180 for m in all_lines])
    beam_w_idx = np.linspace(-10,10,len(line_az))
    beam_w = [act_func(i ,rdr_side) for i in beam_w_idx]

    visability = []
    r_neighbor = 1.0  # meters, neighborhood radius for KDTree

    # Pre-extract azimuth array once for faster masking
    azimuth_all = df["azimuth"].to_numpy(float, copy=False)
    for i, az in enumerate(line_az):
        # Keep the original print (even when debug_mode=False) to match behavior
        print(f"beam: {np.round(beam_w[i])}  az = {np.round(az)}")

        bw = beam_w[i]
        az_lo, az_hi = az - bw, az + bw

        # Angular filter
        mask = (azimuth_all < az_hi) & (azimuth_all > az_lo)
        filt_df = df.loc[mask]

        # Only x/y for KDTree
        pts = filt_df[["x", "y"]].to_numpy(float, copy=False)

        # Build KDTree and count neighbors within r (includes self)
        # (same logic; mask points that have >= 1 other neighbor)
        tree = cKDTree(pts)
        counts = tree.query_ball_point(pts, r=r_neighbor, return_length=True)
        has_neighbor = counts > 1

        df_kept = filt_df.loc[has_neighbor].copy()

        len_az = len(df_kept)
        if len_az == 0:
            visability.append((ranges[i], 1))
            continue

        # Keep points above lower edge line (same inequality)
        df_kept = df_kept[df_kept.y > m * df_kept.x + n2]

        visability.append((ranges[i], len(df_kept) / len_az))

    # Colorbar if colored
    if c is not None:
        cb = plt.colorbar(sc, ax=ax1)
        cb.set_label(color_by)

    # Axes formatting
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    if range_m is not None:
        ax1.set_xlim(-range_m, range_m)
        ax1.set_ylim(-range_m, range_m)
    else:
        # small padding
        pad_x = (np.nanmax(x) - np.nanmin(x)) * 0.05 if np.nanmax(x) != np.nanmin(x) else 1.0
        pad_y = (np.nanmax(y) - np.nanmin(y)) * 0.05 if np.nanmax(y) != np.nanmin(y) else 1.0
        ax1.set_xlim(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
        ax1.set_ylim(np.nanmin(y), np.nanmax(y) + pad_y)

    title_bits = ["Top View Scatter"]
    if doppler_filter is not None and "doppler" in df.columns:
        title_bits.append(f"(doppler = {doppler_filter})")
    if title_suffix:
        title_bits.append(str(title_suffix))
    ax1.set_title(" ".join(title_bits))

    ax1.grid(True, linestyle="--", alpha=0.4)

    r_vis, n_vis = zip(*visability)

    ax2.plot(r_vis, np.array(n_vis) * 100, marker="o")
    ax2.axhline(y=50, color="r", linestyle="--", label="y = 50")  # --- line at y=100
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 100)

    ax2.set_xlabel("Ranges to road (meters)")
    ax2.set_ylabel("Visability Percentage (%)")
    ax2.set_title("Visability")
    ax2.grid(True)


    plt.tight_layout()
    plt.show()

    return visability

Number = Union[int, float, np.number]

def act_func(z: Number, rdr_side) -> float:
    """
    Original activation/beam-width function.
    Note: kept as-is to preserve exact behavior.
    """
    sigmo = 1 / (1 + np.exp(-z))
    if rdr_side == 'right':
        return (1-sigmo) * 15 + 5
    return sigmo * 15 + 5

if __name__ == '__main__':
    # path2cvs = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250423_144609.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250508_164913_human_and_cars_aproching.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250508_165809_car.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_142817_car_inbound.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_172359_human_walking_60m.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_171444_human_and_car.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_20250511_172359_human_walking_60m.csv"

    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_163410.csv"


    # data from day tests -  22/05/2025
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_093816_1_man1_walk_from_70.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_093920_2_man1_walk_to_70.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_105007_3_man_walk_30_degree_to_70m.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_105113_4_man_walk_30_degree_from_70m.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_094309_5_man2_walk_to_70.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250522_094427_6_man2_walk_from_70_to_sensor.csv"
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

    # path2dets = get_latest_detection_csv(r"C:\Users\admin\PycharmProjects\shula\.venv\records")

    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_TI20250617_153331.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_153052_3_Taavura_truck_inbound_0_deg_60kmh.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_154808_7_car_and_Taavura_truck_inbound_10_deg_50_kmh.csv"
    # path2dets = r"C:\Users\admin\PycharmProjects\shula\.venv\records\detections_test_20250617_155343_full_trailer_0_deg_30_kmh.csv"

    # plot_range_vs_time_from_csv(path2dets)
    # plot_z_vs_time_from_csv(path2dets)

    # plot_3d_detections_from_csv(path2dets,color_by='timestamp')


    # output_tracks_path = r"C:\Users\admin\PycharmProjects\shula\.venv\playback_tracks\tracks_log_01.csv"
    #
    # plot_3d_detections_with_tracks(
    #     detection_csv=path2dets,
    #     track_csv=output_tracks_path,
    #     color_by='timestamp'
    # )
    #
    # path2dets = path2dets = Path(r"../records/20250902_100235/det.csv")
    path2dets = Path(r"../records/20250902_105415/det.csv") # open road
    # path2dets = Path(r"../records/20250902_143716/det.csv") # inside house

    path2dets = Path(r"../records/20250916_105707/det.csv")  # Duration: 00:09 | FPS: 5.00 | dets: 1224

    plot_range_vs_time_from_csv(path2dets)

    # plot_3d_detections_from_csv(path2dets, color_by="snr")

    # plot_histograms_doppler0(path2dets)

    # clusters = plot_topview_with_cluster_circles(path2dets)

    # vis = plot_topview_scatter(path2dets,rdr_side='right')
    # pass
    # plot_topview_with_spectral_circles(path2dets)
    # r_vis, n_vis = zip(*visability)
    # # r_vis = list(reversed(r_vis))
    # # n_vis = list(reversed(n_vis))
    # # Figure & axes
    # # Plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(r_vis, np.array(n_vis)*100, marker="o")
    # plt.axhline(y=50, color="r", linestyle="--", label="y = 50")  # --- line at y=100
    # plt.ylim(0,100)
    # plt.xlim(0,100)
    #
    # plt.xlabel("Ranges to road (meters)")
    # plt.ylabel("Visability Percentage (%)")
    # plt.title("Visability")
    # plt.grid(True)
    # plt.show()
