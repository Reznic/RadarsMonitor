import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


Number = Union[int, float, np.number]


def act_func(z: Number, rdr_side, near_angle = 20, far_angle = 7) -> float:
    """
    Original activation/beam-width function.
    Note: kept as-is to preserve exact behavior.
    """
    sigmo = 1 / (1 + np.exp(-z))
    if rdr_side == 'right':
        return (1-sigmo) * (near_angle-far_angle) + far_angle
    return sigmo * (near_angle-far_angle) + far_angle

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

def _x_span_for_line(n: float, m: float, y_min: float, y_max: float, num: int) -> np.ndarray:
    """
    Given line y = m x + n, return x samples whose y-range sweeps [y_min, y_max]
    (used to generate the road and road edges).
    """
    # Invert y = m x + n  ->  x = (y - n)/m
    x_start = (y_min - n) / m
    x_end = (y_max - n) / m
    lo, hi = (x_start, x_end) if x_start <= x_end else (x_end, x_start)
    return np.linspace(lo, hi, num)


def _pad_limits(values: np.ndarray, frac: float = 0.05) -> Tuple[float, float]:
    """Small symmetric padding around [min, max] for plotting."""
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return vmin - 1.0, vmax + 1.0
    pad = (vmax - vmin) * frac
    return vmin - pad, vmax + pad


def get_vis(
    csv_path: Union[str, Path],
    dist2road: Number = 10,
    rdr_side: str = "left",
    road_width: Number = 8,
    rdr2road_angle: float = 18,
    doppler_filter: Union[Number, None] = 0,
    debug_mode: bool = True,
    num_road_samples: int = 20,
) -> List[Tuple[float, float]]:
    """
    Compute visibility along a slanted road segment relative to radar points.

    Returns
    -------
    visability : list of (range_to_sample_point_m, visibility_ratio)
        Same name/format as the original (typo preserved).
    """
    # ---------------------------
    # Load and basic derived cols
    # ---------------------------
    df = pd.read_csv(csv_path)

    # Add range (radial distance) and azimuth (degrees in XY-plane)
    # (use vectorized numpy ops; identical math to original)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float, copy=False)
    df["range"] = np.sqrt((xyz ** 2).sum(axis=1))
    df["azimuth"] = np.degrees(np.arctan2(df["y"].to_numpy(), df["x"].to_numpy()))

    # Optional doppler filtering (kept identical)
    if doppler_filter is not None and "doppler" in df.columns:
        df = df[df["doppler"] == doppler_filter]

    # Precache x/y for plotting extents (avoids repeated column hits)
    x_vals = df["x"].to_numpy(float, copy=False)
    y_vals = df["y"].to_numpy(float, copy=False)
    include_z_in_y = True
    if include_z_in_y:
        z_vals = df["z"].to_numpy(float, copy=False)
        y_vals = np.sqrt(y_vals**2 + z_vals**2)

    # ---------------------------
    # Geometry of the roadway
    # ---------------------------
    d = float(dist2road)
    theta = math.radians(rdr2road_angle)

    sgn = -1 if rdr_side == "right" else 1
    m = -sgn * (math.cos(theta) / math.sin(theta))  # line slope

    # Offsets (n) for center and lane edges: y = m x + n
    # Keep exact expressions from original
    r_w = float(road_width) / (2 * math.cos(theta))
    n_center = d * math.cos(theta) / math.sin(theta)
    n_edge1 = (d + r_w) * math.cos(theta) / math.sin(theta)
    n_edge2 = (d - r_w) * math.cos(theta) / math.sin(theta)

    # Prepare x-samples for the visible road band (y in [10, 100] meters)
    x_center = _x_span_for_line(n_center, m, 0, 100, num_road_samples)
    x_edge1 = _x_span_for_line(n_edge1, m, 0, 100, num_road_samples)
    x_edge2 = _x_span_for_line(n_edge2, m, 0, 100, num_road_samples)

    # Corresponding y for each line
    y_center = m * x_center + n_center
    y_edge1 = m * x_edge1 + n_edge1
    y_edge2 = m * x_edge2 + n_edge2

    # ---------------------------
    # Debug plotting (top view + visibility curve)
    # ---------------------------
    if debug_mode:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.scatter(x_vals, y_vals, s=8, alpha=0.6, edgecolors="none")
        ax1.plot(x_center, y_center)
        ax1.plot(x_edge1, y_edge1)
        ax1.plot(x_edge2, y_edge2)

    # -----------------------------------------
    # Build list of ray directions from edge1
    # -----------------------------------------
    all_slopes: List[float] = []
    line_az = []
    ranges: List[float] = []

    # Note: keep the per-sample loop to preserve original behavior (including
    # potential divide-by-zero semantics and the y-window filter).
    for xi, yi in zip(x_center, y_edge1):
        if yi < 10 or yi > 100:
            continue
        if debug_mode:
            # Visual cue of sample points on the upper edge
            ax1.scatter([xi], [yi], s=20, c="black")

        # Direction slope from origin to the sample (unchanged)
        dir_m = yi / xi
        all_slopes.append(dir_m)
        line_az.append((np.degrees(np.arctan(dir_m)) + 180) % 180)
        ranges.append(float(math.hypot(xi, yi)))

        # if debug_mode and len(line_az) ==18:
        #     beam_w_idx = np.linspace(-10, 10, 18)
        #     m_plus = rotated_slope(dir_m, +act_func(beam_w_idx[17], rdr_side))
        #     m_minus = rotated_slope(dir_m, -act_func(beam_w_idx[17], rdr_side))
        #
        #     x_plus, y_plus = line_segment(m_plus, 100, 20)
        #     x_minus, y_minus = line_segment(m_minus, 100, 20)
        #     ax1.plot(x_plus, y_plus, c="red", label="+10°")
        #     ax1.plot(x_minus, y_minus, c="red", label="-10°")

    # # Convert direction slopes to azimuth angles in degrees (same math)
    # line_az = np.array([(np.degrees(np.arctan(s)) + 180) % 180 for s in all_slopes], dtype=float)

    # Beam width per azimuth index — preserve original indexing and act_func usage
    beam_w_idx = np.linspace(-10, 10, len(line_az))
    beam_w: List[float] = [act_func(i,rdr_side) for i in beam_w_idx]

    # -----------------------------------------
    # Visibility computation per beam
    # -----------------------------------------
    visability: List[Tuple[float, float]] = []
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
        pts = filt_df[["x", "y", "z"]].to_numpy(float, copy=False)

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
        df_kept = df_kept[df_kept.y > m * df_kept.x + n_edge2]

        visability.append((ranges[i], len(df_kept) / len_az))

    # ---------------------------
    # Debug plots: axis cosmetics
    # ---------------------------
    if debug_mode and visability:
        r_vis, n_vis = zip(*visability)

        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("X")

        if include_z_in_y:
            ax1.set_ylabel(r"$\sqrt{y^2 + z^2}$")
        else:
            ax1.set_ylabel("Y")

        # xlim = _pad_limits(x_vals, 0.05)
        # ylim = _pad_limits(y_vals, 0.05)
        ax1.set_xlim(-50,50)
        ax1.set_ylim(0, 100)

        title_bits = ["Top View Scatter"]
        if doppler_filter is not None and "doppler" in df.columns:
            title_bits.append(f"(doppler = {doppler_filter})")
        ax1.set_title(" ".join(title_bits))
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2.plot(r_vis, np.array(n_vis) * 100, marker="o")
        ax2.axhline(y=50, color="r", linestyle="--", label="y = 50")
        ax2.set_ylim(0, 100)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Ranges to road (meters)")
        ax2.set_ylabel("Visability Percentage (%)")
        ax2.set_title("Visability")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return visability


if __name__ == "__main__":
    # path2dets = Path(r"./records/20250902_105415/det.csv")  # open road
    # path2dets = Path(r"./records/20250902_143716/det.csv")  # inside house

    # path2dets = Path(r"./records/20250908_134001/det.csv")  # testing

    # bag interference (1m)
    path2dets = Path(r"./records/20250908_131119/det.csv")  # Duration: 00:15 | FPS: 10.00 | dets: 2504
    # path2dets = Path(r"./records/20250908_131224/det.csv")  # Duration: 00:24 | FPS: 20.00 | dets: 2231
    # path2dets = Path(r"20250908_131342")  # Duration: 00:32 | FPS: 20.00 | dets: 3522

    # radar on ground
    # path2dets = Path(r"20250908_133248")  # Duration: 00:16 | FPS: 10.00 | dets: 1880
    # path2dets = Path(r"20250908_133414")  # Duration: 00:10 | FPS: 20.00 | dets: 1198

    # regular data
    # path2dets = Path(r"../records/20250908_133535")  # Duration: 00:36 | FPS: 20.00 | dets: 4932
    # path2dets = Path(r"20250908_134001")  # Duration: 00:57 | FPS: 20.00 | dets: 10806

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
    # path2dets = Path(r"./records/20250908_141045/det.csv")  # Duration: 00:25 | FPS: 20.00 | dets: 685
    # path2dets = Path(r"20250908_141122")  # Duration: 00:31 | FPS: 10.00 | dets: 2653



    vis = get_vis(
        path2dets,
        dist2road=8,
        rdr_side="left",
        road_width=8,
        rdr2road_angle=12,
        doppler_filter=0,
        debug_mode=True
    )
    pass
