import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline

from GetData import SESSIONS_DIR, YEAR_FILE_PATTERN, openf1_get

PLOTS_DIR = "Plots"

# OpenF1's /location x, y, z are in tenths of a meter, not meters -
# confirmed by comparing summed chord length against known official lap
# distances (e.g. Monaco's ~3337 m, Montreal's ~4361 m both come out to
# within ~2% once divided by 10).
METERS_PER_UNIT = 0.1


def fit_splines(df: pd.DataFrame, bc_type: str = "not-a-knot", closed: bool = False):
    """
    Fit piecewise cubic splines to the x, y, z columns of df, parametrized
    by cumulative chord-length distance between consecutive points.

    scipy's CubicSpline enforces C2 continuity by construction, so each
    segment already matches first and second derivatives with its
    neighbors at every point where they connect.

    If closed is True, a copy of the first point is appended after the
    last one, adding a segment that closes the loop back to the start,
    and bc_type is forced to "periodic" so that closing segment also
    matches first and second derivatives with its neighbors instead of
    just being a plain line back to the start.

    Returns (t, splines) where t is the parameter array and splines is a
    dict of CubicSpline objects keyed by 'x', 'y', 'z'.
    """
    if closed:
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        bc_type = "periodic"

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    z = df["z"].to_numpy(dtype=float)

    deltas = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    t = np.concatenate(([0.0], np.cumsum(deltas)))

    # Consecutive samples with identical (x, y, z) produce a zero-length
    # step, which leaves t merely non-decreasing instead of the strictly
    # increasing sequence CubicSpline requires - drop those duplicates.
    keep = np.concatenate(([True], deltas > 0))
    t, x, y, z = t[keep], x[keep], y[keep], z[keep]

    splines = {
        "x": CubicSpline(t, x, bc_type=bc_type),
        "y": CubicSpline(t, y, bc_type=bc_type),
        "z": CubicSpline(t, z, bc_type=bc_type),
    }
    return t, splines

def compute_curvature(splines: dict, t: np.ndarray) -> np.ndarray:
    """
    Curvature of the 3D parametric spline curve r(t) = (x(t), y(t), z(t)):
        kappa(t) = |r'(t) x r''(t)| / |r'(t)|^3
    """
    dx, dy, dz = (splines[c].derivative(1) for c in ("x", "y", "z"))
    ddx, ddy, ddz = (splines[c].derivative(2) for c in ("x", "y", "z"))

    r1 = np.stack([dx(t), dy(t), dz(t)], axis=-1)
    r2 = np.stack([ddx(t), ddy(t), ddz(t)], axis=-1)

    numerator = np.linalg.norm(np.cross(r1, r2), axis=-1)
    denominator = np.linalg.norm(r1, axis=-1) ** 3

    kappa = np.zeros_like(numerator)
    nonzero = denominator > 0
    kappa[nonzero] = numerator[nonzero] / denominator[nonzero]
    return kappa

def get_track_curvatures():
    """
    Read every Sessions/{year}.json file produced by get_sessions() and,
    for each meeting's Qualifying session on a circuit not already seen,
    pull driver #1's xyz location data for their last full lap from
    OpenF1's /location endpoint, then immediately fit splines and reduce
    it to a total curvature value - one circuit at a time, rather than
    collecting every circuit's raw location data before processing any
    of it.

    A lap's date_start marks when it begins, so a full lap is the
    interval between one lap's date_start (its start) and the next lap's
    date_start (its end). The last lap of a race is especially likely to
    be contaminated by things that aren't a clean racing lap - a red
    flag or safety car parking the field for minutes, or the winner
    doing a slow-down/celebration lap before crossing back through the
    timing loop - so instead of blindly trusting the final pair, this
    walks backward from the end of the race looking for the most recent
    lap whose actual elapsed time (next lap's date_start minus this
    lap's date_start) is within 1.5x the session's median lap_duration,
    and uses that as the "last full lap" instead.

    Also plots each circuit's raw points and fitted splines together on
    an interactive 3D figure and saves it to Plots/{circuit_short_name}.html -
    open it in a browser to rotate/zoom/pan.

    Returns a dict mapping circuit_key -> {"total_curvature": ...,
    "length_m": ...}, one entry per distinct circuit. total_curvature is
    the integral of curvature over arc length (total turning angle, in
    radians) and is scale-invariant, so it captures how twisty a track
    is independent of its size - length_m (the lap's arc length,
    converted from OpenF1's native units via METERS_PER_UNIT) is stored
    alongside it so track size isn't lost.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    circuit_map = {}

    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            sessions = json.load(f)

        for session in sessions:
            if session["session_name"] != "Race":
                continue

            circuit_key = session["circuit_key"]
            circuit_name = session["circuit_short_name"]
            if circuit_key in circuit_map:
                continue

            session_key = session["session_key"]
            laps = openf1_get(
                "laps", params={"session_key": session_key, "driver_number": 1}
            )
            laps = sorted(laps, key=lambda lap: lap["lap_number"])

            if len(laps) < 2:
                continue

            durations = [lap["lap_duration"] for lap in laps if lap["lap_duration"]]
            if not durations:
                continue
            median_duration = np.median(durations)

            second_last_lap_start = last_lap_start = None
            for i in range(len(laps) - 2, -1, -1):
                start, end = laps[i]["date_start"], laps[i + 1]["date_start"]
                if not (start and end):
                    continue
                gap = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
                if gap <= 1.5 * median_duration:
                    second_last_lap_start, last_lap_start = start, end
                    break

            if last_lap_start is None:
                continue

            location = openf1_get(
                "location",
                params={
                    "session_key": session_key,
                    "driver_number": 1,
                    "date>": second_last_lap_start,
                    "date<": last_lap_start,
                },
            )
            if len(location) < 4:
                continue

            df = pd.DataFrame(location)[["x", "y", "z"]]

            t, splines = fit_splines(df, closed=True)
            kappa = compute_curvature(splines, t)
            circuit_map[circuit_key] = {
                "total_curvature": np.trapezoid(kappa, t),
                "length_m": t[-1] * METERS_PER_UNIT,
            }

            t_fine = np.linspace(t[0], t[-1], 500)
            spline_x, spline_y, spline_z = (splines[c](t_fine) for c in "xyz")

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=df["x"],
                        y=df["y"],
                        z=df["z"],
                        mode="markers",
                        name="points",
                        marker=dict(size=3, color="royalblue"),
                    ),
                    go.Scatter3d(
                        x=spline_x,
                        y=spline_y,
                        z=spline_z,
                        mode="lines",
                        name="spline",
                        line=dict(width=4, color="orange"),
                    ),
                ]
            )
            fig.update_layout(
                title=circuit_name,
                scene=dict(
                    # aspectmode="data" scales x/y/z uniformly to their
                    # actual data ranges, instead of z being auto-stretched
                    # to fill the same visual size as the much larger x/y
                    # extent.
                    aspectmode="data",
                    # Camera positioned straight above, looking down the
                    # z-axis onto the xy plane. A larger eye distance is
                    # needed here than feels intuitive because aspectmode
                    # "data" makes the scene's bounding box very wide and
                    # flat (x/y span hundreds/thousands of units, z only
                    # tens), so a small eye distance only frames a tiny
                    # corner of the track on initial load.
                    camera=dict(eye=dict(x=0, y=0, z=10), up=dict(x=0, y=1, z=0)),
                ),
            )
            fig.write_html(os.path.join(PLOTS_DIR, f"{circuit_name}.html"))

    return circuit_map


def main():
    """
    Compute total curvature and length for every circuit and save
    circuit_key -> {"total_curvature", "length_m"} to
    CurcuitCurvatures.json.
    """
    circuit_curvatures = get_track_curvatures()
    circuit_curvatures = {
        circuit_key: {
            "total_curvature": float(stats["total_curvature"]),
            "length_m": float(stats["length_m"]),
        }
        for circuit_key, stats in circuit_curvatures.items()
    }

    with open("CurcuitCurvatures.json", "w") as f:
        json.dump(circuit_curvatures, f, indent=2)


if __name__ == "__main__":
    main()

