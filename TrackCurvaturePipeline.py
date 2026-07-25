import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from GetData import SESSIONS_DIR, YEAR_FILE_PATTERN, openf1_get


def fit_splines(df: pd.DataFrame, bc_type: str = "not-a-knot"):
    """
    Fit piecewise cubic splines to the x, y, z columns of df, parametrized
    by cumulative chord-length distance between consecutive points.

    scipy's CubicSpline enforces C2 continuity by construction, so each
    segment already matches first and second derivatives with its
    neighbors at every point where they connect.

    Returns (t, splines) where t is the parameter array and splines is a
    dict of CubicSpline objects keyed by 'x', 'y', 'z'.
    """
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

    A lap's date_start marks when it begins, so the last full lap is the
    interval between the second-to-last lap's date_start (its start) and
    the last lap's date_start (its end) - the final lap entry itself is
    typically an incomplete in/out lap.

    Returns a dict mapping circuit_key -> total curvature (the integral
    of curvature over arc length), one entry per distinct circuit.
    """
    circuit_map = {}

    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            sessions = json.load(f)

        for session in sessions:
            if session["session_name"] != "Qualifying":
                continue

            circuit_key = session["circuit_key"]
            if circuit_key in circuit_map:
                continue

            session_key = session["session_key"]
            laps = openf1_get(
                "laps", params={"session_key": session_key, "driver_number": 1}
            )
            laps = sorted(laps, key=lambda lap: lap["lap_number"])

            if len(laps) < 2:
                continue

            last_lap_start = laps[-1]["date_start"]
            second_last_lap_start = laps[-2]["date_start"]

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

            t, splines = fit_splines(df)
            kappa = compute_curvature(splines, t)
            circuit_map[circuit_key] = np.trapezoid(kappa, t)

    return circuit_map


def main():
    """
    Compute total curvature for every circuit and save circuit_key ->
    curvature to CurcuitCurvatures.json.
    """
    circuit_curvatures = get_track_curvatures()
    circuit_curvatures = {
        circuit_key: float(curvature)
        for circuit_key, curvature in circuit_curvatures.items()
    }

    with open("CurcuitCurvatures.json", "w") as f:
        json.dump(circuit_curvatures, f, indent=2)


if __name__ == "__main__":
    main()

