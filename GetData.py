import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

OPENF1_BASE_URL = "https://api.openf1.org/v1"
SESSIONS_DIR = "Sessions"
LAPS_DIR = "Laps"
WEATHER_DIR = "Weather"
DRIVERS_DIR = "Drivers"
GRID_POSITIONS_DIR = "GridPositions"
MASTER_FILE = "Master.json"
YEAR_FILE_PATTERN = re.compile(r"^\d{4}\.json$")
TESTING_DAY_PATTERN = re.compile(r"^Day \d+$")
WEATHER_COLUMNS = ["air_temperature", "track_temperature", "humidity", "rainfall"]


def openf1_get(endpoint, params=None, max_retries=5):
    """
    GET from an OpenF1 endpoint. OpenF1 returns 404 instead of an empty
    list when a query matches no data (e.g. a cancelled session with no
    laps recorded, or a future session that hasn't happened yet) - treat
    that case as an empty result rather than an error. Retries with
    backoff on 429 (rate limited), honoring Retry-After when present.
    """
    url = f"{OPENF1_BASE_URL}/{endpoint}"
    delay = 1.0

    for _ in range(max_retries):
        response = requests.get(url, params=params)

        if response.status_code == 404:
            return []

        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", delay)))
            delay *= 2
            continue

        response.raise_for_status()
        return response.json()

    response.raise_for_status()
    return response.json()


def get_sessions():
    """
    Pull /sessions data from OpenF1 for every year from 2023 to 2026,
    excluding sessions that haven't happened yet, pre-season testing
    sessions (session_type "Practice" but named "Day 1", "Day 2", etc.
    instead of "Practice N"), and cancelled sessions. Save each year's
    response to Sessions/{year}.json.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    years = range(2023, 2027)
    now = datetime.now(timezone.utc).isoformat()

    for year in years:
        sessions = openf1_get("sessions", params={"year": year, "date_start<": now})
        sessions = [
            session
            for session in sessions
            if not (
                session["session_type"] == "Practice"
                and TESTING_DAY_PATTERN.match(session["session_name"])
            )
            and not session["is_cancelled"]
        ]
        sessions = [
            {
                "session_key": session["session_key"],
                "meeting_key": session["meeting_key"],
                "circuit_key": session["circuit_key"],
                "circuit_short_name": session["circuit_short_name"],
                "session_name": session["session_name"],
                "session_type": session["session_type"],
            }
            for session in sessions
        ]

        out_path = os.path.join(SESSIONS_DIR, f"{year}.json")
        with open(out_path, "w") as f:
            json.dump(sessions, f, indent=2)


def get_laps():
    """
    Read every Sessions/{year}.json file produced by get_sessions(), pull
    /laps data from OpenF1 for each session_key found within, and save
    each session's laps to Laps/{session_key}.json.
    """
    os.makedirs(LAPS_DIR, exist_ok=True)

    session_keys = set()
    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            sessions = json.load(f)

        for session in sessions:
            session_keys.add(session["session_key"])

    for session_key in session_keys:
        laps = openf1_get("laps", params={"session_key": session_key})
        laps = [
            {
                "session_key": lap["session_key"],
                "driver_number": lap["driver_number"],
                "date_start": lap["date_start"],
                "lap_duration": lap["lap_duration"],
            }
            for lap in laps
        ]

        out_path = os.path.join(LAPS_DIR, f"{session_key}.json")
        with open(out_path, "w") as f:
            json.dump(laps, f, indent=2)


def get_weather():
    """
    Read every Sessions/{year}.json file produced by get_sessions(), pull
    /weather data from OpenF1 for each session_key found within, and save
    each session's weather to Weather/{session_key}.json.
    """
    os.makedirs(WEATHER_DIR, exist_ok=True)

    session_keys = set()
    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            sessions = json.load(f)

        for session in sessions:
            session_keys.add(session["session_key"])

    for session_key in session_keys:
        weather = openf1_get("weather", params={"session_key": session_key})
        weather = [
            {
                "session_key": entry["session_key"],
                "date": entry["date"],
                "air_temperature": entry["air_temperature"],
                "track_temperature": entry["track_temperature"],
                "humidity": entry["humidity"],
                "rainfall": entry["rainfall"],
            }
            for entry in weather
        ]

        out_path = os.path.join(WEATHER_DIR, f"{session_key}.json")
        with open(out_path, "w") as f:
            json.dump(weather, f, indent=2)


def get_drivers():
    """
    Pull the entire /drivers dataset from OpenF1 (no filters) and keep
    only one entry per driver_number / meeting_key combination. Save each
    meeting's drivers to Drivers/{meeting_key}.json.
    """
    os.makedirs(DRIVERS_DIR, exist_ok=True)

    drivers = pd.DataFrame(openf1_get("drivers"))
    drivers = drivers.drop_duplicates(subset=["driver_number", "meeting_key"])

    for meeting_key, group in drivers.groupby("meeting_key"):
        out_path = os.path.join(DRIVERS_DIR, f"{meeting_key}.json")
        with open(out_path, "w") as f:
            json.dump(
                group[["driver_number", "team_name"]].to_dict(orient="records"),
                f,
                indent=2,
            )


def get_grid_positions():
    """
    Read every Sessions/{year}.json file produced by get_sessions() to get
    each session's session_key and session_type. Only Race/Sprint sessions
    (session_type "Race") have an actual starting grid - a formation lap
    into a lights-out start - so those pull real positions from OpenF1's
    /position endpoint as of when the session's earliest lap started (a
    driver's lap 1 is always their earliest lap, so the session-wide
    minimum date_start is equivalent to the earliest lap 1 across
    drivers). For every other session type, "starting grid position" is
    meaningless (drivers leave the pits individually whenever they
    choose), so every driver is filled with a placeholder position of 0
    and has_grid_position=False instead. Save each session's positions to
    GridPositions/{session_key}.json.
    """
    os.makedirs(GRID_POSITIONS_DIR, exist_ok=True)

    sessions_by_key = {}
    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            sessions = json.load(f)

        for session in sessions:
            sessions_by_key[session["session_key"]] = session

    for session_key, session in sessions_by_key.items():
        laps_path = os.path.join(LAPS_DIR, f"{session_key}.json")
        if not os.path.exists(laps_path):
            continue

        with open(laps_path) as f:
            laps = json.load(f)

        driver_numbers = sorted({lap["driver_number"] for lap in laps})
        if not driver_numbers:
            continue

        if session["session_type"] == "Race":
            lap_starts = [lap["date_start"] for lap in laps if lap["date_start"]]
            if not lap_starts:
                continue

            start_time = min(lap_starts)

            positions = openf1_get(
                "position", params={"session_key": session_key, "date<": start_time}
            )
            positions = [
                {
                    "session_key": position["session_key"],
                    "driver_number": position["driver_number"],
                    "position": position["position"],
                    "has_grid_position": True,
                }
                for position in positions
            ]
        else:
            positions = [
                {
                    "session_key": session_key,
                    "driver_number": driver_number,
                    "position": 0,
                    "has_grid_position": False,
                }
                for driver_number in driver_numbers
            ]

        out_path = os.path.join(GRID_POSITIONS_DIR, f"{session_key}.json")
        with open(out_path, "w") as f:
            json.dump(positions, f, indent=2)


def _match_weather_to_laps(laps, weather):
    """
    Match each lap to weather data by date. laps and weather must both
    already be sorted ascending by date_start/date, with laps indexed
    0..n-1 (weather's index doesn't matter).

    Weather is sampled roughly once a minute, so a lap can span zero,
    one, or several samples:
      - If one or more samples fall within [date_start, lap_end], their
        values are averaged.
      - If none do (e.g. a short lap, or an in/out lap with no known
        duration so lap_end == date_start), the single closest sample by
        date is used instead.

    Returns a DataFrame of WEATHER_COLUMNS values, one row per lap,
    aligned positionally to laps.
    """
    w_dates = weather["date"].to_numpy()
    w_values = weather[WEATHER_COLUMNS].to_numpy(dtype=float)
    # Prepend a zero row so cumsum[i] is the sum of the first i weather
    # rows, letting the sum within a window be a plain difference of two
    # lookups instead of slicing and summing per lap.
    cumsum = np.vstack([np.zeros((1, w_values.shape[1])), np.cumsum(w_values, axis=0)])

    lo = np.searchsorted(w_dates, laps["date_start"].to_numpy(), side="left")
    hi = np.searchsorted(w_dates, laps["lap_end"].to_numpy(), side="right")
    counts = hi - lo

    with np.errstate(invalid="ignore"):
        within_window = (cumsum[hi] - cumsum[lo]) / counts[:, None]

    nearest = pd.merge_asof(
        laps[["date_start"]],
        weather[["date", *WEATHER_COLUMNS]],
        left_on="date_start",
        right_on="date",
        direction="nearest",
    )[WEATHER_COLUMNS].to_numpy()

    matched = np.where(counts[:, None] > 0, within_window, nearest)
    return pd.DataFrame(matched, columns=WEATHER_COLUMNS, index=laps.index)


def build_master_df():
    """
    Combine every downloaded dataset into a single master DataFrame, one
    row per lap.

    Weather is matched to each lap by date, as described in
    _match_weather_to_laps(). Session metadata (circuit, session type)
    is merged in by session_key, driver team by driver_number within the
    session's meeting_key, and grid position by session_key and
    driver_number.

    Laps with no date_start can't be matched to weather or ordered
    against other laps, so they're dropped. Sessions missing their Laps
    or Weather file entirely are skipped.

    Returns a DataFrame with one row per matchable lap.
    """
    sessions_by_key = {}
    for filename in os.listdir(SESSIONS_DIR):
        if not YEAR_FILE_PATTERN.match(filename):
            continue

        with open(os.path.join(SESSIONS_DIR, filename)) as f:
            for session in json.load(f):
                sessions_by_key[session["session_key"]] = session

    driver_teams = {}
    for filename in os.listdir(DRIVERS_DIR):
        meeting_key = int(os.path.splitext(filename)[0])
        with open(os.path.join(DRIVERS_DIR, filename)) as f:
            drivers = json.load(f)
        driver_teams[meeting_key] = {
            driver["driver_number"]: driver["team_name"] for driver in drivers
        }

    session_lap_frames = []

    for session_key, session in sessions_by_key.items():
        laps_path = os.path.join(LAPS_DIR, f"{session_key}.json")
        weather_path = os.path.join(WEATHER_DIR, f"{session_key}.json")
        if not (os.path.exists(laps_path) and os.path.exists(weather_path)):
            continue

        with open(laps_path) as f:
            laps = pd.DataFrame(json.load(f))
        with open(weather_path) as f:
            weather = pd.DataFrame(json.load(f))
        if laps.empty or weather.empty:
            continue

        laps = laps[laps["date_start"].notna()].copy()
        if laps.empty:
            continue

        laps["date_start"] = pd.to_datetime(laps["date_start"], format="ISO8601")
        laps["lap_end"] = laps["date_start"] + pd.to_timedelta(
            laps["lap_duration"].fillna(0), unit="s"
        )
        laps = laps.sort_values("date_start").reset_index(drop=True)

        weather["date"] = pd.to_datetime(weather["date"], format="ISO8601")
        weather = weather.sort_values("date").reset_index(drop=True)

        laps[WEATHER_COLUMNS] = _match_weather_to_laps(laps, weather)
        laps = laps.drop(columns="lap_end")

        laps["meeting_key"] = session["meeting_key"]
        laps["circuit_key"] = session["circuit_key"]
        laps["circuit_short_name"] = session["circuit_short_name"]
        laps["session_name"] = session["session_name"]
        laps["session_type"] = session["session_type"]

        team_map = driver_teams.get(session["meeting_key"], {})
        laps["team_name"] = laps["driver_number"].map(team_map)

        grid_path = os.path.join(GRID_POSITIONS_DIR, f"{session_key}.json")
        if os.path.exists(grid_path):
            with open(grid_path) as f:
                grid = pd.DataFrame(json.load(f))
            laps = laps.merge(
                grid[["driver_number", "position", "has_grid_position"]],
                on="driver_number",
                how="left",
            )

        session_lap_frames.append(laps)

    if not session_lap_frames:
        return pd.DataFrame()

    return pd.concat(session_lap_frames, ignore_index=True)


def main():
    """
    Download every dataset, in dependency order (laps, weather, and grid
    positions all read the Sessions/ files; grid positions also reads the
    Laps/ files). Each directory is downloaded into if it doesn't exist
    yet, or if it exists but is still empty.
    """
    downloads = [
        (SESSIONS_DIR, get_sessions),
        (LAPS_DIR, get_laps),
        (WEATHER_DIR, get_weather),
        (DRIVERS_DIR, get_drivers),
        (GRID_POSITIONS_DIR, get_grid_positions),
    ]

    for directory, download in downloads:
        if not os.path.exists(directory):
            os.makedirs(directory)
            download()
        if not os.listdir(directory):
            download()

    lap_count = len(os.listdir(LAPS_DIR))
    weather_count = len(os.listdir(WEATHER_DIR))
    grid_position_count = len(os.listdir(GRID_POSITIONS_DIR))

    if lap_count == weather_count == grid_position_count:
        print(f"Laps, Weather, and GridPositions all have {lap_count} files.")
    else:
        print(
            "Mismatched file counts: "
            f"Laps={lap_count}, Weather={weather_count}, "
            f"GridPositions={grid_position_count}"
        )

    master_df = build_master_df()
    master_df.to_json(MASTER_FILE, orient="records", date_format="iso", indent=2)
    print(f"Saved {len(master_df)} laps to {MASTER_FILE}.")


if __name__ == "__main__":
    main()

