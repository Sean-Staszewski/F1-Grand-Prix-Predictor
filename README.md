# F1 Grand Prix Predictor

Data pipeline for pulling Formula 1 session data from the [OpenF1 API](https://openf1.org/) and computing per-circuit track curvature from GPS location traces.

## Data directories

All data directories are keyed by IDs that come out of the OpenF1 API, and every file is a JSON array (or, for `CurcuitCurvatures.json`, a single object) written by `GetData.py` or `TrackCurvaturePipeline.py`. Nothing in these directories is meant to be hand-edited — delete a file (or the whole directory) and rerun the corresponding script to regenerate it.

### `Sessions/{year}.json`

One file per year (2023-2026), named by year, e.g. `Sessions/2024.json`. Each entry is one session (a practice, qualifying, sprint, or race) held that year, with:

- `session_key` — unique ID for the session; this is the key used to name files in `Laps/`, `Weather/`, and `GridPositions/`.
- `meeting_key` — unique ID for the race weekend the session belongs to; this is the key used to name files in `Drivers/`.
- `circuit_key` — unique ID for the physical circuit; this is the key used in `CurcuitCurvatures.json`.
- `circuit_short_name` — human-readable circuit name, e.g. `"Silverstone"`, `"Sakhir"`; this is used to name `Plots/{circuit_short_name}.html`.
- `session_name` / `session_type` — e.g. `"Practice 1"` / `"Practice"`, `"Qualifying"` / `"Qualifying"`, `"Race"` / `"Race"`.

Pre-season testing days and cancelled or not-yet-happened sessions are filtered out. This is the root file every other directory is built from — every other script lists `Sessions/*.json` to discover which `session_key`s exist.

### `Laps/{session_key}.json`

One file per session. Each entry is one driver's one lap in that session: `session_key`, `driver_number`, `date_start` (timestamp the lap began), `lap_duration` (seconds, `null` for incomplete laps like the in/out laps).

### `Weather/{session_key}.json`

One file per session. Each entry is one weather sample taken during that session: `session_key`, `date` (timestamp the sample was taken), `air_temperature`, `track_temperature`, `humidity`, `rainfall`.

### `GridPositions/{session_key}.json`

One file per session. Each entry is one driver's starting position for that session: `session_key`, `driver_number`, `position`, `has_grid_position`.

Only `Race`-type sessions have a real starting grid (a formation lap into a lights-out start), so `position` is only meaningful when `has_grid_position` is `true`. For every other session type (practice, qualifying), "starting position" isn't a real concept — drivers leave the pits individually whenever they choose — so `position` is a placeholder `0` and `has_grid_position` is `false`.

### `Drivers/{meeting_key}.json`

One file per race weekend (meeting), not per session — a meeting's driver lineup is the same across its practice/qualifying/race sessions. Each entry is one driver who took part in that meeting: `driver_number`, `team_name`.

### `Plots/{circuit_short_name}.html`

One interactive Plotly 3D HTML file per circuit, named after the circuit's short name (e.g. `Plots/Silverstone.html`) and produced as a side effect of `TrackCurvaturePipeline.py`. Open one in a browser to rotate/zoom/pan around the raw GPS points and the fitted spline for that circuit. Purely a visual sanity check, not consumed by any other script.

### `CurcuitCurvatures.json`

A single JSON object (not an array, and not split into per-file records) mapping `circuit_key` (as a string) to a single float: the total curvature of that circuit's racing line, computed by `TrackCurvaturePipeline.py`. Higher values mean a twistier circuit; lower values mean a circuit dominated by long straights.

## Scripts

### `GetData.py`

Downloads the core session/lap/weather/driver/grid datasets from OpenF1 into the directories above. Run it directly:

```
python GetData.py
```

`main()` downloads each directory in dependency order — `Sessions/` first (everything else reads `session_key`s out of it), then `Laps/`, `Weather/`, `Drivers/`, then `GridPositions/` (which additionally reads `Laps/` to find each session's driver list and, for races, its start time). Each directory is only (re)downloaded if it doesn't exist yet or is currently empty, so rerunning the script after a partial/interrupted run resumes rather than starting over — to force a full redownload of one dataset, delete its directory first. At the end it prints whether `Laps/`, `Weather/`, and `GridPositions/` ended up with matching file counts, as a quick check that nothing silently failed.

### `TrackCurvaturePipeline.py`

Computes how "curvy" each circuit is, using one lap of GPS data per circuit. Run it directly:

```
python TrackCurvaturePipeline.py
```

For each circuit (deduplicated by `circuit_key`, taking the first `Race` session found for it in `Sessions/`), it:

1. Pulls driver #1's lap times for that race from the OpenF1 `/laps` endpoint to find the last full lap (the interval between the second-to-last and last lap start times — the final lap entry is normally an incomplete in/out lap).
2. Pulls driver #1's `x`/`y`/`z` GPS location samples from OpenF1's `/location` endpoint for that lap.
3. Fits closed-loop periodic cubic splines through the points (`fit_splines`) and integrates curvature over arc length (`compute_curvature`) to get one total-curvature number for the circuit.
4. Writes a 3D plot of the raw points and fitted spline to `Plots/{circuit_short_name}.html`.

It calls the OpenF1 API directly rather than reading `Laps/` off disk, since `Laps/{session_key}.json` (as written by `GetData.py`) doesn't include `lap_number`, which this script needs to find the last full lap. When run as a script (`python TrackCurvaturePipeline.py`), the results are collected into `CurcuitCurvatures.json`.
