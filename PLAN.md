# Surf Forecast: Build Plan

A free, per-spot surf height forecast for seven Northern California spots, built from the public CDIP nearshore wave model, with a small physics step and a per-spot calibration fitted to real observations.

## Instructions for the executing agent

- Work **one chunk at a time**. At the end of each chunk, stop and present the "Checkpoint" items to the user. Do not start the next chunk until they approve.
- **Keep it tight and barebones. Reliability beats features.**
  - One small package, flat modules, plain functions. No classes unless they remove code. No plugin systems, no config layers, no abstractions for sources we do not use.
  - Dependencies: `requests`, `numpy`, `pandas`, `matplotlib`, `pytest`. Nothing else without asking. Use the standard library for the rest (`sqlite3`, `tomllib`, `argparse`).
  - Every network call has a timeout, one retry, and a clear error. A failed source must never crash the run: use the last stored data and mark the output as stale.
  - If a feature is not needed for a checkpoint, do not build it.
- Python 3.12 in a plain `.venv` (`python3.12 -m venv .venv && .venv/bin/pip install -e . pytest`; `uv` is not installed on this machine). Storage: one SQLite file, `data/surf.db`.
- Save one raw response from each API in `tests/fixtures/` so parsing is tested offline.
- Internal units SI (m, s, degrees true, direction waves come FROM). Convert to feet only when printing.
- Each physics function has a docstring naming its formula and a test with a hand-checked number.

## Spots

All seven are covered by CDIP MOP (verified 2026-11-13). The MOP point, its depth and its shore-normal direction come from the live files. The swell window and tide notes are **guesses for the user to confirm in Chunk 0**.

| Spot | Lat, Lon | MOP point (distance) | Depth | Shore normal | Swell window (guess) | Notes (guess) |
|---|---|---|---|---|---|---|
| Ocean Beach, SF | 37.755, -122.512 | `SF026` (0.7 km) | 10 m | 261° | 200 to 320° | Open to everything, all tides |
| Rockaway, Pacifica | 37.609, -122.497 | `SM387` (1.4 km) | 15 m | 289° | 220 to 320° | |
| Linda Mar, Pacifica | 37.594, -122.503 | `SM376` (1.3 km) | 15 m | 339° | 250 to 320° | Sheltered by Pedro Point; smaller than neighbors |
| Montara | 37.547, -122.515 | `SM338` (0.5 km) | 15 m | 302° | 220 to 320° | Usually the biggest of the group |
| Four Mile, Santa Cruz | 36.965, -122.124 | `SC204` (0.7 km) | 15 m | 190° | 180 to 310° | Low to mid tide |
| Steamer Lane, Santa Cruz | 36.951, -122.026 | `SC149` (0.5 km) | 15 m | 128° | 180 to 300° | NW swell wraps in; low to mid tide |
| Pleasure Point, Santa Cruz | 36.954, -121.971 | `SC117` (0.9 km) | 15 m | 158° | 180 to 290° | Smaller than the Lane on NW swell; lower tide |

Caution: Linda Mar and Rockaway sit in coves, so the nearest 15 m point is over 1 km out and Linda Mar's (339° normal) may sit around Pedro Point. Chunk 0 plots the neighboring MOP points on a map so the user can pick the right one.

Supporting stations:
- **Buoys (validation only):** `46237` SF Bar and `46026` San Francisco for the SF/Pacifica spots; `46042` Monterey and `46236` Monterey Canyon Outer for Santa Cruz; `46214` Point Reyes as upstream reference. (`46012` Half Moon Bay is reporting height only right now.)
- **Tides:** CO-OPS `9414290` San Francisco for SF/Pacifica/Montara; `9413450` Monterey for Santa Cruz.

## Data sources (free, no API keys, all verified live)

| Source | Role | Access |
|---|---|---|
| **CDIP MOP** | Primary input. Nearshore Hs, Tp, Dp and 28-band spectra at each spot's point: nowcast, hindcast and forecast. Already includes sheltering and refraction down to 10 to 15 m depth. | `https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/{ID}_{nowcast,forecast}.nc.ascii?var[i:j],...` fetched as text with `requests` (no netCDF libraries). Nowcast is hourly, 20 bands, back to 2025-04. Forecast is 3-hourly, 28 bands, from 3 days back to 6 days ahead. |
| **NDBC realtime2** | Sanity check on MOP: measured Hs, swell period, direction | `https://www.ndbc.noaa.gov/data/realtime2/{station}.spec` (text) |
| **Open-Meteo Marine** | Longer-range tail (MOP forecast is only a few days) | `https://marine-api.open-meteo.com/v1/marine` (JSON) |
| **Open-Meteo forecast** | Wind speed and direction at the spot | `https://api.open-meteo.com/v1/forecast` (JSON) |
| **NOAA CO-OPS** | Tide predictions | `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter` (JSON) |

Because MOP covers every spot, we do **not** build our own directional-spectrum reconstruction or island-shadowing model. That is the largest simplification in this plan.

## The calculation

1. **Start from MOP at the spot's point**: band energy $E(f)$ and band mean direction $\theta(f)$ at 10 to 15 m depth.
2. **Split into swell trains**: find peaks in $E(f)$ (at most 3), giving each train an $H_s$, $T_p$ and direction. Sanity check: $4\sqrt{\sum E\,\Delta f}$ equals MOP's reported $H_s$.
3. **Spot window**: down-weight trains whose direction falls outside the spot's window (soft edges). MOP has done most of this already; the window handles the last kilometer into coves.
4. **Shoal to breaking**: for each train, undo shoaling from the MOP depth to get the deep-water-equivalent height $H_0'$ (linear theory, $K_s = \sqrt{c_{g0}/c_g}$), then Komar and Gaughan (1973): $H_b = 0.39\,g^{1/5}\,(T\,H_0'^2)^{2/5}$. Combine trains by energy, $\sqrt{\sum H_{b,i}^2}$.
5. **Calibrate per spot**: $H_{\text{surf}} = a\,H_b^{\,b}$, with $a, b$ fitted per spot to observations (optional `a`, `b` keys in `spots.toml`). Before any observations exist, $a = 1$, $b = 1$. The reported range is low $= H_{\text{surf}}$ (typical waves) to high $= 1.27\,H_{\text{surf}}$ (set waves, $H_{1/10}$).
   *Status 2026-11-13:* bands with period under 8 s are treated as wind chop and left out of the surf height. Each spot has a provisional `a` in `spots.toml`, fitted to one week of Surfline's published heights (a big-swell week, 14 to 18 s). Against that week, Surfline's heights grow faster with swell size than ours (a shared $b \approx 1.4$ fits better than $b = 1$), but that exponent is not applied because it would badly under-call small days. Chunk 5 should refit `a` and `b` on logged sessions spanning small and large days.
6. **Tide and wind**: shown next to the height and used for a simple quality score. They do not change the height number.

Error budget, in rough order: (1) offshore forecast error beyond about 3 days, which nobody can fix; (2) the last kilometer of local bathymetry; (3) the surfer's height convention. Calibration addresses (2) and (3), so the logged observations are the main accuracy lever.

## Chunks

### Chunk 0: Skeleton and spot config
- Repo layout: `surf/` (package), `tests/`, `data/`, `pyproject.toml`; `pytest` runs.
- `spots.toml` holding the table above.
- A one-off script that maps each spot with its 5 nearest MOP points (id, depth, shore normal) so the cove spots can be assigned by eye.

**Checkpoint:** the map and the spot table. User confirms or corrects MOP point, swell window and tide notes for each spot.

### Chunk 1: Fetch and store
- `surf/fetch.py`: MOP nowcast + forecast per spot, buoy `.spec`, tides, wind. Each function returns a DataFrame; each has timeout, retry, and fixture-based test.
- `surf/store.py`: write/read SQLite, idempotent upserts.
- CLI `surf fetch`.

**Checkpoint:** `surf fetch` prints one line per spot (MOP Hs, Tp, Dp now; nearest buoy Hs; next tide). Running it twice adds no duplicate rows. Tests pass offline.

### Chunk 2: Height calculation
- `surf/physics.py`: dispersion, group velocity, de-shoaling, Komar-Gaughan. Hand-checked tests.
- `surf/model.py`: MOP spectrum to swell trains to breaking height range in feet.
- CLI `surf now`.

**Checkpoint:** `surf now` table for all seven spots: height range, dominant swell (height, period, direction), tide, wind. User compares with what they see today. Relative ordering should be plausible (Montara and Ocean Beach biggest, Linda Mar smallest in Pacifica).

### Chunk 3: Forecast
- Run the same model on the MOP forecast. Append an Open-Meteo-based tail beyond MOP's horizon, scaled to match MOP where they overlap and labeled low confidence.
- Simple quality score from height, wind (offshore/onshore relative to shore normal) and tide.
- Store every forecast run so skill can be measured later.
- CLI `surf forecast <spot>`.

**Checkpoint:** forecast table and chart for two spots. Hour 0 matches `surf now`.

### Chunk 4: Page and schedule
- One static, phone-friendly HTML file generated by the CLI: a card per spot with height chart, swell trains, wind, tide, quality, and a "data stale" badge when a source failed.
- Hourly schedule via `launchd` (local) or GitHub Actions (hosted). User chooses.

**Checkpoint:** user opens the page on their phone and sees it refresh on schedule.

### Chunk 5: Log and calibration
- `surf log <spot> <low> <high>`: one line appends a timestamped observation (ft) to SQLite.
- Pull MOP hindcast for the times of logged sessions; fit $a, b$ per spot once a spot has about 15 observations; until then keep defaults.
- Backtest report: MAE in ft, bias, percent within 1 ft, calibrated vs uncalibrated.

**Checkpoint:** logging works in one command; with seeded or real observations the report runs and calibrated error on held-out data is no worse than uncalibrated.

### Chunk 6: Skill tracking (after a few weeks of logs)
- Compare each logged session with what the stored forecasts said 1, 3 and 5 days before.
- Monthly refit; a small skill table on the page.

**Checkpoint:** skill report.

## Open decisions for the user
1. Confirm the spot table in Chunk 0 (MOP points for Linda Mar and Rockaway especially).
2. Page hosting: local only, or GitHub Pages.
