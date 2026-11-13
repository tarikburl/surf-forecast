"""Build the forecast series for one spot from stored data."""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from surf import fetch, model, physics, store

FT, KT = 3.28084, 1.94384
STALE_HOURS = 4   # wave data older than this is flagged
PACIFIC = ZoneInfo("America/Los_Angeles")
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def compass(deg):
    return COMPASS[round(deg / 22.5) % 16]


def local(utc_string):
    """UTC time string -> timezone-aware Pacific Timestamp."""
    return pd.Timestamp(utc_string, tz="UTC").tz_convert(PACIFIC)


def wind_kind(wind_from, shore_normal):
    """Wind relative to the beach. shore_normal points out to sea, so wind
    coming FROM that direction blows onshore."""
    off = abs((wind_from - shore_normal + 180) % 360 - 180)
    return "onshore" if off < 60 else "offshore" if off > 120 else "cross-shore"


def tide_height(tides, when):
    """Tide height (m) at time `when`, by cosine interpolation between the
    high/low predictions on either side. None if `when` is not bracketed."""
    t = pd.to_datetime(tides["time"]).reset_index(drop=True)
    i = t.searchsorted(when)
    if i == 0 or i == len(t):
        return None
    h0, h1 = tides["height"].iloc[i - 1], tides["height"].iloc[i]
    x = (when - t[i - 1]) / (t[i] - t[i - 1])
    return float(h0 + (h1 - h0) * (1 - np.cos(np.pi * x)) / 2)


def rating(low, wind_speed, wind_from, tide, spot):
    """'flat', 'poor', 'fair' or 'good'. low (m), wind_speed (m/s), tide (m).

    Under 2 ft is flat. Otherwise wind decides: light wind is best, then
    offshore, cross-shore, onshore; stronger is worse. A tide above the
    spot's optional tide_max (m) knocks the score down.
    """
    if low * FT < 2:
        return "flat"
    if wind_speed is None or np.isnan(wind_speed):
        return ""
    kind = wind_kind(wind_from, spot["shore_normal"])
    if wind_speed < 3:
        score = 1.0
    elif kind == "offshore":
        score = 0.9 if wind_speed < 8 else 0.6
    elif kind == "cross-shore":
        score = 0.6 if wind_speed < 6 else 0.3
    else:
        score = 0.4 if wind_speed < 5 else 0.1
    if tide is not None and tide > spot.get("tide_max", np.inf):
        score *= 0.6
    return "good" if score >= 0.7 else "fair" if score >= 0.4 else "poor"


def build(con, key, spot, start):
    """Forecast for one spot from `start` (UTC 'YYYY-MM-DDTHH:MM') onward.

    Rows from the CDIP MOP forecast (source 'mop', 3-hourly) are followed by a
    low-confidence tail from Open-Meteo swell (source 'tail'), scaled so the
    two agree on average where they overlap.

    Columns: time, low, high (m), hs, tp, dp of the main swell, source,
    wind_speed (m/s), wind_from (deg), tide (m), rating.
    """
    mop = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='forecast' "
                     "AND time>=? ORDER BY time", (spot["mop"], start))
    rows = []
    for _, m in mop.iterrows():
        s = model.surf_height(m, spot)
        main = s["trains"][0]
        rows.append({"time": m.time, "low": s["low"], "high": s["high"],
                     "hs": main["hs"], "tp": main["tp"], "dp": main["dp"],
                     "source": "mop"})
    out = pd.DataFrame(rows)

    sea = store.read(con, "SELECT * FROM marine WHERE spot=? AND time>=? "
                     "ORDER BY time", (key, start))
    if not out.empty and not sea.empty:
        weight = sea.dp.map(lambda d: model.window_weight(d, spot["window"]))
        sea["low"] = spot.get("a", 1.0) * physics.breaking_height(
            sea.hs * weight, sea.tp) ** spot.get("b", 1.0)
        both = sea.merge(out[["time", "low"]], on="time", suffixes=("", "_mop"))
        both = both[both.low > 0]
        tail = sea[(sea.time > out.time.iloc[-1]) & (sea.time.str[11:13].astype(int) % 3 == 0)]
        if not both.empty and not tail.empty:
            tail = tail.assign(low=tail.low * (both.low_mop / both.low).median(),
                               source="tail")
            tail["high"] = model.SET_FACTOR * tail.low
            out = pd.concat([out, tail[out.columns]], ignore_index=True)
    if out.empty:
        return out

    wind = store.read(con, "SELECT time, speed AS wind_speed, direction AS wind_from "
                      "FROM wind WHERE spot=?", (key,))
    out = out.merge(wind, on="time", how="left")
    tides = store.read(con, "SELECT time, height FROM tide WHERE station=? "
                       "ORDER BY time", (spot["tide_station"],))
    out["tide"] = [tide_height(tides, pd.Timestamp(t)) if len(tides) else None
                   for t in out.time]
    out["rating"] = [rating(r.low, r.wind_speed, r.wind_from, r.tide, spot)
                     for r in out.itertuples()]
    return out


def save_run(con, key, df, run):
    """Store one forecast run so its skill can be checked later."""
    if not df.empty:
        store.upsert(con, "forecast", df[["time", "low", "high", "source"]],
                     spot=key, run=run)


def current(con, key, spot, now):
    """Conditions right now from stored data. `now` is a naive UTC Timestamp.

    Returns None if no wave data is stored, else {surf, age_hours, wind, tides}:
    surf is model.surf_height of the latest nowcast, wind is the latest hourly
    row (or None), tides is the next two high/low predictions (may be empty).
    """
    now_s = now.strftime(fetch.TIME_FMT)
    m = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast' "
                   "ORDER BY time DESC LIMIT 1", (spot["mop"],))
    if m.empty:
        return None
    w = store.read(con, "SELECT * FROM wind WHERE spot=? AND time<=? "
                   "ORDER BY time DESC LIMIT 1", (key, now_s))
    t = store.read(con, "SELECT * FROM tide WHERE station=? AND time>? "
                   "ORDER BY time LIMIT 2", (spot["tide_station"], now_s))
    return {"surf": model.surf_height(m.iloc[0], spot),
            "age_hours": (now - pd.Timestamp(m.time[0])).total_seconds() / 3600,
            "wind": None if w.empty else w.iloc[0], "tides": t}
