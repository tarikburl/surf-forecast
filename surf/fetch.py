"""Fetch raw data. Each source has a pure parser (text -> DataFrame, tested
offline) and a thin fetch function that does the network call.

All times are UTC, stored as 'YYYY-MM-DDTHH:MM' strings. All units SI.
"""
import json
import re
import time

import pandas as pd
import requests

MOP_URL = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore"
NDBC_URL = "https://www.ndbc.noaa.gov/data/realtime2"
TIDE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
WIND_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
TIME_FMT = "%Y-%m-%dT%H:%M"


def get(url, params=None, timeout=30):
    """GET with a timeout and one retry. Raises requests.RequestException."""
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt:
                raise
            time.sleep(2)


# --- CDIP MOP (nearshore wave model) -----------------------------------------

def parse_dap_ascii(text):
    """Parse an OPeNDAP .ascii response into {variable: list or list of rows}."""
    body = text.split("-" * 45, 1)[1]
    out, name = {}, None
    for line in body.splitlines():
        line = line.strip()
        header = re.fullmatch(r"([\w.]+?)((?:\[\d+\])+)", line)
        if header:
            name = header.group(1).split(".")[-1]
            out[name] = []
            is_2d = header.group(2).count("[") == 2
        elif line and name:
            values = [float(v) for v in re.sub(r"^\[\d+\],", "", line).split(",")]
            if is_2d:
                out[name].append(values)
            else:
                out[name] = values
    return out


def parse_mop(text):
    """MOP .ascii -> one row per time: hs (m), tp (s), dp (deg), and the
    spectrum as JSON lists: freq (Hz), bandwidth (Hz), energy (m^2/Hz),
    direction (deg). Rows with a bad QC flag or fill values are dropped."""
    d = parse_dap_ascii(text)
    df = pd.DataFrame({
        "time": pd.to_datetime(d["waveTime"], unit="s").strftime(TIME_FMT),
        "hs": d["waveHs"], "tp": d["waveTp"], "dp": d["waveDp"],
        "freq": json.dumps(d["waveFrequency"]),
        "bandwidth": json.dumps(d["waveBandwidth"]),
        "energy": [json.dumps(row) for row in d["waveEnergyDensity"]],
        "direction": [json.dumps(row) for row in d["waveMeanDirection"]],
    })
    good = (pd.Series(d["waveFlagPrimary"]) == 1) & (df["hs"] >= 0)
    return df[good].reset_index(drop=True)


def fetch_mop(mop_id, kind, last=None):
    """kind is 'nowcast' (hourly, buoy-driven) or 'forecast' (3-hourly).
    last = number of most recent time steps to fetch (None = all)."""
    base = f"{MOP_URL}/{mop_id}_{kind}.nc"
    n = int(re.search(r"waveTime = (\d+)", get(base + ".dds").text).group(1))
    t = f"[{max(0, n - last) if last else 0}:{n - 1}]"
    query = ",".join([f"waveTime{t}", f"waveFlagPrimary{t}", f"waveHs{t}",
                      f"waveTp{t}", f"waveDp{t}", f"waveEnergyDensity{t}",
                      f"waveMeanDirection{t}", "waveBandwidth"])
    return parse_mop(get(f"{base}.ascii?{query}", timeout=60).text)


# --- NDBC buoys (measured, for sanity checks) --------------------------------

def parse_buoy(text):
    """NDBC .spec text -> time, hs, swell_hs (m), swell_tp (s), mwd (deg)."""
    rows = []
    for line in text.splitlines():
        p = line.split()
        if line.startswith("#") or len(p) < 15:
            continue
        num = lambda s: None if s in ("MM", "N/A", "-99") else float(s)
        rows.append({"time": f"{p[0]}-{p[1]}-{p[2]}T{p[3]}:{p[4]}",
                     "hs": num(p[5]), "swell_hs": num(p[6]),
                     "swell_tp": num(p[7]), "mwd": num(p[14])})
    return pd.DataFrame(rows, columns=["time", "hs", "swell_hs", "swell_tp", "mwd"])


def fetch_buoy(station):
    return parse_buoy(get(f"{NDBC_URL}/{station}.spec").text)


# --- NOAA tide predictions ---------------------------------------------------

def parse_tides(data):
    """CO-OPS hi/lo JSON -> time, height (m above MLLW), type ('H' or 'L')."""
    if "predictions" not in data:
        raise ValueError(f"tide API error: {data}")
    return pd.DataFrame([{"time": p["t"].replace(" ", "T"),
                          "height": float(p["v"]), "type": p["type"]}
                         for p in data["predictions"]])


def fetch_tides(station, days=8):
    today = pd.Timestamp.now("UTC").strftime("%Y%m%d")
    return parse_tides(get(TIDE_URL, {
        "station": station, "product": "predictions", "datum": "MLLW",
        "interval": "hilo", "units": "metric", "time_zone": "gmt",
        "format": "json", "begin_date": today, "range": days * 24}).json())


# --- Open-Meteo wind ---------------------------------------------------------

def parse_wind(data):
    """Open-Meteo JSON -> time, speed (m/s), direction (deg FROM), gust (m/s)."""
    h = data["hourly"]
    return pd.DataFrame({"time": h["time"], "speed": h["wind_speed_10m"],
                         "direction": h["wind_direction_10m"],
                         "gust": h["wind_gusts_10m"]}).dropna()


def fetch_wind(lat, lon):
    return parse_wind(get(WIND_URL, {
        "latitude": lat, "longitude": lon, "timezone": "GMT",
        "hourly": "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "ms", "past_days": 1, "forecast_days": 7}).json())


# --- Open-Meteo marine (long-range swell, for the forecast tail) -------------

def parse_marine(data):
    """Open-Meteo marine JSON -> time, hs (m), tp (s), dp (deg) of the swell.
    Hours past the model's horizon come back null and are dropped."""
    h = data["hourly"]
    return pd.DataFrame({"time": h["time"], "hs": h["swell_wave_height"],
                         "tp": h["swell_wave_period"],
                         "dp": h["swell_wave_direction"]}).dropna()


def fetch_marine(lat, lon):
    return parse_marine(get(MARINE_URL, {
        "latitude": lat, "longitude": lon, "timezone": "GMT",
        "hourly": "swell_wave_height,swell_wave_period,swell_wave_direction",
        "forecast_days": 16}).json())
