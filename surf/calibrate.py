"""Session log and per-spot calibration  H = a * Hb^b  (Hb in metres)."""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from surf import fetch, model, store
from surf.forecast import FT

LOG = Path(__file__).resolve().parent.parent / "log.csv"
FIELDS = ["time", "spot", "low_ft", "high_ft", "note"]
MIN_OBS = 15       # below this a spot keeps its current a, b
MIN_OBS_B = 40     # below this only a is fitted (b stays 1)
MID = (1 + model.SET_FACTOR) / 2   # middle of the reported range, per unit of `low`
MATCH = pd.Timedelta(minutes=90)   # how far a model row may be from a session


def append_log(spot, low_ft, high_ft, time, note="", path=LOG):
    """Add one observed session. time is a UTC 'YYYY-MM-DDTHH:MM' string."""
    if not 0 <= low_ft <= high_ft <= 80:
        raise ValueError("heights are face height in feet, low then high")
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"time": time, "spot": spot, "low_ft": low_ft,
                    "high_ft": high_ft, "note": note})


def read_log(path=LOG):
    if not Path(path).exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path, dtype={"note": str}).sort_values("time", ignore_index=True)


def raw_heights(con, spot, times, download=True):
    """Uncalibrated breaker height Hb (m) at each session time, from the MOP
    nowcast. Rows not already stored are downloaded and kept. NaN where the
    model has no row within MATCH of the session."""
    when = pd.to_datetime(pd.Series(list(times), dtype=str))
    have = lambda: store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast'",
                              (spot["mop"],))
    rows = have()
    stored = pd.to_datetime(rows.time)
    missing = [t for t in when if stored.empty or (stored - t).abs().min() > MATCH]
    if missing and download:
        all_times = fetch.fetch_mop_times(spot["mop"], "nowcast")
        for t in missing:
            i = int(np.abs(all_times - t).argmin())
            if abs(all_times[i] - t) <= MATCH:
                store.upsert(con, "mop", fetch.fetch_mop(spot["mop"], "nowcast", rows=(i, i)),
                             mop=spot["mop"], kind="nowcast")
        rows = have()
        stored = pd.to_datetime(rows.time)
    base = spot | {"a": 1.0, "b": 1.0}
    out = []
    for t in when:
        gap = (stored - t).abs() if len(stored) else pd.Series(dtype="timedelta64[ns]")
        out.append(model.surf_height(rows.loc[gap.idxmin()], base)["low"]
                   if len(gap) and gap.min() <= MATCH else np.nan)
    return np.array(out)


def fit(hb, obs_mid_ft):
    """Least squares in log space for  obs_mid = MID * a * hb^b.
    hb in metres, obs_mid_ft = middle of the observed range in feet.
    b is fitted only with MIN_OBS_B or more sessions; otherwise b = 1."""
    x, y = np.log(hb), np.log(np.asarray(obs_mid_ft) / FT / MID)
    if len(x) >= MIN_OBS_B and np.ptp(x) > 0:
        b, log_a = np.polyfit(x, y, 1)
    else:
        b, log_a = 1.0, (y - x).mean()
    return float(np.exp(log_a)), float(b)


def predict_mid_ft(hb, a, b):
    return MID * a * np.asarray(hb) ** b * FT


def metrics(pred_ft, obs_ft):
    """Mean absolute error (ft), bias (ft, + = we read high), % within 1 ft."""
    err = np.asarray(pred_ft) - np.asarray(obs_ft)
    return {"mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "within_1ft": float((np.abs(err) <= 1).mean() * 100)}


def evaluate(hb, obs_mid_ft, a_now, b_now):
    """Backtest one spot. Sessions must be in time order. The first 70% train
    a new fit; both it and the current (a_now, b_now) are scored on the last
    30%. Returns {n, current, fitted, a, b}; a, b are refitted on everything."""
    hb, obs = np.asarray(hb), np.asarray(obs_mid_ft)
    cut = int(len(hb) * 0.7)
    a_train, b_train = fit(hb[:cut], obs[:cut])
    a, b = fit(hb, obs)
    return {"n": len(hb), "a": a, "b": b,
            "current": metrics(predict_mid_ft(hb[cut:], a_now, b_now), obs[cut:]),
            "fitted": metrics(predict_mid_ft(hb[cut:], a_train, b_train), obs[cut:])}


def set_spot_values(toml_text, key, **values):
    """Return spots.toml text with numeric values set in one spot's table."""
    start = toml_text.index(f"[{key}]\n")
    end = toml_text.find("\n[", start)
    end = len(toml_text) if end == -1 else end
    block = toml_text[start:end]
    for name, v in values.items():
        line = f"{name} = {v:.2f}"
        if re.search(rf"^{name} = .*$", block, re.M):
            block = re.sub(rf"^{name} = .*$", line, block, flags=re.M)
        else:
            block = block.replace(f"[{key}]\n", f"[{key}]\n{line}\n", 1)
    return toml_text[:start] + block + toml_text[end:]
