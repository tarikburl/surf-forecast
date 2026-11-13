import json

import numpy as np
import pandas as pd
import pytest

from surf import forecast, store
from tests.test_model import F, W, bump

SPOT = {"mop": "XX001", "window": [200, 320], "mop_depth_m": 15.0,
        "shore_normal": 270, "tide_station": "T1"}


def test_tide_height_cosine_interpolation():
    tides = pd.DataFrame({"time": ["2026-01-01T00:00", "2026-01-01T06:00",
                                   "2026-01-01T12:00"], "height": [0.0, 2.0, 0.4]})
    at = lambda s: forecast.tide_height(tides, pd.Timestamp(s))
    assert at("2026-01-01T03:00") == pytest.approx(1.0)      # halfway up
    assert at("2026-01-01T06:00") == pytest.approx(2.0)      # at high tide
    assert at("2026-01-01T01:30") == pytest.approx(0.293, abs=0.001)  # 2*(1-cos 45)/2
    assert at("2026-01-01T09:00") == pytest.approx(1.2)      # halfway down
    assert at("2025-12-31T23:00") is None and at("2026-01-01T13:00") is None


def test_wind_kind():
    # beach faces west (shore normal 270): west wind is onshore, east is offshore
    assert forecast.wind_kind(270, 270) == "onshore"
    assert forecast.wind_kind(90, 270) == "offshore"
    assert forecast.wind_kind(0, 270) == "cross-shore"
    assert forecast.wind_kind(10, 350) == "onshore"  # wraps through north


def test_rating():
    r = lambda low, speed, wfrom, tide=None, **kw: forecast.rating(
        low, speed, wfrom, tide, SPOT | kw)
    assert r(0.3, 1, 90) == "flat"            # under 2 ft
    assert r(1.5, 1, 270) == "good"           # light wind, any direction
    assert r(1.5, 5, 90) == "good"            # moderate offshore
    assert r(1.5, 10, 90) == "fair"           # strong offshore
    assert r(1.5, 10, 270) == "poor"          # strong onshore
    assert r(1.5, 5, 90, tide=1.5, tide_max=1.0) == "fair"   # too much tide
    assert r(1.5, np.nan, np.nan) == ""       # no wind forecast


def fill(con, n_mop=8, n_sea=16):
    times = pd.date_range("2026-01-01", periods=n_sea, freq="3h").strftime("%Y-%m-%dT%H:%M")
    d = list(np.full_like(F, 270.0))
    store.upsert(con, "mop", pd.DataFrame({
        "time": times[:n_mop], "hs": 1.0, "tp": 14.0, "dp": 270.0,
        "freq": json.dumps(list(F)), "bandwidth": json.dumps(list(W)),
        "energy": json.dumps(list(bump(0.07, 1.0))), "direction": json.dumps(d)}),
        mop="XX001", kind="forecast")
    # the global model reads half the height the nearshore model does
    store.upsert(con, "marine", pd.DataFrame(
        {"time": times, "hs": 0.5, "tp": 14.0, "dp": 270.0}), spot="x")
    return times


def test_build_appends_a_tail_scaled_to_match(tmp_path):
    con = store.connect(tmp_path / "t.db")
    times = fill(con)
    df = forecast.build(con, "x", SPOT, times[0])
    assert list(df.time) == list(times)                    # no gaps, no duplicates
    assert list(df.source) == ["mop"] * 8 + ["tail"] * 8
    assert df.low.iloc[-1] == pytest.approx(df.low.iloc[0], rel=0.01)  # scaled up to match
    assert list(df.high) == pytest.approx(list(1.27 * df.low))
    assert (df.rating == "").all()                         # no wind stored


def test_build_without_marine_data_has_no_tail(tmp_path):
    con = store.connect(tmp_path / "t.db")
    times = fill(con)
    con.execute("DELETE FROM marine")
    assert list(forecast.build(con, "x", SPOT, times[0]).source) == ["mop"] * 8


def test_save_run_keeps_one_copy_per_run(tmp_path):
    con = store.connect(tmp_path / "t.db")
    df = forecast.build(con, "x", SPOT, fill(con)[0])
    forecast.save_run(con, "x", df, "2026-01-01")
    forecast.save_run(con, "x", df, "2026-01-01")
    forecast.save_run(con, "x", df, "2026-01-02")
    assert con.execute("SELECT COUNT(*) FROM forecast").fetchone()[0] == 32
