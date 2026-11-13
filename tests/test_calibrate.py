import json
import tomllib

import numpy as np
import pandas as pd
import pytest

from surf import calibrate, store
from surf.forecast import FT
from tests.test_forecast import SPOT
from tests.test_model import F, W, bump


def test_log_round_trip_and_validation(tmp_path):
    p = tmp_path / "log.csv"
    calibrate.append_log("linda_mar", 3, 4, "2026-11-13T16:30", "clean, fun", p)
    calibrate.append_log("linda_mar", 2, 3, "2026-11-12T16:30", path=p)
    log = calibrate.read_log(p)
    assert list(log.time) == ["2026-11-12T16:30", "2026-11-13T16:30"]   # time order
    assert log.iloc[1].to_dict() == {"time": "2026-11-13T16:30", "spot": "linda_mar",
                                     "low_ft": 3, "high_ft": 4, "note": "clean, fun"}
    with pytest.raises(ValueError):
        calibrate.append_log("linda_mar", 5, 3, "2026-11-13T16:30", path=p)  # low > high
    assert calibrate.read_log(tmp_path / "none.csv").empty


def test_fit_recovers_a_scale_factor():
    rng = np.random.default_rng(0)
    hb = rng.uniform(0.5, 3, 20)
    obs = calibrate.predict_mid_ft(hb, 0.6, 1.0) * rng.normal(1, 0.05, 20)
    a, b = calibrate.fit(hb, obs)
    assert b == 1.0 and a == pytest.approx(0.6, rel=0.05)


def test_fit_recovers_the_exponent_with_enough_sessions():
    rng = np.random.default_rng(1)
    hb = rng.uniform(0.5, 4, 60)
    obs = calibrate.predict_mid_ft(hb, 0.5, 1.3) * rng.normal(1, 0.05, 60)
    a, b = calibrate.fit(hb, obs)
    assert a == pytest.approx(0.5, rel=0.08) and b == pytest.approx(1.3, abs=0.08)


def test_metrics_hand_values():
    m = calibrate.metrics([4, 6, 9], [3, 6, 6])   # errors +1, 0, +3
    assert m["mae"] == pytest.approx(4 / 3) and m["bias"] == pytest.approx(4 / 3)
    assert m["within_1ft"] == pytest.approx(200 / 3)


def test_evaluate_shows_the_fit_beating_a_wrong_scale():
    rng = np.random.default_rng(2)
    hb = rng.uniform(0.5, 3, 30)
    obs = calibrate.predict_mid_ft(hb, 0.6, 1.0) * rng.normal(1, 0.08, 30)
    r = calibrate.evaluate(hb, obs, a_now=1.0, b_now=1.0)
    assert r["n"] == 30 and r["a"] == pytest.approx(0.6, rel=0.06)
    assert r["fitted"]["mae"] < 0.5 * r["current"]["mae"]
    assert r["current"]["bias"] > 0    # the uncalibrated model reads high


def test_raw_heights_uses_stored_rows_and_marks_gaps(tmp_path):
    con = store.connect(tmp_path / "t.db")
    store.upsert(con, "mop", pd.DataFrame({
        "time": ["2026-01-01T10:00"], "hs": 1.0, "tp": 14.0, "dp": 270.0,
        "freq": json.dumps(list(F)), "bandwidth": json.dumps(list(W)),
        "energy": json.dumps(list(bump(0.07, 1.0))),
        "direction": json.dumps(list(np.full_like(F, 270.0)))}),
        mop="XX001", kind="nowcast")
    hb = calibrate.raw_heights(con, SPOT | {"a": 0.5}, ["2026-01-01T10:40", "2026-01-03T10:00"],
                               download=False)
    assert hb[0] == pytest.approx(1.74, abs=0.05)   # uncalibrated: ignores a=0.5
    assert np.isnan(hb[1])                          # nothing within 90 minutes


def test_set_spot_values_edits_only_that_spot():
    text = '[one]\nname = "One"\na = 0.81\nlat = 1.0\n\n[two]\nname = "Two"\nlat = 2.0\n'
    out = calibrate.set_spot_values(text, "two", a=0.5, b=1.2)
    out = calibrate.set_spot_values(out, "one", a=0.7)
    assert tomllib.loads(out) == {"one": {"name": "One", "a": 0.7, "lat": 1.0},
                                  "two": {"name": "Two", "a": 0.5, "b": 1.2, "lat": 2.0}}
