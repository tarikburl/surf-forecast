import json
from pathlib import Path

import numpy as np
import pytest

from surf import fetch, model

F = np.arange(0.04, 0.26, 0.01)
W = np.full_like(F, 0.01)
SPOT = {"window": [200, 320], "mop_depth_m": 15.0}


def bump(f0, hs, width=0.01):
    """A Gaussian spectral peak at f0 holding exactly Hs = hs."""
    e = np.exp(-0.5 * ((F - f0) / width) ** 2)
    return e * (hs / 4) ** 2 / (e * W).sum()


def row(energy, direction):
    return {"freq": json.dumps(list(F)), "bandwidth": json.dumps(list(W)),
            "energy": json.dumps(list(energy)), "direction": json.dumps(list(direction))}


def test_two_swells_are_separated():
    e = bump(0.07, 2.0) + bump(0.17, 1.0)
    d = np.where(F < 0.12, 280.0, 200.0)
    trains = model.swell_trains(F, W, e, d)
    assert len(trains) == 2
    assert trains[0]["hs"] == pytest.approx(2.0, rel=0.03)
    assert trains[0]["tp"] == pytest.approx(1 / 0.07, rel=0.01)
    assert trains[0]["dp"] == pytest.approx(280, abs=1)
    assert trains[1]["hs"] == pytest.approx(1.0, rel=0.05)
    assert trains[1]["dp"] == pytest.approx(200, abs=1)


def test_ripple_on_one_swell_stays_one_train():
    e = bump(0.08, 2.0, width=0.02)
    e[5] *= 0.9  # small dip next to the peak, not a real valley
    assert len(model.swell_trains(F, W, e, np.full_like(F, 270.0))) == 1


def test_tiny_second_peak_is_merged():
    e = bump(0.07, 2.0) + bump(0.2, 0.2)
    assert len(model.swell_trains(F, W, e, np.full_like(F, 270.0))) == 1


def test_never_more_than_three_trains_and_energy_is_conserved():
    e = sum(bump(f0, 1.0, width=0.006) for f0 in (0.05, 0.09, 0.13, 0.17, 0.22))
    trains = model.swell_trains(F, W, e, np.full_like(F, 270.0))
    assert len(trains) <= 3
    assert sum(t["hs"] ** 2 for t in trains) ** 0.5 == pytest.approx(5 ** 0.5, rel=0.01)


def test_direction_mean_handles_wrap_through_north():
    d = np.where(np.arange(len(F)) % 2 == 0, 350.0, 10.0)
    dp = model.swell_trains(F, W, bump(0.08, 1.0), d)[0]["dp"]
    assert min(dp, 360 - dp) < 5


def test_real_spectrum_trains_add_up_to_reported_hs():
    df = fetch.parse_mop((Path(__file__).parent / "fixtures/mop_nowcast.ascii").read_text())
    r = df.iloc[-1]
    trains = model.swell_trains(*(json.loads(r[c]) for c in
                                  ("freq", "bandwidth", "energy", "direction")))
    assert 1 <= len(trains) <= 3
    assert sum(t["hs"] ** 2 for t in trains) ** 0.5 == pytest.approx(r.hs, rel=0.02)
    assert trains[0]["tp"] == pytest.approx(r.tp, rel=0.01)


def test_window_weight():
    assert model.window_weight(260, [200, 320]) == 1
    assert model.window_weight(200, [200, 320]) == 1
    assert model.window_weight(190, [200, 320]) == pytest.approx(0.5)
    assert model.window_weight(330, [200, 320]) == pytest.approx(0.5)
    assert model.window_weight(150, [200, 320]) == 0
    assert model.window_weight(10, [200, 320]) == 0


def test_surf_height_hand_value():
    # One swell, Hs=1 m, T=14.3 s (0.07 Hz), 15 m depth, inside the window.
    # H0 = 1/1.03 = 0.97;  Hb = 0.39 * 1.579 * (14.3 * 0.97^2)^0.4 = 1.74 m
    out = model.surf_height(row(bump(0.07, 1.0, width=0.004), np.full_like(F, 270.0)), SPOT)
    assert out["low"] == pytest.approx(1.74, abs=0.04)
    assert out["high"] == pytest.approx(1.27 * out["low"])


def test_blocked_direction_gives_no_surf():
    out = model.surf_height(row(bump(0.07, 2.0), np.full_like(F, 90.0)), SPOT)
    assert out["low"] == 0


def test_calibration_is_applied():
    r = row(bump(0.07, 1.0), np.full_like(F, 270.0))
    base = model.surf_height(r, SPOT)["low"]
    assert model.surf_height(r, SPOT | {"a": 0.5})["low"] == pytest.approx(0.5 * base)


def test_wind_chop_is_not_counted_as_surf():
    swell = bump(0.07, 1.0)
    both = swell + bump(0.2, 1.5)  # add a 5-second chop
    d = np.full_like(F, 270.0)
    out = model.surf_height(row(both, d), SPOT)
    assert out["chop_hs"] == pytest.approx(1.5, rel=0.03)
    assert all(t["tp"] >= model.MIN_SURF_PERIOD for t in out["trains"])
    assert out["low"] == pytest.approx(model.surf_height(row(swell, d), SPOT)["low"], rel=0.03)
