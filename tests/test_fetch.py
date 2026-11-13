"""Parser tests against saved API responses. No network."""
import json
from pathlib import Path

import pytest

from surf import fetch

FIX = Path(__file__).parent / "fixtures"


def test_parse_mop():
    df = fetch.parse_mop((FIX / "mop_nowcast.ascii").read_text())
    assert len(df) == 3
    last = df.iloc[-1]
    assert last.time == "2026-11-13T17:00"
    assert last.hs == pytest.approx(3.3176498)
    assert last.tp == pytest.approx(14.285714)
    assert last.dp == pytest.approx(260.61462)
    freq, bw = json.loads(last.freq), json.loads(last.bandwidth)
    energy, direction = json.loads(last.energy), json.loads(last.direction)
    assert len(freq) == len(bw) == len(energy) == len(direction) == 20
    assert freq[0] == pytest.approx(0.04) and energy[6] == pytest.approx(25.949598)
    # Hs = 4 * sqrt(sum(E * df)) must reproduce the file's own Hs
    m0 = sum(e * w for e, w in zip(energy, bw))
    assert 4 * m0 ** 0.5 == pytest.approx(last.hs, rel=0.02)


def test_parse_mop_drops_bad_flag():
    text = (FIX / "mop_nowcast.ascii").read_text().replace(
        "waveFlagPrimary[3]\n1, 1, 1", "waveFlagPrimary[3]\n1, 4, 1")
    assert len(fetch.parse_mop(text)) == 2


def test_parse_buoy():
    df = fetch.parse_buoy((FIX / "buoy_46237.spec").read_text())
    assert len(df) == 6
    assert df.time.str.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d").all()
    assert 0 < df.hs[0] < 15


def test_parse_buoy_missing_values():
    line = "2026 11 13 17 00  3.0   MM   MM   MM   MM N/A N/A        N/A   MM -99\n"
    df = fetch.parse_buoy(line)
    assert df.hs[0] == 3.0
    assert df[["swell_hs", "swell_tp", "mwd"]].isna().all(axis=None)


def test_parse_tides():
    df = fetch.parse_tides(json.loads((FIX / "tides_9414290.json").read_text()))
    assert len(df) == 8
    assert df.iloc[0].to_dict() == {"time": "2026-11-13T03:17", "height": -0.087,
                                    "type": "L"}


def test_parse_tides_error_response():
    with pytest.raises(ValueError):
        fetch.parse_tides({"error": {"message": "No data"}})


def test_parse_wind():
    df = fetch.parse_wind(json.loads((FIX / "wind.json").read_text()))
    assert len(df) == 24
    assert df.time[0] == "2026-11-13T00:00"
    assert (df.speed >= 0).all() and df.direction.between(0, 360).all()


def test_parse_marine_drops_hours_past_the_horizon():
    data = json.loads((FIX / "marine.json").read_text())
    assert len(fetch.parse_marine(data)) == 24
    data["hourly"]["swell_wave_height"][-1] = None
    df = fetch.parse_marine(data)
    assert len(df) == 23
    assert list(df.columns) == ["time", "hs", "tp", "dp"]
    assert df.hs.between(0, 15).all() and df.tp.between(2, 30).all()
