import pandas as pd

from surf import store


def test_upsert_is_idempotent_and_replaces(tmp_path):
    con = store.connect(tmp_path / "t.db")
    df = pd.DataFrame({"time": ["2026-11-13T00:00", "2026-11-13T01:00"],
                       "hs": [1.0, None], "swell_hs": [0.5, 0.6],
                       "swell_tp": [12.0, 13.0], "mwd": [270.0, 280.0]})
    store.upsert(con, "buoy", df, station="46237")
    store.upsert(con, "buoy", df.assign(swell_hs=9.0), station="46237")
    out = store.read(con, "SELECT * FROM buoy ORDER BY time")
    assert len(out) == 2
    assert (out.swell_hs == 9.0).all()
    assert pd.isna(out.hs[1])
