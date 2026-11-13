import pandas as pd

from surf import page, store
from tests.test_forecast import SPOT, fill

SPOTS = {"x": SPOT | {"name": "Test & Spot"}}


def test_page_renders_with_no_data(tmp_path):
    html = page.render(store.connect(tmp_path / "t.db"), SPOTS, pd.Timestamp("2026-01-01T01:00"))
    assert "Test &amp; Spot" in html and "No wave data yet" in html


def test_page_shows_surf_chart_and_stale_badge(tmp_path):
    con = store.connect(tmp_path / "t.db")
    fill(con)
    con.execute("INSERT INTO mop SELECT mop, 'nowcast', time, hs, tp, dp, freq, bandwidth, "
                "energy, direction FROM mop WHERE time='2026-01-01T00:00'")
    fresh = page.render(con, SPOTS, pd.Timestamp("2026-01-01T01:00"))
    assert "ft now" in fresh and "<svg" in fresh and "h old" not in fresh
    assert fresh.count('class="band"') == 1 and fresh.count('class="tail"') == 1
    assert "low confidence" in fresh
    assert "<script" not in fresh
    stale = page.render(con, SPOTS, pd.Timestamp("2026-01-01T09:00"))
    assert "data 9h old" in stale
