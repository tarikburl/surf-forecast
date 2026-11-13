import re

import pytest

from surf.spots import load_spots


def test_all_spots_load_and_are_sane():
    spots = load_spots()
    assert len(spots) == 7
    for s in spots.values():
        assert 36 < s["lat"] < 39 and -123 < s["lon"] < -121
        assert re.fullmatch(r"(SF|SM|SC)\d{3}", s["mop"])
        assert 0 <= s["shore_normal"] < 360
        lo, hi = s["window"]
        assert 0 <= lo < hi <= 360
        assert s["mop_depth_m"] > 0


def test_missing_field_is_an_error(tmp_path):
    p = tmp_path / "spots.toml"
    p.write_text('[x]\nname = "X"\n')
    with pytest.raises(ValueError, match="missing"):
        load_spots(p)
