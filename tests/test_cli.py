from surf import cli, forecast


def test_compass():
    assert [forecast.compass(d) for d in (0, 359, 90, 262, 292.5)] == ["N", "N", "E", "W", "WNW"]


def test_find_spot():
    import pytest
    from surf.spots import load_spots
    spots = load_spots()
    assert cli.find_spot(spots, "ocean") == "ocean_beach"
    assert cli.find_spot(spots, "Linda Mar") == "linda_mar"
    assert cli.find_spot(spots, "steamer_lane") == "steamer_lane"
    with pytest.raises(SystemExit):
        cli.find_spot(spots, "santa cruz")  # ambiguous
    with pytest.raises(SystemExit):
        cli.find_spot(spots, "mavericks")
