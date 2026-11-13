"""Load the spot table from spots.toml."""
import tomllib
from pathlib import Path

SPOTS_FILE = Path(__file__).resolve().parent.parent / "spots.toml"
REQUIRED = {"name", "lat", "lon", "mop", "mop_depth_m", "shore_normal",
            "window", "tide_station", "buoys"}


def load_spots(path=SPOTS_FILE):
    """Return {spot_id: dict}. Raises ValueError if a spot is missing a field."""
    with open(path, "rb") as f:
        spots = tomllib.load(f)
    for key, s in spots.items():
        missing = REQUIRED - s.keys()
        if missing:
            raise ValueError(f"spot {key!r} is missing {sorted(missing)}")
    return spots
