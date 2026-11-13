"""One-off: map each spot with its neighboring CDIP MOP points.

Usage: uv run python scripts/map_mop.py   ->  data/mop_map.png + a table
"""
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

from surf.spots import load_spots

BASE = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore"
FIELDS = "metaLatitude,metaLongitude,metaWaterDepth,metaShoreNormal"
SPAN = 12  # MOP points to look at on each side of the assigned one


def mop_meta(mop_id):
    """(lat, lon, depth_m, shore_normal) for one MOP point, or None."""
    url = f"{BASE}/{mop_id}_forecast.nc.ascii?{FIELDS}"
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            vals = re.findall(r"^meta\w+, (-?[\d.]+)", r.text, re.M)
            if len(vals) == 4:
                return tuple(float(v) for v in vals)
        except requests.RequestException:
            time.sleep(2)
    return None


def km(lat1, lon1, lat2, lon2):
    """Flat-earth distance, fine at these scales."""
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians(lat1))
    return math.hypot((lat2 - lat1) * 110.57, dx)


def main():
    spots = load_spots()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, s in zip(axes.flat, spots.values()):
        prefix, n = s["mop"][:2], int(s["mop"][2:])
        ids = [f"{prefix}{i:03d}" for i in range(max(1, n - SPAN), n + SPAN + 1)]
        with ThreadPoolExecutor(8) as ex:
            meta = {i: m for i, m in zip(ids, ex.map(mop_meta, ids)) if m}
        dist = {i: km(s["lat"], s["lon"], m[0], m[1]) for i, m in meta.items()}
        nearest = sorted(dist, key=dist.get)[:5]

        print(f"\n{s['name']}  (assigned {s['mop']})")
        print("  id      dist_km  depth_m  shore_normal")
        for i in nearest:
            lat, lon, depth, normal = meta[i]
            print(f"  {i}   {dist[i]:6.2f}   {depth:6.1f}   {normal:6.0f}")

        cos = math.cos(math.radians(s["lat"]))
        for i, (lat, lon, depth, normal) in meta.items():
            color = "red" if i == s["mop"] else ("C0" if i in nearest else "0.6")
            ax.plot(lon, lat, "o", color=color, ms=5)
            # arrow points seaward, along the shore normal
            L = 0.003
            ax.annotate("", xytext=(lon, lat),
                        xy=(lon + L * math.sin(math.radians(normal)) / cos,
                            lat + L * math.cos(math.radians(normal))),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            if i in nearest or int(i[2:]) % 4 == 0:
                ax.annotate(f"{i} {normal:.0f}°", (lon, lat), fontsize=7, color=color,
                            xytext=(4, 4), textcoords="offset points")
        ax.plot(s["lon"], s["lat"], "*", color="k", ms=16)
        ax.set_title(f"{s['name']}  (assigned {s['mop']})", fontsize=10)
        ax.set_aspect(1 / cos)
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(spots):]:
        ax.axis("off")
        ax.text(0, 0.5, "star = spot\nred = assigned MOP point\nblue = 5 nearest\n"
                "arrow = shore normal (seaward)\nlabel = id, shore normal",
                fontsize=11, va="center")
    fig.tight_layout()
    fig.savefig("data/mop_map.png", dpi=110)
    print("\nwrote data/mop_map.png")


if __name__ == "__main__":
    main()
