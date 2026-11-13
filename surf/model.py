"""MOP spectrum at a spot -> swell trains -> surf height."""
import json

import numpy as np

from surf import physics

MAX_TRAINS = 3
MIN_FRACTION = 0.05   # a train must hold at least this share of total energy
VALLEY = 0.7          # two peaks are one swell unless the dip between them is
                      # below this fraction of the smaller peak
WINDOW_EDGE = 20.0    # degrees over which a swell outside the window fades out
MIN_SURF_PERIOD = 8.0  # s; shorter-period bands are wind chop and do not count as surf
SET_FACTOR = 1.27     # H(1/10) / Hs for Rayleigh-distributed wave heights


def swell_trains(freq, bandwidth, energy, direction):
    """Split a spectrum into at most MAX_TRAINS swell trains.

    Inputs are per frequency band: freq (Hz), bandwidth (Hz), energy density
    (m^2/Hz), mean direction (deg). The spectrum is cut at the minima between
    its peaks; neighbours are merged while the valley between them is shallow,
    a piece is too small, or there are too many pieces.

    Returns a list of {hs (m), tp (s), dp (deg), bands (first, past-last
    band index)}, largest first. The trains
    together hold all the energy: sqrt(sum hs^2) equals the total Hs.
    """
    f, w, e, d = (np.asarray(x, float) for x in (freq, bandwidth, energy, direction))
    pad = np.r_[-1.0, e, -1.0]
    peaks = [i for i in range(len(e)) if pad[i] < e[i] >= pad[i + 2]]
    cuts = [0] + [a + int(np.argmin(e[a:b + 1])) for a, b in zip(peaks, peaks[1:])] + [len(e)]
    parts = [slice(a, b) for a, b in zip(cuts, cuts[1:])]
    m0 = lambda p: float((e[p] * w[p]).sum())
    total = m0(slice(None))

    while len(parts) > 1:
        shallow = [i for i in range(len(parts) - 1)
                   if e[parts[i + 1].start] > VALLEY * min(e[parts[i]].max(), e[parts[i + 1]].max())]
        small = min(range(len(parts)), key=lambda i: m0(parts[i]))
        if shallow:
            i = shallow[0]
        elif len(parts) > MAX_TRAINS or m0(parts[small]) < MIN_FRACTION * total:
            # merge the smallest piece into its larger neighbour
            right_bigger = (small < len(parts) - 1 and
                            (small == 0 or m0(parts[small + 1]) > m0(parts[small - 1])))
            i = small if right_bigger else small - 1
        else:
            break
        parts[i:i + 2] = [slice(parts[i].start, parts[i + 1].stop)]

    trains = []
    for p in parts:
        weight, rad = e[p] * w[p], np.radians(d[p])
        dp = np.degrees(np.arctan2((weight * np.sin(rad)).sum(),
                                   (weight * np.cos(rad)).sum())) % 360
        trains.append({"hs": 4 * m0(p) ** 0.5, "tp": 1 / f[p][np.argmax(e[p])],
                       "dp": float(dp), "bands": (p.start, p.stop)})
    return sorted(trains, key=lambda t: -t["hs"])


def window_weight(dp, window):
    """1 inside the spot's swell window [lo, hi], fading linearly to 0 over
    WINDOW_EDGE degrees outside it."""
    lo, hi = window
    if lo <= dp <= hi:
        return 1.0
    outside = min(abs((dp - lo + 180) % 360 - 180), abs((dp - hi + 180) % 360 - 180))
    return max(0.0, 1 - outside / WINDOW_EDGE)


def surf_height(mop_row, spot):
    """One MOP row + one spot -> surf height.

    Bands with period under MIN_SURF_PERIOD are wind chop: they are reported
    as chop_hs but do not count as surf. The rest is split into swell trains;
    each is weighted by the spot's window, de-shoaled from the MOP depth to
    deep water, and broken with Komar-Gaughan. Trains combine by energy. The
    spot's calibration H = a * Hb^b is then applied (a=1, b=1 if the spot has
    none).

    Returns {low, high (m), trains, chop_hs (m)}: low is the typical breaker
    height, high is the set waves (SET_FACTOR * low). Each train carries its
    hs/tp/dp at the MOP point, its window weight, deep-water height h0 and
    breaker height hb.
    """
    f, w, e, d = (np.array(json.loads(mop_row[c]))
                  for c in ("freq", "bandwidth", "energy", "direction"))
    swell = f <= 1 / MIN_SURF_PERIOD
    trains = swell_trains(f[swell], w[swell], e[swell], d[swell])
    for t in trains:
        t["weight"] = window_weight(t["dp"], spot["window"])
        t["h0"] = float(physics.deep_water_height(
            t["hs"] * t["weight"], t["tp"], spot["mop_depth_m"]))
        t["hb"] = float(physics.breaking_height(t["h0"], t["tp"]))
    hb = sum(t["hb"] ** 2 for t in trains) ** 0.5
    low = spot.get("a", 1.0) * hb ** spot.get("b", 1.0)
    return {"low": low, "high": SET_FACTOR * low, "trains": trains,
            "chop_hs": 4 * float((e[~swell] * w[~swell]).sum()) ** 0.5}
