"""Render the 'How it works' page: where the data comes from and how a surf
height is calculated, with a worked example from the latest stored data."""
import json
import math
from html import escape
from pathlib import Path

import pandas as pd

from surf import calibrate, fetch, model, physics, store
from surf.forecast import FT, compass, local

SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
TRAIN_CLASS = ["t1", "t2", "t3"]

CSS = """
article { max-width:720px; margin:0 auto; }
article section { background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:14px 16px; margin-bottom:12px; }
article h2 { margin:0 0 6px; } article h3 { font-size:14px; margin:14px 0 4px; }
article p { margin:6px 0; } article a { color:var(--series); }
pre { background:var(--page); border:1px solid var(--line); border-radius:6px;
  padding:8px; font-size:11px; line-height:1.35; overflow-x:auto; margin:6px 0; }
.formula { font-family:Georgia, serif; font-size:16px; margin:8px 0; white-space:nowrap; overflow-x:auto; }
.note { color:var(--ink2); font-size:13px; }
.wrap { overflow-x:auto; }
:root { --t1:#2a78d6; --t2:#eb6834; --t3:#1baf7a; --chop:#b5b4ad;
  --s1:#86b6ef; --s2:#2a78d6; --s3:#104281; --sea:#cde2fb; --bed:#d9d2c3; }
@media (prefers-color-scheme: dark) { :root { --t1:#3987e5; --t2:#d95926; --t3:#199e70;
  --chop:#5c5b56; --s1:#cde2fb; --s2:#6da7ec; --s3:#256abf; --sea:#184f95; --bed:#4a463d; } }
.t1 { fill:var(--t1); } .t2 { fill:var(--t2); } .t3 { fill:var(--t3); } .chop { fill:var(--chop); }
.key { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
.k1 { background:var(--t1); } .k2 { background:var(--t2); } .k3 { background:var(--t3); } .kc { background:var(--chop); }
.curve { fill:none; stroke-width:2; }
.sea { fill:var(--sea); } .bed { fill:var(--bed); }
.mark { stroke:var(--ink); stroke-width:1; } svg text.lbl { fill:var(--ink); font-size:9.5px; }
"""


def sample(name, lines=6, width=96):
    """First lines of a saved raw API response, trimmed to fit a phone."""
    text = (SAMPLES / name).read_text().splitlines()
    return escape("\n".join(l[:width] + (" ..." if len(l) > width else "") for l in text[:lines]))


def mop_sample():
    """The parts of a raw CDIP response that matter, in the order they arrive."""
    text = (SAMPLES / "mop_nowcast.ascii").read_text()
    body = text.split("-" * 45, 1)[1].strip().split("\n\n")
    keep = [b for b in body if b.split("[")[0].split(".")[-1] in
            ("waveTime", "waveHs", "waveTp", "waveDp", "waveEnergyDensity", "waveFrequency")][:6]
    return escape("\n\n".join("\n".join(l[:92] + (" ..." if len(l) > 92 else "")
                                       for l in b.splitlines()) for b in keep))


def json_sample(name, keep=3):
    """A saved JSON response with its long arrays cut to `keep` items."""
    def cut(v):
        if isinstance(v, list):
            return [cut(x) for x in v[:keep]] + (["..."] if len(v) > keep else [])
        if isinstance(v, dict):
            return {k: cut(x) for k, x in v.items()
                    if k not in ("generationtime_ms", "utc_offset_seconds",
                                 "timezone_abbreviation", "elevation")}
        return v
    return escape(json.dumps(cut(json.loads((SAMPLES / name).read_text())), indent=1))


def path_diagram():
    """Cross-section from deep water to the beach: where each number is taken."""
    W, H, sea = 340, 150, 62
    bed = lambda x: 140 - 78 * (x / W) ** 2.2          # seabed rises toward shore
    pts, phase, x = [], 0.0, 0.0
    while x <= 300:                                     # waves shorten and grow
        depth = max(bed(x) - sea, 2)
        amp = min(3 + 9 * (x / 300) ** 3, depth * 0.45)
        pts.append(f"{x:.1f},{sea - amp * math.sin(phase):.1f}")
        phase += 0.5 / (0.35 + depth / 78)
        x += 1.5
    surface = " ".join(pts)
    seabed = " ".join(f"{x},{bed(x):.1f}" for x in range(0, W + 1, 10))
    marks = [(40, "Buoys", "deep water"), (205, "CDIP point", "10 to 15 m deep"),
             (288, "Breaks", "surf height")]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Path of a swell from deep water to the beach">',
           f'<polygon class="sea" points="{surface} '
           + " ".join(f"{x},{bed(x):.1f}" for x in range(300, -1, -10)) + '"/>',
           f'<polygon class="bed" points="0,{H} {seabed} {W},{H}"/>']
    for x, a, b in marks:
        out.append(f'<line class="mark" x1="{x}" x2="{x}" y1="30" y2="{sea - 12}"/>'
                   f'<text class="lbl" x="{x}" y="14" text-anchor="middle">{a}</text>'
                   f'<text x="{x}" y="25" text-anchor="middle">{b}</text>')
    out.append(f'<text x="120" y="{sea + 30}" text-anchor="middle">swell travels this way &#8594;</text></svg>')
    return "".join(out)


def spectrum_chart(row, trains):
    """Bars of wave energy per period band, coloured by swell train."""
    f, e = json.loads(row["freq"]), json.loads(row["energy"])
    W, H, L, B, T = 340, 150, 8, 30, 8
    n, top = len(f), max(e) or 1
    bw = (W - 2 * L) / n
    cls = ["chop"] * n
    for i, t in enumerate(trains):
        for j in range(*t["bands"]):
            cls[j] = TRAIN_CLASS[i]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Wave energy by period">',
           f'<line class="grid" x1="{L}" x2="{W - L}" y1="{H - B}" y2="{H - B}"/>']
    for j in range(n):
        h = (H - B - T) * e[j] / top
        out.append(f'<rect class="{cls[j]}" x="{L + j * bw + 1:.1f}" y="{H - B - h:.1f}" '
                   f'width="{bw - 2:.1f}" height="{max(h, 0.5):.1f}" rx="1.5">'
                   f'<title>{1 / f[j]:.0f} s: {e[j]:.2f} m&#178;/Hz</title></rect>')
        if j % 3 == 0:
            out.append(f'<text x="{L + (j + .5) * bw:.1f}" y="{H - B + 11}" text-anchor="middle">{1 / f[j]:.0f}s</text>')
    out.append(f'<text x="{W / 2}" y="{H - 4}" text-anchor="middle">wave period (long swell on the left, short chop on the right)</text></svg>')
    return "".join(out)


def breaking_chart():
    """Breaker height against open-ocean swell height for three periods."""
    W, H, L, R, T, B = 340, 160, 26, 34, 8, 28
    x = lambda ft: L + (W - L - R) * ft / 12
    y = lambda ft: H - B - (H - T - B) * ft / 20
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Breaking height against swell height for three periods">']
    for v in (0, 5, 10, 15, 20):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text x="{L - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v}</text>')
    for v in (0, 3, 6, 9, 12):
        out.append(f'<text x="{x(v):.1f}" y="{H - B + 11}" text-anchor="middle">{v}</text>')
    for period, var in ((8, "s1"), (12, "s2"), (16, "s3")):
        pts = [(h, physics.breaking_height(h / FT, period) * FT) for h in
               [i * 0.25 for i in range(49)]]
        out.append(f'<polyline class="curve" style="stroke:var(--{var})" points="'
                   + " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in pts) + '"/>')
        out.append(f'<text class="lbl" x="{x(12) + 3:.1f}" y="{y(pts[-1][1]) + 3:.1f}">{period} s</text>')
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" text-anchor="middle">open-ocean swell height (ft)</text></svg>')
    return "".join(out)


def worked_example(con, spots, now):
    """Live walk-through of the latest stored data for the first spot that has
    some. Returns HTML, or a short note if nothing is stored yet."""
    for key, s in spots.items():
        m = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast' "
                       "ORDER BY time DESC LIMIT 1", (s["mop"],))
        if not m.empty:
            break
    else:
        return '<p class="note">No wave data stored yet, so there is no live example.</p>'
    row, surf = m.iloc[0], model.surf_height(m.iloc[0], s)
    trains, a = surf["trains"], s.get("a", 1.0)
    hb = sum(t["hb"] ** 2 for t in trains) ** 0.5
    keys = "".join(
        f'<tr><td><span class="key k{i + 1}"></span>Swell {i + 1}</td>'
        f'<td>{t["hs"] * FT:.1f} ft</td><td>{t["tp"]:.0f} s</td>'
        f'<td>{compass(t["dp"])} {t["dp"]:.0f}&deg;</td></tr>' for i, t in enumerate(trains))
    if surf["chop_hs"] * FT >= 0.5:
        keys += (f'<tr><td><span class="key kc"></span>Wind chop</td>'
                 f'<td>{surf["chop_hs"] * FT:.1f} ft</td><td>under {model.MIN_SURF_PERIOD:.0f} s</td>'
                 f'<td>not counted</td></tr>')
    steps = "".join(
        f'<tr><td><span class="key k{i + 1}"></span>Swell {i + 1}</td>'
        f'<td>{t["hs"] * FT:.1f}</td><td>{t["weight"]:.2f}</td>'
        f'<td>{t["h0"] * FT:.1f}</td><td>{t["hb"] * FT:.1f}</td></tr>'
        for i, t in enumerate(trains))
    return f"""
<p>This example is live. It uses the most recent CDIP nowcast for
<b>{escape(s["name"])}</b>, model point <code>{s["mop"]}</code> in {s["mop_depth_m"]:.0f} m of water,
for {local(row["time"]):%a %b %-d, %-I %p} Pacific.</p>
<h3>1. Start with the energy at each wave period</h3>
<p>CDIP reports how much wave energy arrives in each of {len(json.loads(row["freq"]))} period bands.
The total is a {row["hs"] * FT:.1f} ft sea. The code cuts that into separate swells at the dips
between peaks.</p>
{spectrum_chart(row, trains)}
<div class="wrap"><table><tr><th></th><th>Height</th><th>Period</th><th>From</th></tr>{keys}</table></div>
<h3>2. Take each swell to the break</h3>
<div class="wrap"><table><tr><th></th><th>At point</th><th>Window</th>
<th>Deep water</th><th>Breaking</th></tr>{steps}</table></div>
<p class="note">Heights in feet: at the CDIP point, converted to the deep-water equivalent, then
at breaking. Window is 1 when the swell's direction is inside the spot's swell window of
{s["window"][0]} to {s["window"][1]}&deg;, and fades to 0 over 20&deg; outside it.</p>
<h3>3. Combine and scale</h3>
<p>The swells combine by energy (square, add, square root) to <b>{hb * FT:.1f} ft</b>.
The spot's scale factor is {a:.2f}, giving <b>{surf["low"] * FT:.1f} ft</b> for a typical wave.
Sets are 1.27 times that: <b>{surf["high"] * FT:.1f} ft</b>.
The page shows <b>{surf["low"] * FT:.0f}-{surf["high"] * FT:.0f} ft</b>.</p>"""


def spot_table(spots):
    n = calibrate.read_log().spot.value_counts()
    rows = "".join(
        f'<tr><td>{escape(s["name"].split(",")[0])}</td><td><code>{s["mop"]}</code></td>'
        f'<td>{s["window"][0]}-{s["window"][1]}&deg;</td>'
        f'<td>{s.get("a", 1.0):.2f}</td><td>{n.get(k, 0)}</td></tr>' for k, s in spots.items())
    return ('<div class="wrap"><table><tr><th>Spot</th><th>Point</th>'
            f'<th>Window</th><th>Scale</th><th>Logged</th></tr>{rows}</table></div>')


def body(con, spots, now):
    return f"""
<article>
<section><h2>The short version</h2>
<p>Scripps runs a wave model with an output point just off every beach in California. This site
reads the swell at the point nearest each spot, works out how big that swell breaks, and scales
the answer per spot. Wind and tide come from two other free sources. The numbers come from
measurements, published models and a few fixed formulas. No AI is involved in producing them.</p>
{path_diagram()}
<p class="note">Offshore buoys measure the swell in deep water. CDIP's model carries it past
headlands and islands to a point in 10 to 15 m of water. This site does the last step, from that
point to the breaking wave.</p></section>

<section><h2>Where the data comes from</h2>
<h3>Swell: CDIP MOP nearshore model</h3>
<p>From the <a href="https://cdip.ucsd.edu">Coastal Data Information Program</a> at Scripps.
The "nowcast" is driven by what buoys are measuring this hour. The "forecast" runs about six days
ahead in three-hour steps. Each time step gives a wave height, peak period, direction, and the
energy in each period band.</p>
<pre>{mop_sample()}</pre>
<p class="note">Raw response for Ocean Beach's point, three hourly steps. waveTime is seconds since
1970, waveHs is height in metres, waveTp is period in seconds, waveDp is the direction the swell
comes from. Each waveEnergyDensity row is one hour: energy at each period band, long periods first.</p>

<h3>Buoys: NOAA National Data Buoy Center</h3>
<p>Used as a check that the model agrees with measurements, from
<a href="https://www.ndbc.noaa.gov/station_page.php?station=46237">ndbc.noaa.gov</a>.</p>
<pre>{sample("buoy_46237.spec", 5)}</pre>
<p class="note">San Francisco Bar buoy. WVHT is total wave height in metres, SwH and SwP are swell
height and period, WWH and WWP are wind-wave height and period, MWD is the mean direction.</p>

<h3>Tides: NOAA CO-OPS</h3>
<p>High and low tide predictions from
<a href="https://tidesandcurrents.noaa.gov">tidesandcurrents.noaa.gov</a>, San Francisco station
for the northern spots and Monterey for Santa Cruz. Heights between highs and lows are filled in
with a smooth cosine curve.</p>
<pre>{json_sample("tides_9414290.json")}</pre>
<p class="note">t is the time in UTC, v is the height in metres above the average lowest tide,
type is H for high or L for low.</p>

<h3>Wind and long-range swell: Open-Meteo</h3>
<p>Hourly wind from <a href="https://open-meteo.com">open-meteo.com</a>, and its global swell
forecast for the dashed long-range part of each chart.</p>
<pre>{json_sample("wind.json")}</pre>
<p class="note">Wind speed and gusts in metres per second, direction in degrees the wind blows
from. The site converts to knots.</p></section>

<section><h2>How a surf height is calculated</h2>
{worked_example(con, spots, now)}</section>

<section><h2>Why period matters so much</h2>
<p>The breaking height comes from a standard formula (Komar and Gaughan, 1973):</p>
<div class="formula">H<sub>b</sub> = 0.39 &middot; g<sup>1/5</sup> &middot; (T &middot; H<sub>0</sub><sup>2</sup>)<sup>2/5</sup></div>
<p>H<sub>0</sub> is the swell height in deep water, T is its period, g is gravity, and H<sub>b</sub>
is the height when it breaks. A long-period swell carries more energy for the same height, so it
grows more as it slows down in shallow water.</p>
{breaking_chart()}
<p class="note">Breaking height in feet (vertical) against open-ocean swell height. A 6 ft swell
breaks at about {physics.breaking_height(6 / FT, 8) * FT:.0f} ft at 8 seconds and about
{physics.breaking_height(6 / FT, 16) * FT:.0f} ft at 16 seconds.</p>
<p>Before using the formula, the height at the CDIP point is converted back to its deep-water
equivalent, because a swell has already started to change height by the time it reaches 10 to
15 m of water. That step uses linear wave theory.</p></section>

<section><h2>Per-spot settings</h2>
{spot_table(spots)}
<p class="note">Point is the CDIP model point used. Window is the range of swell directions that
reach the spot. Logged is the number of observed sessions recorded so far. The scale factors were first set by matching one week of Surfline's published
heights in November 2026. They are replaced by a fit to logged sessions once a spot has
{calibrate.MIN_OBS} of them.</p></section>

<section><h2>Ratings, and what to distrust</h2>
<p><b>Ratings</b> reflect wind and tide only. Light wind is best, then offshore, cross-shore and
onshore, and stronger is worse. Under 2 ft is "flat". Some spots are marked down at higher tides.
A "good" rating says nothing about whether the size is surfable.</p>
<p><b>Wind chop</b> with a period under {model.MIN_SURF_PERIOD:.0f} seconds is left out of the surf height.</p>
<p><b>The long-range part</b> of each chart comes from a coarse global model, scaled to agree
with CDIP where the two overlap. Treat it as a rough guide.</p>
<p><b>Known weak points:</b> heights tend to read high on ordinary days; cove spots such as Linda
Mar are modelled from a point outside the cove; and very large swells are outside the range the
breaking formula was built for.</p>
<p class="note">Code and data history:
<a href="https://github.com/tarikburl/surf-forecast">github.com/tarikburl/surf-forecast</a></p></section>
</article>"""
