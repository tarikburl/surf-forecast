"""Render the 'How it works' page: the breaking-height equation from first
physical reasoning, today's numbers put through it, and the raw data behind it."""
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
.eq { overflow-x:auto; margin:10px 0; text-align:center; }
.eq math { font-size:17px; }
.eq.box { border:1.5px solid var(--ink); border-radius:8px; padding:10px 6px; }
.calc { font-family:ui-monospace, Menlo, monospace; font-size:12.5px; line-height:1.7; overflow-x:auto; white-space:nowrap; }
article ul, article ol { margin:6px 0; padding-left:20px; } article li { margin:4px 0; }
article sub, article sup { line-height:0; }
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
    b = '<tspan dy="2" font-size="7">b</tspan>'
    marks = [(40, "Deep water", "H\u2080, c\u2080"), (205, "CDIP point", "H at depth h"),
             (288, "Break", f"H{b}")]
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
    out.append(f'<text x="{(L + W - R) / 2}" y="{H - 4}" text-anchor="middle">deep-water swell height (ft)</text></svg>')
    return "".join(out)


# --- equations (MathML: renders natively, no JavaScript) ---------------------

def mi(x): return f"<mi>{x}</mi>"
def mn(x): return f"<mn>{x}</mn>"
def mo(x): return f"<mo>{x}</mo>"
def row(*xs): return "<mrow>" + "".join(xs) + "</mrow>"
def sub(a, b): return f"<msub>{a}{b}</msub>"
def sup(a, b): return f"<msup>{a}{b}</msup>"
def subsup(a, b, c): return f"<msubsup>{a}{b}{c}</msubsup>"
def frac(a, b): return f"<mfrac>{a}{b}</mfrac>"
def sqrt(*xs): return "<msqrt>" + "".join(xs) + "</msqrt>"
def paren(*xs): return row(mo("("), *xs, mo(")"))


def eq(*xs, box=False):
    """One displayed equation."""
    return (f'<div class="eq{" box" if box else ""}"><math display="block">'
            + row(*xs) + "</math></div>")


Hb, H0, H = sub(mi("H"), mi("b")), sub(mi("H"), mn(0)), mi("H")
c0 = sub(mi("c"), mn(0))
TH02 = paren(mi("T"), subsup(mi("H"), mn(0), mn(2)))
EQ_RESULT = eq(Hb, mo("="), mn("0.39"), sup(mi("g"), frac(mn(1), mn(5))),
               sup(TH02, frac(mn(2), mn(5))), box=True)
EQ_DESHOAL = eq(H0, mo("="), H, sqrt(frac(row(mi("c"), paren(mi("h"))), c0)), mo(","),
                c0, mo("="), frac(row(mi("g"), mi("T")), row(mn(4), mi("&pi;"))))
EQ_COMBINE = eq(Hb, mo("="), sqrt(subsup(mi("H"), row(mi("b"), mo(","), mn(1)), mn(2)), mo("+"),
                                  subsup(mi("H"), row(mi("b"), mo(","), mn(2)), mn(2)),
                                  mo("+"), mo("&hellip;")))


def worked_example(con, spots, now):
    """Today's numbers put through the equations, for the first spot with data."""
    for key, s in spots.items():
        m = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast' "
                       "ORDER BY time DESC LIMIT 1", (s["mop"],))
        if not m.empty:
            break
    else:
        return '<p class="note">No wave data stored yet, so there is no live example.</p>'
    r, surf = m.iloc[0], model.surf_height(m.iloc[0], s)
    trains, a, h = surf["trains"], s.get("a", 1.0), s["mop_depth_m"]
    total = sum(t["hb"] ** 2 for t in trains) ** 0.5
    rows = "".join(
        f'<tr><td><span class="key k{i + 1}"></span>{i + 1}</td><td>{t["hs"]:.2f} m</td>'
        f'<td>{t["tp"]:.1f} s</td><td>{compass(t["dp"])} {t["dp"]:.0f}&deg;</td>'
        f'<td>{t["h0"]:.2f} m</td><td>{t["hb"]:.2f} m</td></tr>' for i, t in enumerate(trains))
    t = trains[0]
    c_deep, c_here = physics.G * t["tp"] / (4 * math.pi), physics.group_velocity(t["tp"], h)
    window = "" if t["weight"] == 1 else f' &times; {t["weight"]:.2f} (window)'
    combine = "" if len(trains) == 1 else (
        "<p>Combine the swells: &radic;(" + " + ".join(f'{x["hb"]:.2f}&sup2;' for x in trains)
        + f") = {total:.2f} m.</p>")
    chop = "" if surf["chop_hs"] * FT < 0.5 else (
        f' Grey bars are chop under {model.MIN_SURF_PERIOD:.0f} s '
        f'({surf["chop_hs"]:.1f} m), left out.')
    return f"""
<p>{escape(s["name"])}, {local(r["time"]):%a %b %-d, %-I %p} Pacific. CDIP point
<code>{s["mop"]}</code>, depth h = {h:.0f} m. These numbers refresh with the page.</p>
{spectrum_chart(r, trains)}
<p class="note">Energy at each wave period, from CDIP. Each colour is one swell.{chop}</p>
<div class="wrap"><table><tr><th>Swell</th><th>H</th><th>T</th><th>From</th>
<th>H<sub>0</sub></th><th>H<sub>b</sub></th></tr>{rows}</table></div>
<h3>Swell 1, step by step</h3>
<p class="calc">c<sub>0</sub> = gT/4&pi; = 9.81 &times; {t["tp"]:.1f} / 12.57 = {c_deep:.1f} m/s<br>
c(h) = {c_here:.1f} m/s<br>
H<sub>0</sub> = {t["hs"]:.2f}{window} &times; &radic;({c_here:.1f}/{c_deep:.1f}) = {t["h0"]:.2f} m<br>
H<sub>b</sub> = 0.39 &times; 9.81<sup>1/5</sup> &times; ({t["tp"]:.1f} &times; {t["h0"]:.2f}<sup>2</sup>)<sup>2/5</sup> = {t["hb"]:.2f} m</p>
{combine}
<p>Scale by a = {a:.2f}: {surf["low"]:.2f} m = <b>{surf["low"] * FT:.1f} ft</b>, the typical wave.
Sets: 1.27 &times; that = <b>{surf["high"] * FT:.1f} ft</b>.
The forecast tab shows <b>{surf["low"] * FT:.0f}-{surf["high"] * FT:.0f} ft</b>.</p>"""


def spot_table(spots):
    n = calibrate.read_log().spot.value_counts()
    rows = "".join(
        f'<tr><td>{escape(s["name"].split(",")[0])}</td><td><code>{s["mop"]}</code></td>'
        f'<td>{s["window"][0]}-{s["window"][1]}&deg;</td>'
        f'<td>{s.get("a", 1.0):.2f}</td><td>{n.get(k, 0)}</td></tr>' for k, s in spots.items())
    return ('<div class="wrap"><table><tr><th>Spot</th><th>Point</th><th>Window</th>'
            f'<th>a</th><th>Logged</th></tr>{rows}</table></div>')


def body(con, spots, now):
    grow = (2 ** 0.4 - 1) * 100
    b8, b16 = (physics.breaking_height(6 / FT, T) * FT for T in (8, 16))
    return f"""
<article>
<section><h2>The idea</h2>
<p>A swell carries energy toward the beach. In shallow water it slows down, so the same energy
piles into a taller wave. It breaks when it's about as tall as the water is deep.</p>
{path_diagram()}
<p class="note">Buoys measure the swell in deep water. Scripps' CDIP model carries it to a point
10 to 15 m deep off each spot. This site does the last step.</p></section>

<section><h2>The equation</h2>
<p>Two facts set the size. The swell's energy flux stays roughly constant on the way in. And a
wave breaks once its height nears the water depth. Together they give the breaking height
H<sub>b</sub> from the deep-water height H<sub>0</sub> and the period T:</p>
{EQ_RESULT}
<p class="note">Komar and Gaughan (1973). g = 9.81 m/s&sup2;. The 0.39 was fitted to measured
breakers. Heights in metres, T in seconds.</p>
<p>Period matters. Double it and the wave breaks {grow:.0f}% bigger.</p>
{breaking_chart()}
<p class="note">Breaking height against deep-water swell height, both in feet. A 6 ft swell
breaks at {b8:.0f} ft at 8 s and {b16:.0f} ft at 16 s.</p></section>

<section><h2>The recipe</h2>
<ol>
<li><b>Split.</b> CDIP gives wave energy by period. Cut it into separate swells at the dips
between peaks. Drop chop under {model.MIN_SURF_PERIOD:.0f} s. Fade out any swell from outside the
spot's direction window.</li>
<li><b>Convert to deep water.</b> CDIP's height H is at depth h, where the swell has already
changed height. Undo that with the same flux rule. Here c is the speed the swell's energy
travels.
{EQ_DESHOAL}</li>
<li><b>Break.</b> Put H<sub>0</sub> and T into the boxed equation for each swell.</li>
<li><b>Combine.</b> Energies add, so heights add in quadrature.
{EQ_COMBINE}</li>
<li><b>Scale.</b> Each spot has a fitted factor a. The typical wave is a&middot;H<sub>b</sub>.
Sets are 1.27 times that, the average of the biggest tenth of waves in a random sea.</li>
</ol></section>

<section><h2>Today's numbers</h2>
{worked_example(con, spots, now)}</section>

<section><h2>The data</h2>
<h3>Swell: CDIP nearshore model</h3>
<p><a href="https://cdip.ucsd.edu">cdip.ucsd.edu</a>. Hourly "nowcast" driven by buoys, plus a
six-day forecast in three-hour steps.</p>
<pre>{mop_sample()}</pre>
<p class="note">Three hourly steps for Ocean Beach. waveTime: seconds since 1970. waveHs:
height, m. waveTp: period, s. waveDp: direction the swell comes from. Each energy row is one
hour, long periods first.</p>
<h3>Buoys: NOAA NDBC</h3>
<p><a href="https://www.ndbc.noaa.gov/station_page.php?station=46237">ndbc.noaa.gov</a>.
A check on the model.</p>
<pre>{sample("buoy_46237.spec", 5)}</pre>
<p class="note">San Francisco Bar. WVHT: total height, m. SwH, SwP: swell height and period.
WWH, WWP: wind-wave height and period. MWD: mean direction.</p>
<h3>Tides: NOAA CO-OPS</h3>
<p><a href="https://tidesandcurrents.noaa.gov">tidesandcurrents.noaa.gov</a>. Highs and lows;
a cosine curve fills in between.</p>
<pre>{json_sample("tides_9414290.json")}</pre>
<p class="note">t: UTC time. v: metres above mean lower low water. type: H or L.</p>
<h3>Wind and long-range swell: Open-Meteo</h3>
<p><a href="https://open-meteo.com">open-meteo.com</a>. Hourly wind, and the swell behind the
dashed part of each chart.</p>
<pre>{json_sample("wind.json")}</pre>
<p class="note">Speed and gusts in m/s. Direction is where the wind blows from.</p></section>

<section><h2>Per-spot settings</h2>
{spot_table(spots)}
<p class="note">Point: the CDIP model point. Window: swell directions that reach the spot.
a: the scale factor. Logged: sessions recorded. The a values started as a match to one week of
Surfline's heights. A fit to logged sessions replaces each one at {calibrate.MIN_OBS} sessions.</p></section>

<section><h2>Limits</h2>
<ul>
<li>Heights read high on ordinary days and low on the biggest.</li>
<li>Linda Mar's CDIP point sits outside the cove.</li>
<li>A very large swell is already breaking at the CDIP point. The formula isn't built for that.</li>
<li>Ratings use wind and tide only. A clean 20 ft day still says "good".</li>
<li>Past six days the forecast comes from a coarse global model. Rough guide only.</li>
</ul>
<p class="note">Code: <a href="https://github.com/tarikburl/surf-forecast">github.com/tarikburl/surf-forecast</a></p></section>
</article>"""
