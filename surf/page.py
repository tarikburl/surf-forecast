"""Render the site: self-contained HTML pages, no JavaScript."""
from html import escape

import pandas as pd

from surf import fetch, forecast
from surf.forecast import FT, KT, STALE_HOURS, compass, local, wind_kind

CSS = """
:root { --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --line:#e3e2dd; --series:#2a78d6; --good:#0ca30c; --fair:#b97a00; --poor:#d03b3b; }
@media (prefers-color-scheme: dark) { :root { --page:#0d0d0d; --card:#1a1a19;
  --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; --series:#3987e5; --fair:#fab219; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--page); color:var(--ink);
  font:15px/1.4 -apple-system, system-ui, sans-serif; }
header { max-width:1100px; margin:0 auto 12px; }
h1 { font-size:20px; margin:0; } h2 { font-size:16px; margin:0; }
nav { display:flex; gap:4px; margin:8px 0; border-bottom:1px solid var(--line); }
nav a { padding:6px 12px; color:var(--ink2); text-decoration:none; border-bottom:2px solid transparent; margin-bottom:-1px; }
nav a.on { color:var(--ink); border-bottom-color:var(--series); font-weight:600; }
.sub, .meta, th, .lowconf { color:var(--ink2); font-size:13px; }
main { display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
  max-width:1100px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.now { font-size:34px; font-weight:650; margin:2px 0 4px; font-variant-numeric:tabular-nums; }
.now small { font-size:15px; font-weight:400; color:var(--ink2); }
.badge { display:inline-block; font-size:12px; padding:1px 7px; border-radius:9px;
  border:1px solid var(--poor); color:var(--ink); margin-left:6px; vertical-align:middle; }
svg { width:100%; height:auto; display:block; margin:10px 0 6px; }
svg text { fill:var(--ink2); font-size:9px; }
.grid { stroke:var(--line); stroke-width:1; }
.band { fill:var(--series); }
.tail { fill:var(--series); fill-opacity:.3; stroke:var(--series); stroke-width:1; stroke-dasharray:3 3; }
.nowline { stroke:var(--ink); stroke-width:1; stroke-dasharray:2 2; }
.hit { fill:transparent; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:13.5px; white-space:nowrap; }
th { text-align:left; font-weight:400; } td, th { padding:3px 6px 3px 0; border-top:1px solid var(--line); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:4px; }
.good .dot { background:var(--good); } .fair .dot { background:var(--fair); }
.poor .dot { background:var(--poor); } .flat .dot { background:var(--ink2); }
"""


def ft_range(low, high):
    return f"{low * FT:.0f}-{high * FT:.0f}"


def chart(df, now):
    """Inline SVG of the forecast band: typical waves (bottom) to sets (top),
    in feet. The low-confidence tail is drawn lighter with a dashed outline."""
    W, H, L, R, T, B = 340, 150, 24, 6, 8, 18
    t = pd.to_datetime(df.time)
    t0, t1 = t.iloc[0], t.iloc[-1]
    top = df.high.max() * FT
    step = next(s for s in (1, 2, 5, 10, 20) if top / s <= 4)
    ymax = step * (int(top / step) + 1)
    x = lambda when: L + (W - L - R) * (when - t0) / (t1 - t0)
    y = lambda ft: H - B - (H - T - B) * ft / ymax
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Surf height forecast in feet">']
    for v in range(0, ymax + 1, step):
        out.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text x="{L - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v}</text>')
    day = local(str(t0)).normalize() + pd.Timedelta(days=1)
    while day.tz_convert("UTC").tz_localize(None) < t1:
        dx = x(day.tz_convert("UTC").tz_localize(None))
        out.append(f'<line class="grid" x1="{dx:.1f}" x2="{dx:.1f}" y1="{T}" y2="{H - B}"/>'
                   f'<text x="{dx + 2:.1f}" y="{H - 5}">{day:%a}</text>')
        day += pd.Timedelta(days=1)
    for source, cls in (("mop", "band"), ("tail", "tail")):
        on = df.source == source
        if source == "tail" and on.any():
            on = on | (on.shift(-1, fill_value=False))  # join the tail to the band
        d, dt = df[on], t[on]
        if len(d) > 1:
            pts = [f"{x(a):.1f},{y(v * FT):.1f}" for a, v in zip(dt, d.high)]
            pts += [f"{x(a):.1f},{y(v * FT):.1f}" for a, v in zip(dt[::-1], d.low[::-1])]
            out.append(f'<polygon class="{cls}" points="{" ".join(pts)}"/>')
    if t0 <= now <= t1:
        out.append(f'<line class="nowline" x1="{x(now):.1f}" x2="{x(now):.1f}" y1="{T}" y2="{H - B}"/>')
    half = (W - L - R) / max(len(df) - 1, 1) / 2
    for when, r in zip(t, df.itertuples()):  # hover readout on desktop
        out.append(f'<rect class="hit" x="{x(when) - half:.1f}" y="{T}" width="{2 * half:.1f}" '
                   f'height="{H - T - B}"><title>{local(r.time):%a %-I %p}: '
                   f'{ft_range(r.low, r.high)} ft</title></rect>')
    out.append("</svg>")
    return "".join(out)


def rating_cell(words):
    """'good' / 'fair' / 'poor' / 'flat' with a status dot; never color alone."""
    return " ".join(f'<span class="{w}"><span class="dot"></span>{w}</span>'
                    for w in words if w) or "&ndash;"


def daily_rows(df):
    """One table row per day from the daylight hours (6 AM to 6 PM Pacific)."""
    d = df.assign(local=[local(t) for t in df.time])
    d = d[d.local.map(lambda t: 6 <= t.hour <= 18)]
    rows = []
    for date, g in d.groupby(d.local.map(lambda t: t.date()), sort=True):
        mid = g.iloc[len(g) // 2]
        if (g.source == "tail").all():
            rows.append(f'<tr><td>{date:%a %-d}</td><td>{ft_range(g.low.mean(), g.high.mean())} ft</td>'
                        f'<td class="lowconf" colspan="2">long range, low confidence</td></tr>')
            continue
        am = g[g.local.map(lambda t: t.hour < 12)].rating.tolist()[:1]
        pm = g[g.local.map(lambda t: t.hour >= 12)].rating.tolist()[:1]
        rows.append(f'<tr><td>{date:%a %-d}</td><td>{ft_range(g.low.min(), g.high.max())} ft</td>'
                    f'<td>{mid.hs * FT:.0f} ft @ {mid.tp:.0f}s {compass(mid.dp)}</td>'
                    f'<td>{rating_cell(am + pm)}</td></tr>')
    return "".join(rows)


def card(con, key, spot, now):
    c = forecast.current(con, key, spot, now)
    df = forecast.build(con, key, spot, (now - pd.Timedelta(hours=3)).strftime(fetch.TIME_FMT))
    out = [f'<section class="card"><h2>{escape(spot["name"])}']
    if c is None:
        return "".join(out) + '</h2><p class="meta">No wave data yet.</p></section>'
    if c["age_hours"] > STALE_HOURS:
        out.append(f'<span class="badge">data {c["age_hours"]:.0f}h old</span>')
    surf, w, tides = c["surf"], c["wind"], c["tides"]
    out.append(f'</h2><div class="now">{ft_range(surf["low"], surf["high"])} <small>ft now</small></div>')
    lines = ["Swell " + ", ".join(f'{t["hs"] * FT:.1f} ft @ {t["tp"]:.0f}s {compass(t["dp"])}'
                                  for t in surf["trains"])
             + (f' (plus {surf["chop_hs"] * FT:.0f} ft wind chop)' if surf["chop_hs"] * FT >= 1 else "")]
    if w is not None:
        lines.append(f'Wind {w.speed * KT:.0f} kt {compass(w.direction)}, '
                     f'{wind_kind(w.direction, spot["shore_normal"])}')
    if not tides.empty:
        lines.append("Tide " + ("rising" if tides.type[0] == "H" else "falling") + ", "
                     + ", then ".join(f'{"high" if r.type == "H" else "low"} {r.height * FT:.1f} ft '
                                      f'at {local(r.time):%-I:%M %p}' for r in tides.itertuples()))
    out.append('<div class="meta">' + "<br>".join(escape(x) for x in lines) + "</div>")
    if len(df) > 1:
        out.append(chart(df, now))
        out.append('<table><tr><th>Day</th><th>Surf</th><th>Swell</th><th>AM / PM</th></tr>'
                   + daily_rows(df) + "</table>")
    return "".join(out) + "</section>"


def shell(title, active, sub, body, extra_css=""):
    """Wrap a page body in the shared head, header and tab links."""
    tabs = "".join(
        f'<a href="{href}"' + (' class="on" aria-current="page"' if href == active else "")
        + f'>{label}</a>' for href, label in (("index.html", "Forecast"),
                                              ("how.html", "How it works")))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{CSS}{extra_css}</style></head><body>'
            f'<header><h1>Surf</h1><nav>{tabs}</nav><div class="sub">{sub}</div></header>'
            f'{body}</body></html>')


def render(con, spots, now):
    """The forecast page as an HTML string. `now` is a naive UTC Timestamp."""
    stamp = local(now.strftime(fetch.TIME_FMT))
    cards = "".join(card(con, key, s, now) for key, s in spots.items())
    return shell("Surf", "index.html",
                 f"Updated {stamp:%a %b %-d, %-I:%M %p} Pacific. Face height, typical waves "
                 f"to sets. Charts: solid is the CDIP nearshore model, dashed is long range.",
                 f"<main>{cards}</main>")


def render_how(con, spots, now):
    """The 'How it works' page as an HTML string."""
    from surf import how
    return shell("Surf: how it works", "how.html",
                 "The equation, today's numbers, and the raw data.",
                 how.body(con, spots, now), how.CSS)
