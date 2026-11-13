"""Command line: surf fetch | now | forecast <spot> | page | log | calibrate"""
import argparse

import pandas as pd

from surf import calibrate, fetch, forecast, model, spots as spots_mod, store
from surf.forecast import FT, KT, PACIFIC, STALE_HOURS, compass, local, wind_kind
from surf.spots import load_spots

NOWCAST_HOURS = 72


def cmd_fetch(args):
    spots = load_spots()
    con = store.connect()
    failed = []

    def run(label, table, fn, *fn_args, **keys):
        """Fetch one source and store it. A failure is recorded, never raised."""
        try:
            store.upsert(con, table, fn(*fn_args), **keys)
        except Exception as e:
            failed.append(label)
            print(f"  FAILED {label}: {type(e).__name__}: {e}")

    for s in spots.values():
        run(f"mop {s['mop']} nowcast", "mop", fetch.fetch_mop, s["mop"], "nowcast",
            NOWCAST_HOURS, mop=s["mop"], kind="nowcast")
        run(f"mop {s['mop']} forecast", "mop", fetch.fetch_mop, s["mop"], "forecast",
            mop=s["mop"], kind="forecast")
    for key, s in spots.items():
        run(f"wind {key}", "wind", fetch.fetch_wind, s["lat"], s["lon"], spot=key)
        run(f"marine {key}", "marine", fetch.fetch_marine, s["lat"], s["lon"], spot=key)
    for b in sorted({b for s in spots.values() for b in s["buoys"]}):
        run(f"buoy {b}", "buoy", fetch.fetch_buoy, b, station=b)
    for t in sorted({s["tide_station"] for s in spots.values()}):
        run(f"tide {t}", "tide", fetch.fetch_tides, t, station=t)

    now = pd.Timestamp.now("UTC")
    for key, s in spots.items():
        forecast.save_run(con, key, forecast.build(con, key, s, now.strftime(fetch.TIME_FMT)),
                          run=now.strftime("%Y-%m-%d"))
    print_summary(con, spots)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in store.TABLES}
    print("\nrows stored: " + ", ".join(f"{t} {n}" for t, n in counts.items()))
    if failed:
        print(f"{len(failed)} source(s) failed; summary uses last stored data: "
              + ", ".join(failed))
    return 1 if failed else 0


def print_summary(con, spots):
    """One line per spot from what is in the database (so it shows stale data
    with its age if a fetch failed)."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    age = lambda t: f"{(now - pd.Timestamp(t)).total_seconds() / 3600:.0f}h ago"
    print(f"\n{'spot':27s} {'MOP nowcast (Hs Tp Dp)':30s} {'buoy Hs':24s} next tide (UTC)")
    for s in spots.values():
        m = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast' "
                       "ORDER BY time DESC LIMIT 1", (s["mop"],))
        b = store.read(con, "SELECT * FROM buoy WHERE station=? AND hs IS NOT NULL "
                       "ORDER BY time DESC LIMIT 1", (s["buoys"][0],))
        t = store.read(con, "SELECT * FROM tide WHERE station=? AND time>? "
                       "ORDER BY time LIMIT 1",
                       (s["tide_station"], now.strftime(fetch.TIME_FMT)))
        mop = "no data" if m.empty else (
            f"{m.hs[0]:.1f}m {m.tp[0]:.0f}s {m.dp[0]:.0f}° ({age(m.time[0])})")
        buoy = "no data" if b.empty else (
            f"{b.station[0]} {b.hs[0]:.1f}m ({age(b.time[0])})")
        tide = "no data" if t.empty else (
            f"{'high' if t.type[0] == 'H' else 'low'} {t.height[0]:.1f}m at {t.time[0][5:]}")
        print(f"{s['name']:27s} {mop:30s} {buoy:24s} {tide}")


def cmd_now(args):
    """Current surf at every spot, from the latest stored data."""
    spots = load_spots()
    con = store.connect()
    now = pd.Timestamp.now("UTC").tz_localize(None)
    now_s = now.strftime(fetch.TIME_FMT)
    print(f"Surf now, {local(now_s):%a %b %-d %-I:%M %p} Pacific\n")
    for key, s in spots.items():
        c = forecast.current(con, key, s, now)
        if c is None:
            print(f"{s['name']}\n  no wave data stored; run `surf fetch`\n")
            continue
        surf, w, t = c["surf"], c["wind"], c["tides"]
        stale = (f"   [STALE: wave data is {c['age_hours']:.0f}h old]"
                 if c["age_hours"] > STALE_HOURS else "")
        print(f"{s['name']:28s}{surf['low'] * FT:.0f}-{surf['high'] * FT:.0f} ft{stale}")
        print("  swell: " + ",  ".join(
            f"{t_['hs'] * FT:.1f} ft @ {t_['tp']:.0f}s {compass(t_['dp'])} {t_['dp']:.0f}°"
            for t_ in surf["trains"])
            + (f",  plus {surf['chop_hs'] * FT:.1f} ft of wind chop (not counted)"
               if surf["chop_hs"] * FT >= 1 else ""))
        if w is not None:
            print(f"  wind:  {w.speed * KT:.0f} kt {compass(w.direction)}, "
                  f"gusts {w.gust * KT:.0f} kt "
                  f"({wind_kind(w.direction, s['shore_normal'])})")
        if not t.empty:
            print("  tide:  " + ("rising" if t.type[0] == "H" else "falling") + ", " +
                  ", then ".join(
                      f"{'high' if r.type == 'H' else 'low'} {r.height * FT:.1f} ft "
                      f"at {local(r.time):%-I:%M %p}" for r in t.itertuples()))
        print()
    return 0


def cmd_page(args):
    """Write the phone page from stored data."""
    from surf import page
    out = store.DB.parent.parent / "site" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page.render(store.connect(), load_spots(),
                               pd.Timestamp.now("UTC").tz_localize(None)))
    print(f"wrote {out}")
    return 0


def find_spot(spots, text):
    """Match 'ocean', 'Linda Mar', 'steamer_lane'... to exactly one spot key."""
    want = text.lower().replace("_", " ")
    hits = [k for k, s in spots.items()
            if want in k.replace("_", " ") or want in s["name"].lower()]
    if len(hits) != 1:
        raise SystemExit(f"'{text}' matches {hits or 'no spot'}; choose from: "
                         + ", ".join(spots))
    return hits[0]


def cmd_forecast(args):
    """Forecast table and chart for one spot, from stored data."""
    spots = load_spots()
    key = find_spot(spots, args.spot)
    s = spots[key]
    con = store.connect()
    now = pd.Timestamp.now("UTC").tz_localize(None)
    df = forecast.build(con, key, s, (now - pd.Timedelta(hours=3)).strftime(fetch.TIME_FMT))
    if df.empty:
        print("no forecast data stored; run `surf fetch`")
        return 1
    df["local"] = [local(t) for t in df.time]
    rng = lambda r: f"{r.low * FT:.0f}-{r.high * FT:.0f} ft"

    print(f"{s['name']} forecast (Pacific time)")
    m = store.read(con, "SELECT * FROM mop WHERE mop=? AND kind='nowcast' "
                   "ORDER BY time DESC LIMIT 1", (s["mop"],))
    if not m.empty:
        cur = model.surf_height(m.iloc[0], s)
        near = df.iloc[(pd.to_datetime(df.time) - pd.Timestamp(m.time[0])).abs().argmin()]
        print(f"now: {cur['low'] * FT:.0f}-{cur['high'] * FT:.0f} ft from live buoy data; "
              f"the forecast says {rng(near)} for the same hour")

    day = None
    for r in df[df.source == "mop"].itertuples():
        if not 6 <= r.local.hour <= 18:
            continue
        if r.local.date() != day:
            day = r.local.date()
            print(f"\n{r.local:%a %b %-d}")
        wind = "" if pd.isna(r.wind_speed) else (
            f"{r.wind_speed * KT:2.0f} kt {compass(r.wind_from):3s} "
            f"{wind_kind(r.wind_from, s['shore_normal'])}")
        tide = "" if r.tide is None or pd.isna(r.tide) else f"tide {r.tide * FT:4.1f} ft"
        print(f"  {r.local:%-I %p}".ljust(8) + f"{rng(r):9s} "
              f"{r.hs * FT:4.1f} ft @ {r.tp:2.0f}s {compass(r.dp):3s}  "
              f"{wind:22s} {tide:13s} {r.rating}")

    tail = df[df.source == "tail"]
    if not tail.empty:
        print("\nLong range (low confidence, global model scaled to match the above)")
        for d, g in tail[tail.local.map(lambda t: 6 <= t.hour <= 18)].groupby(
                tail.local.map(lambda t: t.date())):
            print(f"  {d:%a %b %-d}  {g.low.mean() * FT:.0f}-{g.high.mean() * FT:.0f} ft")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    x = [t.tz_localize(None) for t in df.local]
    for source, color, label in (("mop", "C0", "CDIP nearshore model"),
                                 ("tail", "0.6", "long range (low confidence)")):
        on = (df.source == source).values
        ax.fill_between(x, df.low * FT, df.high * FT, where=on, color=color,
                        alpha=0.6, label=label)
    ax.set_ylabel("surf face height (ft)")
    ax.set_ylim(0)
    ax.set_title(f"{s['name']}: typical waves (bottom) to sets (top), Pacific time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %-d'))
    path = store.DB.parent / f"forecast_{key}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"\nchart: {path}")
    return 0


def cmd_log(args):
    """Record a session: surf log linda 3 4 [--time "2026-11-13 08:30"]"""
    spots = load_spots()
    key = find_spot(spots, args.spot)
    when = (pd.Timestamp(args.time, tz=PACIFIC) if args.time
            else pd.Timestamp.now(PACIFIC)).tz_convert("UTC")
    if when > pd.Timestamp.now("UTC"):
        raise SystemExit("that time is in the future")
    calibrate.append_log(key, args.low, args.high, when.strftime(fetch.TIME_FMT),
                         args.note, args.log)
    n = int((calibrate.read_log(args.log).spot == key).sum())
    print(f"logged {spots[key]['name']}: {args.low:g}-{args.high:g} ft at "
          f"{when.tz_convert(PACIFIC):%a %b %-d %-I:%M %p} Pacific "
          f"({n} session{'s' * (n != 1)} at this spot; calibration starts at "
          f"{calibrate.MIN_OBS})")
    return 0


def cmd_calibrate(args):
    """Compare logged sessions with the model and refit each spot's a, b."""
    spots = load_spots()
    log = calibrate.read_log(args.log)
    con = store.connect()
    text = spots_mod.SPOTS_FILE.read_text()
    changed = []
    print(f"{len(log)} logged sessions\n")
    for key, s in spots.items():
        mine = log[log.spot == key]
        a_now, b_now = s.get("a", 1.0), s.get("b", 1.0)
        if len(mine) < calibrate.MIN_OBS:
            print(f"{s['name']}: {len(mine)} sessions, need {calibrate.MIN_OBS}. "
                  f"Keeping a={a_now:.2f}, b={b_now:.2f}.")
            continue
        hb = calibrate.raw_heights(con, s, mine.time)
        ok = ~pd.isna(hb) & (hb > 0)
        obs = ((mine.low_ft + mine.high_ft) / 2).values[ok]
        if ok.sum() < calibrate.MIN_OBS:
            print(f"{s['name']}: only {ok.sum()} sessions have model data. Keeping current.")
            continue
        r = calibrate.evaluate(hb[ok], obs, a_now, b_now)
        better = r["fitted"]["mae"] <= r["current"]["mae"]
        print(f"{s['name']}: {r['n']} sessions; scored on the most recent "
              f"{r['n'] - int(r['n'] * 0.7)}, which the new fit never saw")
        for label, m, a, b in (("current", r["current"], a_now, b_now),
                               ("new fit", r["fitted"], r["a"], r["b"])):
            print(f"  {label}  a={a:.2f} b={b:.2f}   error {m['mae']:.1f} ft, "
                  f"bias {m['bias']:+.1f} ft, {m['within_1ft']:.0f}% within 1 ft")
        if better:
            text = calibrate.set_spot_values(text, key, a=r["a"], b=r["b"])
            changed.append(key)
        else:
            print("  new fit is not better on unseen sessions; keeping current")
    if changed and args.write:
        spots_mod.SPOTS_FILE.write_text(text)
        print(f"\nupdated spots.toml for: {', '.join(changed)}")
    elif changed:
        print(f"\nwould update: {', '.join(changed)}. Run again with --write to save.")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="surf")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="fetch all sources and store them").set_defaults(
        func=cmd_fetch)
    sub.add_parser("now", help="current surf at every spot").set_defaults(func=cmd_now)
    sub.add_parser("page", help="write site/index.html").set_defaults(func=cmd_page)
    p = sub.add_parser("log", help="record a session you saw or surfed")
    p.add_argument("spot")
    p.add_argument("low", type=float, help="typical face height, ft")
    p.add_argument("high", type=float, help="set face height, ft")
    p.add_argument("--time", help='Pacific time, "YYYY-MM-DD HH:MM" (default: now)')
    p.add_argument("--note", default="")
    p.add_argument("--log", default=calibrate.LOG, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_log)
    p = sub.add_parser("calibrate", help="refit height scaling from logged sessions")
    p.add_argument("--write", action="store_true", help="save better fits to spots.toml")
    p.add_argument("--log", default=calibrate.LOG, help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("forecast", help="forecast table and chart for one spot")
    p.add_argument("spot", help="spot name or part of it, e.g. 'ocean' or 'linda'")
    p.set_defaults(func=cmd_forecast)
    args = parser.parse_args()
    raise SystemExit(args.func(args))
