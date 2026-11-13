"""SQLite storage. One table per source; writes are idempotent upserts."""
import sqlite3
from pathlib import Path

import pandas as pd

DB = Path(__file__).resolve().parent.parent / "data" / "surf.db"

# table -> (columns, primary key)
TABLES = {
    "mop": ("mop TEXT, kind TEXT, time TEXT, hs REAL, tp REAL, dp REAL, "
            "freq TEXT, bandwidth TEXT, energy TEXT, direction TEXT",
            "mop, kind, time"),
    "buoy": ("station TEXT, time TEXT, hs REAL, swell_hs REAL, swell_tp REAL, "
             "mwd REAL", "station, time"),
    "tide": ("station TEXT, time TEXT, height REAL, type TEXT", "station, time"),
    "wind": ("spot TEXT, time TEXT, speed REAL, direction REAL, gust REAL",
             "spot, time"),
    "marine": ("spot TEXT, time TEXT, hs REAL, tp REAL, dp REAL", "spot, time"),
    # one stored forecast per spot per day (run = UTC date it was made)
    "forecast": ("spot TEXT, run TEXT, time TEXT, low REAL, high REAL, source TEXT",
                 "spot, run, time"),
}


def connect(path=DB):
    con = sqlite3.connect(path)
    for name, (cols, key) in TABLES.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols}, PRIMARY KEY ({key}))")
    return con


def upsert(con, table, df, **keys):
    """Insert df into table, replacing rows with the same primary key.
    keys are constant columns added to every row, e.g. station='46237'."""
    df = df.assign(**keys)
    cols = ", ".join(df.columns)
    marks = ", ".join("?" * len(df.columns))
    with con:
        con.executemany(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})",
                        df.astype(object).where(df.notna(), None).values.tolist())


def read(con, sql, params=()):
    return pd.read_sql_query(sql, con, params=params)
