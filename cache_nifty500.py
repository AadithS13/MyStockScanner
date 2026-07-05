"""Build the committed Nifty-500 price cache the cloud app reads at startup.

Why this exists: the Streamlit Cloud app used to download ~60 bhavcopy CSVs
from NSE on every cold start. NSE archives are slow (and sometimes hostile)
from non-Indian cloud IPs, so first load could hang for minutes. Instead, the
nightly GitHub Actions job runs this script and commits:

    data/n500_cache.parquet
    columns: date, symbol, open, high, low, close, volume  (EQ series only)

The app (data.py) reads the parquet instantly and only falls back to live
NSE fetching when the cache is missing or stale.

Streamlit-free so CI can run it. Re-runnable / idempotent: always writes a
fresh full window.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "n500_cache.parquet")
N_DAYS = 70  # enough for 50DMA + buffer

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}
BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"


def _fetch_day(date: datetime) -> pd.DataFrame | None:
    url = BHAVCOPY_URL.format(date=date.strftime("%d%m%Y"))
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
        eq = df[df["SERIES"].str.strip() == "EQ"].copy()
        cols = {"OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low",
                "CLOSE_PRICE": "close", "TTL_TRD_QNTY": "volume"}
        for c in cols:
            eq[c] = pd.to_numeric(eq[c], errors="coerce")
        out = eq[["SYMBOL", *cols]].rename(columns={"SYMBOL": "symbol", **cols})
        out.insert(0, "date", pd.Timestamp(date.date()))
        return out.dropna(subset=["close"])
    except Exception as e:  # noqa: BLE001
        print(f"  {date.date()}: fetch failed ({e})")
        return None


def build_cache(n_days: int = N_DAYS) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    frames, candidate, attempts = [], datetime.today() - timedelta(days=1), 0
    got = 0
    while got < n_days and attempts < n_days + 45:
        if candidate.weekday() < 5:
            df = _fetch_day(candidate)
            if df is not None:
                frames.append(df)
                got += 1
                if got % 10 == 0:
                    print(f"  {got}/{n_days} days…")
                time.sleep(0.25)
        candidate -= timedelta(days=1)
        attempts += 1

    if not frames:
        raise RuntimeError("No bhavcopy data fetched — cache not written")

    out = (pd.concat(frames, ignore_index=True)
           .sort_values(["date", "symbol"]).reset_index(drop=True))
    out.to_parquet(CACHE_PATH, index=False)
    print(f"Saved {len(out):,} rows · {out['date'].nunique()} trading days "
          f"({out['date'].min().date()} → {out['date'].max().date()}) "
          f"→ {CACHE_PATH}")
    return out


if __name__ == "__main__":
    build_cache()
