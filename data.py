from __future__ import annotations
import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from io import StringIO

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NSE_CSV_PRIMARY = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_CSV_FALLBACK = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"


@st.cache_data(ttl=3600)
def get_nifty500_symbols() -> set[str]:
    for url in [NSE_CSV_PRIMARY, NSE_CSV_FALLBACK]:
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            return set(df["Symbol"].dropna().str.strip())
        except Exception:
            pass
    return set()


@st.cache_data(ttl=3600)
def _fetch_bhavcopy_raw(date_str: str) -> pd.DataFrame | None:
    url = BHAVCOPY_URL.format(date=date_str)
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
        df["SERIES"] = df["SERIES"].str.strip()
        eq = df[df["SERIES"] == "EQ"].copy()
        for col in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]:
            eq[col] = pd.to_numeric(eq[col], errors="coerce")
        return eq[["SYMBOL", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]].reset_index(drop=True)
    except Exception:
        return None


CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "n500_cache.parquet")
CACHE_MAX_AGE_DAYS = 5  # weekend + a holiday; staler than this → try live fetch


def _load_cache() -> dict | None:
    """Instant load path: the committed parquet refreshed nightly by CI.

    Returns the same {closes, volumes, raw} shape as the live path, or None
    when the cache is missing/stale/unreadable (caller falls back to NSE).
    This exists because cold-starting the cloud app on ~60 live NSE downloads
    could hang for minutes (NSE archives are slow from non-Indian cloud IPs).
    """
    try:
        if not os.path.exists(CACHE_PATH):
            return None
        df = pd.read_parquet(CACHE_PATH)
        df["date"] = pd.to_datetime(df["date"])
        last = df["date"].max()
        if (pd.Timestamp.today().normalize() - last.normalize()).days > CACHE_MAX_AGE_DAYS:
            return None  # bot hasn't refreshed lately — fall back to live

        raw: dict[object, pd.DataFrame] = {}
        for d, g in df.groupby("date"):
            raw[d.date()] = g.rename(columns={
                "symbol": "SYMBOL", "open": "OPEN_PRICE", "high": "HIGH_PRICE",
                "low": "LOW_PRICE", "close": "CLOSE_PRICE",
                "volume": "TTL_TRD_QNTY",
            })[["SYMBOL", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
                "CLOSE_PRICE", "TTL_TRD_QNTY"]].reset_index(drop=True)

        closes = df.pivot_table(index="date", columns="symbol",
                                values="close").sort_index()
        closes.index = [d.date() for d in closes.index]
        volumes = df.pivot_table(index="date", columns="symbol",
                                 values="volume").sort_index()
        volumes.index = [d.date() for d in volumes.index]
        return {"closes": closes, "volumes": volumes, "raw": raw}
    except Exception:  # noqa: BLE001 — any cache problem → live fallback
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(n_days: int = 30) -> dict:
    """Price history for the app. Reads the committed nightly cache instantly;
    falls back to live NSE downloads only when the cache is missing/stale."""
    cached = _load_cache()
    if cached is not None:
        return cached

    today = datetime.today()
    candidate = today - timedelta(days=1)
    raw: dict[object, pd.DataFrame] = {}
    attempts = 0

    while len(raw) < n_days and attempts < 60:
        if candidate.weekday() < 5:
            df = _fetch_bhavcopy_raw(candidate.strftime("%d%m%Y"))
            if df is not None:
                raw[candidate.date()] = df
        candidate -= timedelta(days=1)
        attempts += 1

    if not raw:
        return {"closes": pd.DataFrame(), "volumes": pd.DataFrame(), "raw": {}}

    closes = pd.DataFrame(
        {d: df.set_index("SYMBOL")["CLOSE_PRICE"] for d, df in raw.items()}
    ).T.sort_index()
    volumes = pd.DataFrame(
        {d: df.set_index("SYMBOL")["TTL_TRD_QNTY"] for d, df in raw.items()}
    ).T.sort_index()

    return {"closes": closes, "volumes": volumes, "raw": raw}


def get_stock_ohlcv(symbol: str, raw: dict) -> pd.DataFrame:
    rows = []
    for date, df in sorted(raw.items()):
        row = df[df["SYMBOL"] == symbol]
        if not row.empty:
            r = row.iloc[0]
            rows.append({
                "date": date,
                "open": r["OPEN_PRICE"],
                "high": r["HIGH_PRICE"],
                "low": r["LOW_PRICE"],
                "close": r["CLOSE_PRICE"],
                "volume": r["TTL_TRD_QNTY"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
