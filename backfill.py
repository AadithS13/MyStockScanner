"""One-time (and incrementally re-runnable) historical price backfill for the
defence universe, via yfinance. Produces a tidy parquet:

    data/defence_history.parquet
    columns: date, symbol, open, high, low, close, volume

Re-running is safe: it fetches fresh full history and overwrites. yfinance
gives split/bonus-adjusted OHLC (auto_adjust=True), so the series is
continuous across corporate actions — important for these stocks.
"""

from __future__ import annotations

import os
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from defence import YF_TICKERS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_PARQUET = os.path.join(DATA_DIR, "defence_history.parquet")
PERIOD = "2y"


def _fetch_one(yf, ticker: str, retries: int = 3) -> pd.DataFrame | None:
    import yfinance as yfin
    for attempt in range(retries):
        try:
            df = yfin.download(
                ticker, period=PERIOD, interval="1d",
                progress=False, auto_adjust=True, threads=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            print(f"  {ticker}: attempt {attempt+1} failed ({e})")
        time.sleep(1.5 * (attempt + 1))
    return None


def backfill() -> pd.DataFrame:
    import yfinance as yfin
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for sym, ticker in YF_TICKERS.items():
        df = _fetch_one(yfin, ticker)
        if df is None:
            print(f"✗ {sym}: no data")
            continue
        # yfinance may return a single- or multi-index column frame
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df.index).date
        frames.append(df.reset_index(drop=True))
        print(f"✓ {sym}: {len(df)} rows  ({df['date'].min()} → {df['date'].max()})")

    if not frames:
        raise RuntimeError("Backfill produced no data")

    out = pd.concat(frames, ignore_index=True)
    out = out[["date", "symbol", "open", "high", "low", "close", "volume"]]
    out = out.dropna(subset=["close"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(HISTORY_PARQUET, index=False)
    print(f"\nSaved {len(out)} rows for {out['symbol'].nunique()} symbols → {HISTORY_PARQUET}")
    return out


def load_history() -> pd.DataFrame:
    df = pd.read_parquet(HISTORY_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    backfill()
