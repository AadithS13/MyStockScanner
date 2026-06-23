"""Feature engineering for the defence ML prototype.

Given the tidy history (date, symbol, OHLCV), build a per-(symbol, date)
feature matrix of ~30 technical signals plus a leakage-safe next-day label.

Design rules:
  * Every feature at row t uses ONLY data up to and including day t.
  * The label is the return from close[t] -> close[t+1], so the most recent
    day has no label (that's the row we predict on, live).
  * The sector signal is an equal-weight basket of all defence names'
    daily returns, shifted so it never peeks at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Human-readable names for the explanation panel (+ / - signals).
FEATURE_LABELS = {
    "ret_1d": "1-day momentum",
    "ret_3d": "3-day momentum",
    "ret_5d": "1-week momentum",
    "ret_10d": "2-week momentum",
    "ret_20d": "1-month momentum",
    "dist_sma10": "vs 10-day avg",
    "dist_sma20": "vs 20-day avg",
    "dist_sma50": "vs 50-day avg",
    "sma20_slope": "20-day trend slope",
    "rsi_14": "RSI (14)",
    "rsi_7": "RSI (7, fast)",
    "macd_hist": "MACD histogram",
    "macd_hist_slope": "MACD momentum",
    "atr_pct": "volatility (ATR)",
    "ret_std_10": "10-day choppiness",
    "ret_std_20": "20-day choppiness",
    "bb_position": "Bollinger position",
    "bb_width": "Bollinger width",
    "vol_ratio": "volume surge",
    "vol_zscore": "volume z-score",
    "gap_pct": "overnight gap",
    "close_in_range": "close strength",
    "hl_range_pct": "daily range",
    "sector_ret_1d": "sector 1-day",
    "sector_ret_5d": "sector 1-week",
    "rel_strength_5d": "relative strength",
    "dist_high_20": "below 20-day high",
    "dist_low_20": "above 20-day low",
    "up_days_10": "up-days in 10",
    "dow": "day of week",
}

FEATURE_COLS = list(FEATURE_LABELS.keys())

# Label: was the next day an up day? (clean, interpretable probability)
LABEL_THRESHOLD = 0.0


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, period=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _per_stock_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"]

    out = pd.DataFrame(index=g.index)
    out["date"] = g["date"].values
    out["symbol"] = g["symbol"].values
    out["close"] = c.values

    # momentum
    for n in (1, 3, 5, 10, 20):
        out[f"ret_{n}d"] = c.pct_change(n)

    # moving-average distances
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50, min_periods=35).mean()
    out["dist_sma10"] = (c - sma10) / sma10
    out["dist_sma20"] = (c - sma20) / sma20
    out["dist_sma50"] = (c - sma50) / sma50
    out["sma20_slope"] = sma20.pct_change(5)

    # RSI
    out["rsi_14"] = _rsi(c, 14)
    out["rsi_7"] = _rsi(c, 7)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    out["macd_hist"] = hist / c          # normalise by price
    out["macd_hist_slope"] = hist.diff()

    # volatility
    out["atr_pct"] = _atr(h, l, c) / c
    rets = c.pct_change()
    out["ret_std_10"] = rets.rolling(10).std()
    out["ret_std_20"] = rets.rolling(20).std()

    # Bollinger
    bb_mid = sma20
    bb_std = c.rolling(20).std()
    upper = bb_mid + 2 * bb_std
    lower = bb_mid - 2 * bb_std
    out["bb_position"] = (c - lower) / (upper - lower)
    out["bb_width"] = (upper - lower) / bb_mid

    # volume
    vol_sma20 = v.rolling(20).mean()
    out["vol_ratio"] = v.rolling(5).mean() / vol_sma20
    out["vol_zscore"] = (v - vol_sma20) / v.rolling(20).std()

    # candle / range
    out["gap_pct"] = (g["open"] - c.shift(1)) / c.shift(1)
    rng = (h - l).replace(0, np.nan)
    out["close_in_range"] = (c - l) / rng
    out["hl_range_pct"] = rng / c

    # position within recent range
    high20 = h.rolling(20).max()
    low20 = l.rolling(20).min()
    out["dist_high_20"] = (c - high20) / high20
    out["dist_low_20"] = (c - low20) / low20
    out["up_days_10"] = (rets > 0).rolling(10).sum()

    # calendar
    out["dow"] = pd.to_datetime(g["date"]).dt.dayofweek.values

    # label: next-day return (for training + validation)
    out["fwd_ret_1d"] = c.shift(-1) / c - 1
    return out


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Returns a long feature frame with FEATURE_COLS + close, fwd_ret_1d, target."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    # leakage-safe equal-weight sector basket return per date
    history = history.sort_values(["symbol", "date"])
    history["ret_1d_raw"] = history.groupby("symbol")["close"].pct_change()
    sector = history.groupby("date")["ret_1d_raw"].mean().rename("sector_ret_1d")
    sector_5d = (
        history.groupby("date")["ret_1d_raw"].mean()
        .rolling(5).sum().rename("sector_ret_5d")
    )

    frames = [
        _per_stock_features(g)
        for _, g in history.groupby("symbol", sort=False)
    ]
    feat = pd.concat(frames, ignore_index=True)

    feat = feat.merge(sector, on="date", how="left")
    feat = feat.merge(sector_5d, on="date", how="left")
    feat["rel_strength_5d"] = feat["ret_5d"] - feat["sector_ret_5d"]

    # IMPORTANT: keep target missing where the next-day return is unknown,
    # otherwise (NaN > 0) -> False silently mislabels the last day as "down".
    up = feat["fwd_ret_1d"] > LABEL_THRESHOLD
    feat["target"] = up.where(feat["fwd_ret_1d"].notna()).astype("Int8")
    return feat


def train_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training: have all features AND a known label."""
    cols = FEATURE_COLS + ["target", "fwd_ret_1d", "date", "symbol", "close"]
    df = feat[cols].dropna(subset=FEATURE_COLS + ["target"]).copy()
    df["target"] = df["target"].astype(int)
    return df


def latest_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Most recent dated row per symbol (label unknown) — the live prediction set."""
    live = feat[feat["fwd_ret_1d"].isna()].copy()
    live = live.dropna(subset=FEATURE_COLS)
    idx = live.groupby("symbol")["date"].idxmax()
    return live.loc[idx].reset_index(drop=True)
