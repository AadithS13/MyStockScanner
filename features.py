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
    # ── news / sentiment (GDELT tone) ──
    "news_tone": "news tone today",
    "news_tone_3d": "news tone (3-day)",
    "news_active": "news coverage",
    "sector_news_tone": "sector news tone",
    "sector_news_tone_3d": "sector tone (3-day)",
    "rel_news_tone": "vs sector news",
}

# Feature groups (handy for the dashboard: "did news help?")
NEWS_FEATURES = [
    "news_tone", "news_tone_3d", "news_active",
    "sector_news_tone", "sector_news_tone_3d", "rel_news_tone",
]

PRICE_FEATURES = [c for c in FEATURE_LABELS if c not in NEWS_FEATURES]

# PRODUCTION feature set. News features are excluded: the 2026-06 A/B test
# showed sector-level tone REDUCES out-of-sample AUC (0.5683 -> 0.5571) —
# one shared value per day adds no cross-sectional ranking power, and coverage
# before Oct 2025 is neutral-zero noise. We keep collecting tone data and can
# re-run ab_news.py once per-stock coverage matures.
FEATURE_COLS = PRICE_FEATURES

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


def _continuous_tone(tone: pd.DataFrame) -> pd.DataFrame:
    """Reindex one entity's tone to continuous calendar days (missing = 0 tone),
    then derive a 3-day rolling mean and a coverage flag. Calendar-day rolling
    means weekend/holiday news still reaches the next trading day. Past-only
    windows → leakage-safe."""
    tone = tone.sort_values("date").set_index("date")
    full = pd.date_range(tone.index.min(), tone.index.max(), freq="D")
    s = tone["news_tone"].reindex(full).fillna(0.0)
    out = pd.DataFrame({"date": full})
    out["news_tone"] = s.values
    out["news_tone_3d"] = s.rolling(3, min_periods=1).mean().values
    out["news_active"] = (s.values != 0).astype(float)
    return out


def _merge_sentiment(feat: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """Attach per-stock + sector tone onto the feature frame by (symbol, date).
    Missing tone defaults to 0 (neutral); the model learns its own weighting."""
    news_cols = ["news_tone", "news_tone_3d", "news_active",
                 "sector_news_tone", "sector_news_tone_3d", "rel_news_tone"]
    if sentiment is None or sentiment.empty:
        for c in news_cols:
            feat[c] = 0.0
        return feat

    sentiment = sentiment.copy()
    sentiment["date"] = pd.to_datetime(sentiment["date"])

    # sector series
    sec_raw = sentiment[sentiment["symbol"] == "__SECTOR__"][["date", "news_tone"]]
    if not sec_raw.empty:
        sec = _continuous_tone(sec_raw).rename(columns={
            "news_tone": "sector_news_tone",
            "news_tone_3d": "sector_news_tone_3d",
            "news_active": "_sec_active",
        })[["date", "sector_news_tone", "sector_news_tone_3d"]]
    else:
        sec = pd.DataFrame(columns=["date", "sector_news_tone", "sector_news_tone_3d"])

    # per-stock series
    parts = []
    for sym, g in sentiment[sentiment["symbol"] != "__SECTOR__"].groupby("symbol"):
        if g["date"].nunique() < 2:
            continue
        c = _continuous_tone(g[["date", "news_tone"]])
        c["symbol"] = sym
        parts.append(c)
    stock = (pd.concat(parts, ignore_index=True)
             if parts else
             pd.DataFrame(columns=["date", "news_tone", "news_tone_3d",
                                   "news_active", "symbol"]))

    feat = feat.merge(stock, on=["symbol", "date"], how="left")
    feat = feat.merge(sec, on="date", how="left")
    for c in news_cols:
        if c not in feat.columns:
            feat[c] = 0.0
    feat[news_cols] = feat[news_cols].fillna(0.0)
    feat["rel_news_tone"] = feat["news_tone_3d"] - feat["sector_news_tone_3d"]
    return feat


def build_features(history: pd.DataFrame,
                   sentiment: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns a long feature frame with FEATURE_COLS + close, fwd_ret_1d, target.

    `sentiment` is the GDELT tone frame (date, symbol, news_tone). If omitted,
    it is loaded from disk; if unavailable, news features fall back to neutral 0
    so the price-only path still works.
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    if sentiment is None:
        try:
            from news import load_sentiment
            sentiment = load_sentiment()
        except Exception:  # noqa: BLE001
            sentiment = None

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

    # ── news / sentiment features (leakage-safe; neutral 0 where absent) ──
    feat = _merge_sentiment(feat, sentiment)

    # IMPORTANT: keep target missing where the next-day return is unknown,
    # otherwise (NaN > 0) -> False silently mislabels the last day as "down".
    up = feat["fwd_ret_1d"] > LABEL_THRESHOLD
    feat["target"] = up.where(feat["fwd_ret_1d"].notna()).astype("Int8")
    return feat


def train_frame(feat: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Rows usable for training: have all features AND a known label.

    `feature_cols` defaults to the production set; the A/B harness passes
    PRICE_FEATURES + NEWS_FEATURES to keep news columns in the frame.
    """
    fc = feature_cols or FEATURE_COLS
    cols = fc + ["target", "fwd_ret_1d", "date", "symbol", "close"]
    df = feat[cols].dropna(subset=fc + ["target"]).copy()
    df["target"] = df["target"].astype(int)
    return df


def latest_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """Most recent dated row per symbol (label unknown) — the live prediction set."""
    live = feat[feat["fwd_ret_1d"].isna()].copy()
    live = live.dropna(subset=FEATURE_COLS)
    idx = live.groupby("symbol")["date"].idxmax()
    return live.loc[idx].reset_index(drop=True)
