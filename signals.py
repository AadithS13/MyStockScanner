import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return float("nan")
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = gain.iloc[-1] / last_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - (100 / (1 + rs))).round(2)


def generate_signals(closes: pd.DataFrame, volumes: pd.DataFrame, nifty500: set) -> pd.DataFrame:
    results = []

    for symbol in closes.columns:
        if symbol not in nifty500:
            continue
        prices = closes[symbol].dropna()
        if len(prices) < 15:
            continue

        current = prices.iloc[-1]
        rsi = compute_rsi(prices)
        ma20 = prices.tail(20).mean()
        vs_ma = (current - ma20) / ma20 * 100
        week_pct = (
            (current - prices.iloc[-6]) / prices.iloc[-6] * 100
            if len(prices) >= 6 else float("nan")
        )

        vol_ratio = 1.0
        if symbol in volumes.columns:
            vol = volumes[symbol].dropna()
            if len(vol) >= 15:
                vol_ratio = vol.tail(5).mean() / vol.iloc[-15:-5].mean()

        # Swing signal logic (2-week horizon)
        if pd.isna(rsi):
            signal, score = "No data", 0
        elif rsi < 35 and week_pct > 0:
            signal, score = "🟡 Oversold Bounce", 65
        elif rsi <= 50 and vs_ma > -3 and (pd.isna(week_pct) or week_pct > 0):
            bonus = (10 if vol_ratio > 1.2 else 0) + (5 if vs_ma > 0 else 0)
            signal, score = "🟢 Buy", 70 + bonus
        elif rsi <= 58 and vs_ma > -5:
            signal, score = "🟡 Watch", 50
        elif rsi > 70:
            signal, score = "🔴 Overbought", 20
        else:
            signal, score = "⚪ Neutral", 35

        results.append({
            "Symbol": symbol,
            "Price (₹)": round(current, 2),
            "RSI": round(rsi, 1) if not pd.isna(rsi) else "–",
            "vs 20DMA (%)": round(vs_ma, 2),
            "Week (%)": round(week_pct, 2) if not pd.isna(week_pct) else "–",
            "Vol Ratio": round(vol_ratio, 2),
            "Signal": signal,
            "Score": min(int(score), 100),
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
