import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _timeline_dates(weeks_min: int, weeks_max: int) -> str:
    today = datetime.today()
    start = today + timedelta(weeks=weeks_min)
    end = today + timedelta(weeks=weeks_max)
    return f"{start.strftime('%-d %b')} – {end.strftime('%-d %b')}"


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


def compute_macd(prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)"""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def generate_signals(closes: pd.DataFrame, volumes: pd.DataFrame, nifty500: set) -> pd.DataFrame:
    results = []

    for symbol in closes.columns:
        if symbol not in nifty500:
            continue
        prices = closes[symbol].dropna()
        if len(prices) < 30:
            continue

        current = prices.iloc[-1]

        # ── Indicators ──────────────────────────────────────────────────────
        rsi = compute_rsi(prices)
        ma20 = prices.rolling(20, min_periods=15).mean().iloc[-1]
        ma50 = prices.rolling(50, min_periods=40).mean().iloc[-1]
        vs_ma20 = (current - ma20) / ma20 * 100 if not pd.isna(ma20) else float("nan")
        vs_ma50 = (current - ma50) / ma50 * 100 if not pd.isna(ma50) else float("nan")

        week_pct = (
            (current - prices.iloc[-6]) / prices.iloc[-6] * 100
            if len(prices) >= 6 else float("nan")
        )

        macd_line, signal_line, histogram = compute_macd(prices)
        macd_bullish = (
            macd_line.iloc[-1] > signal_line.iloc[-1] and
            histogram.iloc[-1] > histogram.iloc[-2]
        ) if len(histogram) >= 2 else False

        vol_ratio = 1.0
        if symbol in volumes.columns:
            vol = volumes[symbol].dropna()
            if len(vol) >= 15:
                vol_ratio = vol.tail(5).mean() / vol.iloc[-15:-5].mean()

        # ── Score (0–100) ─────────────────────────────────────────────────
        # Each factor contributes a fixed weight
        score = 0
        score_breakdown = {}

        # RSI in sweet spot 35–55
        if not pd.isna(rsi):
            if 35 <= rsi <= 55:
                score += 20
                score_breakdown["RSI"] = f"{rsi:.1f} ✓"
            elif rsi < 35:
                score += 10
                score_breakdown["RSI"] = f"{rsi:.1f} (oversold)"
            elif rsi > 70:
                score -= 10
                score_breakdown["RSI"] = f"{rsi:.1f} (overbought)"
            else:
                score_breakdown["RSI"] = f"{rsi:.1f}"

        # Above 20DMA
        if vs_ma20 > 0:
            score += 15
            score_breakdown["20DMA"] = "above ✓"
        elif vs_ma20 > -3:
            score += 8
            score_breakdown["20DMA"] = "near"
        else:
            score_breakdown["20DMA"] = "below"

        # Above 50DMA
        if not pd.isna(vs_ma50):
            if vs_ma50 > 0:
                score += 15
                score_breakdown["50DMA"] = "above ✓"
            elif vs_ma50 > -5:
                score += 5
                score_breakdown["50DMA"] = "near"
            else:
                score_breakdown["50DMA"] = "below"

        # MACD bullish crossover
        if macd_bullish:
            score += 20
            score_breakdown["MACD"] = "bullish ✓"
        else:
            score_breakdown["MACD"] = "bearish"

        # Volume surge
        if vol_ratio > 1.5:
            score += 15
            score_breakdown["Volume"] = f"{vol_ratio:.1f}x ✓"
        elif vol_ratio > 1.2:
            score += 8
            score_breakdown["Volume"] = f"{vol_ratio:.1f}x"
        else:
            score_breakdown["Volume"] = f"{vol_ratio:.1f}x"

        # Positive week momentum
        if not pd.isna(week_pct) and week_pct > 0:
            score += 15
            score_breakdown["Week"] = f"+{week_pct:.1f}% ✓"
        elif not pd.isna(week_pct):
            score_breakdown["Week"] = f"{week_pct:.1f}%"

        score = max(0, min(100, score))

        # ── Signal label ──────────────────────────────────────────────────
        if score >= 75:
            signal = "🟢 Strong Buy"
        elif score >= 55:
            signal = "🟢 Buy"
        elif score >= 40:
            signal = "🟡 Watch"
        elif rsi > 70:
            signal = "🔴 Overbought"
        else:
            signal = "⚪ Neutral"

        # ── Technical Target & Timeline ───────────────────────────────────
        recent_high = prices.tail(30).max()
        recent_low = prices.tail(20).min()

        if current < recent_high * 0.98:
            target = recent_high
        else:
            # Already near high — measured move projection
            target = current + (recent_high - recent_low) * 0.5

        upside_pct = (target - current) / current * 100
        stop_loss = max(recent_low, current * 0.93)  # recent low or 7% stop
        risk_pct = (current - stop_loss) / current * 100
        rr_ratio = upside_pct / risk_pct if risk_pct > 0 else 0

        if score >= 70 and not pd.isna(week_pct) and week_pct > 3:
            timeline = _timeline_dates(1, 2)
        elif score >= 55:
            timeline = _timeline_dates(2, 3)
        elif score >= 40:
            timeline = _timeline_dates(3, 4)
        else:
            timeline = "–"

        results.append({
            "Symbol": symbol,
            "Price (₹)": round(current, 2),
            "RSI": round(rsi, 1) if not pd.isna(rsi) else "–",
            "MACD": "Bullish" if macd_bullish else "Bearish",
            "vs 20DMA (%)": round(vs_ma20, 2) if not pd.isna(vs_ma20) else "–",
            "vs 50DMA (%)": round(vs_ma50, 2) if not pd.isna(vs_ma50) else "–",
            "Vol Ratio": round(vol_ratio, 2),
            "Signal": signal,
            "Score": score,
            "Target (₹)": round(target, 2),
            "Upside (%)": round(upside_pct, 2),
            "Stop Loss (₹)": round(stop_loss, 2),
            "R/R Ratio": round(rr_ratio, 2),
            "Timeline": timeline,
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
