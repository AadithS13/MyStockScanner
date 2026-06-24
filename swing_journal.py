"""Swing-signal journal — a self-grading track record for the 2-week swing calls.

Mirrors the AI Lab loop, but for the Nifty-500 rule-based swing engine:

  * log_swing(signals_df, made_on)  — record each Buy/Strong Buy call
                                      (entry, target, stop, score, horizon).
  * validate_swing(history)         — once ~2 weeks have passed, grade each call:
                                      did the close reach the TARGET, hit the
                                      STOP, or neither? what was the realised
                                      return vs an equal-weight market baseline?
  * swing_scorecard()               — hit-rate / target-rate / avg edge for the UI.
  * seed_from_history()             — replay past as-of dates so there is an
                                      immediate track record instead of weeks of
                                      waiting.

Streamlit-free on purpose, so the weekly GitHub Actions job can run it. Grading
uses closing prices (conservative: an intraday target touch that doesn't close
through the level is not counted).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests

from signals import generate_signals

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JOURNAL_PATH = os.path.join(DATA_DIR, "swing_predictions.csv")

HORIZON_DAYS = 10            # trading days ≈ 2 weeks
SEED_GAP = 5                 # trading days between seeded as-of snapshots
TOP_K = 15                   # log the strongest N Buy/Strong-Buy calls per run

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

COLUMNS = [
    "pred_id", "made_on", "symbol", "signal", "score", "entry", "target",
    "stop", "horizon_end", "status", "outcome", "exit_close", "realised_ret",
    "market_ret", "correct",
]


# ── streamlit-free data layer ────────────────────────────────────────────────
def get_nifty500() -> set[str]:
    try:
        r = requests.get(NIFTY500_URL, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return set(pd.read_csv(StringIO(r.text))["Symbol"].dropna().str.strip())
    except Exception:  # noqa: BLE001
        return set()


def _fetch_bhavcopy(date: datetime) -> pd.DataFrame | None:
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
        for c in ("CLOSE_PRICE", "TTL_TRD_QNTY"):
            eq[c] = pd.to_numeric(eq[c], errors="coerce")
        return eq[["SYMBOL", "CLOSE_PRICE", "TTL_TRD_QNTY"]]
    except Exception:  # noqa: BLE001
        return None


def fetch_history(n_days: int = 70) -> dict:
    """Closes + volumes pivots (dates × symbols) from bhavcopy, streamlit-free."""
    today = datetime.today()
    candidate = today - timedelta(days=1)
    raw, attempts = {}, 0
    while len(raw) < n_days and attempts < n_days + 40:
        if candidate.weekday() < 5:
            df = _fetch_bhavcopy(candidate)
            if df is not None:
                raw[candidate.date()] = df
                time.sleep(0.25)
        candidate -= timedelta(days=1)
        attempts += 1
    if not raw:
        return {"closes": pd.DataFrame(), "volumes": pd.DataFrame()}
    closes = pd.DataFrame(
        {d: df.set_index("SYMBOL")["CLOSE_PRICE"] for d, df in raw.items()}).T.sort_index()
    volumes = pd.DataFrame(
        {d: df.set_index("SYMBOL")["TTL_TRD_QNTY"] for d, df in raw.items()}).T.sort_index()
    closes.index = pd.to_datetime(closes.index)
    volumes.index = pd.to_datetime(volumes.index)
    return {"closes": closes, "volumes": volumes}


# ── journal io ───────────────────────────────────────────────────────────────
def _load() -> pd.DataFrame:
    if os.path.exists(JOURNAL_PATH):
        df = pd.read_csv(JOURNAL_PATH)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def log_swing(signals_df: pd.DataFrame, made_on: str, top_k: int = TOP_K) -> int:
    """Record the strongest Buy/Strong-Buy calls for `made_on` (idempotent)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    buys = signals_df[signals_df["Signal"].str.contains("Buy", na=False)].copy()
    buys = buys.sort_values("Score", ascending=False).head(top_k)

    existing = _load()
    existing = existing[existing["made_on"].astype(str) != str(made_on)]

    rows = []
    for _, s in buys.iterrows():
        rows.append({
            "pred_id": f"{made_on}_{s['Symbol']}",
            "made_on": made_on,
            "symbol": s["Symbol"],
            "signal": s["Signal"],
            "score": s["Score"],
            "entry": s["Price (₹)"],
            "target": s["Target (₹)"],
            "stop": s["Stop Loss (₹)"],
            "horizon_end": pd.NA,
            "status": "pending",
            "outcome": pd.NA, "exit_close": pd.NA, "realised_ret": pd.NA,
            "market_ret": pd.NA, "correct": pd.NA,
        })
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)[COLUMNS]
    out.to_csv(JOURNAL_PATH, index=False)
    return len(rows)


def validate_swing(history: dict | None = None) -> int:
    """Grade pending calls whose 2-week horizon has elapsed."""
    if history is None:
        history = fetch_history()
    closes = history["closes"]
    if closes.empty:
        return 0
    dates = list(closes.index)
    market = closes.mean(axis=1)  # equal-weight Nifty-500 baseline

    jour = _load()
    graded = 0
    for i, row in jour.iterrows():
        if row["status"] == "validated":
            continue
        sym = row["symbol"]
        if sym not in closes.columns:
            continue
        made_on = pd.to_datetime(row["made_on"])
        after = [d for d in dates if d > made_on]
        if len(after) < HORIZON_DAYS:
            continue  # not matured yet
        window = after[:HORIZON_DAYS]
        path = closes.loc[window, sym].dropna()
        if path.empty:
            continue
        entry = float(row["entry"]); target = float(row["target"]); stop = float(row["stop"])

        outcome = "OPEN"
        for px in path:
            if px >= target:
                outcome = "TARGET"; break
            if px <= stop:
                outcome = "STOP"; break
        exit_close = float(path.iloc[-1])
        realised = exit_close / entry - 1
        mkt = float(market.loc[window[-1]] / market.loc[window[0]] - 1)
        win = outcome == "TARGET" or (outcome == "OPEN" and realised > 0)

        jour.at[i, "horizon_end"] = str(window[-1].date())
        jour.at[i, "outcome"] = outcome
        jour.at[i, "exit_close"] = round(exit_close, 2)
        jour.at[i, "realised_ret"] = round(realised, 4)
        jour.at[i, "market_ret"] = round(mkt, 4)
        jour.at[i, "correct"] = bool(win)
        jour.at[i, "status"] = "validated"
        graded += 1

    jour.to_csv(JOURNAL_PATH, index=False)
    return graded


def get_swing_journal() -> pd.DataFrame:
    return _load()


def swing_scorecard() -> dict:
    jour = _load()
    val = jour[jour["status"] == "validated"].copy()
    if val.empty:
        return {"n": 0}
    val["realised_ret"] = val["realised_ret"].astype(float)
    val["market_ret"] = val["market_ret"].astype(float)
    val["correct"] = val["correct"].astype(str).isin(["True", "true", "1"])
    val["target_hit"] = val["outcome"].astype(str) == "TARGET"
    val["stopped"] = val["outcome"].astype(str) == "STOP"
    return {
        "n": len(val),
        "win_rate": val["correct"].mean(),
        "target_rate": val["target_hit"].mean(),
        "stop_rate": val["stopped"].mean(),
        "avg_ret": val["realised_ret"].mean(),
        "market_ret": val["market_ret"].mean(),
        "edge": val["realised_ret"].mean() - val["market_ret"].mean(),
        "first_date": str(val["made_on"].min()),
        "last_date": str(val["made_on"].max()),
        "recent": val.sort_values("made_on").tail(40),
    }


# ── seeding ──────────────────────────────────────────────────────────────────
def seed_from_history(n_days: int = 75) -> None:
    """Replay past as-of dates so we have a graded track record immediately."""
    print(f"Fetching {n_days} trading days of bhavcopy for seed…")
    hist = fetch_history(n_days)
    closes, volumes = hist["closes"], hist["volumes"]
    if closes.empty:
        print("No data — seed aborted")
        return
    nifty500 = get_nifty500()
    dates = list(closes.index)
    # as-of snapshots that are old enough to be fully matured
    asof_idxs = range(50, len(dates) - HORIZON_DAYS, SEED_GAP)
    logged = 0
    for i in asof_idxs:
        asof = dates[i]
        sub_c = closes.loc[:asof]
        sub_v = volumes.loc[:asof]
        try:
            sig = generate_signals(sub_c, sub_v, nifty500)
        except Exception as e:  # noqa: BLE001
            print(f"  {asof.date()}: signal gen failed ({e})"); continue
        logged += log_swing(sig, str(asof.date()))
        print(f"  logged {asof.date()}")
    graded = validate_swing(hist)
    print(f"\nSeed done: {logged} calls logged, {graded} graded → {JOURNAL_PATH}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_from_history()
    else:
        # weekly entry point: validate matured calls, then log this week's signals
        hist = fetch_history()
        g = validate_swing(hist)
        nifty500 = get_nifty500()
        sig = generate_signals(hist["closes"], hist["volumes"], nifty500)
        n = log_swing(sig, str(hist["closes"].index.max().date()))
        print(f"Validated {g}, logged {n} new swing calls")
