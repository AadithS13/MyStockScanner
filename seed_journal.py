"""Seed the prediction journal with a walk-forward replay of recent days.

For each of the last N trading days we train ONLY on data strictly before it,
predict, log, and grade against the known outcome. Every entry is genuine
out-of-sample — no peeking — so the resulting track record is honest, and the
dashboard isn't empty on day one. Re-runnable.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from backfill import load_history
from features import build_features, train_frame, FEATURE_COLS
from ml_model import train_model
from journal import JOURNAL_PATH, COLUMNS

N_DAYS = 40


def seed(n_days: int = N_DAYS) -> int:
    hist = load_history()
    feat = build_features(hist)
    tr = train_frame(feat)
    dates = np.sort(tr["date"].unique())
    replay_dates = dates[-n_days:]

    rows = []
    for d in replay_dates:
        past = tr[tr["date"] < d]
        today = tr[tr["date"] == d]
        if len(past) < 300 or today.empty:
            continue
        model = train_model(past)
        proba = model.predict_proba(today[FEATURE_COLS])[:, 1]
        made_on = str(pd.to_datetime(d).date())
        for (_, r), pr in zip(today.iterrows(), proba):
            went_up = r["fwd_ret_1d"] > 0
            pred_dir = "Bullish" if pr >= 0.5 else "Bearish"
            correct = (went_up and pred_dir == "Bullish") or \
                      (not went_up and pred_dir == "Bearish")
            rows.append({
                "pred_id": f"{made_on}_{r['symbol']}",
                "made_on": made_on,
                "symbol": r["symbol"],
                "close_at_pred": round(float(r["close"]), 2),
                "proba": round(float(pr), 4),
                "pred_dir": pred_dir,
                "signals": json.dumps([]),  # seed rows skip SHAP for speed
                "status": "validated",
                "actual_close": round(float(r["close"]) * (1 + r["fwd_ret_1d"]), 2),
                "fwd_ret": round(float(r["fwd_ret_1d"]), 4),
                "correct": bool(correct),
            })

    out = pd.DataFrame(rows)[COLUMNS]
    os.makedirs(os.path.dirname(JOURNAL_PATH), exist_ok=True)
    out.to_csv(JOURNAL_PATH, index=False)
    return len(out)


if __name__ == "__main__":
    n = seed()
    print(f"Seeded {n} validated predictions into the journal")
    from journal import scorecard
    s = scorecard()
    print("\n=== LIVE TRACK RECORD (seeded, walk-forward) ===")
    print(f"period      : {s['first_date']} -> {s['last_date']}")
    print(f"predictions : {s['n']}")
    print(f"accuracy    : {s['accuracy']:.1%}")
    print(f"bullish picks avg next-day : {s['bullish_avg_ret']:+.2%}  (n={s['bullish_n']})")
    print(f"universe      avg next-day : {s['universe_avg_ret']:+.2%}")
    if s.get("calibration") is not None:
        print("\ncalibration (proba bucket -> actual up-rate):")
        print(s["calibration"].to_string(index=False))
