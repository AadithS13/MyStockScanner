"""Prediction journal — the self-validating learning loop.

Every run logs the model's predictions to data/predictions.csv. Once the
next trading day's close is known, validate_pending() grades each call
(was the direction right? what was the realised return?). scorecard()
summarises the live track record, which the dashboard displays.

This is what makes the system *learning* rather than just confidence-spitting:
the graded outcomes accumulate and feed the next retrain.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JOURNAL_PATH = os.path.join(DATA_DIR, "predictions.csv")

COLUMNS = [
    "pred_id", "made_on", "symbol", "close_at_pred", "proba", "pred_dir",
    "signals", "status", "actual_close", "fwd_ret", "correct",
]


def _load() -> pd.DataFrame:
    if os.path.exists(JOURNAL_PATH):
        df = pd.read_csv(JOURNAL_PATH)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def log_predictions(preds: pd.DataFrame, made_on: str | None = None) -> int:
    """Append today's predictions (output of predict_with_explanations). Idempotent
    per (made_on, symbol) — re-running the same day overwrites, doesn't duplicate."""
    os.makedirs(DATA_DIR, exist_ok=True)
    made_on = made_on or str(pd.to_datetime(preds["date"].max()).date())
    existing = _load()
    existing = existing[~(existing["made_on"].astype(str) == made_on)]

    rows = []
    for _, p in preds.iterrows():
        rows.append({
            "pred_id": f"{made_on}_{p['symbol']}",
            "made_on": made_on,
            "symbol": p["symbol"],
            "close_at_pred": round(float(p["close"]), 2),
            "proba": round(float(p["proba"]), 4),
            "pred_dir": "Bullish" if p["proba"] >= 0.5 else "Bearish",
            "signals": json.dumps(p["signals"]),
            "status": "pending",
            "actual_close": pd.NA,
            "fwd_ret": pd.NA,
            "correct": pd.NA,
        })
    new = pd.DataFrame(rows)
    out = pd.concat([existing, new], ignore_index=True)[COLUMNS]
    out.to_csv(JOURNAL_PATH, index=False)
    return len(new)


def validate_pending(history: pd.DataFrame) -> int:
    """Grade pending predictions whose next-day close is now known."""
    jour = _load()
    if jour.empty:
        return 0
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])

    graded = 0
    for i, row in jour.iterrows():
        if row["status"] == "validated":
            continue
        sym = row["symbol"]
        made_on = pd.to_datetime(row["made_on"])
        future = history[(history["symbol"] == sym) & (history["date"] > made_on)]
        if future.empty:
            continue
        nxt = future.sort_values("date").iloc[0]
        actual_close = float(nxt["close"])
        fwd_ret = actual_close / float(row["close_at_pred"]) - 1
        went_up = fwd_ret > 0
        correct = (went_up and row["pred_dir"] == "Bullish") or \
                  (not went_up and row["pred_dir"] == "Bearish")
        jour.at[i, "actual_close"] = round(actual_close, 2)
        jour.at[i, "fwd_ret"] = round(fwd_ret, 4)
        jour.at[i, "correct"] = bool(correct)
        jour.at[i, "status"] = "validated"
        graded += 1

    jour.to_csv(JOURNAL_PATH, index=False)
    return graded


def get_journal() -> pd.DataFrame:
    return _load()


def scorecard() -> dict:
    """Live track record from validated predictions."""
    jour = _load()
    val = jour[jour["status"] == "validated"].copy()
    if val.empty:
        return {"n": 0}

    val["proba"] = val["proba"].astype(float)
    val["fwd_ret"] = val["fwd_ret"].astype(float)
    val["correct"] = val["correct"].astype(str).isin(["True", "true", "1"])

    bullish = val[val["pred_dir"] == "Bullish"]
    calib = None
    if len(val) >= 10:
        b = pd.cut(val["proba"], bins=[0, .45, .5, .55, .6, 1.0])
        calib = val.groupby(b, observed=True).agg(
            n=("correct", "size"),
            up_rate=("fwd_ret", lambda s: (s > 0).mean()),
        ).reset_index()

    return {
        "n": len(val),
        "accuracy": val["correct"].mean(),
        "bullish_n": len(bullish),
        "bullish_precision": (bullish["fwd_ret"] > 0).mean() if len(bullish) else float("nan"),
        "bullish_avg_ret": bullish["fwd_ret"].mean() if len(bullish) else float("nan"),
        "universe_avg_ret": val["fwd_ret"].mean(),
        "calibration": calib,
        "first_date": str(val["made_on"].min()),
        "last_date": str(val["made_on"].max()),
    }


if __name__ == "__main__":
    # daily entry point: validate yesterday, then (re)log today's predictions
    from backfill import load_history
    from features import build_features, train_frame, latest_frame
    from ml_model import (train_model, save_model, predict_with_explanations,
                          feature_importance)

    hist = load_history()
    graded = validate_pending(hist)
    print(f"Validated {graded} pending predictions")

    feat = build_features(hist)
    tr, live = train_frame(feat), latest_frame(feat)
    model = train_model(tr)
    save_model(model, {"trained_at": datetime.now(), "n_train": len(tr),
                       "last_train_date": str(tr["date"].max().date())})

    # persist feature importance so the cloud app needs no xgboost at runtime
    feature_importance(model).to_csv(
        os.path.join(DATA_DIR, "feature_importance.csv"), index=False)

    preds = predict_with_explanations(model, live)
    n = log_predictions(preds)
    print(f"Logged {n} predictions for {preds['date'].max().date()}")
