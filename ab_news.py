"""A/B test: does the GDELT news layer add out-of-sample edge?

Runs the identical expanding-window walk-forward on the same rows with two
feature sets — price-only vs price+news — and prints a side-by-side scorecard.
This is the honest "did news help?" answer.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from backfill import load_history
from news import load_sentiment
from features import build_features, train_frame, PRICE_FEATURES, NEWS_FEATURES
from ml_model import PARAMS

PRICE_COLS = PRICE_FEATURES
ALL_COLS = PRICE_FEATURES + NEWS_FEATURES


def backtest(df: pd.DataFrame, cols: list[str], n_folds=6, initial_frac=0.45) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    test_dates = dates[int(len(dates) * initial_frac):]
    oos = []
    for block in np.array_split(test_dates, n_folds):
        if len(block) == 0:
            continue
        cutoff = block[0]
        tr = df[df["date"] < cutoff]
        te = df[df["date"].isin(block)]
        if len(tr) < 200 or te.empty:
            continue
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(tr[cols], tr["target"])
        p = m.predict_proba(te[cols])[:, 1]
        part = te[["date", "fwd_ret_1d", "target"]].copy()
        part["proba"] = p
        oos.append(part)
    oos = pd.concat(oos, ignore_index=True)
    oos["pred"] = (oos["proba"] >= 0.5).astype(int)
    bull = oos[oos["pred"] == 1]
    top = oos.loc[oos.groupby("date")["proba"].idxmax()]
    return {
        "n": len(oos),
        "accuracy": (oos["pred"] == oos["target"]).mean(),
        "auc": roc_auc_score(oos["target"], oos["proba"]),
        "bull_ret": bull["fwd_ret_1d"].mean(),
        "univ_ret": oos["fwd_ret_1d"].mean(),
        "edge": bull["fwd_ret_1d"].mean() - oos["fwd_ret_1d"].mean(),
        "toppick_ret": top["fwd_ret_1d"].mean(),
        "toppick_win": (top["fwd_ret_1d"] > 0).mean(),
    }


def main():
    hist = load_history()
    sent = load_sentiment()
    print(f"History: {len(hist)} rows | Sentiment: {len(sent)} tone rows "
          f"({sent['symbol'].nunique()} entities)\n")

    feat = build_features(hist, sentiment=sent)
    df = train_frame(feat, feature_cols=ALL_COLS)
    print(f"Training rows: {len(df)}  ({df['date'].min().date()} → {df['date'].max().date()})\n")

    price = backtest(df, PRICE_COLS)
    full = backtest(df, ALL_COLS)

    def row(name, a, b, fmt, better_high=True):
        da, db = a, b
        arrow = "↑" if (db > da) == better_high and da != db else ("↓" if da != db else "·")
        print(f"  {name:24} {fmt.format(a):>12} {fmt.format(b):>12}   {arrow}")

    print("  " + " " * 24 + f"{'PRICE-ONLY':>12} {'PRICE+NEWS':>12}")
    print("  " + "-" * 54)
    row("OOS predictions", price["n"], full["n"], "{:.0f}")
    row("AUC", price["auc"], full["auc"], "{:.4f}")
    row("Directional accuracy", price["accuracy"], full["accuracy"], "{:.4f}")
    row("Bullish edge vs univ %", price["edge"]*100, full["edge"]*100, "{:+.3f}")
    row("Top-pick next-day %", price["toppick_ret"]*100, full["toppick_ret"]*100, "{:+.3f}")
    row("Top-pick win rate", price["toppick_win"], full["toppick_win"], "{:.3f}")

    d_auc = (full["auc"] - price["auc"])
    print(f"\n  AUC delta from news: {d_auc:+.4f}  "
          f"({'news helps' if d_auc > 0.003 else 'no meaningful lift' if abs(d_auc) <= 0.003 else 'news hurts'})")


if __name__ == "__main__":
    main()
