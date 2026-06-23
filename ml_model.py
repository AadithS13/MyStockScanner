"""XGBoost model for next-day direction on the defence universe.

Provides:
  * train_model(df)            -> fitted model (+ feature list)
  * walk_forward_backtest(...) -> honest out-of-sample metrics
  * predict_with_explanations  -> probabilities + SHAP +/- top signals
  * save_model / load_model

The backtest is expanding-window walk-forward: at each step the model is
trained ONLY on dates strictly before the test block, mimicking the live
nightly-retrain loop. No row is ever scored by a model that saw its future.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURE_COLS, FEATURE_LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "defence_model.json")
META_PATH = os.path.join(DATA_DIR, "defence_model_meta.json")

PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=3,
    n_estimators=180,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.5,
    reg_alpha=0.2,
    n_jobs=4,
    random_state=42,
)


def train_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**PARAMS)
    model.fit(df[FEATURE_COLS], df["target"])
    return model


def save_model(model: xgb.XGBClassifier, meta: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**PARAMS)
    model.load_model(MODEL_PATH)
    return model


def walk_forward_backtest(train_df: pd.DataFrame, n_folds: int = 6,
                          initial_frac: float = 0.45) -> dict:
    """Expanding-window walk-forward. Returns metrics + the OOS prediction frame."""
    df = train_df.sort_values("date").reset_index(drop=True)
    dates = np.sort(df["date"].unique())
    start_i = int(len(dates) * initial_frac)
    test_dates = dates[start_i:]
    blocks = np.array_split(test_dates, n_folds)

    oos = []
    for block in blocks:
        if len(block) == 0:
            continue
        cutoff = block[0]
        tr = df[df["date"] < cutoff]
        te = df[df["date"].isin(block)]
        if len(tr) < 200 or te.empty:
            continue
        model = train_model(tr)
        proba = model.predict_proba(te[FEATURE_COLS])[:, 1]
        part = te[["date", "symbol", "fwd_ret_1d", "target"]].copy()
        part["proba"] = proba
        oos.append(part)

    oos = pd.concat(oos, ignore_index=True)
    oos["pred"] = (oos["proba"] >= 0.5).astype(int)

    # ── headline metrics ──
    acc = (oos["pred"] == oos["target"]).mean()
    base_rate = oos["target"].mean()
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(oos["target"], oos["proba"])
    except Exception:  # noqa: BLE001
        auc = float("nan")

    bullish = oos[oos["pred"] == 1]
    bull_precision = bullish["target"].mean() if len(bullish) else float("nan")
    bull_ret = bullish["fwd_ret_1d"].mean() if len(bullish) else float("nan")
    overall_ret = oos["fwd_ret_1d"].mean()

    # daily top pick (highest proba each day) — a simple strategy proxy
    top = oos.loc[oos.groupby("date")["proba"].idxmax()]
    top_pick_ret = top["fwd_ret_1d"].mean()
    top_pick_winrate = (top["fwd_ret_1d"] > 0).mean()

    # calibration: in each proba bucket, what fraction actually went up
    buckets = pd.cut(oos["proba"], bins=[0, .4, .45, .5, .55, .6, 1.0])
    calib = oos.groupby(buckets, observed=True).agg(
        n=("target", "size"), predicted=("proba", "mean"), actual=("target", "mean")
    ).reset_index()

    return {
        "n_predictions": len(oos),
        "oos_period": (str(oos["date"].min().date()), str(oos["date"].max().date())),
        "accuracy": acc,
        "base_rate": base_rate,
        "edge_vs_base": acc - max(base_rate, 1 - base_rate),
        "auc": auc,
        "bullish_precision": bull_precision,
        "bullish_avg_fwd_ret": bull_ret,
        "overall_avg_fwd_ret": overall_ret,
        "top_pick_avg_fwd_ret": top_pick_ret,
        "top_pick_winrate": top_pick_winrate,
        "calibration": calib,
        "oos": oos,
    }


def predict_with_explanations(model: xgb.XGBClassifier, live_df: pd.DataFrame,
                              top_n_signals: int = 4) -> pd.DataFrame:
    """Per-stock probability + SHAP-based +/- top signal contributions."""
    import shap

    X = live_df[FEATURE_COLS]
    proba = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)  # (rows, features) in log-odds space

    rows = []
    for i, (_, r) in enumerate(live_df.iterrows()):
        contribs = sorted(
            zip(FEATURE_COLS, shap_vals[i]),
            key=lambda kv: abs(kv[1]), reverse=True,
        )[:top_n_signals]
        signals = [
            {"label": FEATURE_LABELS[f], "feature": f,
             "direction": "+" if v > 0 else "-", "impact": float(v)}
            for f, v in contribs
        ]
        rows.append({
            "symbol": r["symbol"],
            "date": r["date"],
            "close": r["close"],
            "proba": float(proba[i]),
            "signals": signals,
        })
    out = pd.DataFrame(rows).sort_values("proba", ascending=False).reset_index(drop=True)
    return out


def feature_importance(model: xgb.XGBClassifier) -> pd.DataFrame:
    imp = model.feature_importances_
    df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df
