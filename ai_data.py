"""Streamlit-cached access layer for the defence AI module.

The cloud app does NO training or inference — it only reads artifacts produced
by the nightly job (predictions.csv with precomputed SHAP signals,
feature_importance.csv, model meta). That keeps the serving environment light:
no xgboost / shap / numba needed on Streamlit Cloud.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JOURNAL = os.path.join(DATA_DIR, "predictions.csv")
IMPORTANCE = os.path.join(DATA_DIR, "feature_importance.csv")
META = os.path.join(DATA_DIR, "defence_model_meta.json")


def ai_available() -> bool:
    return os.path.exists(JOURNAL) and os.path.exists(IMPORTANCE)


@st.cache_data(ttl=900, show_spinner=False)
def live_predictions() -> pd.DataFrame:
    """Latest day's predictions, with SHAP signals already attached."""
    j = pd.read_csv(JOURNAL)
    if j.empty:
        return pd.DataFrame(columns=["symbol", "date", "close", "proba", "signals"])
    latest = j["made_on"].max()
    cur = j[j["made_on"] == latest].copy()
    cur["signals"] = cur["signals"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s.strip() else []
    )
    cur = cur.rename(columns={"made_on": "date", "close_at_pred": "close"})
    cur["proba"] = cur["proba"].astype(float)
    return cur[["symbol", "date", "close", "proba", "signals"]].sort_values(
        "proba", ascending=False
    ).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def importance() -> pd.DataFrame:
    return pd.read_csv(IMPORTANCE)


@st.cache_data(ttl=900, show_spinner=False)
def track_record() -> dict:
    from journal import scorecard
    return scorecard()


@st.cache_data(ttl=900, show_spinner=False)
def journal_df() -> pd.DataFrame:
    return pd.read_csv(JOURNAL)


@st.cache_data(ttl=3600, show_spinner=False)
def model_meta() -> dict:
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}
