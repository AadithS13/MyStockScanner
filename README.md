# AI Trader — self-learning NSE swing-trading dashboard

A personal trading terminal for the Nifty 500 that **grades its own predictions
and learns from the outcomes** — and shows you its live track record next to
every call it makes. Dark-fintech UI, zero paid services, fully self-sustaining
via GitHub Actions.

> **Honesty first:** this is a research aid, not an oracle. Every number on
> screen (win rates, edge vs market, calibration) is computed from graded,
> timestamped predictions — including when the answer is "no edge."

## What it does

| Page | What you get |
|---|---|
| **Overview** | Week-on-week gainers/losers across the Nifty 500 |
| **Swing Signals** | Rule-based 2-week setups (RSI · MACD · DMAs · volume) with score, target, volatility-aware stop, R/R, position sizing — plus **learned odds** (`Hist Win %`) beside every signal, a live track record, and an equity curve of "every call taken" vs the market |
| **AI Lab — Defence** | XGBoost next-day direction model for ~16 defence stocks, with per-stock SHAP "why" chips, live accuracy/edge, and a calibration chart |
| **Feature Importance** | What the ML model actually weighs |
| **Stock Detail** | Price/RSI/MACD panels + this stock's own graded call history |
| **My Portfolio** | Paste holdings, live P&L |

A **weekly digest email** (Resend) delivers the top setups + track record.

## The learning loops

Three GitHub Actions keep the system learning without any human involvement:

```
defence-ai.yml  (weekdays 19:00 IST)
  refresh prices → grade yesterday's pending predictions against actual
  closes → retrain XGBoost on the grown outcome set → log tomorrow's
  picks (with SHAP explanations) → refresh the app's price cache → commit

swing-validate.yml  (Mondays 19:00 IST)
  grade swing calls whose 2-week horizon elapsed (target hit / stopped /
  neither, vs equal-weight market baseline) → log this week's calls → commit

nifty-weekly.yml  (Mondays 08:00 IST)
  build + send the digest email (top gainers, swing setups with learned
  odds, track-record strip)
```

Every prediction is a row in a journal CSV (`data/predictions.csv`,
`data/swing_predictions.csv`) with status `pending` → `validated`. The
dashboards read only these graded artifacts — the UI cannot show a number
that wasn't earned.

## Architecture

```
NSE Bhavcopy / yfinance          GitHub Actions (free)            Streamlit Cloud
────────────────────────         ─────────────────────            ────────────────
daily EOD prices        ──►      nightly/weekly loops:            reads committed
                                 predict → validate →             artifacts only —
GDELT news tone         ──►      retrain → commit         ──►     no training, no
(collected; excluded             journals + model +               live NSE fetch,
from model by A/B test)          n500_cache.parquet               ~0.3s cold start
```

Key design decisions (each verified, not assumed):

- **Price cache, not live fetch** — the app reads `data/n500_cache.parquet`
  (committed nightly). Cold-starting on ~60 live NSE downloads hung the cloud
  app for minutes; now it loads in ~0.3s.
- **News features excluded from the model** — an A/B on identical walk-forward
  folds showed sector-tone features *reduce* OOS AUC (0.5683 → 0.5571). The
  collection pipeline stays (`news.py`) so a future per-stock re-test is cheap.
- **Empirical-Bayes smoothing on learned odds** — thin score-buckets shrink
  toward the global win rate (k=10) so three lucky calls can't claim 100%.
- **Walk-forward only** — the model is never evaluated on data it trained on;
  the seed journal was built the same way the live loop runs.

## Repo map

```
app.py             Streamlit app (native st.navigation, dark fintech theme)
signals.py         rule-based swing engine (score, target, vol-aware stop, why)
swing_journal.py   swing logging / grading / learned-odds engine
ml_model.py        XGBoost train / walk-forward backtest / SHAP explanations
features.py        30 leakage-safe price features (+ news features for A/B)
journal.py         defence prediction journal (nightly entry point)
backfill.py        2yr defence price history (yfinance)
cache_nifty500.py  nightly Nifty-500 OHLCV cache for instant app loads
news.py            GDELT tone collection + headlines (kept for re-test)
ab_news.py         the "does news help?" A/B harness
nifty_scanner.py   weekly digest email (supports --dry-run)
ai_data.py         cached, read-only accessors for the app
data/              committed artifacts: journals, model, caches
```

## Running locally

```bash
pip install -r requirements.txt        # app only
streamlit run app.py

pip install -r requirements-ml.txt     # + training stack (xgboost, shap…)
python journal.py                      # one nightly learning cycle
python swing_journal.py                # one weekly swing cycle
python ab_news.py                      # reproduce the news A/B
python nifty_scanner.py --dry-run      # build digest email without sending
```

Secrets: only `RESEND_API_KEY` (repo secret) for the email job. Never in code.

## Performance, honestly

Numbers move nightly as the loops grade more calls — the app is the source of
truth. At the time of writing: defence model ~53% directional accuracy /
+0.12% next-day edge vs universe on 745 graded predictions (walk-forward AUC
≈ 0.57); swing engine ~58% win rate with edge vs market ≈ 0 over 255 graded
calls. Translation: modest, honestly-measured signal — the value is the
discipline and the visible feedback loop, not a money printer.

*Data: NSE Bhavcopy (EOD). Not investment advice.*
