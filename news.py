"""News / sentiment layer for the defence AI module.

Two free sources, no API key:

  * GDELT DOC 2.0 (`timelinetone`) — daily average *tone* of global news
    mentioning each company (and the sector). Tone is roughly
    [-10 very negative .. +10 very positive]; 0 = no coverage that day.
    GDELT has years of history, so this is what we backfill on.

  * Google News RSS — fresh recent headlines per stock, shown on the pick
    cards and scored with VADER. Display only (no deep history), so it is
    NOT a training feature; GDELT tone is.

Leakage rule: tone[t] reflects news published *on* day t, which is known by
the time we predict day t+1's close. So merging tone[t] onto the day-t feature
row is safe — no peeking at the future.

Throttling: GDELT asks for <=1 request / 5s. We space calls 8s apart and
retry on HTTP 429 with backoff.
"""

from __future__ import annotations

import os
import time
import warnings
from datetime import datetime
from xml.etree import ElementTree as ET

import pandas as pd
import requests

warnings.filterwarnings("ignore")

from defence import DEFENCE_STOCKS, NAMES  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SENTIMENT_PARQUET = os.path.join(DATA_DIR, "defence_sentiment.parquet")

SECTOR_KEY = "__SECTOR__"
SECTOR_QUERY = '"India defence"'  # single phrase — heavy OR queries time out on GDELT

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_GAP = 10.0  # seconds between calls (limit is 1 / 5s; extra margin avoids 429)


# Per-symbol GDELT query: full company name is the most precise handle.
def _stock_query(symbol: str) -> str:
    name = NAMES.get(symbol, symbol)
    return f'"{name}"'


def _gdelt_tone(query: str, start: datetime, end: datetime,
                retries: int = 4) -> pd.DataFrame:
    """Daily tone timeline for one query. Returns DataFrame[date, news_tone]."""
    params = dict(
        query=query, mode="timelinetone", format="json",
        startdatetime=start.strftime("%Y%m%d000000"),
        enddatetime=end.strftime("%Y%m%d000000"),
    )
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(GDELT_GAP * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            series = data.get("timeline", [])
            if not series:
                return pd.DataFrame(columns=["date", "news_tone"])
            pts = series[0].get("data", [])
            rows = [
                {"date": pd.to_datetime(p["date"]).normalize().tz_localize(None),
                 "news_tone": float(p["value"])}
                for p in pts
            ]
            return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            time.sleep(GDELT_GAP * (attempt + 1))
    return pd.DataFrame(columns=["date", "news_tone"])


def build_sentiment_history(years: int = 1, chunk_months: int = 3,
                            resume: bool = True) -> pd.DataFrame:
    """Backfill daily tone for every stock + the sector basket.

    Chunks the date range so GDELT keeps daily (not weekly) resolution and so a
    single huge request can't time out. RESUMABLE: writes the parquet after each
    entity completes, and on restart skips entities already present. So an
    interrupted run loses at most one entity's progress.

    Produces data/defence_sentiment.parquet: [date, symbol, news_tone].
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    end = datetime.utcnow()
    start = end - pd.DateOffset(years=years)
    bounds = list(pd.date_range(start, end, freq=f"{chunk_months}MS"))
    if not bounds or bounds[0] > pd.Timestamp(start):
        bounds = [pd.Timestamp(start)] + bounds
    if bounds[-1] < pd.Timestamp(end):
        bounds.append(pd.Timestamp(end))

    entities = [(SECTOR_KEY, SECTOR_QUERY)] + [
        (sym, _stock_query(sym)) for sym in DEFENCE_STOCKS
    ]

    # resume: load whatever we already have, skip done entities
    done: set[str] = set()
    existing = pd.DataFrame(columns=["date", "symbol", "news_tone"])
    if resume and os.path.exists(SENTIMENT_PARQUET):
        existing = pd.read_parquet(SENTIMENT_PARQUET)
        done = set(existing["symbol"].unique())
        print(f"Resuming — {len(done)} entities already cached: {sorted(done)}")

    first = True
    for key, query in entities:
        if key in done:
            print(f"· {key:14} (cached, skipped)", flush=True)
            continue
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if not first:
                time.sleep(GDELT_GAP)
            first = False
            part = _gdelt_tone(query, a, b)
            if not part.empty:
                parts.append(part)
        if parts:
            df = pd.concat(parts, ignore_index=True).drop_duplicates("date")
            df["symbol"] = key
            existing = pd.concat([existing, df[["date", "symbol", "news_tone"]]],
                                 ignore_index=True)
            print(f"✓ {key:14} {len(df):4} days of tone", flush=True)
        else:
            print(f"✗ {key:14} no coverage", flush=True)
        # checkpoint after every entity
        existing.sort_values(["symbol", "date"]).to_parquet(SENTIMENT_PARQUET, index=False)

    out = existing.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"\nSaved {len(out)} tone rows for {out['symbol'].nunique()} entities "
          f"→ {SENTIMENT_PARQUET}")
    return out


def load_sentiment() -> pd.DataFrame:
    if not os.path.exists(SENTIMENT_PARQUET):
        return pd.DataFrame(columns=["date", "symbol", "news_tone"])
    df = pd.read_parquet(SENTIMENT_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Live headlines (display only) ────────────────────────────────────────────
def fetch_headlines(symbol: str, limit: int = 4) -> list[dict]:
    """Recent Google News RSS headlines for a stock, scored with VADER if
    available. Best-effort: returns [] on any failure."""
    name = NAMES.get(symbol, symbol)
    q = requests.utils.quote(f'{name} stock')
    url = f"https://news.google.com/rss/search?q={q}+when:14d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:limit]
    except Exception:  # noqa: BLE001
        return []

    scorer = _vader()
    out = []
    for it in items:
        title = (it.findtext("title") or "").strip()
        link = it.findtext("link") or ""
        pub = it.findtext("pubDate") or ""
        score = scorer(title) if scorer else None
        out.append({"title": title, "link": link, "published": pub, "score": score})
    return out


def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        return lambda t: an.polarity_scores(t)["compound"]
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    build_sentiment_history()
