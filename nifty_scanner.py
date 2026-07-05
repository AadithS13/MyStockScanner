from __future__ import annotations
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

# read lazily in send_email so --dry-run works without the secret
TO_EMAIL = "aadithsuresh10@gmail.com"
FROM_EMAIL = "onboarding@resend.dev"

NSE_CSV_PRIMARY = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_CSV_FALLBACK = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"
BHAVCOPY_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def get_nifty500_symbols() -> set[str]:
    for url in [NSE_CSV_PRIMARY, NSE_CSV_FALLBACK]:
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            symbols = set(df["Symbol"].dropna().str.strip())
            print(f"Fetched {len(symbols)} symbols from {url}")
            return symbols
        except Exception as e:
            print(f"Failed {url}: {e}")
    raise RuntimeError("Could not fetch Nifty 500 symbol list")


def fetch_bhavcopy(date: datetime) -> pd.DataFrame | None:
    """Download bhavcopy for a given date. Returns None if market was closed."""
    url = BHAVCOPY_URL.format(date=date.strftime("%d%m%Y"))
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        df["SYMBOL"] = df["SYMBOL"].str.strip()
        eq = df[df["SERIES"].str.strip() == "EQ"][["SYMBOL", "CLOSE_PRICE"]].copy()
        eq["CLOSE_PRICE"] = pd.to_numeric(eq["CLOSE_PRICE"], errors="coerce")
        print(f"Bhavcopy {date.strftime('%d-%b-%Y')}: {len(eq)} EQ records")
        return eq
    except Exception as e:
        print(f"Bhavcopy fetch failed for {date.strftime('%d-%m-%Y')}: {e}")
        return None


def last_trading_day(before: datetime, target_weekday: int = 4) -> tuple[datetime, pd.DataFrame]:
    """Walk back from `before` to find the most recent `target_weekday` (4=Friday) with a valid bhavcopy."""
    candidate = before - timedelta(days=1)
    while True:
        if candidate.weekday() == target_weekday:
            df = fetch_bhavcopy(candidate)
            if df is not None:
                return candidate, df
        candidate -= timedelta(days=1)
        if (before - candidate).days > 30:
            raise RuntimeError("Could not find a valid trading Friday in the last 30 days")


def get_swing_digest() -> tuple[pd.DataFrame | None, dict | None]:
    """Top swing Buy calls + live track record, from the committed artifacts.

    Reads data/n500_cache.parquet (refreshed nightly by CI) so this adds no
    extra NSE downloads to the email job. Best-effort: returns (None, None)
    when artifacts are missing so the gainers email still goes out.
    """
    try:
        from signals import generate_signals
        from swing_journal import annotate_with_learning, swing_scorecard

        cache = os.path.join(os.path.dirname(__file__), "data", "n500_cache.parquet")
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        closes = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        volumes = df.pivot_table(index="date", columns="symbol", values="volume").sort_index()

        nifty500 = get_nifty500_symbols()
        sig = generate_signals(closes, volumes, nifty500)
        buys = sig[sig["Signal"].str.contains("Buy", na=False)].head(8)
        buys = annotate_with_learning(buys)

        score = swing_scorecard()
        return buys, (score if score.get("n", 0) else None)
    except Exception as e:  # noqa: BLE001
        print(f"Swing digest skipped: {e}")
        return None, None


def main():
    dry_run = "--dry-run" in sys.argv
    today = datetime.today()

    nifty500 = get_nifty500_symbols()

    print("Finding last trading Friday...")
    fri1_date, bhavcopy1 = last_trading_day(today, target_weekday=4)

    print("Finding prior trading Friday...")
    fri2_date, bhavcopy2 = last_trading_day(fri1_date, target_weekday=4)

    print(f"Comparing {fri2_date.strftime('%d %b %Y')} → {fri1_date.strftime('%d %b %Y')}")

    merged = bhavcopy1.merge(bhavcopy2, on="SYMBOL", suffixes=("_last", "_prev"))
    merged = merged[merged["SYMBOL"].isin(nifty500)]
    merged = merged[(merged["CLOSE_PRICE_prev"] > 0) & (merged["CLOSE_PRICE_last"] > 0)]
    merged["pct_gain"] = (merged["CLOSE_PRICE_last"] - merged["CLOSE_PRICE_prev"]) / merged["CLOSE_PRICE_prev"] * 100

    top10 = merged.nlargest(10, "pct_gain").reset_index(drop=True)
    print(f"Top 10 computed from {len(merged)} matched Nifty 500 stocks")

    if top10.empty:
        print("No data — aborting")
        sys.exit(1)

    buys, score = get_swing_digest()
    if buys is not None:
        print(f"Swing digest: {len(buys)} Buy calls"
              + (f", track record n={score['n']}" if score else ""))

    subject = f"Trading Digest — Week of {fri1_date.strftime('%d %b %Y')}"
    html = build_html(top10, fri1_date, fri2_date, buys, score)

    if dry_run:
        out = "/tmp/digest_preview.html"
        with open(out, "w") as f:
            f.write(html)
        print(f"DRY RUN — wrote {out}, not sending")
        return

    send_email(subject, html)
    print("Done.")


TH = ("padding:9px 12px;font-size:11px;color:#8a919c;font-weight:600;"
      "text-transform:uppercase;letter-spacing:.4px;")
TD = "padding:10px 12px;font-size:13px;color:#374151;"


def _swing_section(buys: pd.DataFrame | None, score: dict | None) -> str:
    if buys is None or buys.empty:
        return ""

    strip = ""
    if score:
        strip = f"""
        <tr><td style="padding:0 32px 4px;">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td style="background:#f6f7f9;border-radius:10px;padding:12px 16px;">
              <span style="font-size:12px;color:#8a919c;">Track record ({score['n']} graded calls):</span>
              <span style="font-size:13px;color:#111827;font-weight:600;"> {score['win_rate']:.0%} win</span>
              <span style="font-size:12px;color:#8a919c;"> · {score['target_rate']:.0%} hit target · avg </span>
              <span style="font-size:13px;color:{'#0a7a4b' if score['avg_ret'] >= 0 else '#b3261e'};font-weight:600;">{score['avg_ret']:+.2%}</span>
              <span style="font-size:12px;color:#8a919c;"> vs market {score['market_ret']:+.2%}</span>
            </td>
          </tr></table>
        </td></tr>"""

    rows = ""
    for _, r in buys.iterrows():
        hist = r.get("Hist Win %")
        hist_txt = f"{hist:.1f}%" if isinstance(hist, (int, float)) else "–"
        rows += f"""
        <tr style="border-bottom:1px solid #eef0f3;">
          <td style="{TD}font-weight:600;color:#111827;">{r['Symbol']}</td>
          <td style="{TD}text-align:right;">₹{r['Price (₹)']:,.2f}</td>
          <td style="{TD}"><span style="background:#e8f5ee;color:#0a7a4b;font-size:11px;padding:2px 10px;border-radius:999px;white-space:nowrap;">{str(r['Signal']).replace('🟢 ', '')} · {r['Score']}</span></td>
          <td style="{TD}text-align:right;">₹{r['Target (₹)']:,.2f}</td>
          <td style="{TD}text-align:right;">₹{r['Stop Loss (₹)']:,.2f}</td>
          <td style="{TD}text-align:right;color:#0a7a4b;font-weight:600;">{hist_txt}</td>
        </tr>"""

    return f"""
        <tr><td style="padding:20px 32px 4px;">
          <h2 style="margin:0;font-size:16px;color:#111827;">Swing setups for this week</h2>
          <p style="margin:4px 0 0;font-size:12px;color:#8a919c;">2-week horizon · 'Hist win' is learned from past graded calls in the same score bucket</p>
        </td></tr>
        {strip}
        <tr><td style="padding:8px 32px 4px;">
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
            <thead><tr style="border-bottom:2px solid #e2e8f0;">
              <th style="{TH}text-align:left;">Symbol</th>
              <th style="{TH}text-align:right;">Price</th>
              <th style="{TH}text-align:left;">Signal</th>
              <th style="{TH}text-align:right;">Target</th>
              <th style="{TH}text-align:right;">Stop</th>
              <th style="{TH}text-align:right;">Hist win</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </td></tr>"""


def build_html(top10: pd.DataFrame, fri1: datetime, fri2: datetime,
               buys: pd.DataFrame | None = None, score: dict | None = None) -> str:
    rows_html = ""
    for rank, row in enumerate(top10.itertuples(), 1):
        color = "#0a7a4b" if row.pct_gain >= 0 else "#b3261e"
        arrow = "▲" if row.pct_gain >= 0 else "▼"
        rows_html += f"""
        <tr style="border-bottom:1px solid #eef0f3;">
          <td style="{TD}text-align:center;color:#8a919c;">{rank}</td>
          <td style="{TD}font-weight:600;color:#111827;">{row.SYMBOL}</td>
          <td style="{TD}text-align:right;">₹{row.CLOSE_PRICE_prev:,.2f}</td>
          <td style="{TD}text-align:right;">₹{row.CLOSE_PRICE_last:,.2f}</td>
          <td style="{TD}text-align:right;font-weight:600;color:{color};">{arrow} {abs(row.pct_gain):.2f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f2f5;padding:32px 0;">
    <tr><td align="center">
      <table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:14px;overflow:hidden;">
        <tr>
          <td style="background:#111418;padding:24px 32px;">
            <table cellpadding="0" cellspacing="0"><tr>
              <td style="background:#00c853;border-radius:9px;width:36px;height:36px;text-align:center;vertical-align:middle;color:#04240f;font-weight:700;font-size:14px;">AT</td>
              <td style="padding-left:12px;">
                <div style="color:#eef2f6;font-size:18px;font-weight:600;">Weekly Trading Digest</div>
                <div style="color:#8a95a1;font-size:12px;margin-top:2px;">{fri2.strftime('%d %b')} → {fri1.strftime('%d %b %Y')} · NSE Nifty 500</div>
              </td>
            </tr></table>
          </td>
        </tr>
        {_swing_section(buys, score)}
        <tr><td style="padding:20px 32px 4px;">
          <h2 style="margin:0;font-size:16px;color:#111827;">Top 10 weekly gainers</h2>
        </td></tr>
        <tr>
          <td style="padding:8px 32px 8px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
              <thead>
                <tr style="border-bottom:2px solid #e2e8f0;">
                  <th style="{TH}text-align:center;">#</th>
                  <th style="{TH}text-align:left;">Symbol</th>
                  <th style="{TH}text-align:right;">Prev Fri</th>
                  <th style="{TH}text-align:right;">Last Fri</th>
                  <th style="{TH}text-align:right;">% Gain</th>
                </tr>
              </thead>
              <tbody>{rows_html}
              </tbody>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:14px 32px 26px;">
            <p style="margin:0;font-size:11px;color:#8a919c;text-align:center;">
              NSE Bhavcopy data · Targets/stops are technical levels, hist win rates are learned from graded outcomes · Not investment advice.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_email(subject: str, html: str):
    resend_key = os.environ["RESEND_API_KEY"]
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
        json={"from": FROM_EMAIL, "to": TO_EMAIL, "subject": subject, "html": html},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Email sent: {resp.json()}")


if __name__ == "__main__":
    main()
