import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

RESEND_API_KEY = os.environ["RESEND_API_KEY"]
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


def main():
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

    subject = f"Nifty 500 Top 10 Gainers — Week of {fri1_date.strftime('%d %b %Y')}"
    html = build_html(top10, fri1_date, fri2_date)
    send_email(subject, html)
    print("Done.")


def build_html(top10: pd.DataFrame, fri1: datetime, fri2: datetime) -> str:
    rows_html = ""
    for rank, row in enumerate(top10.itertuples(), 1):
        color = "#16a34a" if row.pct_gain >= 0 else "#dc2626"
        arrow = "▲" if row.pct_gain >= 0 else "▼"
        rows_html += f"""
        <tr style="border-bottom:1px solid #e5e7eb;">
          <td style="padding:12px 16px;text-align:center;font-weight:600;color:#6b7280;">{rank}</td>
          <td style="padding:12px 16px;font-weight:700;letter-spacing:0.5px;">{row.SYMBOL}</td>
          <td style="padding:12px 16px;text-align:right;color:#374151;">₹{row.CLOSE_PRICE_prev:,.2f}</td>
          <td style="padding:12px 16px;text-align:right;color:#374151;">₹{row.CLOSE_PRICE_last:,.2f}</td>
          <td style="padding:12px 16px;text-align:right;font-weight:700;color:{color};">{arrow} {abs(row.pct_gain):.2f}%</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 6px rgba(0,0,0,0.07);">
        <tr>
          <td style="background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);padding:28px 32px;">
            <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;">Nifty 500 — Top 10 Weekly Gainers</h1>
            <p style="margin:6px 0 0;color:#bfdbfe;font-size:14px;">{fri2.strftime('%d %b %Y')} → {fri1.strftime('%d %b %Y')}</p>
          </td>
        </tr>
        <tr>
          <td style="padding:24px 32px 8px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
              <thead>
                <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
                  <th style="padding:10px 16px;text-align:center;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;">#</th>
                  <th style="padding:10px 16px;text-align:left;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;">Symbol</th>
                  <th style="padding:10px 16px;text-align:right;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;">Prev Fri</th>
                  <th style="padding:10px 16px;text-align:right;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;">Last Fri</th>
                  <th style="padding:10px 16px;text-align:right;font-size:12px;color:#6b7280;font-weight:600;text-transform:uppercase;">% Gain</th>
                </tr>
              </thead>
              <tbody>{rows_html}
              </tbody>
            </table>
          </td>
        </tr>
        <tr>
          <td style="padding:16px 32px 28px;">
            <p style="margin:0;font-size:12px;color:#9ca3af;text-align:center;">
              Prices sourced from NSE Bhavcopy. Not investment advice.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_email(subject: str, html: str):
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        json={"from": FROM_EMAIL, "to": TO_EMAIL, "subject": subject, "html": html},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Email sent: {resp.json()}")


if __name__ == "__main__":
    main()
