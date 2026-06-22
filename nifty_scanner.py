import os
import sys
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

RESEND_API_KEY = os.environ["RESEND_API_KEY"]
TO_EMAIL = "aadithsuresh10@gmail.com"
FROM_EMAIL = "onboarding@resend.dev"

NSE_CSV_PRIMARY = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE_CSV_FALLBACK = "https://raw.githubusercontent.com/kprohith/nse-stock-analysis/master/ind_nifty500list.csv"


def get_nifty500_symbols():
    for url in [NSE_CSV_PRIMARY, NSE_CSV_FALLBACK]:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            symbols = df["Symbol"].dropna().tolist()
            print(f"Fetched {len(symbols)} symbols from {url}")
            return symbols
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
    raise RuntimeError("Could not fetch Nifty 500 symbol list from any source")


def last_friday(ref: datetime) -> datetime:
    days_back = (ref.weekday() - 4) % 7
    if days_back == 0:
        days_back = 7
    return ref - timedelta(days=days_back)


def get_friday_closes(symbols: list[str]) -> dict[str, tuple[float, float]]:
    today = datetime.today()
    fri1 = last_friday(today)
    fri2 = last_friday(fri1 - timedelta(days=1))
    print(f"Last Friday: {fri1.date()}, Prior Friday: {fri2.date()}")

    start = (fri2 - timedelta(days=3)).strftime("%Y-%m-%d")
    end = (fri1 + timedelta(days=2)).strftime("%Y-%m-%d")

    closes: dict[str, tuple[float, float]] = {}
    batch_size = 50

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        tickers = [s + ".NS" for s in batch]
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            close = raw["Close"] if "Close" in raw.columns else raw
            for sym, ticker in zip(batch, tickers):
                try:
                    series = close[ticker].dropna()
                    fri1_close = series.loc[series.index.normalize() <= pd.Timestamp(fri1.date())].iloc[-1]
                    fri2_close = series.loc[series.index.normalize() <= pd.Timestamp(fri2.date())].iloc[-1]
                    closes[sym] = (float(fri2_close), float(fri1_close))
                except Exception:
                    pass
        except Exception as e:
            print(f"Batch {i//batch_size + 1} failed: {e}")

    print(f"Successfully fetched closes for {len(closes)}/{len(symbols)} symbols")
    return closes


def rank_top10(closes: dict[str, tuple[float, float]]) -> list[dict]:
    rows = []
    for sym, (prev, last) in closes.items():
        if prev > 0:
            pct = (last - prev) / prev * 100
            rows.append({"symbol": sym, "prev_close": prev, "last_close": last, "pct_gain": pct})
    rows.sort(key=lambda r: r["pct_gain"], reverse=True)
    return rows[:10]


def build_html(top10: list[dict], fri1: datetime, fri2: datetime) -> str:
    rows_html = ""
    for rank, r in enumerate(top10, 1):
        color = "#16a34a" if r["pct_gain"] >= 0 else "#dc2626"
        arrow = "▲" if r["pct_gain"] >= 0 else "▼"
        rows_html += f"""
        <tr style="border-bottom:1px solid #e5e7eb;">
          <td style="padding:12px 16px;text-align:center;font-weight:600;color:#6b7280;">{rank}</td>
          <td style="padding:12px 16px;font-weight:700;letter-spacing:0.5px;">{r['symbol']}</td>
          <td style="padding:12px 16px;text-align:right;color:#374151;">₹{r['prev_close']:,.2f}</td>
          <td style="padding:12px 16px;text-align:right;color:#374151;">₹{r['last_close']:,.2f}</td>
          <td style="padding:12px 16px;text-align:right;font-weight:700;color:{color};">{arrow} {abs(r['pct_gain']):.2f}%</td>
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
              Prices sourced from Yahoo Finance. Not investment advice.
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


def main():
    today = datetime.today()
    fri1 = last_friday(today)
    fri2 = last_friday(fri1 - timedelta(days=1))

    symbols = get_nifty500_symbols()
    closes = get_friday_closes(symbols)
    top10 = rank_top10(closes)

    if not top10:
        print("No data — aborting email send")
        sys.exit(1)

    subject = f"Nifty 500 Top 10 Gainers — Week of {fri1.strftime('%d %b %Y')}"
    html = build_html(top10, fri1, fri2)
    send_email(subject, html)
    print("Done.")


if __name__ == "__main__":
    main()
