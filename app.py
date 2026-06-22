import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from data import get_nifty500_symbols, get_price_history, get_stock_ohlcv
from signals import generate_signals, compute_rsi, compute_rsi_series, compute_macd

st.set_page_config(
    page_title="My Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stHeader"] { background: transparent; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #1e293b; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; }
.stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }
div[data-testid="metric-container"] { background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 My Trading Dashboard")
st.caption(f"Nifty 500 · NSE Bhavcopy · {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}")

with st.spinner("Loading 60 days of market data (first load takes ~60s, then cached)..."):
    nifty500 = get_nifty500_symbols()
    history = get_price_history(n_days=60)

closes = history["closes"]
volumes = history["volumes"]
raw = history["raw"]

if closes.empty:
    st.error("Could not load market data from NSE. Please try again in a few minutes.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊  Overview", "🎯  Swing Signals", "🔍  Stock Detail", "💼  My Portfolio"])

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    if len(closes) >= 6:
        last_close = closes.iloc[-1]
        prev_close = closes.iloc[-6]
        week_chg = ((last_close - prev_close) / prev_close * 100).dropna()
        week_chg = week_chg[week_chg.index.isin(nifty500)]

        top_sym = week_chg.idxmax()
        bot_sym = week_chg.idxmin()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks Tracked", len(week_chg))
        c2.metric("Top Gainer", top_sym, f"+{week_chg.max():.2f}%")
        c3.metric("Biggest Drop", bot_sym, f"{week_chg.min():.2f}%")
        c4.metric("Gainers vs Losers", f"{(week_chg > 0).sum()} / {(week_chg < 0).sum()}")

        st.divider()
        col_g, col_l = st.columns(2)

        with col_g:
            st.markdown("#### 🟢 Top 10 Gainers this Week")
            top10 = week_chg.nlargest(10).reset_index()
            top10.columns = ["Symbol", "Week (%)"]
            top10["Week (%)"] = top10["Week (%)"].round(2)
            st.dataframe(top10, use_container_width=True, hide_index=True)

        with col_l:
            st.markdown("#### 🔴 Top 10 Losers this Week")
            bot10 = week_chg.nsmallest(10).reset_index()
            bot10.columns = ["Symbol", "Week (%)"]
            bot10["Week (%)"] = bot10["Week (%)"].round(2)
            st.dataframe(bot10, use_container_width=True, hide_index=True)

# ── Tab 2: Swing Signals ───────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 🎯 Swing Trade Signals — 2-week horizon")
    st.caption("RSI + 20-day MA + momentum + volume ratio. Not financial advice.")

    with st.spinner("Computing signals for all 500 stocks..."):
        signals_df = generate_signals(closes, volumes, nifty500)

    fc, sc = st.columns([3, 2])
    with fc:
        sig_opts = ["🟢 Strong Buy", "🟢 Buy", "🟡 Watch", "⚪ Neutral", "🔴 Overbought"]
        selected_sigs = st.multiselect("Filter signals", sig_opts, default=["🟢 Strong Buy", "🟢 Buy"])
    with sc:
        sort_col = st.selectbox("Sort by", ["Score", "Upside (%)", "RSI", "R/R Ratio", "vs 20DMA (%)"])

    filtered = signals_df[signals_df["Signal"].isin(selected_sigs)] if selected_sigs else signals_df
    asc = sort_col == "RSI"
    filtered = filtered.sort_values(sort_col, ascending=asc).reset_index(drop=True)

    display_cols = ["Symbol", "Price (₹)", "Signal", "Score", "RSI", "MACD",
                    "vs 20DMA (%)", "vs 50DMA (%)", "Vol Ratio",
                    "Target (₹)", "Upside (%)", "Stop Loss (₹)", "R/R Ratio", "Timeline"]

    def _color_signal(val):
        if "Strong Buy" in str(val):
            return "color: #22c55e; font-weight: bold"
        if "Buy" in str(val):
            return "color: #4ade80; font-weight: bold"
        if "Overbought" in str(val):
            return "color: #f87171; font-weight: bold"
        if "Watch" in str(val):
            return "color: #fbbf24; font-weight: bold"
        return "color: #94a3b8"

    def _color_macd(val):
        if val == "Bullish":
            return "color: #4ade80"
        return "color: #f87171"

    def _color_num(val):
        try:
            return f"color: {'#4ade80' if float(val) >= 0 else '#f87171'}"
        except Exception:
            return ""

    styled = (
        filtered[display_cols].style
        .map(_color_signal, subset=["Signal"])
        .map(_color_macd, subset=["MACD"])
        .map(_color_num, subset=["vs 20DMA (%)", "vs 50DMA (%)", "Upside (%)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)
    st.caption(
        f"Showing {len(filtered)} of {len(signals_df)} Nifty 500 stocks  ·  "
        "Target = next resistance level  ·  Timeline = estimated based on momentum  ·  Not financial advice"
    )

# ── Tab 3: Stock Detail ────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🔍 Stock Detail")
    symbol_list = sorted(nifty500)
    default_idx = symbol_list.index("RELIANCE") if "RELIANCE" in symbol_list else 0
    selected = st.selectbox("Select symbol", symbol_list, index=default_idx)

    ohlcv = get_stock_ohlcv(selected, raw)

    if ohlcv.empty:
        st.warning(f"No data found for {selected} in the last 30 days.")
    else:
        ohlcv["ma20"] = ohlcv["close"].rolling(20).mean()
        ohlcv["ma50"] = ohlcv["close"].rolling(50).mean()
        ohlcv["rsi"] = compute_rsi_series(ohlcv["close"])
        macd_line, signal_line, macd_hist = compute_macd(ohlcv["close"])
        ohlcv["macd"] = macd_line
        ohlcv["macd_signal"] = signal_line
        ohlcv["macd_hist"] = macd_hist

        cur = ohlcv["close"].iloc[-1]
        prev_w = ohlcv["close"].iloc[-6] if len(ohlcv) >= 6 else ohlcv["close"].iloc[0]
        week_pct = (cur - prev_w) / prev_w * 100
        rsi_val = ohlcv["rsi"].dropna().iloc[-1] if ohlcv["rsi"].dropna().shape[0] > 0 else float("nan")
        ma20_val = ohlcv["ma20"].dropna().iloc[-1] if ohlcv["ma20"].dropna().shape[0] > 0 else float("nan")
        vs_ma = (cur - ma20_val) / ma20_val * 100 if not pd.isna(ma20_val) else float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"₹{cur:,.2f}")
        m2.metric("Week Change", f"{week_pct:+.2f}%")
        m3.metric("RSI (14)", f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "–")
        m4.metric("vs 20-day MA", f"{vs_ma:+.2f}%" if not pd.isna(vs_ma) else "–")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
            subplot_titles=("Price", "RSI (14)", "MACD"),
        )

        # Price + MAs
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["close"], name="Close",
                       line=dict(color="#60a5fa", width=2), fill="tozeroy",
                       fillcolor="rgba(96,165,250,0.06)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["ma20"], name="20DMA",
                       line=dict(color="#f59e0b", width=1.5, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["ma50"], name="50DMA",
                       line=dict(color="#ec4899", width=1.5, dash="dot")),
            row=1, col=1,
        )

        # RSI
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["rsi"], name="RSI",
                       line=dict(color="#a78bfa", width=1.5)),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#f87171", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", opacity=0.5, row=2, col=1)

        # MACD
        colors = ["#4ade80" if v >= 0 else "#f87171" for v in ohlcv["macd_hist"].fillna(0)]
        fig.add_trace(
            go.Bar(x=ohlcv["date"], y=ohlcv["macd_hist"], name="Histogram",
                   marker_color=colors, opacity=0.6),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["macd"], name="MACD",
                       line=dict(color="#60a5fa", width=1.5)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ohlcv["date"], y=ohlcv["macd_signal"], name="Signal",
                       line=dict(color="#f59e0b", width=1.5)),
            row=3, col=1,
        )

        fig.update_layout(
            title=f"{selected}  ·  Last {len(ohlcv)} trading days",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#94a3b8"),
            height=580,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        fig.update_xaxes(gridcolor="#1e293b", zeroline=False)
        fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Portfolio ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 💼 My Portfolio")
    st.caption("Enter holdings below — one per line. Refreshes live against today's prices.")

    col_inp, col_help = st.columns([3, 1])
    with col_help:
        st.markdown("""
**Format:**
```
SYMBOL, Qty, Buy Price
```
Example:
```
NIACL, 50, 152.80
HFCL, 100, 171.86
TRENT, 5, 2755.30
```
""")

    with col_inp:
        holdings_txt = st.text_area("Holdings", height=200, placeholder="NIACL, 50, 152.80\nHFCL, 100, 171.86")

    if holdings_txt.strip():
        rows, errors = [], []
        for line in holdings_txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                try:
                    rows.append({"Symbol": parts[0].upper(), "Qty": int(parts[1]), "Buy Price (₹)": float(parts[2])})
                except ValueError:
                    errors.append(line)
            else:
                errors.append(line)

        if errors:
            st.warning(f"Skipped unreadable lines: {errors}")

        if rows:
            latest = closes.iloc[-1] if not closes.empty else pd.Series(dtype=float)
            pf = pd.DataFrame(rows)
            pf["Current Price (₹)"] = pf["Symbol"].map(lambda s: round(latest.get(s, float("nan")), 2))
            pf["Invested (₹)"] = (pf["Buy Price (₹)"] * pf["Qty"]).round(2)
            pf["Value (₹)"] = (pf["Current Price (₹)"] * pf["Qty"]).round(2)
            pf["P&L (₹)"] = (pf["Value (₹)"] - pf["Invested (₹)"]).round(2)
            pf["P&L (%)"] = ((pf["P&L (₹)"] / pf["Invested (₹)"]) * 100).round(2)

            total_inv = pf["Invested (₹)"].sum()
            total_val = pf["Value (₹)"].sum()
            total_pnl = pf["P&L (₹)"].sum()
            total_pct = total_pnl / total_inv * 100 if total_inv else 0

            s1, s2, s3 = st.columns(3)
            s1.metric("Total Invested", f"₹{total_inv:,.0f}")
            s2.metric("Current Value", f"₹{total_val:,.0f}", f"₹{total_pnl:+,.0f}")
            s3.metric("Overall P&L", f"₹{total_pnl:+,.0f}", f"{total_pct:+.2f}%")

            def _pnl_color(val):
                try:
                    return f"color: {'#4ade80' if float(val) >= 0 else '#f87171'}; font-weight: bold"
                except Exception:
                    return ""

            styled_pf = pf.style.map(_pnl_color, subset=["P&L (₹)", "P&L (%)"])
            st.dataframe(styled_pf, use_container_width=True, hide_index=True)
