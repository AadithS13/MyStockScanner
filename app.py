import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from data import get_nifty500_symbols, get_price_history, get_stock_ohlcv
from signals import generate_signals, compute_rsi, compute_rsi_series, compute_macd

try:
    import ai_data
    from defence import NAMES as DEFENCE_NAMES
    _AI_OK = ai_data.ai_available()
except Exception:  # noqa: BLE001 — keep the rest of the app working without the ML extras
    _AI_OK = False

st.set_page_config(
    page_title="My AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS theme: dark fintech ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════ DARK FINTECH THEME ══════════
   page #111418 · sidebar #0c0f12 · cards #171c22 · accent #00c853
   text #eef2f6 / #8a95a1 / #5c6672 · negative #ff5252 · warn #ffb300 */

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #111418 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #eef2f6 !important;
}
[data-testid="stMainBlockContainer"], .main, .block-container,
[data-testid="stVerticalBlock"] { background: transparent !important; }

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stMainBlockContainer"],
.main .block-container {
    padding: 1.6rem 2.2rem 2.2rem 2.2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0f12 !important;
    border-right: 1px solid #1e242b !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.4rem 0.9rem !important; }

/* nav items: full-width rows, fixed icon column, quiet grey → green when active */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 3px !important; flex-direction: column !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 10px 12px !important;
    border-radius: 9px !important;
    color: #8a95a1 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: background .15s ease, color .15s ease !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #171c22 !important;
    color: #c6ced6 !important;
}
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(0,200,83,.10) !important;
    color: #00c853 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }
/* label content fills the row; icon gets a fixed, aligned column */
[data-testid="stSidebar"] .stRadio > div > label > div {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stRadio label [data-testid="stIconMaterial"],
[data-testid="stSidebar"] .stRadio label span[class*="material-symbols"] {
    font-size: 19px !important;
    width: 30px !important;
    flex-shrink: 0 !important;
    color: inherit !important;
}
[data-testid="stSidebar"] .stRadio label p {
    color: inherit !important;
    font-size: 14px !important;
    margin: 0 !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e242b !important;
    margin: 0.7rem 0 !important;
}

/* ── Metric cards: flat, borderless, calm ── */
[data-testid="metric-container"] {
    background: #171c22 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
    transition: background .18s ease, transform .18s ease !important;
}
[data-testid="metric-container"]:hover {
    background: #1a2028 !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] p {
    color: #8a95a1 !important;
    font-size: 11.5px !important;
    letter-spacing: .2px !important;
}
[data-testid="stMetricValue"] {
    color: #eef2f6 !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricDeltaIcon-Up"] { color: #00c853 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #ff5252 !important; }

/* ── Dataframes: flat card, no glow ── */
.stDataFrame > div, .stDataFrame iframe {
    border: none !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #171c22 !important;
}
.stDataFrame { border-radius: 10px !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div,
.stNumberInput > div > div {
    background: #171c22 !important;
    border: 1px solid #232a32 !important;
    border-radius: 8px !important;
    color: #eef2f6 !important;
}
.stNumberInput input, .stSelectbox input { color: #eef2f6 !important; }
.stNumberInput button { background: #1e242b !important; color: #8a95a1 !important; }
.stTextArea > div > div > textarea {
    background: #171c22 !important;
    border: 1px solid #232a32 !important;
    border-radius: 8px !important;
    color: #eef2f6 !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(0,200,83,.10) !important;
    color: #00c853 !important;
    border-radius: 999px !important;
}
.stMultiSelect [data-baseweb="tag"] span { color: #00c853 !important; }

/* ── Buttons: quiet outline, green on hover ── */
.stButton > button, .stDownloadButton > button {
    background: #171c22 !important;
    color: #c6ced6 !important;
    border: 1px solid #232a32 !important;
    border-radius: 8px !important;
    transition: all .18s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #00c853 !important;
    color: #00c853 !important;
    background: rgba(0,200,83,.06) !important;
}

/* ── Expanders: flat card ── */
[data-testid="stExpander"] {
    border: none !important;
    border-radius: 10px !important;
    background: #171c22 !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #8a95a1 !important;
    transition: color .15s ease !important;
}
[data-testid="stExpander"] summary:hover { color: #00c853 !important; }

/* ── Text ── */
h1,h2,h3,h4 {
    color: #eef2f6 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -.2px !important;
}
.stMarkdown p { color: #c6ced6 !important; }
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #5c6672 !important;
    font-size: 12px !important;
}
hr { border-color: #1e242b !important; }
.stSpinner > div { border-top-color: #00c853 !important; }
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Alerts ── */
.stAlert {
    background: #171c22 !important;
    border: 1px solid #232a32 !important;
    border-radius: 10px !important;
    color: #c6ced6 !important;
}

/* ── Motion: one soft entrance, nothing looping ── */
@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stMainBlockContainer"] > div > div:nth-child(-n+5) {
    animation: riseIn .35s cubic-bezier(.22,.9,.36,1) both;
}
[data-testid="stMainBlockContainer"] > div > div:nth-child(2) { animation-delay: .05s; }
[data-testid="stMainBlockContainer"] > div > div:nth-child(3) { animation-delay: .10s; }
[data-testid="stMainBlockContainer"] > div > div:nth-child(4) { animation-delay: .15s; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #111418; }
::-webkit-scrollbar-thumb { background: #2a323c; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3a444f; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:2px 4px 14px 4px;">
      <div style="width:34px;height:34px;border-radius:9px;background:#00c853;display:flex;
                  align-items:center;justify-content:center;color:#04240f;
                  font-weight:600;font-size:14px;flex-shrink:0;">AT</div>
      <div>
        <div style="font-size:15px;font-weight:600;color:#eef2f6;letter-spacing:-.2px;">AI Trader</div>
        <div style="font-size:10.5px;color:#5c6672;margin-top:1px;">NSE · Nifty 500</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [":material/dashboard: Overview", ":material/track_changes: Swing Signals", ":material/smart_toy: AI Lab — Defence",
         ":material/psychology: Feature Importance", ":material/candlestick_chart: Stock Detail", ":material/work: My Portfolio"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"""
    <div style="padding:0 4px;font-size:11px;color:#5c6672;line-height:1.9;">
      <div>{datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>
      <div>Data refreshes hourly</div>
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading 60 days of market data…"):
    nifty500 = get_nifty500_symbols()
    history = get_price_history(n_days=60)

closes  = history["closes"]
volumes = history["volumes"]
raw     = history["raw"]

if closes.empty:
    st.error("Could not load market data from NSE. Please try again in a few minutes.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────

# ── Overview ──────────────────────────────────────────────────────────────────
if page == ":material/dashboard: Overview":
    st.markdown("## 📊 Market Overview")
    st.caption(f"Week-on-week performance across Nifty 500 — as of {closes.index[-1].strftime('%d %b %Y')}")

    if len(closes) >= 6:
        last_close = closes.iloc[-1]
        prev_close = closes.iloc[-6]
        wchg = ((last_close - prev_close) / prev_close * 100).dropna()
        wchg = wchg[wchg.index.isin(nifty500)]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stocks Tracked", f"{len(wchg)}")
        c2.metric("Top Gainer", wchg.idxmax(), f"+{wchg.max():.2f}%")
        c3.metric("Biggest Drop", wchg.idxmin(), f"{wchg.min():.2f}%")
        c4.metric("Gainers / Losers", f"{(wchg>0).sum()} / {(wchg<0).sum()}")

        st.divider()
        col_g, col_l = st.columns(2)

        with col_g:
            st.markdown("#### 🟢 Top 10 Gainers")
            top10 = wchg.nlargest(10).reset_index()
            top10.columns = ["Symbol", "Week (%)"]
            top10["Week (%)"] = top10["Week (%)"].round(2)
            st.dataframe(
                top10.style.map(lambda v: "color:#00c853;font-weight:600", subset=["Week (%)"]),
                width="stretch", hide_index=True,
            )

        with col_l:
            st.markdown("#### 🔴 Top 10 Losers")
            bot10 = wchg.nsmallest(10).reset_index()
            bot10.columns = ["Symbol", "Week (%)"]
            bot10["Week (%)"] = bot10["Week (%)"].round(2)
            st.dataframe(
                bot10.style.map(lambda v: "color:#ff5252;font-weight:600", subset=["Week (%)"]),
                width="stretch", hide_index=True,
            )

# ── Swing Signals ─────────────────────────────────────────────────────────────
elif page == ":material/track_changes: Swing Signals":
    st.markdown("## 🎯 Swing Trade Signals")
    st.caption("RSI · MACD · 20DMA · 50DMA · Volume — 2-week swing horizon · Not financial advice")

    # ── Track record: how did past Buy calls actually do? ──
    import ai_data
    rec = ai_data.swing_record()
    if rec.get("n", 0) > 0:
        st.markdown("#### 📈 Track record — how past calls played out")
        st.caption(
            f"Every Buy/Strong-Buy call is logged and graded after its 2-week horizon "
            f"(did the close reach the target, hit the stop, or neither?). "
            f"{rec['n']} graded calls · {rec['first_date']} → {rec['last_date']}."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win rate", f"{rec['win_rate']:.0%}",
                  help="Share of calls that closed up or hit target within 2 weeks.")
        c2.metric("Target hit", f"{rec['target_rate']:.0%}",
                  help="Reached the projected target before the stop.")
        c3.metric("Avg return", f"{rec['avg_ret']:+.2%}",
                  delta=f"{rec['edge']:+.2%} vs market",
                  help="Average 2-week return of the calls vs an equal-weight Nifty-500 baseline.")
        c4.metric("Stopped out", f"{rec['stop_rate']:.0%}")

        # ── equity curve: what following every call would have compounded to ──
        val = ai_data.swing_validated()
        if len(val) >= 20:
            cohort = (val.groupby("made_on")
                      .agg(sys_ret=("realised_ret", "mean"),
                           mkt_ret=("market_ret", "mean"),
                           n=("symbol", "size"))
                      .sort_index())
            eq_sys = (1 + cohort["sys_ret"]).cumprod() - 1
            eq_mkt = (1 + cohort["mkt_ret"]).cumprod() - 1

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=list(cohort.index), y=(eq_mkt * 100).round(2),
                name="Market (equal-weight N500)", mode="lines",
                line=dict(color="#8a95a1", width=1.6, dash="dot"),
                hovertemplate="%{x}<br>Market: %{y:.2f}%<extra></extra>"))
            fig_eq.add_trace(go.Scatter(
                x=list(cohort.index), y=(eq_sys * 100).round(2),
                name="System calls (equal-weight)", mode="lines+markers",
                line=dict(color="#00c853", width=2.6, shape="spline"),
                marker=dict(size=6),
                fill="tonexty", fillcolor="rgba(0,200,83,.06)",
                customdata=cohort["n"],
                hovertemplate="%{x}<br>System: %{y:.2f}%<br>calls in cohort: "
                              "%{customdata}<extra></extra>"))
            fig_eq.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(l=10, r=10, t=36, b=10),
                title=dict(text="Compounded return — every call taken vs the market",
                           font=dict(size=13, color="#8a95a1")),
                legend=dict(orientation="h", y=1.14, x=0,
                            font=dict(size=11, color="#8a95a1"),
                            bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="rgba(255,255,255,.05)", title=None),
                yaxis=dict(gridcolor="rgba(255,255,255,.05)",
                           ticksuffix="%", title=None),
                hoverlabel=dict(bgcolor="#111", font=dict(color="#eef2f6")),
            )
            st.plotly_chart(fig_eq, width="stretch",
                            config={"displayModeBar": False})
            st.caption(
                "Each point = one batch of calls (equal-weighted, 2-week hold, "
                "graded at close). The gap between the lines IS the system's "
                "edge — when green is below grey, following the calls "
                "underperformed simply holding the market.")

        with st.expander("See the graded calls"):
            recent = rec.get("recent")
            if recent is not None and len(recent):
                show = recent[["made_on", "symbol", "signal", "entry", "target",
                               "stop", "outcome", "exit_close", "realised_ret",
                               "market_ret"]].copy()
                show = show.rename(columns={
                    "made_on": "Date", "symbol": "Symbol", "signal": "Signal",
                    "entry": "Entry", "target": "Target", "stop": "Stop",
                    "outcome": "Outcome", "exit_close": "Exit",
                    "realised_ret": "Return", "market_ret": "Market"})
                show["Return"] = (show["Return"].astype(float) * 100).round(2)
                show["Market"] = (show["Market"].astype(float) * 100).round(2)

                def _out(v):
                    return ("color:#00c853;font-weight:700" if v == "TARGET"
                            else "color:#ff5252;font-weight:600" if v == "STOP"
                            else "color:#8a95a1")
                def _ret(v):
                    try: return f"color:{'#00c853' if float(v) >= 0 else '#ff5252'}"
                    except: return ""
                st.dataframe(
                    show.sort_values("Date", ascending=False).style
                        .map(_out, subset=["Outcome"])
                        .map(_ret, subset=["Return", "Market"]),
                    width="stretch", hide_index=True, height=320,
                )
        st.divider()
    else:
        st.info("📈 Track record is building — calls get graded after their 2-week "
                "horizon, then their hit-rate and returns appear here.")

    with st.spinner("Computing signals for 500 stocks…"):
        signals_df = generate_signals(closes, volumes, nifty500)

    fcol, ccol, rcol = st.columns([2.4, 1, 1])
    with fcol:
        sig_opts = ["🟢 Strong Buy", "🟢 Buy", "🟡 Watch", "⚪ Neutral", "🔴 Overbought"]
        sel = st.multiselect("Filter by signal", sig_opts,
                             default=["🟢 Strong Buy", "🟢 Buy"])
    with ccol:
        capital = st.number_input("Capital (₹)", min_value=10_000, value=100_000,
                                  step=10_000,
                                  help="Total trading capital for position sizing.")
    with rcol:
        risk_pct_in = st.number_input("Risk per trade (%)", min_value=0.25,
                                      max_value=5.0, value=1.0, step=0.25,
                                      help="Max % of capital lost if the stop is hit "
                                           "(pros: 0.5–2%). Qty = capital×risk ÷ "
                                           "(entry − stop).")

    filtered = signals_df[signals_df["Signal"].isin(sel)] if sel else signals_df

    # risk-based position sizing: fixed-fractional (the professional standard)
    risk_amt = capital * risk_pct_in / 100
    per_share_risk = (filtered["Price (₹)"] - filtered["Stop Loss (₹)"]).clip(lower=0)
    qty = (risk_amt / per_share_risk.replace(0, float("nan"))).fillna(0)
    # cap so a tight stop can't demand more capital than we have
    qty = qty.clip(upper=(capital / filtered["Price (₹)"]).fillna(0)).astype(int)
    filtered = filtered.assign(**{"Qty": qty,
                                  "Position (₹)": (qty * filtered["Price (₹)"]).round(0)})
    # initial order; users re-sort by clicking headers (Excel-style)
    filtered = filtered.sort_values("Score", ascending=False).reset_index(drop=True)

    # ── learned stats beside each prediction ──
    learn = ai_data.swing_learning()
    if learn.get("n", 0) > 0:
        from swing_journal import annotate_with_learning
        filtered = annotate_with_learning(filtered)
        bucket_bits = []
        for b, s in sorted(learn["buckets"].items(),
                           key=lambda kv: kv[0], reverse=True):
            bucket_bits.append(
                f"**Score {b}** → {s['win_rate']:.0%} win · "
                f"{s['avg_ret']:+.1%} avg (n={s['n']})")
        st.markdown(
            f"🧠 **Learned from {learn['n']} graded calls** — historical odds by "
            f"score bucket:&nbsp;&nbsp;" + " &nbsp;·&nbsp; ".join(bucket_bits))
        st.caption(
            "『Hist Win %』and『Hist Avg %』beside each signal are the smoothed "
            "outcomes of past calls in the same score bucket — they update "
            "automatically every time the weekly loop grades more calls.")

    COLS = ["Symbol", "Price (₹)", "Signal", "Score", "Hist Win %", "Hist Avg (%)",
            "RSI", "MACD", "vs 20DMA (%)", "vs 50DMA (%)", "Vol Ratio",
            "Target (₹)", "Upside (%)", "Stop Loss (₹)", "R/R Ratio",
            "Qty", "Position (₹)", "Timeline", "Why"]
    COLS = [c for c in COLS if c in filtered.columns]

    def _sig(v):
        if "Strong" in str(v): return "color:#00c853;font-weight:700"
        if "Buy" in str(v):    return "color:#00c853;font-weight:600"
        if "Overbought" in str(v): return "color:#ff5252;font-weight:600"
        if "Watch" in str(v):  return "color:#ffb300;font-weight:600"
        return "color:#8a95a1"

    def _macd(v): return "color:#00c853" if v == "Bullish" else "color:#ff5252"
    def _pct(v):
        try: return f"color:{'#00c853' if float(v)>=0 else '#ff5252'}"
        except: return ""

    # clean number formatting (display only — underlying stays numeric so
    # header-click sorting stays numeric, not lexical)
    def _f(fs):
        def fn(v):
            try: return fs.format(v)
            except (ValueError, TypeError): return v
        return fn
    num_fmt = {c: _f(s) for c, s in {
        "Price (₹)": "₹{:,.2f}", "Score": "{:.0f}", "RSI": "{:.1f}",
        "vs 20DMA (%)": "{:+.2f}", "vs 50DMA (%)": "{:+.2f}", "Vol Ratio": "{:.2f}×",
        "Target (₹)": "₹{:,.2f}", "Upside (%)": "{:+.2f}",
        "Stop Loss (₹)": "₹{:,.2f}", "R/R Ratio": "{:.2f}",
        "Hist Win %": "{:.1f}", "Hist Avg (%)": "{:+.2f}",
        "Qty": "{:,.0f}", "Position (₹)": "₹{:,.0f}",
    }.items() if c in COLS}

    def _win(v):
        try:
            return ("color:#00c853;font-weight:600" if float(v) >= 50
                    else "color:#ff5252")
        except (ValueError, TypeError):
            return ""

    st.markdown("#### 🎯 Live signals — today's setups")
    st.caption("💡 Click any column header to sort (click again to reverse) — like Excel.")
    pct_cols = [c for c in ["vs 20DMA (%)", "vs 50DMA (%)", "Upside (%)",
                            "Hist Avg (%)"] if c in COLS]
    styled = (filtered[COLS].style
              .format(num_fmt)
              .map(_sig, subset=["Signal"])
              .map(_macd, subset=["MACD"])
              .map(_pct, subset=pct_cols))
    if "Hist Win %" in COLS:
        styled = styled.map(_win, subset=["Hist Win %"])
    st.dataframe(styled, width="stretch", hide_index=True, height=520)
    st.caption(f"Showing {len(filtered)} of {len(signals_df)} stocks  ·  "
               f"Target = next resistance  ·  Stop = volatility-aware "
               f"(2.5×σ₂₀, clamped 3–7%, respects recent low)  ·  "
               f"Qty = ₹{risk_amt:,.0f} risk ÷ stop distance  ·  "
               f"'Why' = factor breakdown behind the score")

# ── Stock Detail ──────────────────────────────────────────────────────────────
elif page == ":material/candlestick_chart: Stock Detail":
    st.markdown("## 🔍 Stock Detail")

    syms = sorted(nifty500)
    sel = st.selectbox("Symbol", syms, index=syms.index("RELIANCE") if "RELIANCE" in syms else 0)
    ohlcv = get_stock_ohlcv(sel, raw)

    if ohlcv.empty:
        st.warning(f"No data for {sel}.")
    else:
        ohlcv["ma20"]       = ohlcv["close"].rolling(20, min_periods=15).mean()
        ohlcv["ma50"]       = ohlcv["close"].rolling(50, min_periods=40).mean()
        ohlcv["rsi"]        = compute_rsi_series(ohlcv["close"])
        ml, sl, mh          = compute_macd(ohlcv["close"])
        ohlcv["macd"]       = ml
        ohlcv["macd_sig"]   = sl
        ohlcv["macd_hist"]  = mh

        cur     = ohlcv["close"].iloc[-1]
        prev_w  = ohlcv["close"].iloc[-6] if len(ohlcv) >= 6 else ohlcv["close"].iloc[0]
        wpct    = (cur - prev_w) / prev_w * 100
        rsi_val = ohlcv["rsi"].dropna().iloc[-1] if not ohlcv["rsi"].dropna().empty else float("nan")
        ma20v   = ohlcv["ma20"].dropna().iloc[-1] if not ohlcv["ma20"].dropna().empty else float("nan")
        vs_ma   = (cur - ma20v) / ma20v * 100 if not pd.isna(ma20v) else float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"₹{cur:,.2f}")
        m2.metric("Week Change", f"{wpct:+.2f}%")
        m3.metric("RSI (14)", f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "–")
        m4.metric("vs 20DMA", f"{vs_ma:+.2f}%" if not pd.isna(vs_ma) else "–")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.03,
            subplot_titles=(f"{sel} — Close Price", "RSI (14)", "MACD"),
        )
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["close"], name="Close",
            line=dict(color="#00c853", width=2), fill="tozeroy", fillcolor="rgba(0,200,83,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ma20"], name="20DMA",
            line=dict(color="#ffb300", width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ma50"], name="50DMA",
            line=dict(color="#60a5fa", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["rsi"], name="RSI",
            line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff5252", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00c853", opacity=0.5, row=2, col=1)
        bar_colors = ["#00c853" if v >= 0 else "#ff5252" for v in ohlcv["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=ohlcv["date"], y=ohlcv["macd_hist"], name="Histogram",
            marker_color=bar_colors, opacity=0.55), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["macd"], name="MACD",
            line=dict(color="#00c853", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["macd_sig"], name="Signal",
            line=dict(color="#ffb300", width=1.5)), row=3, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8a95a1", family="Inter, system-ui"),
            height=560, hovermode="x unified", showlegend=True,
            legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)", font_color="#8a95a1"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        fig.update_xaxes(gridcolor="rgba(0,200,83,0.05)", zeroline=False, color="#8a95a1")
        fig.update_yaxes(gridcolor="rgba(0,200,83,0.05)", zeroline=False, color="#8a95a1")
        fig.update_yaxes(title_text="₹", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        st.plotly_chart(fig, width="stretch")

        # ── this stock's call history: how did OUR past calls on it do? ──
        import ai_data as _aid
        _hist = _aid.swing_validated()
        if len(_hist) and (sym_hist := _hist[_hist["symbol"] == sel]).shape[0] > 0:
            wins = (sym_hist["realised_ret"] > 0).sum()
            st.markdown(f"#### 📜 Our call history on {sel}")
            st.caption(
                f"{len(sym_hist)} graded swing call(s) · {wins} closed up · "
                f"avg {sym_hist['realised_ret'].mean():+.2%} over the 2-week horizon")
            hshow = sym_hist[["made_on", "signal", "entry", "target", "stop",
                              "outcome", "exit_close", "realised_ret"]].copy()
            hshow.columns = ["Date", "Signal", "Entry", "Target", "Stop",
                             "Outcome", "Exit", "Return"]
            hshow["Return"] = (hshow["Return"] * 100).round(2)

            def _o(v):
                return ("color:#00c853;font-weight:700" if v == "TARGET"
                        else "color:#ff5252;font-weight:600" if v == "STOP"
                        else "color:#8a95a1")
            def _r(v):
                try: return f"color:{'#00c853' if float(v) >= 0 else '#ff5252'}"
                except (ValueError, TypeError): return ""
            st.dataframe(
                hshow.sort_values("Date", ascending=False).style
                     .map(_o, subset=["Outcome"]).map(_r, subset=["Return"]),
                width="stretch", hide_index=True,
                height=min(320, 60 + 35 * len(hshow)),
            )

# ── Portfolio ─────────────────────────────────────────────────────────────────
elif page == ":material/work: My Portfolio":
    st.markdown("## 💼 My Portfolio")
    st.caption("Enter holdings — one per line. P&L updates live against today's prices.")

    col_inp, col_help = st.columns([3, 1])
    with col_help:
        st.markdown("""
<div style="background:rgba(0,200,83,.06);border:1px solid rgba(0,200,83,.15);border-radius:10px;padding:14px 16px;font-size:13px;color:#8a95a1;line-height:1.8;">
<b style="color:#00c853;">Format</b><br>
<code>SYMBOL, Qty, Buy Price</code><br><br>
<b style="color:#8a95a1;">Example</b><br>
<code>NIACL, 50, 152.80</code><br>
<code>HFCL, 100, 171.86</code><br>
<code>TRENT, 5, 2755.30</code>
</div>
""", unsafe_allow_html=True)

    with col_inp:
        txt = st.text_area("Holdings", height=200, placeholder="NIACL, 50, 152.80\nHFCL, 100, 171.86",
                           label_visibility="collapsed")

    if txt.strip():
        rows, errors = [], []
        for line in txt.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                try:
                    rows.append({"Symbol": parts[0].upper(), "Qty": int(parts[1]), "Buy Price (₹)": float(parts[2])})
                except ValueError:
                    errors.append(line)
            else:
                errors.append(line)

        if errors:
            st.warning(f"Skipped: {errors}")

        if rows:
            latest = closes.iloc[-1] if not closes.empty else pd.Series(dtype=float)
            pf = pd.DataFrame(rows)
            pf["Current (₹)"]  = pf["Symbol"].map(lambda s: round(float(latest.get(s, float("nan"))), 2))
            pf["Invested (₹)"] = (pf["Buy Price (₹)"] * pf["Qty"]).round(2)
            pf["Value (₹)"]    = (pf["Current (₹)"] * pf["Qty"]).round(2)
            pf["P&L (₹)"]      = (pf["Value (₹)"] - pf["Invested (₹)"]).round(2)
            pf["P&L (%)"]      = ((pf["P&L (₹)"] / pf["Invested (₹)"]) * 100).round(2)

            ti, tv = pf["Invested (₹)"].sum(), pf["Value (₹)"].sum()
            tp = tv - ti
            tpct = tp / ti * 100 if ti else 0

            s1, s2, s3 = st.columns(3)
            s1.metric("Total Invested", f"₹{ti:,.0f}")
            s2.metric("Current Value",  f"₹{tv:,.0f}", f"₹{tp:+,.0f}")
            s3.metric("Overall P&L",    f"₹{tp:+,.0f}", f"{tpct:+.2f}%")

            def _pnl(v):
                try: return f"color:{'#00c853' if float(v)>=0 else '#ff5252'};font-weight:600"
                except: return ""

            st.dataframe(
                pf.style.map(_pnl, subset=["P&L (₹)", "P&L (%)"]),
                width="stretch", hide_index=True,
            )

# ── AI Lab — Defence ──────────────────────────────────────────────────────────
elif page == ":material/smart_toy: AI Lab — Defence":
    st.markdown("## 🤖 AI Lab — Defence Sector")
    st.caption("XGBoost next-day predictions that grade themselves every day and learn from the outcomes.")

    if not _AI_OK:
        st.warning("AI model not found. Run the nightly job (`python journal.py`) to generate predictions, "
                   "then commit `data/defence_model.json` and `data/predictions.csv`.")
        st.stop()

    preds = ai_data.live_predictions()
    score = ai_data.track_record()
    meta = ai_data.model_meta()

    # ── honest framing ──
    st.markdown("""
<div style="background:#171c22;border-left:3px solid #00c853;border-radius:0 10px 10px 0;padding:12px 16px;font-size:12.5px;color:#8a95a1;line-height:1.7;margin-bottom:18px;">
<b style="color:#00c853;">How to read this:</b> the model ranks defence stocks by probability of an <b>up</b> day tomorrow.
It's a research aid, not a guarantee — out-of-sample edge is real but modest (AUC ≈ 0.56). Trust the <b>track record</b> below, not any single call.
</div>
""", unsafe_allow_html=True)

    # ── live track record ──
    if score.get("n", 0) > 0:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Directional Accuracy", f"{score['accuracy']:.1%}",
                  help=f"{score['n']} graded predictions, {score['first_date']} → {score['last_date']}")
        edge = score["bullish_avg_ret"] - score["universe_avg_ret"]
        m2.metric("Bullish picks · next-day", f"{score['bullish_avg_ret']:+.2%}",
                  f"{edge:+.2%} vs universe")
        m3.metric("Universe avg · next-day", f"{score['universe_avg_ret']:+.2%}")
        m4.metric("Graded predictions", f"{score['n']:,}")
    else:
        st.info("No graded predictions yet — the track record fills in as outcomes arrive.")

    if meta:
        st.caption(f"Model retrained {meta.get('trained_at','?')[:16]} · "
                   f"{meta.get('n_train','?')} training rows · through {meta.get('last_train_date','?')}")

    st.divider()
    asof = pd.to_datetime(preds['date'].max())
    target = (asof + pd.tseries.offsets.BDay(1)).strftime('%d %b')
    st.markdown(f"### Latest picks — as of {asof.strftime('%d %b %Y')} close")
    st.caption(
        f"Computed from the last **completed** session ({asof.strftime('%d %b')}); "
        f"each call predicts that stock's **next trading-day** close (≈ {target}). "
        f"If today's session is still open, the latest finished close is the prior day. "
        f"Logged and graded automatically once the outcome is known."
    )

    # rule-based score side-by-side (where the symbol exists in the Nifty-500 engine)
    rule_scores = {}
    try:
        sig = generate_signals(closes, volumes, nifty500)
        rule_scores = dict(zip(sig["Symbol"], sig["Score"]))
    except Exception:  # noqa: BLE001
        pass

    for _, p in preds.iterrows():
        sym = p["symbol"]
        prob = p["proba"]
        bullish = prob >= 0.5
        col = "#00c853" if bullish else "#ff5252"
        label = "Bullish" if bullish else "Bearish"
        name = DEFENCE_NAMES.get(sym, sym)
        rscore = rule_scores.get(sym)
        rule_html = (f'<span style="font-size:12px;color:#8a95a1;">Rule score '
                     f'<b style="color:#eef2f6;">{rscore:.0f}</b>/100</span>'
                     if rscore is not None else
                     '<span style="font-size:12px;color:#5c6672;">Rule score —</span>')

        chips = ""
        for s in p["signals"]:
            pos = s["direction"] == "+"
            c = "#00c853" if pos else "#ff5252"
            bg = "rgba(0,200,83,.08)" if pos else "rgba(248,113,113,.08)"
            chips += (f'<span style="display:inline-block;background:{bg};border:1px solid {c}33;'
                      f'color:{c};border-radius:6px;padding:3px 10px;margin:3px 6px 3px 0;font-size:12px;">'
                      f'{s["direction"]} {s["label"]}</span>')

        st.markdown(f"""
<div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);border-left:3px solid {col};
            border-radius:10px;padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:16px;font-weight:700;color:#eef2f6;">{sym}</span>
      <span style="font-size:12px;color:#8a95a1;margin-left:8px;">{name}</span>
      <span style="font-size:12px;color:#8a95a1;margin-left:8px;">· ₹{p['close']:,.1f}</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:17px;font-weight:700;color:{col};">{label} {prob:.0%}</span><br>{rule_html}
    </div>
  </div>
  <div style="margin-top:10px;">{chips}</div>
</div>
""", unsafe_allow_html=True)

    # calibration
    calib = score.get("calibration")
    if calib is not None and not calib.empty:
        st.divider()
        st.markdown("### 📐 Calibration — does its confidence mean anything?")
        st.caption("For each confidence bucket, the share of predictions that actually rose. "
                   "Higher buckets should show higher up-rates if the model is honest.")
        cdf = calib.copy()
        cdf["proba"] = cdf["proba"].astype(str)
        cdf["up_rate"] = (cdf["up_rate"] * 100).round(1)
        cfig = go.Figure(go.Bar(
            x=cdf["proba"], y=cdf["up_rate"], marker_color="#00c853",
            text=[f"{v:.0f}%" for v in cdf["up_rate"]], textposition="outside",
        ))
        cfig.update_layout(
            template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Actual up-rate (%)", xaxis_title="Predicted probability bucket",
        )
        st.plotly_chart(cfig, width="stretch")

# ── Feature Importance ────────────────────────────────────────────────────────
elif page == ":material/psychology: Feature Importance":
    st.markdown("## 🧠 Feature Importance")
    st.caption("What the model actually weighs when ranking defence stocks — global view across all predictions.")

    if not _AI_OK:
        st.warning("AI model not found. Run `python journal.py` first.")
        st.stop()

    imp = ai_data.importance()
    st.markdown("""
<div style="background:rgba(0,200,83,.05);border:1px solid rgba(0,200,83,.15);border-radius:10px;padding:12px 16px;font-size:12.5px;color:#8a95a1;line-height:1.7;margin-bottom:16px;">
This is where XGBoost shines — every feature's contribution is measurable. Bars show each signal's share of the model's
splitting decisions. Per-stock <b>+/−</b> attributions live on the <b>AI Lab</b> page.
</div>
""", unsafe_allow_html=True)

    top = imp.head(18).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["label"], orientation="h",
        marker=dict(color=top["importance"], colorscale=[[0, "#1f3d2b"], [1, "#00c853"]]),
    ))
    fig.update_layout(
        template="plotly_dark", height=560, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Relative importance",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption("Sector momentum signals typically dominate — defence stocks move as a bloc, so 'is the whole "
               "sector hot?' carries real predictive weight.")
