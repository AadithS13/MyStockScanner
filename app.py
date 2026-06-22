import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from data import get_nifty500_symbols, get_price_history, get_stock_ohlcv
from signals import generate_signals, compute_rsi, compute_rsi_series, compute_macd

st.set_page_config(
    page_title="My AI Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Background animation (same particle/canvas engine as portfolio) ──────────
st.iframe("""
<script>
(function () {
  try {
    var W = window.parent, doc = W.document;
    if (doc.getElementById('atrader-canvas')) return;

    /* inject Google Font */
    if (!doc.getElementById('atrader-font')) {
      var lnk = doc.createElement('link');
      lnk.id = 'atrader-font';
      lnk.rel = 'stylesheet';
      lnk.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap';
      doc.head.appendChild(lnk);
    }

    /* canvas */
    var cv = doc.createElement('canvas');
    cv.id = 'atrader-canvas';
    cv.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;';
    doc.body.insertBefore(cv, doc.body.firstChild);
    var ctx = cv.getContext('2d');

    /* cursor spotlight */
    var spot = doc.createElement('div');
    spot.id = 'atrader-spot';
    spot.style.cssText = 'position:fixed;top:0;left:0;width:700px;height:700px;border-radius:50%;pointer-events:none;z-index:1;opacity:0;transition:opacity .5s;background:radial-gradient(circle,rgba(74,222,128,.055) 0%,transparent 65%);';
    doc.body.insertBefore(spot, cv.nextSibling);

    var Ww = W.innerWidth, Wh = W.innerHeight;
    var mouse = { x: -9999, y: -9999 }, sp = { x: 0, y: 0 }, spInit = false;

    function resize() {
      Ww = W.innerWidth; Wh = W.innerHeight;
      cv.width = Ww; cv.height = Wh;
    }
    resize();

    /* particles */
    var N = Math.min(55, Math.floor(Ww * Wh / 28000));
    var pts = [];
    for (var i = 0; i < N; i++) pts.push({ x: Math.random()*Ww, y: Math.random()*Wh, vx:(Math.random()-.5)*.28, vy:(Math.random()-.5)*.28, r:Math.random()*1.4+.5 });

    /* rising candlesticks */
    var CK = 30;
    var cks = [];
    for (var i = 0; i < CK; i++) cks.push({ x:Math.random()*Ww, y:Math.random()*Wh, bh:8+Math.random()*22, wh:4+Math.random()*14, sp:.08+Math.random()*.22, bull:Math.random()>.32, a:.03+Math.random()*.05 });

    var LINK = 125;

    doc.addEventListener('mousemove', function(e) {
      mouse.x = e.clientX; mouse.y = e.clientY;
      if (!spInit) { sp.x = mouse.x; sp.y = mouse.y; spInit = true; }
      sp.x += (mouse.x - sp.x) * .08;
      sp.y += (mouse.y - sp.y) * .08;
      spot.style.opacity = '1';
      spot.style.transform = 'translate3d('+(sp.x-350)+'px,'+(sp.y-350)+'px,0)';
    });
    doc.addEventListener('mouseleave', function() { spot.style.opacity = '0'; });
    W.addEventListener('resize', resize);

    function frame() {
      ctx.clearRect(0, 0, Ww, Wh);

      /* candlesticks */
      cks.forEach(function(c) {
        var col = c.bull ? 'rgba(74,222,128,'+c.a+')' : 'rgba(239,68,68,'+(c.a*.5)+')';
        ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(c.x, c.y-c.bh/2-c.wh); ctx.lineTo(c.x, c.y+c.bh/2+c.wh/2); ctx.stroke();
        ctx.fillRect(c.x-3, c.y-c.bh/2, 6, c.bh);
        c.y -= c.sp;
        if (c.y+c.bh+c.wh < 0) { c.y = Wh+30; c.x = Math.random()*Ww; c.bull = Math.random()>.32; c.bh = 8+Math.random()*22; }
      });

      /* particles */
      pts.forEach(function(p) {
        p.x += p.vx; p.y += p.vy;
        p.vx *= .998; p.vy *= .998;
        if (p.x < -10) p.x = Ww+10; if (p.x > Ww+10) p.x = -10;
        if (p.y < -10) p.y = Wh+10; if (p.y > Wh+10) p.y = -10;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(74,222,128,.22)'; ctx.fill();
      });

      /* links */
      for (var i = 0; i < pts.length; i++) for (var j = i+1; j < pts.length; j++) {
        var dx = pts[i].x-pts[j].x, dy = pts[i].y-pts[j].y, d = Math.sqrt(dx*dx+dy*dy);
        if (d < LINK) { ctx.beginPath(); ctx.moveTo(pts[i].x,pts[i].y); ctx.lineTo(pts[j].x,pts[j].y); ctx.strokeStyle='rgba(74,222,128,'+((1-d/LINK)*.07)+')'; ctx.lineWidth=1; ctx.stroke(); }
      }

      /* mouse links */
      pts.forEach(function(p) {
        var dx=p.x-mouse.x, dy=p.y-mouse.y, d=Math.sqrt(dx*dx+dy*dy);
        if (d < 160) { ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(mouse.x,mouse.y); ctx.strokeStyle='rgba(74,222,128,'+((1-d/160)*.18)+')'; ctx.lineWidth=1; ctx.stroke(); }
      });

      W.requestAnimationFrame(frame);
    }
    frame();
  } catch(e) { console.warn('AI Trader BG:', e); }
})();
</script>
""", height=0)

# ── Global CSS theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [data-testid="stApp"], .stApp {
    background: #0a0a0a !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #e5e5e5 !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Main padding */
[data-testid="stMainBlockContainer"],
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid rgba(74,222,128,.12) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* Sidebar nav (radio) */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; flex-direction: column !important; }
[data-testid="stSidebar"] .stRadio > div > label {
    padding: 11px 16px !important;
    border-radius: 8px !important;
    border-left: 2px solid transparent !important;
    color: #6b7280 !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all .15s ease !important;
    margin: 1px 0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(74,222,128,.07) !important;
    color: #4ade80 !important;
    border-left-color: rgba(74,222,128,.35) !important;
}
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(74,222,128,.1) !important;
    color: #4ade80 !important;
    border-left-color: #4ade80 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,.02) !important;
    border: 1px solid rgba(74,222,128,.12) !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: .6px !important;
}
[data-testid="stMetricValue"] { color: #e5e5e5 !important; font-weight: 700 !important; }
[data-testid="stMetricDeltaIcon-Up"] { color: #4ade80 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #ef4444 !important; }

/* ── Dataframe ── */
.stDataFrame iframe, .stDataFrame > div {
    border: 1px solid rgba(74,222,128,.1) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: rgba(255,255,255,.015) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: rgba(255,255,255,.03) !important;
    border-color: rgba(74,222,128,.2) !important;
    border-radius: 8px !important;
    color: #e5e5e5 !important;
}
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,.03) !important;
    border-color: rgba(74,222,128,.2) !important;
    border-radius: 8px !important;
    color: #e5e5e5 !important;
    font-family: 'Inter', monospace !important;
}

/* ── Divider / HR ── */
hr { border-color: rgba(74,222,128,.1) !important; }

/* ── Text ── */
h1,h2,h3,h4 { color: #e5e5e5 !important; font-family: 'Inter', system-ui, sans-serif !important; }
.stMarkdown p, .stCaption { color: #6b7280 !important; }
[data-testid="stCaptionContainer"] p { color: #4b5563 !important; font-size: 12px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4ade80 !important; }

/* ── Plotly ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 4px 16px 4px;">
      <div style="font-size:28px;margin-bottom:4px;">📈</div>
      <div style="font-size:18px;font-weight:700;color:#e5e5e5;letter-spacing:-.3px;">My AI Trader</div>
      <div style="font-size:11px;color:#4b5563;margin-top:2px;">Nifty 500 · NSE Bhavcopy</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["📊  Overview", "🎯  Swing Signals", "🔍  Stock Detail", "💼  My Portfolio"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"""
    <div style="font-size:11px;color:#4b5563;line-height:1.8;">
      🕐 {datetime.now().strftime('%d %b %Y, %I:%M %p')}<br>
      ⚡ Data refreshes hourly
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
if page == "📊  Overview":
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
                top10.style.map(lambda v: "color:#4ade80;font-weight:600", subset=["Week (%)"]),
                width="stretch", hide_index=True,
            )

        with col_l:
            st.markdown("#### 🔴 Top 10 Losers")
            bot10 = wchg.nsmallest(10).reset_index()
            bot10.columns = ["Symbol", "Week (%)"]
            bot10["Week (%)"] = bot10["Week (%)"].round(2)
            st.dataframe(
                bot10.style.map(lambda v: "color:#ef4444;font-weight:600", subset=["Week (%)"]),
                width="stretch", hide_index=True,
            )

# ── Swing Signals ─────────────────────────────────────────────────────────────
elif page == "🎯  Swing Signals":
    st.markdown("## 🎯 Swing Trade Signals")
    st.caption("RSI · MACD · 20DMA · 50DMA · Volume — 2-week swing horizon · Not financial advice")

    with st.spinner("Computing signals for 500 stocks…"):
        signals_df = generate_signals(closes, volumes, nifty500)

    fc, sc = st.columns([3, 2])
    with fc:
        sig_opts = ["🟢 Strong Buy", "🟢 Buy", "🟡 Watch", "⚪ Neutral", "🔴 Overbought"]
        sel = st.multiselect("Filter", sig_opts, default=["🟢 Strong Buy", "🟢 Buy"])
    with sc:
        sort_by = st.selectbox("Sort by", ["Score", "Upside (%)", "R/R Ratio", "RSI"])

    filtered = signals_df[signals_df["Signal"].isin(sel)] if sel else signals_df
    filtered = filtered.sort_values(sort_by, ascending=(sort_by == "RSI")).reset_index(drop=True)

    COLS = ["Symbol", "Price (₹)", "Signal", "Score", "RSI", "MACD",
            "vs 20DMA (%)", "vs 50DMA (%)", "Vol Ratio",
            "Target (₹)", "Upside (%)", "Stop Loss (₹)", "R/R Ratio", "Timeline"]

    def _sig(v):
        if "Strong" in str(v): return "color:#22c55e;font-weight:700"
        if "Buy" in str(v):    return "color:#4ade80;font-weight:600"
        if "Overbought" in str(v): return "color:#ef4444;font-weight:600"
        if "Watch" in str(v):  return "color:#fbbf24;font-weight:600"
        return "color:#6b7280"

    def _macd(v): return "color:#4ade80" if v == "Bullish" else "color:#ef4444"
    def _pct(v):
        try: return f"color:{'#4ade80' if float(v)>=0 else '#ef4444'}"
        except: return ""

    st.dataframe(
        filtered[COLS].style.map(_sig, subset=["Signal"]).map(_macd, subset=["MACD"]).map(_pct, subset=["vs 20DMA (%)", "vs 50DMA (%)", "Upside (%)"]),
        width="stretch", hide_index=True, height=520,
    )
    st.caption(f"Showing {len(filtered)} of {len(signals_df)} stocks  ·  Target = next resistance  ·  Stop = recent low or −7%")

# ── Stock Detail ──────────────────────────────────────────────────────────────
elif page == "🔍  Stock Detail":
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
            line=dict(color="#4ade80", width=2), fill="tozeroy", fillcolor="rgba(74,222,128,0.05)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ma20"], name="20DMA",
            line=dict(color="#fbbf24", width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["ma50"], name="50DMA",
            line=dict(color="#60a5fa", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["rsi"], name="RSI",
            line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", opacity=0.5, row=2, col=1)
        bar_colors = ["#4ade80" if v >= 0 else "#ef4444" for v in ohlcv["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=ohlcv["date"], y=ohlcv["macd_hist"], name="Histogram",
            marker_color=bar_colors, opacity=0.55), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["macd"], name="MACD",
            line=dict(color="#4ade80", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ohlcv["date"], y=ohlcv["macd_sig"], name="Signal",
            line=dict(color="#fbbf24", width=1.5)), row=3, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af", family="Inter, system-ui"),
            height=560, hovermode="x unified", showlegend=True,
            legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)", font_color="#9ca3af"),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        fig.update_xaxes(gridcolor="rgba(74,222,128,0.05)", zeroline=False, color="#6b7280")
        fig.update_yaxes(gridcolor="rgba(74,222,128,0.05)", zeroline=False, color="#6b7280")
        fig.update_yaxes(title_text="₹", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        st.plotly_chart(fig, width="stretch")

# ── Portfolio ─────────────────────────────────────────────────────────────────
elif page == "💼  My Portfolio":
    st.markdown("## 💼 My Portfolio")
    st.caption("Enter holdings — one per line. P&L updates live against today's prices.")

    col_inp, col_help = st.columns([3, 1])
    with col_help:
        st.markdown("""
<div style="background:rgba(74,222,128,.06);border:1px solid rgba(74,222,128,.15);border-radius:10px;padding:14px 16px;font-size:13px;color:#9ca3af;line-height:1.8;">
<b style="color:#4ade80;">Format</b><br>
<code>SYMBOL, Qty, Buy Price</code><br><br>
<b style="color:#6b7280;">Example</b><br>
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
                try: return f"color:{'#4ade80' if float(v)>=0 else '#ef4444'};font-weight:600"
                except: return ""

            st.dataframe(
                pf.style.map(_pnl, subset=["P&L (₹)", "P&L (%)"]),
                width="stretch", hide_index=True,
            )
