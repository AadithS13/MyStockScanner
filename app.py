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

# ── Background animation (portfolio canvas engine, injected into parent) ─────
st.iframe("""<!DOCTYPE html><html><head><style>body{margin:0;background:transparent}</style></head><body><script>
(function(){
  try{
    var W=window.parent,doc=W.document;
    if(doc.getElementById('at-cv'))return;

    // canvas
    var cv=doc.createElement('canvas');
    cv.id='at-cv';
    cv.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;';
    doc.body.insertBefore(cv,doc.body.firstChild);
    var ctx=cv.getContext('2d');

    // spotlight
    var sp=doc.createElement('div');
    sp.id='at-sp';
    sp.style.cssText='position:fixed;top:0;left:0;width:700px;height:700px;border-radius:50%;pointer-events:none;z-index:1;opacity:0;transition:opacity .5s;background:radial-gradient(circle,rgba(74,222,128,.06) 0%,transparent 65%);';
    doc.body.insertBefore(sp,cv.nextSibling);

    var Ww=W.innerWidth,Wh=W.innerHeight;
    var mouse={x:-9999,y:-9999},spos={x:0,y:0},sinit=false;

    function resize(){Ww=W.innerWidth;Wh=W.innerHeight;cv.width=Ww;cv.height=Wh;}
    resize();

    // stars (twinkling, 3 depths — exact portfolio logic)
    var stars=[];
    function mkStars(){
      stars=[];
      var n=Math.max(90,Math.min(220,Math.floor(Ww*Wh/9000)));
      for(var i=0;i<n;i++){
        var depth=Math.random(),roll=Math.random();
        stars.push({x:Math.random()*Ww,y:Math.random()*Wh,r:.4+depth*1.1,depth:depth,
          phase:Math.random()*Math.PI*2,speed:.4+Math.random()*1.4,base:.12+depth*.5,
          hue:roll<.78?0:roll<.9?1:2});
      }
    }
    mkStars();

    // constellation particles
    var pts=[];
    function mkPts(){
      pts=[];
      var n=Math.max(36,Math.min(80,Math.floor(Ww*Wh/26000)));
      for(var i=0;i<n;i++) pts.push({x:Math.random()*Ww,y:Math.random()*Wh,vx:(Math.random()-.5)*.34,vy:(Math.random()-.5)*.34,r:Math.random()*1.4+.6});
    }
    mkPts();

    // candlestick drifters (instead of shooting stars)
    var cks=[];
    function mkCks(){
      cks=[];
      for(var i=0;i<28;i++) cks.push({x:Math.random()*Ww,y:Math.random()*Wh,bh:8+Math.random()*22,wh:4+Math.random()*14,sp:.08+Math.random()*.2,bull:Math.random()>.32,a:.04+Math.random()*.05});
    }
    mkCks();

    var LINK=130,t=0;

    function starColor(hue,a){
      if(hue===1)return'rgba(74,222,128,'+a+')';
      if(hue===2)return'rgba(96,165,250,'+a+')';
      return'rgba(226,232,240,'+a+')';
    }

    function frame(){
      t+=.016;
      ctx.clearRect(0,0,Ww,Wh);

      // stars
      for(var i=0;i<stars.length;i++){
        var s=stars[i];
        var tw=.55+.45*Math.sin(t*s.speed+s.phase);
        var a=s.base*tw;
        s.y+=.012+s.depth*.03; if(s.y>Wh+4)s.y=-4;
        ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
        ctx.fillStyle=starColor(s.hue,a);ctx.fill();
        if(s.depth>.82&&tw>.92){
          var len=s.r*5*(tw-.9)*10;
          ctx.strokeStyle=starColor(s.hue,a*.5);ctx.lineWidth=.6;
          ctx.beginPath();ctx.moveTo(s.x-len,s.y);ctx.lineTo(s.x+len,s.y);
          ctx.moveTo(s.x,s.y-len);ctx.lineTo(s.x,s.y+len);ctx.stroke();
        }
      }

      // rising candlesticks
      for(var i=0;i<cks.length;i++){
        var c=cks[i];
        var col=c.bull?'rgba(74,222,128,'+c.a+')':'rgba(239,68,68,'+(c.a*.4)+')';
        ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(c.x,c.y-c.bh/2-c.wh);ctx.lineTo(c.x,c.y+c.bh/2+c.wh/2);ctx.stroke();
        ctx.fillRect(c.x-3,c.y-c.bh/2,6,c.bh);
        c.y-=c.sp;
        if(c.y+c.bh+c.wh<0){c.y=Wh+30;c.x=Math.random()*Ww;c.bull=Math.random()>.32;c.bh=8+Math.random()*22;}
      }

      // particles
      for(var i=0;i<pts.length;i++){
        var p=pts[i];
        p.x+=p.vx;p.y+=p.vy;p.vx*=.995;p.vy*=.995;
        if(p.x<-10)p.x=Ww+10;if(p.x>Ww+10)p.x=-10;
        if(p.y<-10)p.y=Wh+10;if(p.y>Wh+10)p.y=-10;
        ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle='rgba(74,222,128,.28)';ctx.fill();
      }

      // links between particles
      for(var i=0;i<pts.length;i++) for(var j=i+1;j<pts.length;j++){
        var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<LINK){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(74,222,128,'+((1-d/LINK)*.09)+')';ctx.lineWidth=1;ctx.stroke();}
      }

      // mouse links
      for(var i=0;i<pts.length;i++){
        var dx=pts[i].x-mouse.x,dy=pts[i].y-mouse.y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<170){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(mouse.x,mouse.y);ctx.strokeStyle='rgba(74,222,128,'+((1-d/170)*.2)+')';ctx.lineWidth=1;ctx.stroke();}
      }

      // spotlight lerp
      if(sinit){
        spos.x+=(mouse.x-spos.x)*.08;spos.y+=(mouse.y-spos.y)*.08;
        sp.style.transform='translate3d('+(spos.x-350)+'px,'+(spos.y-350)+'px,0)';
      }

      W.requestAnimationFrame(frame);
    }

    doc.addEventListener('mousemove',function(e){
      mouse.x=e.clientX;mouse.y=e.clientY;
      if(!sinit){spos.x=mouse.x;spos.y=mouse.y;sinit=true;}
      sp.style.opacity='1';
    });
    doc.addEventListener('mouseleave',function(){sp.style.opacity='0';});
    W.addEventListener('resize',function(){resize();mkStars();mkPts();mkCks();});

    frame();
  }catch(e){console.warn('AT BG:',e);}
})();
</script></body></html>""", height=1)

# ── Global CSS theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base: dark background on body only ── */
html, body {
    background: #0a0a0a !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: #e5e5e5 !important;
}

/* Make all Streamlit containers transparent so background shows through */
[data-testid="stApp"], .stApp,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
.main, .block-container,
section.main > div { background: transparent !important; }

/* ── Animated nebula (on body, z-index: -1 so it's behind everything) ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 15% 25%, rgba(74,222,128,.09) 0%, transparent 45%),
        radial-gradient(circle at 85% 70%, rgba(74,222,128,.06) 0%, transparent 45%),
        radial-gradient(circle at 50% 90%, rgba(96,165,250,.04) 0%, transparent 40%);
    animation: nebula-drift 28s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: -1;
}
@keyframes nebula-drift {
    0%   { opacity:1; transform: scale(1)    translate(0,0); }
    50%  { opacity:.75; transform: scale(1.07) translate(3%,2%); }
    100% { opacity:1; transform: scale(1)    translate(-2%,-1%); }
}
/* ── Dot grid (on body, z-index: -1) ── */
body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(74,222,128,.07) 1px, transparent 1px);
    background-size: 30px 30px;
    mask-image: radial-gradient(ellipse 90% 70% at 50% 40%, #000 30%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse 90% 70% at 50% 40%, #000 30%, transparent 100%);
    pointer-events: none;
    z-index: -1;
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
        ["📊  Overview", "🎯  Swing Signals", "🤖  AI Lab — Defence",
         "🧠  Feature Importance", "🔍  Stock Detail", "💼  My Portfolio"],
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
                    return ("color:#22c55e;font-weight:700" if v == "TARGET"
                            else "color:#ef4444;font-weight:600" if v == "STOP"
                            else "color:#9ca3af")
                def _ret(v):
                    try: return f"color:{'#4ade80' if float(v) >= 0 else '#ef4444'}"
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

    sig_opts = ["🟢 Strong Buy", "🟢 Buy", "🟡 Watch", "⚪ Neutral", "🔴 Overbought"]
    sel = st.multiselect("Filter by signal", sig_opts,
                         default=["🟢 Strong Buy", "🟢 Buy"])

    filtered = signals_df[signals_df["Signal"].isin(sel)] if sel else signals_df
    # initial order; users re-sort by clicking headers (Excel-style)
    filtered = filtered.sort_values("Score", ascending=False).reset_index(drop=True)

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
    }.items()}

    st.markdown("#### 🎯 Live signals — today's setups")
    st.caption("💡 Click any column header to sort (click again to reverse) — like Excel.")
    st.dataframe(
        filtered[COLS].style
            .format(num_fmt)
            .map(_sig, subset=["Signal"])
            .map(_macd, subset=["MACD"])
            .map(_pct, subset=["vs 20DMA (%)", "vs 50DMA (%)", "Upside (%)"]),
        width="stretch", hide_index=True, height=520,
    )
    st.caption(f"Showing {len(filtered)} of {len(signals_df)} stocks  ·  "
               f"Target = next resistance  ·  Stop = recent low or −7%")

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

# ── AI Lab — Defence ──────────────────────────────────────────────────────────
elif page == "🤖  AI Lab — Defence":
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
<div style="background:rgba(96,165,250,.06);border:1px solid rgba(96,165,250,.2);border-radius:10px;padding:12px 16px;font-size:12.5px;color:#9ca3af;line-height:1.7;margin-bottom:18px;">
<b style="color:#60a5fa;">How to read this:</b> the model ranks defence stocks by probability of an <b>up</b> day tomorrow.
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
        col = "#4ade80" if bullish else "#ef4444"
        label = "Bullish" if bullish else "Bearish"
        name = DEFENCE_NAMES.get(sym, sym)
        rscore = rule_scores.get(sym)
        rule_html = (f'<span style="font-size:12px;color:#9ca3af;">Rule score '
                     f'<b style="color:#e5e5e5;">{rscore:.0f}</b>/100</span>'
                     if rscore is not None else
                     '<span style="font-size:12px;color:#4b5563;">Rule score —</span>')

        chips = ""
        for s in p["signals"]:
            pos = s["direction"] == "+"
            c = "#4ade80" if pos else "#f87171"
            bg = "rgba(74,222,128,.08)" if pos else "rgba(248,113,113,.08)"
            chips += (f'<span style="display:inline-block;background:{bg};border:1px solid {c}33;'
                      f'color:{c};border-radius:6px;padding:3px 10px;margin:3px 6px 3px 0;font-size:12px;">'
                      f'{s["direction"]} {s["label"]}</span>')

        st.markdown(f"""
<div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);border-left:3px solid {col};
            border-radius:10px;padding:14px 18px;margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:16px;font-weight:700;color:#e5e5e5;">{sym}</span>
      <span style="font-size:12px;color:#6b7280;margin-left:8px;">{name}</span>
      <span style="font-size:12px;color:#6b7280;margin-left:8px;">· ₹{p['close']:,.1f}</span>
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
            x=cdf["proba"], y=cdf["up_rate"], marker_color="#4ade80",
            text=[f"{v:.0f}%" for v in cdf["up_rate"]], textposition="outside",
        ))
        cfig.update_layout(
            template="plotly_dark", height=320, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Actual up-rate (%)", xaxis_title="Predicted probability bucket",
        )
        st.plotly_chart(cfig, width="stretch")

# ── Feature Importance ────────────────────────────────────────────────────────
elif page == "🧠  Feature Importance":
    st.markdown("## 🧠 Feature Importance")
    st.caption("What the model actually weighs when ranking defence stocks — global view across all predictions.")

    if not _AI_OK:
        st.warning("AI model not found. Run `python journal.py` first.")
        st.stop()

    imp = ai_data.importance()
    st.markdown("""
<div style="background:rgba(74,222,128,.05);border:1px solid rgba(74,222,128,.15);border-radius:10px;padding:12px 16px;font-size:12.5px;color:#9ca3af;line-height:1.7;margin-bottom:16px;">
This is where XGBoost shines — every feature's contribution is measurable. Bars show each signal's share of the model's
splitting decisions. Per-stock <b>+/−</b> attributions live on the <b>AI Lab</b> page.
</div>
""", unsafe_allow_html=True)

    top = imp.head(18).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["label"], orientation="h",
        marker=dict(color=top["importance"], colorscale=[[0, "#1f3d2b"], [1, "#4ade80"]]),
    ))
    fig.update_layout(
        template="plotly_dark", height=560, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Relative importance",
    )
    st.plotly_chart(fig, width="stretch")

    st.caption("Sector momentum signals typically dominate — defence stocks move as a bloc, so 'is the whole "
               "sector hot?' carries real predictive weight.")
