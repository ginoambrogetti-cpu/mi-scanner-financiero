import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import time
import io

st.set_page_config(page_title="Scanner Pro · Oportunidades", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.main, [data-testid="stAppViewContainer"] { background-color: #070b0f !important; }
section[data-testid="stSidebar"] { background-color: #0c1219 !important; border-right: 1px solid #1a2d3f; }
div[data-testid="stMetric"] { background-color: #101820 !important; border: 1px solid #1a2d3f !important; border-radius: 10px !important; padding: 14px !important; }
div[data-testid="stMetric"] label { color: #5a7a94 !important; font-size: 11px !important; letter-spacing: 1px; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2f0ff !important; font-size: 20px !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #00c8f0, #0090b8) !important; color: #070b0f !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; font-size: 14px !important; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.15); }
.stButton > button { background-color: #101820 !important; color: #b8cfe0 !important; border: 1px solid #1a2d3f !important; border-radius: 8px !important; }
details { background-color: #101820 !important; border: 1px solid #1a2d3f !important; border-radius: 10px !important; }
summary { color: #00c8f0 !important; font-weight: 600; letter-spacing: 1px; font-size: 12px; }
.stTextArea textarea, .stTextInput input { background-color: #0c1219 !important; color: #e2f0ff !important; border: 1px solid #1a2d3f !important; border-radius: 8px !important; }
.stMultiSelect > div { background-color: #0c1219 !important; border: 1px solid #1a2d3f !important; border-radius: 8px !important; }
.stTabs [data-baseweb="tab"] { color: #5a7a94 !important; font-size: 12px !important; }
.stTabs [aria-selected="true"] { color: #00c8f0 !important; border-bottom-color: #00c8f0 !important; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1a2d3f !important; background: none !important; }
.stDataFrame { border: 1px solid #1a2d3f !important; border-radius: 10px !important; overflow: hidden; }
iframe { background-color: #101820 !important; }
.stProgress > div > div { background: linear-gradient(90deg, #00c8f0, #b57bee) !important; border-radius: 4px; }
.stCaption { color: #3d5a72 !important; font-size: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_close_vol(raw, ticker, multi):
    try:
        sub = raw[ticker] if multi else raw
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = sub.columns.get_level_values(0)
        close = sub["Close"].dropna()
        vol   = sub["Volume"].dropna()
        if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
        if isinstance(vol,   pd.DataFrame): vol   = vol.iloc[:, 0]
        return close, vol
    except:
        return None, None

def rsi_calc(close, p=14):
    if len(close) < p + 1: return None
    d = close.diff().dropna()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    rs = g / l.replace(0, np.nan)
    v  = (100 - 100 / (1 + rs)).iloc[-1]
    return float(v) if not np.isnan(v) else None

def ema_calc(close, p):
    if len(close) < p: return None
    return float(close.ewm(span=p, adjust=False).mean().iloc[-1])

def wma_calc(close, p):
    """Media Móvil Ponderada (WMA) — pesa más los precios recientes linealmente."""
    if len(close) < p: return None
    weights = np.arange(1, p + 1, dtype=float)
    vals = close.iloc[-p:].values
    return float(np.dot(vals, weights) / weights.sum())

def macd_signal(close):
    if len(close) < 35: return None, None
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    m   = e12 - e26
    s   = m.ewm(span=9, adjust=False).mean()
    return float(m.iloc[-1]), float(s.iloc[-1])

def bollinger_pct(close, p=20):
    try:
        mid = close.rolling(p).mean()
        std = close.rolling(p).std()
        b = (close - (mid - 2*std)) / ((mid + 2*std) - (mid - 2*std))
        return float(b.iloc[-1])
    except:
        return None

def calcular_score(rsi_v, macd_v, macd_s, precio, wma21, e150, e200, vol_rel):
    s = 0
    if rsi_v is not None:
        if rsi_v < 25:   s += 3
        elif rsi_v < 35: s += 2
        elif rsi_v < 45: s += 1
        elif rsi_v > 75: s -= 3
        elif rsi_v > 65: s -= 2
        elif rsi_v > 55: s -= 1
    if macd_v and macd_s:
        s += 1 if macd_v > macd_s else -1
    for ref in [wma21, e150, e200]:
        if ref and precio:
            s += 1 if precio > ref else -1
    return max(-8, min(8, s))

def señal_label(score, vol_rel):
    whale = " 🐋" if vol_rel and vol_rel > 1.6 else ""
    if score >= 5:  return f"🚀 COMPRA FUERTE{whale}", "#004d26", "#69ffb0"
    if score >= 3:  return f"✅ COMPRAR{whale}",       "#003d1a", "#80ffbb"
    if score >= -1: return f"⏳ ESPERAR{whale}",       "#1a1a00", "#ffd740"
    if score >= -3: return f"⚠️ REDUCIR{whale}",       "#3d2000", "#ff9100"
    return              f"🔴 VENDER{whale}",           "#4d0000", "#ff3d57"

def fmt_pct(v, mul=True):
    if v is None: return "—"
    n = v * 100 if mul else v
    return f"{'+'if n>=0 else ''}{n:.1f}%"

def fmt2(v):
    if v is None: return "—"
    return f"{v:,.2f}"

# ── Cache / Fetch ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def descargar_masivo(tickers_tuple, period="1y"):
    tickers = list(tickers_tuple)
    try:
        raw = yf.download(tickers, period=period,
                          group_by="ticker" if len(tickers) > 1 else None,
                          progress=False, auto_adjust=True, threads=True)
        return raw
    except Exception as e:
        st.error(f"Error de descarga: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentales_bulk(tickers_tuple):
    def fetch_one(t):
        try:
            info = yf.Ticker(t).info
            return t, {
                "name":   info.get("longName") or info.get("shortName") or t,
                "pe":     info.get("trailingPE"),
                "pb":     info.get("priceToBook"),
                "sector": info.get("sector", "—"),
            }
        except:
            return t, {"name": t, "pe": None, "pb": None, "sector": "—"}
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for ticker, data in ex.map(fetch_one, list(tickers_tuple)):
            result[ticker] = data
    return result

def analizar_ticker(ticker, raw, multi, fund):
    close, vol = get_close_vol(raw, ticker, multi)
    if close is None or len(close) < 30: return None
    precio = float(close.iloc[-1])
    prev   = float(close.iloc[-2])  if len(close) >= 2  else precio
    w_ago  = float(close.iloc[-6])  if len(close) >= 6  else prev
    m_ago  = float(close.iloc[-22]) if len(close) >= 22 else prev

    rsi_v        = rsi_calc(close)
    macd_v, macd_s = macd_signal(close)
    wma21_v = wma_calc(close, 21)
    e150_v  = ema_calc(close, 150)
    e200_v  = ema_calc(close, 200)

    vol_series = vol.reindex(close.index).fillna(0)
    vol_avg = float(vol_series.rolling(20).mean().iloc[-1]) if len(vol_series) >= 20 else 1
    vol_rel = float(vol_series.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

    dist = lambda ref: (precio - ref) / ref if ref else None
    score = calcular_score(rsi_v, macd_v, macd_s, precio, wma21_v, e150_v, e200_v, vol_rel)
    lbl, bg, fg = señal_label(score, vol_rel)

    # ── Estrategias clásicas (artículo de referencia) ─────────────────────────
    # Estrategia 1: SMA Crossover — SMA20 vs SMA50
    sma20 = float(close.iloc[-20:].mean()) if len(close) >= 20 else None
    sma50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else None
    strat1 = "✅ COMPRAR" if (sma20 and sma50 and sma20 > sma50) else "⚠️ VENDER"

    # Estrategia 2: Momentum — log return 20 días
    mom20 = float(np.log(close.iloc[-1] / close.iloc[-21])) if len(close) >= 21 else None
    strat2 = "✅ COMPRAR" if (mom20 is not None and mom20 > 0) else "⚠️ VENDER"

    # Estrategia 3: Mean Reversion — z-score 20 días
    if len(close) >= 20:
        rmean = close.iloc[-20:].mean()
        rstd  = close.iloc[-20:].std()
        zsc   = (precio - rmean) / rstd if rstd > 0 else 0
    else:
        zsc = 0
    strat3 = "🔥 COMPRAR" if zsc < -1.0 else ("⚠️ VENDER" if zsc > 1.0 else "⏳ ESPERAR")

    return {
        "ticker":    ticker,
        "name":      fund.get("name", ticker),
        "sector":    fund.get("sector", "—"),
        "precio":    precio,
        "dia":       (precio - prev) / prev,
        "semana":    (precio - w_ago) / w_ago,
        "mes":       (precio - m_ago) / m_ago,
        "rsi":       rsi_v,
        "macd_bull": macd_v is not None and macd_s is not None and macd_v > macd_s,
        "dist_wma21": dist(wma21_v),
        "dist150":   dist(e150_v),
        "dist200":   dist(e200_v),
        "vol_rel":   vol_rel,
        "strat1":    strat1,
        "strat2":    strat2,
        "strat3":    strat3,
        "zscore":    round(zsc, 2),
        "momentum":  round(mom20 * 100, 2) if mom20 else None,
        "pe":        fund.get("pe"),
        "pb":        fund.get("pb"),
        "score":     score,
        "señal":     lbl,
        "señal_bg":  bg,
        "señal_fg":  fg,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Scanner Pro")
    st.markdown("---")
    periodo = st.selectbox("Período de datos", ["1y", "2y", "6mo", "3mo"], index=0)
    mostrar_fund = st.checkbox("Incluir fundamentales (P/E, P/B)", value=True)
    st.markdown("---")
    st.markdown("**Filtros**")
    filtro_señal   = st.multiselect("Señales", ["COMPRA FUERTE","COMPRAR","ESPERAR","REDUCIR","VENDER"], default=[])
    rsi_rng        = st.slider("Rango RSI", 0, 100, (0, 100))
    solo_whale     = st.checkbox("Solo 🐋 Manos Grandes")
    solo_macd_bull = st.checkbox("Solo MACD alcista")
    min_vol_rel    = st.slider("Vol. Relativo mínimo", 0.0, 3.0, 0.0, 0.1)
    sort_by        = st.selectbox("Ordenar por", ["Score (mejor primero)","RSI (más bajo)","RSI (más alto)","Variación semana","Variación mes","Vol. Relativo"])
    st.markdown("---")
    st.caption("Datos: Yahoo Finance · ~15min diferido\nNo es asesoramiento financiero")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Scanner de Oportunidades Pro")

@st.cache_data(ttl=1800)
def get_market_metrics():
    """Descarga SPY, QQQ, BTC-USD, ETH-USD, BNB-USD, SOL-USD y VIX de una sola vez."""
    indices = ["SPY", "QQQ", "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "^VIX"]
    try:
        raw = yf.download(indices, period="1y", group_by="ticker",
                          progress=False, auto_adjust=True, threads=True)
        result = {}
        for t in indices:
            try:
                sub = raw[t]["Close"].dropna()
                if isinstance(sub, pd.DataFrame): sub = sub.iloc[:,0]
                p    = float(sub.iloc[-1])
                prev = float(sub.iloc[-2])
                e200 = float(sub.iloc[-200:].mean()) if len(sub) >= 200 else None
                rsi_v = rsi_calc(sub)
                result[t] = {
                    "price":  p,
                    "chg":    (p - prev) / prev,
                    "e200":   e200,
                    "dist200": (p - e200) / e200 if e200 else None,
                    "rsi":    rsi_v,
                }
            except:
                result[t] = {}
        return result
    except:
        return {}

with st.spinner("Cargando métricas de mercado..."):
    mm = get_market_metrics()

# ── FILA 1: Indicador Buffett + SPY + QQQ ────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
  <div style="font-size:10px;color:#5a7a94;letter-spacing:2px;text-transform:uppercase">Indicadores de Mercado</div>
  <div style="flex:1;height:1px;background:#1a2d3f"></div>
  <div style="font-size:10px;color:#3d5a72">Warren Buffett: SPY precio / SMA histórica</div>
</div>
""", unsafe_allow_html=True)

spy  = mm.get("SPY",  {})
qqq  = mm.get("QQQ",  {})
btc  = mm.get("BTC-USD", {})
eth  = mm.get("ETH-USD", {})
bnb  = mm.get("BNB-USD", {})
sol  = mm.get("SOL-USD", {})
vix_d = mm.get("^VIX",  {})

# Buffett label (SPY precio vs su propia SMA larga)
spy_ratio = (spy.get("price", 0) / spy.get("e200", 1) * 100) if spy.get("e200") else 100
buff_lbl  = "⚠️ BURBUJA" if spy_ratio > 120 else ("🔥 BARATO" if spy_ratio < 85 else "⚖️ NEUTRAL")
vix_val   = vix_d.get("price")
vix_lbl   = "😱 MIEDO" if vix_val and vix_val > 30 else ("😰 Elevado" if vix_val and vix_val > 20 else "😌 Calmo")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("🏛️ Indicador Buffett", buff_lbl,
            f"SPY {spy_ratio:.0f}% de SMA200",
            help="Precio SPY vs su media histórica. >120% = mercado caro según lógica Buffett.")
col2.metric("SPY",
            f"${spy.get('price',0):,.0f}",
            fmt_pct(spy.get('chg')))
col3.metric("QQQ",
            f"${qqq.get('price',0):,.0f}",
            fmt_pct(qqq.get('chg')))
col4.metric("VIX", f"{vix_val:.1f}" if vix_val else "—", vix_lbl)
col5.metric("RSI SPY", f"{spy.get('rsi',0):.1f}" if spy.get('rsi') else "—",
            "sobrecomprado" if spy.get('rsi') and spy['rsi'] > 70 else ("sobrevendido" if spy.get('rsi') and spy['rsi'] < 30 else "neutral"))
col6.metric("RSI QQQ", f"{qqq.get('rsi',0):.1f}" if qqq.get('rsi') else "—")

# ── FILA 2: Distancia SMA200 + Crypto ────────────────────────────────────────
st.markdown("""<div style="display:flex;align-items:center;gap:10px;margin:8px 0 6px">
  <div style="font-size:10px;color:#5a7a94;letter-spacing:2px;text-transform:uppercase">Distancia SMA200 · Crypto</div>
  <div style="flex:1;height:1px;background:#1a2d3f"></div>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("SPY vs SMA200",   fmt_pct(spy.get("dist200")),
          "↑ sobre media" if spy.get("dist200") and spy["dist200"] > 0 else "↓ bajo media")
c2.metric("QQQ vs SMA200",   fmt_pct(qqq.get("dist200")),
          "↑ sobre media" if qqq.get("dist200") and qqq["dist200"] > 0 else "↓ bajo media")
c3.metric("BTC",  f"${btc.get('price',0):,.0f}", fmt_pct(btc.get('chg')))
c4.metric("ETH",  f"${eth.get('price',0):,.0f}", fmt_pct(eth.get('chg')))
c5.metric("BNB",  f"${bnb.get('price',0):,.0f}", fmt_pct(bnb.get('chg')))
c6.metric("SOL",  f"${sol.get('price',0):,.0f}", fmt_pct(sol.get('chg')))

st.markdown("---")

# ── Grupos curados ────────────────────────────────────────────────────────────
GRUPOS = {
    # ── USA por sector ────────────────────────────────────────────────────────
    "🇺🇸 Top 50 USA": (
        "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, BRK-B, AVGO, JPM, "
        "LLY, V, UNH, XOM, MA, COST, HD, PG, JNJ, ABBV, "
        "WMT, MRK, BAC, NFLX, CRM, CVX, ORCL, AMD, KO, PEP, "
        "TMO, ACN, LIN, MCD, ABT, GS, IBM, TXN, QCOM, GE, "
        "ISRG, SPGI, AXP, MS, HON, AMGN, CAT, RTX, LOW, AMAT"
    ),
    "💻 Tecnología": (
        "AAPL, MSFT, NVDA, AMD, INTC, AVGO, QCOM, TXN, AMAT, KLAC, "
        "LRCX, MU, SNPS, CDNS, MRVL, ON, TSM, ASML, ADBE, CRM, "
        "ORCL, IBM, SAP, SNOW, PLTR, DDOG, MDB, NOW, WDAY, TEAM"
    ),
    "🤖 Inteligencia Artificial": (
        "NVDA, MSFT, GOOGL, META, AMZN, AMD, AVGO, PLTR, SOUN, BBAI, "
        "ARQT, UPST, AI, IONQ, RGTI, QUBT, BTBT, SMCI, ARM, TSM"
    ),
    "📡 Comunicación & Media": (
        "GOOGL, META, NFLX, DIS, CMCSA, T, VZ, WBD, PARA, SNAP, "
        "PINS, RDDT, SPOT, MTCH, LYV, OMC, IPG, NYT, FOXA, TTWO"
    ),
    "💊 Salud & Biotech": (
        "LLY, UNH, JNJ, ABBV, MRK, TMO, ABT, BMY, AMGN, GILD, "
        "REGN, VRTX, ISRG, BSX, ELV, CVS, CI, HUM, PFE, BIIB, "
        "MRNA, ZBH, IQV, A, DXCM, BIOX, NVO, AZN, GSK, RHHBY"
    ),
    "💳 Financieras & Bancos": (
        "JPM, BAC, WFC, GS, MS, C, AXP, BLK, SCHW, USB, "
        "TFC, PNC, COF, MTB, RF, CFG, HBAN, KEY, FITB, ZION, "
        "V, MA, PYPL, SQ, NU, ITUB, BBD, BPAC3.SA"
    ),
    "🛒 Consumo & Retail": (
        "WMT, COST, HD, MCD, NKE, SBUX, TGT, LOW, AMZN, BABA, "
        "PG, KO, PEP, PM, MO, CL, EL, KMB, GIS, K, "
        "MELI, SHOP, ETSY, EBAY, W, RH, LULU, GPS, ANF, DECK"
    ),
    "⚡ Energía & Recursos": (
        "XOM, CVX, COP, EOG, SLB, MPC, PSX, VLO, OXY, PXD, "
        "FSLR, ENPH, SEDG, NEE, CEG, VST, NRG, PCG, AES, ETR, "
        "URA, CCJ, UUUU, DNN, NXE, GLD, SLV, GOLD, AEM, WPM"
    ),
    "🏦 ETFs Clave": (
        "SPY, QQQ, IWM, DIA, VTI, EEM, EFA, GLD, SLV, TLT, "
        "HYG, LQD, XLF, XLK, XLV, XLE, XLI, XLC, XLRE, XLU, "
        "ILF, EWZ, ARKK, ARKG, ARKW, VNQ, IBIT, FBTC, BITO, GBTC"
    ),
    "₿ Top 15 Crypto": (
        "BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD, DOGE-USD, ADA-USD, "
        "AVAX-USD, SHIB-USD, DOT-USD, MATIC-USD, LTC-USD, UNI-USD, LINK-USD, ATOM-USD"
    ),
    "🇦🇷 Merval & CEDEARs": (
        "GGAL.BA, YPF.BA, PAMP.BA, TXAR.BA, METR.BA, HARG.BA, CARC.BA, "
        "BBAR.BA, BMA.BA, SUPV.BA, CEPU.BA, COME.BA, TECO2.BA, IRSA.BA, "
        "ALUA.BA, VALO.BA, CRES.BA, LOMA.BA, CVH.BA, EDN.BA"
    ),
}

# ── Tickers Input ─────────────────────────────────────────────────────────────
# Inicializar session state
if "tickers_text" not in st.session_state:
    st.session_state["tickers_text"] = (
        "TSLA, AAPL, NVDA, AMZN, META, GOOGL, MSFT, INTC, AMD, TSM, "
        "QCOM, AVGO, PLTR, CRM, ORCL, SNOW, BABA, AMGN, PFE, MRK, "
        "ABBV, WMT, PG, PEP, JNJ, KO, MCD, HD, NKE, SBUX, "
        "BAC, JPM, WFC, GS, V, MA, AXP, PYPL, NU, "
        "BIOX, MELI, URA, NIO, RGTI, ILF, FSLR, ADBE, "
        "BTC-USD, ETH-USD, BNB-USD, SOL-USD"
    )

if "grupo_activo" not in st.session_state:
    st.session_state["grupo_activo"] = None

with st.expander("⚙️ Configurar Tickers para Escanear", expanded=True):

    st.markdown("""
<div style="font-size:10px;color:#5a7a94;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">
  Seleccioná un grupo para REEMPLAZAR la lista, o usá los botones + para AGREGAR
</div>""", unsafe_allow_html=True)

    # ── Fila de grupos — REEMPLAZAR ───────────────────────────────────────────
    btn_cols = st.columns(4)
    grupo_nombres = list(GRUPOS.keys())

    # Fila 1
    for i, nombre in enumerate(grupo_nombres[:4]):
        if btn_cols[i].button(nombre, use_container_width=True, key=f"grp_{i}"):
            st.session_state["tickers_text"] = GRUPOS[nombre]
            st.session_state["grupo_activo"] = nombre
            st.rerun()

    btn_cols2 = st.columns(4)
    for i, nombre in enumerate(grupo_nombres[4:8]):
        if btn_cols2[i].button(nombre, use_container_width=True, key=f"grp_{i+4}"):
            st.session_state["tickers_text"] = GRUPOS[nombre]
            st.session_state["grupo_activo"] = nombre
            st.rerun()

    btn_cols3 = st.columns(3)
    for i, nombre in enumerate(grupo_nombres[8:11]):
        if btn_cols3[i].button(nombre, use_container_width=True, key=f"grp_{i+8}"):
            st.session_state["tickers_text"] = GRUPOS[nombre]
            st.session_state["grupo_activo"] = nombre
            st.rerun()

    # ── Botones agregar ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;color:#3d5a72;margin:8px 0 4px'>➕ Agregar al listado actual:</div>", unsafe_allow_html=True)
    add_cols = st.columns(5)
    add_grupos = {
        "+ Crypto":    GRUPOS["₿ Top 15 Crypto"],
        "+ ETFs":      GRUPOS["🏦 ETFs Clave"],
        "+ Merval":    GRUPOS["🇦🇷 Merval & CEDEARs"],
        "+ IA":        GRUPOS["🤖 Inteligencia Artificial"],
        "+ Financ.":   GRUPOS["💳 Financieras & Bancos"],
    }
    for i, (label, tks) in enumerate(add_grupos.items()):
        if add_cols[i].button(label, use_container_width=True, key=f"add_{i}"):
            current = st.session_state.get("tickers_text", "")
            existing = set(t.strip().upper() for t in current.split(",") if t.strip())
            new_tks  = [t.strip() for t in tks.split(",") if t.strip().upper() not in existing]
            st.session_state["tickers_text"] = current.rstrip(", ") + (", " if current.strip() else "") + ", ".join(new_tks)
            st.rerun()

    # ── Grupo activo badge ────────────────────────────────────────────────────
    if st.session_state.get("grupo_activo"):
        st.markdown(f"""<div style="display:inline-block;background:rgba(0,200,240,.1);border:1px solid rgba(0,200,240,.3);
        border-radius:6px;padding:4px 10px;font-size:10px;color:#00c8f0;margin:6px 0">
        ✓ Grupo activo: <b>{st.session_state["grupo_activo"]}</b></div>""", unsafe_allow_html=True)

    # ── Texto de tickers ──────────────────────────────────────────────────────
    # NO usamos key= para evitar que Streamlit ignore el value= tras rerun()
    def _on_ticker_change():
        st.session_state["tickers_text"] = st.session_state["_ticker_area"]
        st.session_state["grupo_activo"] = None

    st.text_area(
        "Editá manualmente o usá los grupos de arriba:",
        value=st.session_state["tickers_text"],
        height=90,
        key="_ticker_area",
        on_change=_on_ticker_change,
    )

    tickers_list = list(dict.fromkeys(
        t.strip().upper() for t in st.session_state["tickers_text"].split(",") if t.strip()
    ))
    if len(tickers_list) > 150:
        tickers_list = tickers_list[:150]
        st.warning("Se toman los primeros 150 tickers.")

    # Conteo por tipo
    n_crypto = sum(1 for t in tickers_list if "-USD" in t)
    n_arg    = sum(1 for t in tickers_list if t.endswith(".BA"))
    n_usa    = len(tickers_list) - n_crypto - n_arg
    st.caption(
        f"📊 {len(tickers_list)} / 150 tickers  ·  "
        f"🇺🇸 {n_usa} USA  ·  ₿ {n_crypto} Crypto  ·  🇦🇷 {n_arg} ARG  ·  "
        f"~{max(5, len(tickers_list)//5):.0f}s estimados"
    )

# Recalcular tickers_list fuera del expander (por si el expander no se ejecutó)
tickers_list = list(dict.fromkeys(
    t.strip().upper() for t in st.session_state.get("tickers_text", "").split(",") if t.strip()
))
if len(tickers_list) > 150:
    tickers_list = tickers_list[:150]

col_btn1, col_btn2 = st.columns([4,1])
run = col_btn1.button(f"🚀 ESCANEAR {len(tickers_list)} TICKERS", type="primary", use_container_width=True)
if col_btn2.button("🔄 Limpiar caché", use_container_width=True):
    st.cache_data.clear()
    st.success("Caché limpiada")

# ── Ejecución ─────────────────────────────────────────────────────────────────
if run:
    t_start = time.time()
    pb = st.progress(0, text="📡 Descargando datos históricos...")
    raw_data = descargar_masivo(tuple(tickers_list), period=periodo)
    pb.progress(0.4, text="📊 Cargando fundamentales en paralelo...")
    fund_data = get_fundamentales_bulk(tuple(tickers_list)) if mostrar_fund else {t: {"name":t,"pe":None,"pb":None,"sector":"—"} for t in tickers_list}
    multi = len(tickers_list) > 1
    resultados, errores = [], []
    for i, ticker in enumerate(tickers_list):
        pb.progress(0.5 + 0.5*(i/len(tickers_list)), text=f"⚡ Analizando {ticker} ({i+1}/{len(tickers_list)})...")
        r = analizar_ticker(ticker, raw_data, multi, fund_data.get(ticker, {"name":ticker}))
        if r: resultados.append(r)
        else: errores.append(ticker)
    pb.empty()
    elapsed = time.time() - t_start
    if not resultados:
        st.error("No se pudieron obtener datos.")
    else:
        df = pd.DataFrame(resultados)
        st.session_state["scanner_result"] = df
        st.session_state["scanner_errores"] = errores
        st.success(f"✅ {len(resultados)} activos en {elapsed:.1f}s · {len(errores)} errores")

# ── Resultados ────────────────────────────────────────────────────────────────
if "scanner_result" in st.session_state:
    df = st.session_state["scanner_result"].copy()
    errores = st.session_state.get("scanner_errores", [])

    if filtro_señal:
        df = df[df["señal"].apply(lambda x: any(f in x for f in filtro_señal))]
    df = df[df["rsi"].apply(lambda x: x is None or (rsi_rng[0] <= x <= rsi_rng[1]))]
    if solo_whale:     df = df[df["señal"].str.contains("🐋")]
    if solo_macd_bull: df = df[df["macd_bull"] == True]
    if min_vol_rel > 0: df = df[df["vol_rel"] >= min_vol_rel]

    sort_map = {
        "Score (mejor primero)": ("score", False),
        "RSI (más bajo)":        ("rsi", True),
        "RSI (más alto)":        ("rsi", False),
        "Variación semana":      ("semana", False),
        "Variación mes":         ("mes", False),
        "Vol. Relativo":         ("vol_rel", False),
    }
    sc, sa = sort_map.get(sort_by, ("score", False))
    df = df.sort_values(sc, ascending=sa, na_position="last")

    tab_tabla, tab_top, tab_alertas, tab_guia = st.tabs([
        f"📊 Tabla completa ({len(df)})", "🏆 Top oportunidades", "🔔 Alertas", "📚 Guía de Indicadores"
    ])

    # ══ TAB TABLA ═════════════════════════════════════════════════════════════
    with tab_tabla:
        disp_cols = ["ticker","name","precio","dia","semana","mes","rsi","macd_bull",
                     "dist_wma21","dist150","dist200","vol_rel","strat1","strat2","strat3"]
        if mostrar_fund: disp_cols += ["pe","pb"]
        disp_cols += ["score","señal"]

        display = df[disp_cols].copy()
        display.columns = (
            ["Ticker","Nombre","Precio","Día","Semana","Mes","RSI","MACD",
             "vsWMA21","vsEMA150","vsEMA200","Vol.Rel","Estrat.1","Estrat.2","Estrat.3"] +
            (["P/E","P/B"] if mostrar_fund else []) +
            ["Score","Señal"]
        )
        display["Precio"]     = display["Precio"].apply(fmt2)
        display["Día"]        = display["Día"].apply(fmt_pct)
        display["Semana"]     = display["Semana"].apply(fmt_pct)
        display["Mes"]        = display["Mes"].apply(fmt_pct)
        display["RSI"]        = display["RSI"].apply(lambda x: f"{x:.1f}" if x else "—")
        display["MACD"]       = display["MACD"].apply(lambda x: "🟢 Bull" if x else "🔴 Bear")
        display["vsWMA21"]    = display["vsWMA21"].apply(fmt_pct)
        display["vsEMA150"]   = display["vsEMA150"].apply(fmt_pct)
        display["vsEMA200"]   = display["vsEMA200"].apply(fmt_pct)
        # Vol.Rel: muestra señal de compra/venta basada en dirección del precio + volumen alto
        def fmt_vol(row_idx):
            vol = df.iloc[row_idx]["vol_rel"]
            bull = df.iloc[row_idx]["macd_bull"]
            score_v = df.iloc[row_idx]["score"]
            if vol > 1.6:
                direction = "🐋 C" if score_v >= 2 else ("🐋 V" if score_v <= -2 else "🐋 N")
            else:
                direction = f"{vol:.1f}x"
            return direction
        display["Vol.Rel"]    = [fmt_vol(i) for i in range(len(display))]
        if mostrar_fund:
            display["P/E"] = display["P/E"].apply(lambda x: f"{x:.1f}x" if x else "—")
            display["P/B"] = display["P/B"].apply(lambda x: f"{x:.1f}x" if x else "—")
        display["Score"] = display["Score"].apply(lambda x: f"{x:+d}" if x is not None else "—")

        def color_rsi_cell(val):
            try:
                v = float(val)
                if v < 30:  return "background-color:#004d26;color:#69ffb0;font-weight:bold"
                if v < 40:  return "color:#80ffbb;font-weight:bold"
                if v > 70:  return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
                if v > 60:  return "color:#ff9100"
                return "color:#ffd740"
            except: return ""

        def color_pct(val):
            try:
                n = float(val.replace("%","").replace("+",""))
                if n > 5:  return "color:#00e676;font-weight:bold"
                if n > 0:  return "color:#69ffb0"
                if n < -5: return "color:#ff3d57;font-weight:bold"
                if n < 0:  return "color:#ff8a95"
                return "color:#ffd740"
            except: return ""

        def color_score(val):
            try:
                v = int(val.replace("+",""))
                if v >= 4:  return "color:#00e676;font-weight:bold"
                if v >= 2:  return "color:#69ffb0"
                if v <= -4: return "color:#ff3d57;font-weight:bold"
                if v <= -2: return "color:#ff8a95"
                return "color:#ffd740"
            except: return ""

        def color_señal(val):
            if "COMPRA FUERTE" in str(val): return "background-color:#004d26;color:#69ffb0;font-weight:bold"
            if "COMPRAR"       in str(val): return "background-color:#003d1a;color:#80ffbb;font-weight:bold"
            if "REDUCIR"       in str(val): return "background-color:#3d2000;color:#ff9100;font-weight:bold"
            if "VENDER"        in str(val): return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
            return "color:#ffd740"

        def color_pe(val):
            # Verde < 20x, amarillo 20-40x, rojo > 40x
            try:
                v = float(val.replace("x",""))
                if v < 15:  return "background-color:#004d26;color:#69ffb0;font-weight:bold"
                if v < 20:  return "color:#80ffbb"
                if v < 40:  return "color:#ffd740"
                return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
            except: return "color:#3d5a72"

        def color_pb(val):
            # Verde < 1.5x, amarillo 1.5-4x, rojo > 4x
            try:
                v = float(val.replace("x",""))
                if v < 1.5: return "background-color:#004d26;color:#69ffb0;font-weight:bold"
                if v < 4:   return "color:#ffd740"
                return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
            except: return "color:#3d5a72"

        def color_vol(val):
            # 🐋 C = compra institucional (verde), 🐋 V = venta institucional (rojo), 🐋 N = neutral (cyan)
            if "🐋 C" in str(val): return "background-color:#003d1a;color:#69ffb0;font-weight:bold"
            if "🐋 V" in str(val): return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
            if "🐋 N" in str(val): return "background-color:#00405c;color:#bfe5ff;font-weight:bold"
            try:
                v = float(str(val).replace("x",""))
                if v >= 1.6: return "color:#00c8f0;font-weight:bold"
            except: pass
            return "color:#5a7a94"

        pct_cols = ["Día","Semana","Mes","vsWMA21","vsEMA150","vsEMA200"]
        def color_strat(val):
            if "COMPRAR" in str(val) or "COMPRA" in str(val):
                return "background-color:#003d1a;color:#69ffb0;font-weight:bold"
            if "VENDER" in str(val): return "background-color:#4d0000;color:#ff8a95;font-weight:bold"
            return "color:#ffd740"

        strat_cols = ["Estrat.1","Estrat.2","Estrat.3"]
        styled = (
            display.style
            .map(color_rsi_cell, subset=["RSI"])
            .map(color_pct,      subset=pct_cols)
            .map(color_score,    subset=["Score"])
            .map(color_señal,    subset=["Señal"])
            .map(color_vol,      subset=["Vol.Rel"])
            .map(color_strat,    subset=strat_cols)
        )
        if mostrar_fund:
            styled = styled.map(color_pe, subset=["P/E"]).map(color_pb, subset=["P/B"])

        st.dataframe(styled, use_container_width=True, hide_index=True, height=700)

        # Leyenda Vol.Rel
        st.markdown("""
<div style="display:flex;gap:16px;padding:8px 0;font-size:10px;color:#5a7a94">
  <span style="color:#5a7a94">Vol.Rel:</span>
  <span style="background:#003d1a;color:#69ffb0;padding:1px 7px;border-radius:4px;font-weight:700">🐋 C</span> Volumen inusual + señal alcista
  <span style="background:#4d0000;color:#ff8a95;padding:1px 7px;border-radius:4px;font-weight:700">🐋 V</span> Volumen inusual + señal bajista
  <span style="background:#00405c;color:#bfe5ff;padding:1px 7px;border-radius:4px;font-weight:700">🐋 N</span> Volumen inusual + neutral
  <span style="color:#ffd740">· P/E verde &lt;20x · amarillo 20–40x · rojo &gt;40x · P/B verde &lt;1.5x · rojo &gt;4x · Estrat.1=SMA Cross · Estrat.2=Momentum · Estrat.3=Mean Rev.</span>
</div>""", unsafe_allow_html=True)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Exportar CSV completo", data=csv_buf.getvalue(),
                           file_name=f"scanner_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
        if errores:
            st.warning(f"⚠️ No se pudieron cargar: {', '.join(errores)}")

    # ══ TAB TOP ══════════════════════════════════════════════════════════════
    with tab_top:
        top_compra = df[df["score"] >= 3].head(6)
        top_venta  = df[df["score"] <= -3].head(6)
        if not top_compra.empty:
            st.markdown("### 🚀 Mejores oportunidades de ENTRADA")
            cols = st.columns(min(3, len(top_compra)))
            for i, (_, r) in enumerate(top_compra.iterrows()):
                with cols[i % 3]:
                    lbl, bg, fg = señal_label(r["score"], r["vol_rel"])
                    st.markdown(f"""
<div style="background:{bg};border:1px solid #243d52;border-radius:10px;padding:14px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div><div style="font-size:18px;font-weight:800;color:#e2f0ff">{r['ticker']}</div>
         <div style="font-size:9px;color:#5a7a94;margin-top:2px">{r['name'][:28]}</div></div>
    <div style="font-size:10px;font-weight:700;color:{fg};border:1px solid {fg}33;border-radius:5px;padding:3px 8px">{r['señal']}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px">
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Precio</div>
         <div style="font-size:14px;color:#e2f0ff;font-weight:500">${r['precio']:,.2f}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">RSI</div>
         <div style="font-size:14px;color:{'#00e676' if r['rsi'] and r['rsi']<35 else '#ffd740'};font-weight:500">{f"{r['rsi']:.0f}" if r['rsi'] else '—'}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Semana</div>
         <div style="font-size:13px;color:{'#00e676' if r['semana']>=0 else '#ff3d57'};font-weight:500">{fmt_pct(r['semana'])}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Score</div>
         <div style="font-size:14px;color:{fg};font-weight:700">{r['score']:+d}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">vsEMA200</div>
         <div style="font-size:13px;color:{'#00e676' if r['dist200'] and r['dist200']>=0 else '#ff3d57'}">{fmt_pct(r['dist200'])}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Vol.Rel</div>
         <div style="font-size:13px;color:{'#00c8f0' if r['vol_rel']>1.5 else '#5a7a94'}">{r['vol_rel']:.1f}x {'🐋' if r['vol_rel']>1.5 else ''}</div></div>
  </div>
  {"<div style='margin-top:10px;font-size:9px;color:#5a7a94'>P/E: "+str(f"{r['pe']:.0f}x" if r['pe'] else '—')+" · P/B: "+str(f"{r['pb']:.1f}x" if r['pb'] else '—')+"</div>" if mostrar_fund else ''}
</div>""", unsafe_allow_html=True)

        if not top_venta.empty:
            st.markdown("### ⚠️ Señales de SALIDA / REDUCIR")
            cols2 = st.columns(min(3, len(top_venta)))
            for i, (_, r) in enumerate(top_venta.iterrows()):
                with cols2[i % 3]:
                    lbl, bg, fg = señal_label(r["score"], r["vol_rel"])
                    st.markdown(f"""
<div style="background:{bg};border:1px solid #4d0000;border-radius:10px;padding:14px;margin-bottom:10px">
  <div style="display:flex;justify-content:space-between">
    <div><div style="font-size:18px;font-weight:800;color:#e2f0ff">{r['ticker']}</div>
         <div style="font-size:9px;color:#5a7a94">{r['name'][:28]}</div></div>
    <div style="font-size:10px;font-weight:700;color:{fg};padding:3px 8px;border:1px solid {fg}33;border-radius:5px">{r['señal']}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px">
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Precio</div>
         <div style="font-size:14px;color:#e2f0ff">${r['precio']:,.2f}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">RSI</div>
         <div style="font-size:14px;color:#ff3d57;font-weight:500">{f"{r['rsi']:.0f}" if r['rsi'] else '—'}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Semana</div>
         <div style="font-size:13px;color:{'#00e676' if r['semana']>=0 else '#ff3d57'}">{fmt_pct(r['semana'])}</div></div>
    <div><div style="font-size:8px;color:#3d5a72;text-transform:uppercase;letter-spacing:1px">Score</div>
         <div style="font-size:14px;color:{fg};font-weight:700">{r['score']:+d}</div></div>
  </div>
</div>""", unsafe_allow_html=True)
        if top_compra.empty and top_venta.empty:
            st.info("No hay señales fuertes en los activos analizados.")

    # ══ TAB ALERTAS ══════════════════════════════════════════════════════════
    with tab_alertas:
        alertas = []
        for _, r in df.iterrows():
            if r["rsi"] and r["rsi"] < 30:
                alertas.append(("ag","🟢",f"**{r['ticker']}** RSI sobrevendido: {r['rsi']:.0f} — posible oportunidad"))
            if r["rsi"] and r["rsi"] > 72:
                alertas.append(("ar","🔴",f"**{r['ticker']}** RSI sobrecomprado: {r['rsi']:.0f} — posible toma de ganancias"))
            if r["dist200"] and r["dist200"] < -0.05:
                alertas.append(("ay","📉",f"**{r['ticker']}** bajo EMA200 ({fmt_pct(r['dist200'])}) — tendencia bajista"))
            if r["vol_rel"] > 1.8:
                sc_v = r["score"]
                dir_v = "COMPRADORA 🟢" if sc_v >= 2 else ("VENDEDORA 🔴" if sc_v <= -2 else "neutral")
                alertas.append(("ab","🐋",f"**{r['ticker']}** volumen inusual {r['vol_rel']:.1f}x — actividad institucional {dir_v}"))
            # Mean Reversion alert via zscore
            zs = r.get("zscore", 0) or 0
            if zs < -1.5:
                alertas.append(("ag","📊",f"**{r['ticker']}** z-score bajo ({zs:.1f}) — zona de Mean Reversion / posible rebote"))
            if zs > 1.5:
                alertas.append(("ar","📊",f"**{r['ticker']}** z-score alto ({zs:.1f}) — sobreextendido vs media"))
            if mostrar_fund and r.get("pe") and r["pe"] > 40:
                alertas.append(("ap","📐",f"**{r['ticker']}** P/E elevado: {r['pe']:.0f}x — valuación cara"))
        colors = {"ag":"#004d26","ar":"#4d0000","ay":"#3d2000","ab":"#00405c","ap":"#2d1a4d"}
        if alertas:
            for cls_a, icon, msg in sorted(alertas, key=lambda x: x[0]):
                st.markdown(f"""<div style="background:{colors.get(cls_a,'#1a2d3f')};border-left:3px solid;border-radius:8px;padding:10px 14px;margin-bottom:6px;font-size:12px;color:#e2f0ff">{icon} {msg}</div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Sin alertas activas.")

    # ══ TAB GUÍA ══════════════════════════════════════════════════════════════
    with tab_guia:
        st.markdown("""<style>
.guia-card{background:#101820;border:1px solid #1a2d3f;border-radius:12px;padding:20px 22px;margin-bottom:16px}
.guia-title{font-size:15px;font-weight:700;color:#00c8f0;margin-bottom:10px;border-bottom:1px solid #1a2d3f;padding-bottom:8px}
.guia-formula{background:#070b0f;border:1px solid #1a2d3f;border-left:3px solid #00c8f0;border-radius:6px;padding:10px 14px;font-family:monospace;font-size:12px;color:#b57bee;margin:10px 0}
.guia-param-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}
.guia-param-table th{background:#0c1219;color:#5a7a94;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;padding:8px 12px;text-align:left;border-bottom:1px solid #1a2d3f}
.guia-param-table td{padding:8px 12px;border-bottom:1px solid rgba(26,45,63,.5);color:#b8cfe0;vertical-align:top}
.guia-param-table tr:last-child td{border-bottom:none}
.guia-param-table td:first-child{color:#e2f0ff;font-weight:500;min-width:110px}
.zone-green{display:inline-block;background:rgba(0,230,118,.15);color:#69ffb0;border:1px solid rgba(0,230,118,.3);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700}
.zone-red{display:inline-block;background:rgba(255,61,87,.12);color:#ff8a95;border:1px solid rgba(255,61,87,.3);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700}
.zone-yellow{display:inline-block;background:rgba(255,215,64,.08);color:#ffd740;border:1px solid rgba(255,215,64,.2);border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700}
.ref-link{color:#00c8f0;font-size:11px}
</style>""", unsafe_allow_html=True)

        st.markdown("## 📚 Guía de Indicadores Técnicos")
        st.caption("Parámetros, fórmulas, zonas de señal y referencias bibliográficas de cada indicador.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">📊 RSI — Relative Strength Index</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">Oscilador de momentum que mide la velocidad y magnitud de los movimientos. Desarrollado por J. Welles Wilder (1978). Oscila entre 0 y 100.</p>
  <div class="guia-formula">RSI = 100 − [100 / (1 + RS)]<br>RS = Promedio ganancias (n) / Promedio pérdidas (n)</div>
  <table class="guia-param-table">
    <tr><th>Parámetro</th><th>Valor</th><th>Descripción</th></tr>
    <tr><td>Período (n)</td><td>14 sesiones</td><td>Estándar Wilder. Más corto = más sensible.</td></tr>
  </table>
  <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px">
    <div><span class="zone-green">RSI &lt; 30</span> <span style="font-size:10px;color:#b8cfe0">Sobrevendido · +2 pts</span></div>
    <div><span class="zone-green" style="background:rgba(0,230,118,.07)">RSI 30–45</span> <span style="font-size:10px;color:#b8cfe0">Barato · +1 pt</span></div>
    <div><span class="zone-yellow">RSI 45–55</span> <span style="font-size:10px;color:#b8cfe0">Neutro · 0</span></div>
    <div><span class="zone-red" style="background:rgba(255,61,87,.05)">RSI 55–65</span> <span style="font-size:10px;color:#b8cfe0">Caro · -1 pt</span></div>
    <div><span class="zone-red">RSI &gt; 75</span> <span style="font-size:10px;color:#b8cfe0">Sobrecomprado · -3 pts</span></div>
  </div>
  <div style="margin-top:10px;font-size:10px;color:#3d5a72">📖 Wilder, J.W. (1978). <em>New Concepts in Technical Trading Systems.</em> · <a class="ref-link" href="https://www.investopedia.com/terms/r/rsi.asp" target="_blank">Investopedia</a></div>
</div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">📈 MACD — Moving Average Convergence Divergence</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">Indicador de tendencia y momentum. Desarrollado por Gerald Appel, década de 1970.</p>
  <div class="guia-formula">MACD = EMA(12) − EMA(26)<br>Signal = EMA(9) del MACD<br>Histograma = MACD − Signal</div>
  <table class="guia-param-table">
    <tr><th>Parámetro</th><th>Valor</th><th>Descripción</th></tr>
    <tr><td>EMA rápida</td><td>12 períodos</td><td>Momentum reciente</td></tr>
    <tr><td>EMA lenta</td><td>26 períodos</td><td>Tendencia más estable</td></tr>
    <tr><td>Señal</td><td>9 períodos</td><td>EMA del MACD para cruces</td></tr>
  </table>
  <div style="margin-top:14px;display:flex;gap:10px;flex-wrap:wrap">
    <div><span class="zone-green">MACD &gt; Signal</span> <span style="font-size:10px;color:#b8cfe0">Bull · +1 pt</span></div>
    <div><span class="zone-red">MACD &lt; Signal</span> <span style="font-size:10px;color:#b8cfe0">Bear · -1 pt</span></div>
  </div>
  <div style="margin-top:10px;font-size:10px;color:#3d5a72">📖 Appel, G. (2005). <em>Technical Analysis: Power Tools.</em> FT Press. · <a class="ref-link" href="https://www.investopedia.com/terms/m/macd.asp" target="_blank">Investopedia</a></div>
</div>""", unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">〰️ EMAs — Medias Móviles Exponenciales</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">Ponderan más los precios recientes. Niveles de soporte/resistencia dinámicos clave para institucionales.</p>
  <div class="guia-formula">EMA(t) = Precio(t) × k + EMA(t-1) × (1−k)<br>k = 2 / (n + 1)</div>
  <table class="guia-param-table">
    <tr><th>EMA</th><th>Uso principal</th><th>Score</th></tr>
    <tr><td>WMA 21</td><td>Tendencia corto plazo · pondera más los precios recientes linealmente</td><td><span class="zone-green">+1 pt</span></td></tr>
    <tr><td>EMA 150</td><td>Tendencia intermedia</td><td><span class="zone-green">+1 pt</span></td></tr>
    <tr><td>EMA 200</td><td>Largo plazo · institucionales</td><td><span class="zone-green">+1 pt</span></td></tr>
  </table>
  <div style="margin-top:10px;font-size:10px;color:#3d5a72">📖 Murphy, J.J. (1999). <em>Technical Analysis of the Financial Markets.</em> NYIF. · <a class="ref-link" href="https://www.investopedia.com/terms/e/ema.asp" target="_blank">Investopedia</a></div>
</div>""", unsafe_allow_html=True)

        with col_d:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">📐 Estrat. 3 · Mean Reversion (Z-Score)</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">
    Estrategia contraria: si el precio se alejó mucho de su media histórica, tenderá a volver. 
    Contraria al momentum — funciona en mercados laterales. Basada en el paper de Jegadeesh &amp; Titman (1993).
  </p>
  <div class="guia-formula">z = (Precio − Media_20) / DesvStd_20<br>z &lt; −1.0 → COMPRAR · z &gt; 0 → SALIR</div>
  <table class="guia-param-table">
    <tr><th>Parámetro</th><th>Valor</th><th>Descripción</th></tr>
    <tr><td>Ventana</td><td>20 sesiones</td><td>Media y desvío estándar rodante</td></tr>
    <tr><td>Umbral entrada</td><td>z &lt; −1.0</td><td>Precio 1 desvío bajo la media</td></tr>
  </table>
  <div style="margin-top:12px;display:flex;flex-direction:column;gap:5px">
    <div><span class="zone-green">z &lt; −1.0</span> <span style="font-size:10px;color:#b8cfe0">🔥 COMPRAR — precio muy bajo vs media</span></div>
    <div><span class="zone-yellow">−1.0 a +1.0</span> <span style="font-size:10px;color:#b8cfe0">⏳ ESPERAR</span></div>
    <div><span class="zone-red">z &gt; +1.0</span> <span style="font-size:10px;color:#b8cfe0">⚠️ VENDER — precio muy alto vs media</span></div>
  </div>
  <div style="margin-top:10px;font-size:10px;color:#3d5a72">📖 Jegadeesh, N. (1990). Evidence of Predictable Behavior. <em>Journal of Finance.</em><br>
  Poterba &amp; Summers (1988). Mean Reversion in Stock Prices. <em>Journal of Financial Economics.</em></div>
</div>""", unsafe_allow_html=True)

        col_e, col_f = st.columns(2)
        with col_e:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">📈 Estrat. 1 · SMA Crossover</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">
    Estrategia de seguimiento de tendencia. Usa dos medias simples (rápida y lenta).
    Cuando la rápida cruza hacia arriba → señal alcista. Funciona mejor en mercados con tendencia clara.
  </p>
  <div class="guia-formula">SMA_rápida(20) &gt; SMA_lenta(50) → COMPRAR<br>SMA_rápida(20) &lt; SMA_lenta(50) → VENDER</div>
  <table class="guia-param-table">
    <tr><th>Parámetro</th><th>Valor</th><th>Descripción</th></tr>
    <tr><td>SMA rápida</td><td>20 sesiones</td><td>Refleja tendencia reciente</td></tr>
    <tr><td>SMA lenta</td><td>50 sesiones</td><td>Refleja tendencia de largo plazo</td></tr>
  </table>
  <div style="margin-top:10px;font-size:11px;color:#ffd740">⚠️ En mercados laterales genera falsas señales (whipsaws). Mejor en tendencias sostenidas.</div>
  <div style="margin-top:8px;font-size:10px;color:#3d5a72">📖 Murphy, J.J. (1999). <em>Technical Analysis of the Financial Markets.</em> NYIF.</div>
</div>""", unsafe_allow_html=True)

        with col_f:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">🚀 Estrat. 2 · Momentum</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">
    Lo que sube tiende a seguir subiendo. Si el retorno acumulado de 20 días es positivo → mercado en tendencia alcista.
    Una de las anomalías más documentadas en finanzas.
  </p>
  <div class="guia-formula">Momentum = ln(Precio_hoy / Precio_hace_20d)<br>Momentum &gt; 0 → COMPRAR · Momentum ≤ 0 → VENDER</div>
  <table class="guia-param-table">
    <tr><th>Parámetro</th><th>Valor</th><th>Descripción</th></tr>
    <tr><td>Ventana</td><td>20 sesiones</td><td>Período de lookback para retorno log</td></tr>
    <tr><td>Umbral</td><td>&gt; 0%</td><td>Retorno positivo en el período</td></tr>
  </table>
  <div style="margin-top:10px;font-size:11px;color:#ffd740">⚠️ En correcciones bruscas puede dar señales tardías. Complementar con RSI para filtrar sobrecompra.</div>
  <div style="margin-top:8px;font-size:10px;color:#3d5a72">📖 Jegadeesh &amp; Titman (1993). Returns to Buying Winners. <em>Journal of Finance</em>, 48(1). · Carhart (1997). On Persistence in Mutual Fund Performance.</div>
</div>""", unsafe_allow_html=True)

        col_e2, col_f2 = st.columns(2)
        with col_e2:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">🐋 Volumen Relativo (RVOL)</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">Compara el volumen actual vs el promedio histórico. El scanner además detecta si el movimiento institucional es COMPRADOR o VENDEDOR según el Score del activo.</p>
  <div class="guia-formula">RVOL = Volumen(hoy) / SMA_Volumen(20)</div>
  <table class="guia-param-table">
    <tr><th>Valor en tabla</th><th>Significado</th></tr>
    <tr><td><span class="zone-green">🐋 C</span></td><td>Vol. inusual + Score alcista → presión compradora institucional</td></tr>
    <tr><td><span class="zone-red">🐋 V</span></td><td>Vol. inusual + Score bajista → presión vendedora institucional</td></tr>
    <tr><td style="color:#bfe5ff">🐋 N</td><td>Vol. inusual + señal neutral → dirección no definida</td></tr>
    <tr><td><span class="zone-yellow">1.0–1.5x</span></td><td>Volumen normal — sin actividad inusual</td></tr>
  </table>
  <div style="margin-top:10px;font-size:10px;color:#3d5a72">📖 Elder, A. (1993). <em>Trading for a Living.</em> Wiley. · <a class="ref-link" href="https://www.investopedia.com/terms/r/relative_volume.asp" target="_blank">Investopedia</a></div>
</div>""", unsafe_allow_html=True)

        with col_f:
            st.markdown("""<div class="guia-card">
  <div class="guia-title">🎯 Sistema de Score — Señal Final</div>
  <p style="font-size:12px;color:#b8cfe0;margin-bottom:10px">Modelo propio que combina todos los indicadores en un puntaje <b style="color:#e2f0ff">−8 a +8</b>.</p>
  <table class="guia-param-table">
    <tr><th>Indicador</th><th>Puntos</th><th>Condición</th></tr>
    <tr><td>RSI</td><td>−3 a +3</td><td>&lt;25=+3 · &lt;35=+2 · &lt;45=+1 · &gt;55=−1 · &gt;65=−2 · &gt;75=−3</td></tr>
    <tr><td>MACD</td><td>−1 a +1</td><td>Bull=+1 · Bear=−1</td></tr>
    <tr><td>WMA 21</td><td>−1 a +1</td><td>Precio sobre=+1 · bajo=−1</td></tr>
    <tr><td>EMA 150</td><td>−1 a +1</td><td>Precio sobre=+1 · bajo=−1</td></tr>
    <tr><td>EMA 200</td><td>−1 a +1</td><td>Precio sobre=+1 · bajo=−1</td></tr>
  </table>
  <div style="margin-top:12px;display:flex;flex-direction:column;gap:5px">
    <div><span class="zone-green">≥ +5</span> <span style="font-size:10px;color:#b8cfe0;margin-left:6px">🚀 COMPRA FUERTE</span></div>
    <div><span class="zone-green" style="background:rgba(0,230,118,.07)">+3 a +4</span> <span style="font-size:10px;color:#b8cfe0;margin-left:6px">✅ COMPRAR</span></div>
    <div><span class="zone-yellow">−1 a +2</span> <span style="font-size:10px;color:#b8cfe0;margin-left:6px">⏳ ESPERAR</span></div>
    <div><span class="zone-red" style="background:rgba(255,61,87,.05)">−2 a −3</span> <span style="font-size:10px;color:#b8cfe0;margin-left:6px">⚠️ REDUCIR</span></div>
    <div><span class="zone-red">≤ −4</span> <span style="font-size:10px;color:#b8cfe0;margin-left:6px">🔴 VENDER</span></div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="guia-card">
  <div class="guia-title">📐 Fundamentales — P/E · P/B · Indicador Buffett</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:10px">
    <div>
      <div style="font-size:11px;font-weight:700;color:#00c8f0;margin-bottom:6px">P/E — Price to Earnings</div>
      <div class="guia-formula" style="font-size:11px">P/E = Precio / EPS anual</div>
      <div style="font-size:10px;color:#b8cfe0;margin-top:8px;display:flex;flex-direction:column;gap:4px">
        <span><span class="zone-green">&lt; 15x</span> Barato</span>
        <span><span class="zone-green" style="background:rgba(0,230,118,.07)">&lt; 20x</span> Razonable</span>
        <span><span class="zone-yellow">20–40x</span> Caro · puede justificarse</span>
        <span><span class="zone-red">&gt; 40x</span> Muy caro · alerta 🔴</span>
      </div>
      <div style="font-size:9px;color:#3d5a72;margin-top:8px">📖 Graham, B. (1949). <em>The Intelligent Investor.</em></div>
    </div>
    <div>
      <div style="font-size:11px;font-weight:700;color:#00c8f0;margin-bottom:6px">P/B — Price to Book</div>
      <div class="guia-formula" style="font-size:11px">P/B = Precio / Valor libro por acción</div>
      <div style="font-size:10px;color:#b8cfe0;margin-top:8px;display:flex;flex-direction:column;gap:4px">
        <span><span class="zone-green">&lt; 1.5x</span> Barato vs activos</span>
        <span><span class="zone-yellow">1.5–4x</span> Normal para tech</span>
        <span><span class="zone-red">&gt; 4x</span> Alto vs activos</span>
      </div>
      <div style="font-size:9px;color:#3d5a72;margin-top:8px">📖 Fama &amp; French (1992). <em>Journal of Finance.</em></div>
    </div>
    <div>
      <div style="font-size:11px;font-weight:700;color:#00c8f0;margin-bottom:6px">🏛️ Indicador Buffett</div>
      <div class="guia-formula" style="font-size:11px">Precio SPY / SMA200 histórica × 100</div>
      <div style="font-size:10px;color:#b8cfe0;margin-top:8px;display:flex;flex-direction:column;gap:4px">
        <span><span class="zone-green">&lt; 85%</span> 🔥 Mercado barato</span>
        <span><span class="zone-yellow">85–120%</span> ⚖️ Valor justo</span>
        <span><span class="zone-red">&gt; 120%</span> ⚠️ Burbuja potencial</span>
      </div>
      <div style="font-size:9px;color:#3d5a72;margin-top:8px">Adaptación del ratio Mkt Cap / PIB popularizado por Warren Buffett.</div>
    </div>
  </div>
  <div style="margin-top:12px;font-size:10px;color:#ffd740;background:rgba(255,215,64,.05);border:1px solid rgba(255,215,64,.15);border-radius:6px;padding:8px 12px">
    ⚠️ Para CEDEARs estos ratios corresponden a la empresa original (NYSE/NASDAQ), no al precio en pesos.
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background:#0c1219;border:1px solid #1a2d3f;border-left:3px solid #ffd740;border-radius:10px;padding:16px 20px;margin-top:8px">
  <div style="font-size:11px;font-weight:700;color:#ffd740;margin-bottom:6px">⚠️ Aviso importante</div>
  <div style="font-size:11px;color:#5a7a94;line-height:1.8">
    Este scanner es una herramienta de análisis técnico y no constituye asesoramiento financiero.<br>
    Los indicadores técnicos son señales probabilísticas, no garantías de resultado.<br>
    Siempre complementar con análisis fundamental, contexto macroeconómico y gestión de riesgo.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("🔍 Scanner Pro · Yahoo Finance (~15min diferido) · No es asesoramiento financiero · Crypto: BTC-USD, ETH-USD, BNB-USD, SOL-USD · CEDEARs: GGAL.BA")
