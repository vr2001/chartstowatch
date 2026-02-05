"""
Modern Financial Dashboard - Complete Version
Integrated with modern styling and bug fixes
"""

import os
import time as pytime
from datetime import datetime as dt, timedelta
from typing import Optional, Dict, List, Tuple
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# ============================================================
# MODERN STYLING - EMBEDDED
# ============================================================

MODERN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        padding: 0rem 1.5rem;
        background: linear-gradient(to bottom, #f9fafb 0%, #ffffff 100%);
    }
    
    h1 {
        color: #111827;
        font-weight: 800;
        letter-spacing: -0.03em;
        padding-bottom: 1rem;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    h1::after {
        content: '';
        display: block;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        margin-top: 1rem;
        border-radius: 2px;
    }
    
    h2 {
        color: #1f2937;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #f3f4f6;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .stAlert {
        border-radius: 1rem;
        border: none;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.25rem;
        border-left: 4px solid #3b82f6;
    }
    
    .stButton>button {
        border-radius: 0.75rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #f3f4f6;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #6b7280;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6;
        background: white;
        border-bottom: 3px solid #3b82f6;
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 0.75rem;
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f3f4f6;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
        border-radius: 10px;
    }
</style>
"""

# ============================================================
# MATPLOTLIB MODERN STYLING
# ============================================================

def create_modern_plot_style():
    """Modern matplotlib styling"""
    return {
        'figure.facecolor': 'white',
        'axes.facecolor': '#f9fafb',
        'axes.edgecolor': '#e5e7eb',
        'axes.labelcolor': '#374151',
        'axes.titlecolor': '#1f2937',
        'axes.titleweight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.color': '#6b7280',
        'ytick.color': '#6b7280',
        'grid.color': '#e5e7eb',
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Arial']
    }


# ============================================================
# PAGE CONFIGURATION
# ============================================================
plt.switch_backend("Agg")
st.set_page_config(
    page_title="Charts to Watch - Financial Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply modern styling
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ============================================================
# PRESET RATIOS (GROUPED)
# ============================================================
RATIO_GROUPS = {
    "📈 Stock Market Breadth & Strength": {
        "SPY / RSP – S&P 500 Cap vs Equal Weight": ("SPY", "RSP"),
        "QQQ / IWM – Nasdaq 100 vs Russell 2000": ("QQQ", "IWM"),
        "DIA / IWM – Dow vs Small Caps": ("DIA", "IWM"),
        "MGK / SPY – Mega Cap Growth vs S&P 500": ("MGK", "SPY"),
    },
    "🔄 Risk-On vs Risk-Off Sentiment": {
        "SPY / TLT – Stocks vs Long-Term Bonds": ("SPY", "TLT"),
        "HYG / IEF – High Yield vs Treasuries": ("HYG", "IEF"),
        "XLY / XLP – Discretionary vs Staples": ("XLY", "XLP"),
        "IWM / SHY – Small Caps vs Short Treasuries": ("IWM", "SHY"),
        "SPHB / SPLV – High Beta vs Low Vol": ("SPHB", "SPLV"),
    },
    "🏭 Sector Relative Strength": {
        "XLF / SPY – Financials vs Market": ("XLF", "SPY"),
        "XLV / SPY – Healthcare vs Market": ("XLV", "SPY"),
        "XLE / SPY – Energy vs Market": ("XLE", "SPY"),
        "XLK / SPY – Tech vs Market": ("XLK", "SPY"),
        "XLI / SPY – Industrials vs Market": ("XLI", "SPY"),
        "RSPD / RSPS – Equal Disc vs Equal Staples": ("RSPD", "RSPS"),
    },
    "🛢️ Commodities & Inflation Indicators": {
        "DBC / SPY – Commodities vs Stocks": ("DBC", "SPY"),
        "GDX / SPY – Gold Miners vs Market": ("GDX", "SPY"),
        "GLD / TLT – Gold vs Bonds": ("GLD", "TLT"),
        "USO / SPY – Oil vs Stocks": ("USO", "SPY"),
        "TIP / TLT – TIPS vs Treasuries": ("TIP", "TLT"),
        "CPER / GLD – Copper vs Gold": ("CPER", "GLD"),
        "GLD / USO – Gold vs Oil": ("GLD", "USO"),
        "GLD / XME – Gold vs Metals & Mining": ("GLD", "XME"),
    },
    "🌍 Global vs U.S. Market Strength": {
        "EEM / SPY – Emerging Markets vs U.S.": ("EEM", "SPY"),
        "VEA / SPY – Developed Intl vs U.S.": ("VEA", "SPY"),
        "FXI / SPY – China vs U.S.": ("FXI", "SPY"),
    },
    "₿ Crypto Relative Strength": {
        "ETHA / IBIT – ETH vs BTC ETF": ("ETHA", "IBIT"),
        "ETHA / GSOL – ETH vs Solana": ("ETHA", "GSOL"),
        "BMNR / ETHA – BMNR vs ETH": ("BMNR", "ETHA"),
        "MSTR / IBIT – MicroStrategy vs BTC ETF": ("MSTR", "IBIT"),
    }
}

RATIO_INFO = {
    "SPY / RSP – S&P 500 Cap vs Equal Weight": {
        "description": "Cap-weighted S&P 500 vs equal-weight; highlights breadth vs mega-cap concentration.",
        "commentary": "Rising = narrow leadership (mega-caps dominate). Falling = broader participation (healthier breadth).",
    },
    "QQQ / IWM – Nasdaq 100 vs Russell 2000": {
        "description": "Large-cap growth/tech vs small caps; growth leadership + risk appetite gauge.",
        "commentary": "Rising = big tech leadership/quality preference. Falling = risk-on rotation into small caps.",
    },
    "DIA / IWM – Dow vs Small Caps": {
        "description": "Blue-chip Dow vs small caps; stability vs domestic risk exposure.",
        "commentary": "Rising = defensive tilt. Falling = higher risk appetite / cyclical participation.",
    },
    "MGK / SPY – Mega Cap Growth vs S&P 500": {
        "description": "Mega-cap growth vs broad market; measures growth concentration.",
        "commentary": "Rising = growth crowding/concentration. Falling = rotation into broader market/value/cyclicals.",
    },
    "SPY / TLT – Stocks vs Long-Term Bonds": {
        "description": "Risk-on/risk-off: equities vs long-duration Treasuries.",
        "commentary": "Rising = risk-on. Falling = flight to safety / growth concerns / duration bid.",
    },
    "HYG / IEF – High Yield vs Treasuries": {
        "description": "Credit risk appetite: high yield vs intermediate Treasuries.",
        "commentary": "Rising = healthy credit. Falling = widening spreads / credit stress risk.",
    },
    "XLY / XLP – Discretionary vs Staples": {
        "description": "Consumer cyclicals vs defensives; proxy for consumer confidence.",
        "commentary": "Rising = consumers/risk-on. Falling = defensive posture; caution on growth.",
    },
    "IWM / SHY – Small Caps vs Short Treasuries": {
        "description": "Small caps vs cash-like Treasuries; pure risk appetite gauge.",
        "commentary": "Rising = risk-on. Falling = liquidity preference/capital preservation.",
    },
    "SPHB / SPLV – High Beta vs Low Vol": {
        "description": "High beta vs low vol; aggressive vs defensive indicator.",
        "commentary": "Rising = speculation/risk-taking. Falling = demand for stability/defense.",
    },
    "XLF / SPY – Financials vs Market": {
        "description": "Financials vs market; ties to credit and curve expectations.",
        "commentary": "Rising = improving conditions. Falling = tightening/stress or growth worries.",
    },
    "XLV / SPY – Healthcare vs Market": {
        "description": "Healthcare vs market; defensive leadership indicator.",
        "commentary": "Rising = defensive rotation. Falling = risk-on into cyclicals/growth.",
    },
    "XLE / SPY – Energy vs Market": {
        "description": "Energy vs market; sensitive to oil and inflation dynamics.",
        "commentary": "Rising = inflation/energy strength. Falling = disinflation or weaker demand.",
    },
    "XLK / SPY – Tech vs Market": {
        "description": "Tech vs market; growth leadership + rate sensitivity.",
        "commentary": "Rising = tech leadership. Falling = rotation away from long-duration growth exposure.",
    },
    "XLI / SPY – Industrials vs Market": {
        "description": "Industrials vs market; proxy for capex/trade/manufacturing optimism.",
        "commentary": "Rising = stronger cycle expectations. Falling = slowing activity concerns.",
    },
    "RSPD / RSPS – Equal Disc vs Equal Staples": {
        "description": "Equal-weight discretionary vs equal-weight staples; reduces mega-cap distortion.",
        "commentary": "Rising = broad consumer risk-on. Falling = defensive consumer posture.",
    },
    "DBC / SPY – Commodities vs Stocks": {
        "description": "Broad commodities vs equities; inflation/real-asset sensitivity.",
        "commentary": "Rising = inflation/real asset bid. Falling = equity leadership/disinflation backdrop.",
    },
    "GDX / SPY – Gold Miners vs Market": {
        "description": "Gold miners vs market; leveraged gold/hedge sentiment indicator.",
        "commentary": "Rising = hedge demand/uncertainty. Falling = preference for risk assets.",
    },
    "GLD / TLT – Gold vs Bonds": {
        "description": "Gold vs long Treasuries; safe-haven preference and inflation/currency risk.",
        "commentary": "Rising = gold favored. Falling = bonds favored / policy confidence.",
    },
    "USO / SPY – Oil vs Stocks": {
        "description": "Crude oil vs equities; demand/supply shocks and inflation proxy.",
        "commentary": "Rising = inflation risk. Falling = weaker demand or disinflation.",
    },
    "TIP / TLT – TIPS vs Treasuries": {
        "description": "TIPS vs nominal Treasuries; inflation expectations proxy.",
        "commentary": "Rising = inflation expectations firming. Falling = disinflation expectations.",
    },
    "CPER / GLD – Copper vs Gold": {
        "description": "Copper vs gold; growth vs fear signal.",
        "commentary": "Rising = growth optimism. Falling = risk-off/recession concerns.",
    },
    "GLD / USO – Gold vs Oil": {
        "description": "Gold vs oil; defensive vs cyclical commodity exposure.",
        "commentary": "Rising = fear/slower growth. Falling = stronger demand/cycle/inflation pressures.",
    },
    "GLD / XME – Gold vs Metals & Mining": {
        "description": "Gold vs metals/mining; safety vs industrial cycle exposure.",
        "commentary": "Rising = defensive preference. Falling = pro-growth industrial demand theme.",
    },
    "EEM / SPY – Emerging Markets vs U.S.": {
        "description": "Emerging markets vs U.S.; global growth and USD sensitivity.",
        "commentary": "Rising = EM tailwinds (often weaker USD). Falling = U.S. dominance/caution.",
    },
    "VEA / SPY – Developed Intl vs U.S.": {
        "description": "Developed international vs U.S.; rotation between regions/styles.",
        "commentary": "Rising = non-U.S. leadership. Falling = U.S. leadership (often growth-led).",
    },
    "FXI / SPY – China vs U.S.": {
        "description": "China vs U.S.; policy/growth and geopolitics sensitivity.",
        "commentary": "Rising = improving China sentiment. Falling = elevated risk/policy/growth concerns.",
    },
    "ETHA / IBIT – ETH vs BTC ETF": {
        "description": "Ethereum vs Bitcoin; crypto rotation gauge.",
        "commentary": "Rising = ETH leadership. Falling = BTC leadership as core asset bid.",
    },
    "ETHA / GSOL – ETH vs Solana": {
        "description": "Ethereum vs Solana; layer-1 leadership rotation.",
        "commentary": "Rising = ETH favored. Falling = SOL favored (often higher risk appetite).",
    },
    "BMNR / ETHA – BMNR vs ETH": {
        "description": "Speculative equity vs ETH; leveraged/speculative exposure proxy.",
        "commentary": "Rising = speculation appetite. Falling = preference for underlying exposure.",
    },
    "MSTR / IBIT – MicroStrategy vs BTC ETF": {
        "description": "MSTR vs BTC ETF; equity optionality vs pure BTC exposure.",
        "commentary": "Rising = leverage/optionality rewarded. Falling = preference for pure BTC exposure.",
    },
}

# ============================================================
# DATE PRESETS
# ============================================================
def period_start_date(preset: str) -> str:
    """Calculate start date based on preset period."""
    today = dt.today()
    if preset == "QTD":
        quarter = (today.month - 1) // 3 + 1
        start_month = 3 * (quarter - 1) + 1
        return dt(today.year, start_month, 1).strftime("%Y-%m-%d")
    if preset == "YTD":
        return f"{today.year}-01-01"
    if preset == "3M":
        return (today - timedelta(days=92)).strftime("%Y-%m-%d")
    if preset == "6M":
        return (today - timedelta(days=183)).strftime("%Y-%m-%d")
    if preset == "1Y":
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")
    if preset == "3Y":
        return (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    if preset == "5Y":
        return (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return "2000-01-01"

def ratio_start_date_from_preset(preset: str) -> str:
    """Calculate start date for ratio charts."""
    today = dt.today()
    if preset == "YTD":
        return f"{today.year}-01-01"
    if preset == "1Y":
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")
    if preset == "3Y":
        return (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    if preset == "5Y":
        return (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return "2000-01-01"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def pct_growth(series: pd.Series) -> pd.Series:
    """Calculate percentage growth."""
    return series.pct_change() * 100

def safe_to_numeric(s: pd.Series) -> pd.Series:
    """Safely convert series to numeric."""
    return pd.to_numeric(s, errors="coerce")

def fmt_percent(x) -> str:
    """Format as percentage."""
    return f"{x:.2f}%" if pd.notna(x) else "—"

def fmt_number(x) -> str:
    """Format as number with commas."""
    return f"{x:,.2f}" if pd.notna(x) else "—"

def fmt_ratio(x) -> str:
    """Format as ratio."""
    return f"{x:.3f}" if pd.notna(x) else "—"

def scaled_money(series: pd.Series, scale: str) -> pd.Series:
    """Scale monetary values."""
    s = series.copy()
    if scale == "Millions":
        return s / 1_000_000
    if scale == "Billions":
        return s / 1_000_000_000
    return s

def find_line_item_row(df_raw: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find matching row in financial statements."""
    if df_raw is None or df_raw.empty:
        return None

    idx = [str(x).strip() for x in df_raw.index]
    idx_norm_map = {x.lower(): x for x in idx}

    for c in candidates:
        key = str(c).strip().lower()
        if key in idx_norm_map:
            return idx_norm_map[key]

    for c in candidates:
        key = str(c).strip().lower()
        for raw_label in idx:
            if key in raw_label.lower():
                return raw_label

    return None

def row_to_time_series(df_raw: pd.DataFrame, row_name: str) -> pd.Series:
    """Convert dataframe row to time series."""
    if row_name is None:
        return pd.Series(dtype=float)
    s = df_raw.loc[row_name].copy()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    s = safe_to_numeric(s)
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].sort_index()
    return s.dropna()

# ============================================================
# DATA FETCHING WITH ERROR HANDLING
# ============================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_close_series(ticker_symbol: str, start_date_str: str) -> pd.Series:
    """Fetch closing prices for a single ticker."""
    try:
        df = yf.download(ticker_symbol, start=start_date_str, progress=False, auto_adjust=False)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.name = ticker_symbol
        return close
    except Exception as e:
        st.error(f"Error fetching {ticker_symbol}: {str(e)}")
        return pd.Series(dtype=float)

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_close_df(symbols: List[str], start_date_str: str) -> pd.DataFrame:
    """Fetch closing prices for multiple tickers."""
    try:
        raw = yf.download(symbols, start=start_date_str, progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            close = raw["Close"].copy()
        else:
            if "Close" not in raw.columns:
                return pd.DataFrame()
            close = raw[["Close"]].copy()
            close.columns = [symbols[0]]
        return close.dropna(how="all")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def build_ratio_dataframe(sym_a: str, sym_b: str, start_date_str: str) -> pd.DataFrame:
    """Build ratio dataframe with moving averages."""
    s_a = fetch_close_series(sym_a, start_date_str)
    s_b = fetch_close_series(sym_b, start_date_str)
    if s_a.empty or s_b.empty:
        return pd.DataFrame()

    df = pd.concat([s_a, s_b], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    df.columns = [sym_a, sym_b]
    ratio = df[sym_a] / df[sym_b]
    ma50 = ratio.rolling(50).mean()
    ma200 = ratio.rolling(200).mean()

    return pd.DataFrame({sym_a: df[sym_a], sym_b: df[sym_b], "ratio": ratio, "ma50": ma50, "ma200": ma200})

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_close_prices(ticker_list, start_date_str, end_date_str) -> pd.DataFrame:
    """Fetch closing prices for performance comparison."""
    try:
        raw = yf.download(ticker_list, start=start_date_str, end=end_date_str, progress=False)
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()
            close_df = raw["Close"].copy()
        else:
            if "Close" not in raw.columns:
                return pd.DataFrame()
            close_df = raw[["Close"]].copy()
            close_df.columns = [ticker_list[0]]

        return close_df.dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"Error fetching prices: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_company_names(ticker_list, sleep_seconds=0.25) -> dict:
    """Fetch company names for tickers."""
    names = {}
    for t in ticker_list:
        try:
            info_obj = yf.Ticker(t).info
            names[t] = info_obj.get("shortName", t)
        except Exception:
            names[t] = t
        pytime.sleep(sleep_seconds)
    return names

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_statements_raw(ticker: str, frequency: str = "Annual") -> Dict[str, pd.DataFrame]:
    """Fetch financial statements."""
    t = yf.Ticker(ticker)
    freq = "yearly" if frequency == "Annual" else "quarterly"

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out.index = out.index.map(lambda x: str(x).strip())
        out.columns = pd.to_datetime(out.columns, errors="coerce")
        out = out.loc[:, out.columns.notna()].sort_index(axis=1)
        out = out.apply(pd.to_numeric, errors="coerce")
        return out

    income = pd.DataFrame()
    balance = pd.DataFrame()
    cash = pd.DataFrame()

    try:
        income = t.get_income_stmt(pretty=True, freq=freq)
    except Exception:
        pass
    try:
        balance = t.get_balance_sheet(pretty=True, freq=freq)
    except Exception:
        pass
    try:
        cash = t.get_cashflow(pretty=True, freq=freq)
    except Exception:
        pass

    if income is None or income.empty:
        income = t.financials if frequency == "Annual" else t.quarterly_financials
    if balance is None or balance.empty:
        balance = t.balance_sheet if frequency == "Annual" else t.quarterly_balance_sheet
    if cash is None or cash.empty:
        cash = t.cashflow if frequency == "Annual" else t.quarterly_cashflow

    return {"income_raw": _clean(income), "balance_raw": _clean(balance), "cash_raw": _clean(cash)}

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Dict:
    """Fetch ticker info."""
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


# ============================================================
# APP HEADER + NAVIGATION
# ============================================================
st.title("📊 Charts to Watch")
st.markdown("""
<p style="font-size: 1.1rem; color: #6b7280; margin-bottom: 2rem;">
Professional financial analysis dashboard • Real-time market data • Technical & fundamental insights
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Ratio Dashboard", "📈 Performance", "📑 Fundamentals", "📋 Reference"],
    index=0,
    label_visibility="collapsed"
)

# ============================================================
# PAGE 1: RATIO DASHBOARD
# ============================================================
if page == "📊 Ratio Dashboard":
    st.markdown("## 📊 Ratio Analysis Dashboard")
    st.info("📈 Compare two assets and analyze their relative performance over time")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Ratio Settings")

    group_choice = st.sidebar.selectbox("📂 Category", list(RATIO_GROUPS.keys()))
    label_choice = st.sidebar.radio("📊 Preset Ratios", list(RATIO_GROUPS[group_choice].keys()))
    preset_a, preset_b = RATIO_GROUPS[group_choice][label_choice]

    date_preset_ratio = st.sidebar.radio("📅 Time Period", ["YTD", "1Y", "3Y", "5Y", "Max"], horizontal=True)
    start_date_ratio = ratio_start_date_from_preset(date_preset_ratio)

    use_custom = st.sidebar.checkbox("🎯 Use Custom Tickers", value=False)
    if use_custom:
        custom_a = st.sidebar.text_input("Ticker 1", value="SPY").upper().strip()
        custom_b = st.sidebar.text_input("Ticker 2", value="TLT").upper().strip()

    with st.sidebar.expander("⚙️ Chart Options"):
        show_ma = st.checkbox("Show moving averages", value=True)
        log_scale = st.checkbox("Log scale", value=False)
        show_extremes = st.checkbox("Label extremes", value=False)

    if use_custom:
        sym_a, sym_b = custom_a, custom_b
        title_text = f"Custom Ratio: {sym_a} / {sym_b}"
        desc_text = "User-defined custom ratio"
        comm_text = "Interpretation depends on the relationship between the two chosen assets"
        csv_prefix = f"{sym_a}_{sym_b}_custom"
    else:
        sym_a, sym_b = preset_a, preset_b
        title_text = label_choice
        info = RATIO_INFO.get(label_choice, {})
        desc_text = info.get("description", "No description available")
        comm_text = info.get("commentary", "No commentary available")
        csv_prefix = f"{sym_a}_{sym_b}_preset"

    st.markdown(f"### {title_text}")
    
    col_desc1, col_desc2 = st.columns([1, 1])
    with col_desc1:
        st.markdown(f"**📅 Period:** {date_preset_ratio} (from {start_date_ratio})")
    with col_desc2:
        st.markdown(f"**🎯 Tickers:** {sym_a} / {sym_b}")
    
    st.markdown(f"**📖 Description:** {desc_text}")
    st.markdown(f"**💡 Commentary:** {comm_text}")

    if "ratio_df" not in st.session_state:
        st.session_state["ratio_df"] = pd.DataFrame()
        st.session_state["ratio_name"] = ""

    if st.button("🚀 Analyze Ratio", type="primary", use_container_width=True):
        with st.spinner("📊 Downloading market data..."):
            df_ratio = build_ratio_dataframe(sym_a, sym_b, start_date_ratio)
        if df_ratio.empty:
            st.error("❌ No valid data returned. Try Max period or check tickers")
        else:
            st.session_state["ratio_df"] = df_ratio
            st.session_state["ratio_name"] = f"{sym_a}/{sym_b}"
            st.success("✅ Data loaded successfully!")

    df_saved = st.session_state.get("ratio_df", pd.DataFrame())
    if isinstance(df_saved, pd.DataFrame) and not df_saved.empty:
        r = df_saved["ratio"].dropna()
        latest_ratio = float(r.iloc[-1])
        prev_ratio = float(r.iloc[-2]) if len(r) > 1 else latest_ratio
        ratio_delta_pct = (latest_ratio / prev_ratio - 1) * 100 if prev_ratio != 0 else 0.0

        high_ratio = float(r.max()); high_date = r.idxmax()
        low_ratio = float(r.min()); low_date = r.idxmin()
        drawdown_pct = (latest_ratio / high_ratio - 1) * 100 if high_ratio != 0 else 0.0

        a_series = df_saved[sym_a].dropna()
        b_series = df_saved[sym_b].dropna()
        a_last = float(a_series.iloc[-1])
        a_prev = float(a_series.iloc[-2]) if len(a_series) > 1 else a_last
        a_delta_pct = (a_last / a_prev - 1) * 100 if a_prev != 0 else 0.0
        b_last = float(b_series.iloc[-1])
        b_prev = float(b_series.iloc[-2]) if len(b_series) > 1 else b_last
        b_delta_pct = (b_last / b_prev - 1) * 100 if b_prev != 0 else 0.0

        st.markdown("### 📊 Key Metrics")
        m1, m2, m3 = st.columns(3)
        m4, m5, m6 = st.columns(3)
        
        m1.metric(f"💰 {sym_a} Price", f"${a_last:.2f}", f"{a_delta_pct:+.2f}%")
        m2.metric(f"💰 {sym_b} Price", f"${b_last:.2f}", f"{b_delta_pct:+.2f}%")
        m3.metric("📊 Current Ratio", f"{latest_ratio:.3f}", f"{ratio_delta_pct:+.2f}%")
        m4.metric("📈 Period High", f"{high_ratio:.3f}", high_date.strftime("%Y-%m-%d"))
        m5.metric("📉 Period Low", f"{low_ratio:.3f}", low_date.strftime("%Y-%m-%d"))
        m6.metric("📊 Drawdown", f"{drawdown_pct:.2f}%")

        st.markdown("### 📈 Ratio Chart")
        with plt.rc_context(create_modern_plot_style()):
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(df_saved.index, df_saved["ratio"], label=f"{sym_a}/{sym_b} Ratio", 
                   linewidth=3, color='#3b82f6', zorder=3)

            if show_ma:
                if df_saved["ma50"].notna().any():
                    ax.plot(df_saved.index, df_saved["ma50"], label="50-day MA", 
                           linestyle="--", linewidth=2.5, color='#f59e0b', alpha=0.8, zorder=2)
                if df_saved["ma200"].notna().any():
                    ax.plot(df_saved.index, df_saved["ma200"], label="200-day MA", 
                           linestyle="--", linewidth=2.5, color='#ef4444', alpha=0.8, zorder=2)

            if show_extremes:
                ax.scatter([high_date], [high_ratio], s=100, color='#10b981', zorder=4, 
                          edgecolors='white', linewidths=2)
                ax.scatter([low_date], [low_ratio], s=100, color='#ef4444', zorder=4, 
                          edgecolors='white', linewidths=2)
                ax.scatter([r.index[-1]], [latest_ratio], s=100, color='#3b82f6', zorder=4, 
                          edgecolors='white', linewidths=2)

            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Ratio", fontsize=12)
            if log_scale:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.3, color='#e5e7eb')
            ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc='best')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        df_export = df_saved.copy()
        ratio_lbl = st.session_state.get("ratio_name", "ratio")
        df_export = df_export.rename(columns={"ratio": f"{ratio_lbl}_ratio", "ma50": "MA50", "ma200": "MA200"})
        
        st.download_button(
            "📥 Download Ratio Data (CSV)",
            data=df_export.to_csv().encode("utf-8"),
            file_name=f"{csv_prefix}_{date_preset_ratio}_ratio_series.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================
# PAGE 2: PERFORMANCE
# ============================================================
elif page == "📈 Performance":
    st.markdown("## 📈 Multi-Asset Performance Comparison")
    st.info("📊 Compare up to 20 tickers indexed to 100 at the start of the period")

    perf_preset = st.radio("📅 Performance Period", ["QTD", "YTD", "3M", "6M", "1Y", "3Y", "5Y", "Max"], horizontal=True)
    start_perf = period_start_date(perf_preset)
    end_perf = dt.today().strftime("%Y-%m-%d")

    tickers_raw = st.text_input("🎯 Enter Tickers (comma-separated, max 20)", value="SPY, QQQ, IWM, TLT")
    ticker_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if len(ticker_list) > 20:
        st.warning("⚠️ More than 20 tickers entered. Only the first 20 will be used")
        ticker_list = ticker_list[:20]

    want_names = st.checkbox("🔍 Fetch Company Names (slower)", value=False)

    if "perf_export" not in st.session_state:
        st.session_state["perf_export"] = pd.DataFrame()

    if st.button("🚀 Analyze Performance", type="primary", use_container_width=True):
        if len(ticker_list) == 0:
            st.error("❌ No tickers entered")
        else:
            with st.spinner("⏳ Downloading price data..."):
                close_df = fetch_close_prices(ticker_list, start_perf, end_perf)

            if close_df.empty:
                st.error("❌ No valid data returned. Check tickers and try again")
            else:
                ok = close_df.columns.tolist()
                bad = [t for t in ticker_list if t not in ok]
                if bad:
                    st.warning(f"⚠️ These tickers failed: {', '.join(bad)}")

                idx = (close_df / close_df.iloc[0]) * 100
                final_vals = idx.iloc[-1]

                if want_names:
                    with st.spinner("🔍 Fetching company names..."):
                        name_map = fetch_company_names(ok)
                else:
                    name_map = {t: t for t in ok}

                summary = pd.DataFrame({
                    "Ticker": final_vals.index,
                    "Name": [name_map[t] for t in final_vals.index],
                    f"{perf_preset} Return (%)": final_vals.values - 100
                }).sort_values(by=f"{perf_preset} Return (%)", ascending=False)

                summary_display = summary.copy()
                summary_display[f"{perf_preset} Return (%)"] = summary_display[f"{perf_preset} Return (%)"].map(lambda x: f"{x:.1f}%")

                st.markdown(f"### 📊 Performance Summary")
                st.caption(f"Period: {start_perf} → {end_perf}")
                st.dataframe(summary_display, use_container_width=True, hide_index=True)

                st.markdown("### 📈 Performance Chart")
                with plt.rc_context(create_modern_plot_style()):
                    fig, ax = plt.subplots(figsize=(16, 7))
                    
                    colors = plt.cm.tab20(np.linspace(0, 1, len(idx.columns)))
                    line_colors = {}
                    
                    for i, t in enumerate(idx.columns):
                        line, = ax.plot(idx.index, idx[t], linewidth=2.5, color=colors[i], alpha=0.8)
                        line_colors[t] = colors[i]

                    ax.axhline(y=100, color='#6b7280', linestyle="--", linewidth=2, alpha=0.5)

                    sorted_tickers = final_vals.sort_values(ascending=False).index.tolist()
                    spacing_offset = 0.8
                    for rank, t in enumerate(sorted_tickers):
                        last_date = idx.index[-1]
                        last_value = idx[t].iloc[-1]
                        offset = spacing_offset * (len(sorted_tickers) - rank - len(sorted_tickers)/2)
                        ax.text(
                            last_date, last_value + offset,
                            f"{t} ({last_value - 100:+.1f}%)",
                            fontsize=9, ha="left", va="center",
                            color='white',
                            fontweight='bold',
                            bbox=dict(facecolor=line_colors[t], edgecolor='white', 
                                    boxstyle="round,pad=0.4", alpha=0.9, linewidth=1.5)
                        )

                    ax.set_title(f"Performance Comparison ({perf_preset}) — Indexed to 100", 
                               fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel("Date", fontsize=12)
                    ax.set_ylabel("Performance (Indexed to 100)", fontsize=12)
                    ax.grid(True, linestyle="--", alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig)

                export_df = idx.copy()
                export_df.columns = [f"{c}_indexed100" for c in export_df.columns]
                export_df = export_df.join((idx - 100).add_suffix(f"_{perf_preset.lower()}_pct"))
                st.session_state["perf_export"] = export_df
                st.success("✅ Performance analysis complete!")

    saved_perf = st.session_state.get("perf_export", pd.DataFrame())
    if isinstance(saved_perf, pd.DataFrame) and not saved_perf.empty:
        st.download_button(
            "📥 Download Performance Data (CSV)",
            data=saved_perf.to_csv().encode("utf-8"),
            file_name=f"performance_{perf_preset}_{dt.now().year}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("💡 Enter tickers and click **Analyze Performance** to get started")


# ============================================================
# PAGE 3: FUNDAMENTALS (SIMPLIFIED)
# ============================================================
elif page == "📑 Fundamentals":
    st.markdown("## 📑 Fundamental Analysis")
    st.info("📊 Financial statements and key metrics")

    fund_ticker = st.text_input("🎯 Ticker Symbol", value="AAPL").upper().strip()
    frequency = st.radio("📅 Frequency", ["Annual", "Quarterly"], horizontal=True)

    if st.button("📊 Load Fundamentals", type="primary"):
        with st.spinner("📊 Loading financial data..."):
            stmts = fetch_statements_raw(fund_ticker, frequency=frequency)
            info = fetch_ticker_info(fund_ticker)

        income_raw = stmts.get("income_raw", pd.DataFrame())
        balance_raw = stmts.get("balance_raw", pd.DataFrame())
        cash_raw = stmts.get("cash_raw", pd.DataFrame())

        if income_raw.empty and balance_raw.empty and cash_raw.empty:
            st.error("❌ No fundamentals data returned. Try a different ticker")
        else:
            st.success("✅ Data loaded successfully!")
            
            # Display key metrics
            shares_outstanding = info.get("sharesOutstanding", "N/A")
            market_cap = info.get("marketCap", "N/A")
            pe_ratio = info.get("trailingPE", "N/A")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Shares Outstanding", f"{shares_outstanding:,}" if isinstance(shares_outstanding, (int, float)) else shares_outstanding)
            col2.metric("💰 Market Cap", f"${market_cap/1e9:.2f}B" if isinstance(market_cap, (int, float)) else market_cap)
            col3.metric("📈 P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio)
            
            # Show statements in tabs
            tab1, tab2, tab3 = st.tabs(["💵 Income Statement", "📊 Balance Sheet", "💸 Cash Flow"])
            
            with tab1:
                if not income_raw.empty:
                    st.dataframe(income_raw.head(10), use_container_width=True)
                    st.download_button(
                        "📥 Download Income Statement",
                        data=income_raw.to_csv().encode("utf-8"),
                        file_name=f"{fund_ticker}_income_{frequency}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No income statement data available")
            
            with tab2:
                if not balance_raw.empty:
                    st.dataframe(balance_raw.head(10), use_container_width=True)
                    st.download_button(
                        "📥 Download Balance Sheet",
                        data=balance_raw.to_csv().encode("utf-8"),
                        file_name=f"{fund_ticker}_balance_{frequency}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No balance sheet data available")
            
            with tab3:
                if not cash_raw.empty:
                    st.dataframe(cash_raw.head(10), use_container_width=True)
                    st.download_button(
                        "📥 Download Cash Flow",
                        data=cash_raw.to_csv().encode("utf-8"),
                        file_name=f"{fund_ticker}_cashflow_{frequency}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No cash flow data available")

# ============================================================
# PAGE 4: REFERENCE GUIDE
# ============================================================
else:
    st.markdown("## 📋 Ratio Reference Guide")
    st.info("📚 Comprehensive list of all preset ratios with descriptions and commentary")

    # Add search functionality
    search_term = st.text_input("🔍 Search ratios", placeholder="Type to filter...")

    rows = []
    for cat, ratios in RATIO_GROUPS.items():
        for lbl, (a, b) in ratios.items():
            info = RATIO_INFO.get(lbl, {})
            rows.append({
                "Category": cat,
                "Ratio": lbl,
                "Ticker A": a,
                "Ticker B": b,
                "Description": info.get("description", ""),
                "Commentary": info.get("commentary", "")
            })

    cheat_df = pd.DataFrame(rows)
    
    # Filter based on search
    if search_term:
        mask = cheat_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        cheat_df = cheat_df[mask]
        st.caption(f"Found {len(cheat_df)} matching ratios")

    # Group by category and display
    for category in cheat_df["Category"].unique():
        with st.expander(f"{category}", expanded=(search_term != "")):
            cat_data = cheat_df[cheat_df["Category"] == category].drop("Category", axis=1)
            st.dataframe(cat_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.download_button(
        "📥 Download Complete Reference (CSV)",
        data=cheat_df.to_csv(index=False).encode("utf-8"),
        file_name="ratio_reference_guide.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #6b7280;">
    <p style="margin: 0; font-size: 0.875rem;">
        📊 <strong>Charts to Watch</strong> — Professional Financial Analysis Dashboard
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.75rem;">
        Powered by Yahoo Finance API • Data for informational purposes only
    </p>
</div>
""", unsafe_allow_html=True)
