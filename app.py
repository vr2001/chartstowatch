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

plt.switch_backend("Agg")
st.set_page_config(page_title="Charts to Watch", layout="wide")

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
# SMALL HELPERS
# ============================================================
def pct_growth(series: pd.Series) -> pd.Series:
    return series.pct_change() * 100

def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def fmt_percent(x) -> str:
    return f"{x:.2f}%" if pd.notna(x) else ""

def fmt_number(x) -> str:
    return f"{x:,.2f}" if pd.notna(x) else ""

def fmt_ratio(x) -> str:
    return f"{x:.3f}" if pd.notna(x) else ""

def scaled_money(series: pd.Series, scale: str) -> pd.Series:
    s = series.copy()
    if scale == "Millions":
        return s / 1_000_000
    if scale == "Billions":
        return s / 1_000_000_000
    return s

def find_line_item_row(df_raw: pd.DataFrame, candidates: List[str]) -> Optional[str]:
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
    s = df_raw.loc[row_name].copy()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    s = safe_to_numeric(s)
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[s.index.notna()].sort_index()
    return s.dropna()

# ============================================================
# PLOTTING HELPERS (DATA LABELS THAT WORK FOR LINE + BAR)
# ============================================================
def _format_for_labels(val: float, is_money: bool, scale: str, as_percent: bool, as_ratio: bool) -> str:
    if pd.isna(val):
        return ""
    if as_percent:
        return f"{val:.1f}%"
    if as_ratio:
        return f"{val:.2f}"
    if is_money:
        if scale == "Billions":
            return f"{val:.2f}B"
        if scale == "Millions":
            return f"{val:.0f}M"
        return f"{val:,.0f}"
    return f"{val:,.2f}"

def add_data_labels(ax, x_vals, y_vals, chart_type: str, label_texts: List[str]):
    if chart_type == "Line":
        for x, y, txt in zip(x_vals, y_vals, label_texts):
            if not txt:
                continue
            ax.annotate(
                txt, xy=(x, y),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, clip_on=True
            )
    else:
        for rect, txt in zip(ax.patches, label_texts):
            if not txt:
                continue
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_height()
            ax.annotate(
                txt, xy=(x, y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, clip_on=True
            )

# ============================================================
# PRICE / PERFORMANCE DATA HELPERS
# ============================================================
@st.cache_data(ttl=60 * 30)
def fetch_close_series(ticker_symbol: str, start_date_str: str) -> pd.Series:
    df = yf.download(ticker_symbol, start=start_date_str, progress=False, auto_adjust=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = ticker_symbol
    return close

@st.cache_data(ttl=60 * 30)
def fetch_close_df(symbols: List[str], start_date_str: str) -> pd.DataFrame:
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

def build_ratio_dataframe(sym_a: str, sym_b: str, start_date_str: str) -> pd.DataFrame:
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

@st.cache_data(ttl=60 * 30)
def fetch_close_prices(ticker_list, start_date_str, end_date_str) -> pd.DataFrame:
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

def fetch_company_names(ticker_list, sleep_seconds=0.25) -> dict:
    names = {}
    for t in ticker_list:
        try:
            info_obj = yf.Ticker(t).info
            names[t] = info_obj.get("shortName", t)
        except Exception:
            names[t] = t
        pytime.sleep(sleep_seconds)
    return names

# ============================================================
# FUNDAMENTALS: STATEMENTS + TTM + VALUATION MULTIPLES
# ============================================================
@st.cache_data(ttl=60 * 60)
def fetch_statements_raw(ticker: str, frequency: str = "Annual") -> Dict[str, pd.DataFrame]:
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

@st.cache_data(ttl=60 * 60)
def fetch_ticker_info(ticker: str) -> Dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def compute_ratios_over_time(income_t: pd.DataFrame, balance_t: pd.DataFrame) -> pd.DataFrame:
    if income_t is None or income_t.empty or balance_t is None or balance_t.empty:
        return pd.DataFrame()

    df = income_t.join(balance_t, how="inner", lsuffix="_inc", rsuffix="_bal").copy()
    ratios = pd.DataFrame(index=df.index)

    def safe_div(a, b):
        if a is None or b is None:
            return None
        b2 = b.replace({0: pd.NA})
        return a / b2

    revenue = df.get("Total Revenue")
    gross_profit = df.get("Gross Profit")
    op_income = df.get("Operating Income")
    net_income = df.get("Net Income")

    total_assets = df.get("Total Assets")
    total_liab = df.get("Total Liab")
    equity = df.get("Total Stockholder Equity")

    curr_assets = df.get("Total Current Assets")
    curr_liab = df.get("Total Current Liabilities")

    if revenue is not None and gross_profit is not None:
        ratios["Gross Margin"] = safe_div(gross_profit, revenue)
    if revenue is not None and op_income is not None:
        ratios["Operating Margin"] = safe_div(op_income, revenue)
    if revenue is not None and net_income is not None:
        ratios["Net Margin"] = safe_div(net_income, revenue)

    if net_income is not None and equity is not None:
        ratios["ROE"] = safe_div(net_income, equity)
    if net_income is not None and total_assets is not None:
        ratios["ROA"] = safe_div(net_income, total_assets)

    if total_liab is not None and equity is not None:
        ratios["Debt / Equity"] = safe_div(total_liab, equity)

    if curr_assets is not None and curr_liab is not None:
        ratios["Current Ratio"] = safe_div(curr_assets, curr_liab)

    return ratios.dropna(axis=1, how="all")

def compute_ttm_from_quarterly_series(s: pd.Series) -> pd.Series:
    if s is None or s.dropna().empty:
        return pd.Series(dtype=float)
    s2 = s.sort_index().dropna()
    return s2.rolling(4).sum()

def align_price_to_period_ends(price: pd.Series, period_ends: pd.DatetimeIndex) -> pd.Series:
    if price is None or price.empty or len(period_ends) == 0:
        return pd.Series(dtype=float)

    px = price.sort_index().dropna()
    out = []
    for d in period_ends:
        sub = px.loc[:d]
        out.append(sub.iloc[-1] if not sub.empty else pd.NA)
    s = pd.Series(out, index=period_ends)
    return pd.to_numeric(s, errors="coerce").dropna()

def compute_enterprise_value_series(
    price_at_dates: pd.Series,
    shares_outstanding: Optional[float],
    balance_t: pd.DataFrame
) -> pd.Series:
    if price_at_dates is None or price_at_dates.empty or shares_outstanding in [None, 0, pd.NA]:
        return pd.Series(dtype=float)

    debt_candidates = [
        "Total Debt", "TotalDebt",
        "Long Term Debt", "LongTermDebt",
        "Short Long Term Debt", "ShortLongTermDebt",
        "Short Term Debt", "ShortTermDebt",
        "Long Term Debt And Capital Lease Obligation"
    ]
    cash_candidates = [
        "Cash And Cash Equivalents", "CashAndCashEquivalents",
        "Cash", "Cash And Short Term Investments", "CashAndShortTermInvestments",
        "Cash Financial"
    ]

    total_debt = pd.Series(index=price_at_dates.index, dtype=float)
    cash = pd.Series(index=price_at_dates.index, dtype=float)

    if balance_t is not None and not balance_t.empty:
        debt_col = next((c for c in debt_candidates if c in balance_t.columns), None)
        cash_col = next((c for c in cash_candidates if c in balance_t.columns), None)

        if debt_col is not None:
            total_debt = pd.to_numeric(balance_t.loc[price_at_dates.index, debt_col], errors="coerce")
        if cash_col is not None:
            cash = pd.to_numeric(balance_t.loc[price_at_dates.index, cash_col], errors="coerce")

    mkt_cap = price_at_dates * float(shares_outstanding)
    ev = mkt_cap + total_debt.fillna(0) - cash.fillna(0)
    return pd.to_numeric(ev, errors="coerce").dropna()

# ============================================================
# TECHNICALS (V1) HELPERS
# ============================================================
def _start_date_from_lookback(preset: str) -> str:
    today = dt.today()
    if preset == "1Y":
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")
    if preset == "2Y":
        return (today - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    if preset == "5Y":
        return (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return "2000-01-01"

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    c = close.dropna().copy()
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace({0: np.nan})
    out = 100 - (100 / (1 + rs))
    return out

def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    c = close.dropna()
    rets = np.log(c / c.shift(1))
    rv = rets.rolling(window).std() * np.sqrt(252) * 100
    return rv

def percentile_rank(series: pd.Series, window_days: int) -> float:
    s = series.dropna()
    if s.empty or len(s) < 5:
        return np.nan
    tail = s.iloc[-window_days:] if len(s) >= window_days else s
    return float(tail.rank(pct=True).iloc[-1] * 100)

def slope_of_series(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < window:
        return np.nan
    y = s.iloc[-window:].values
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])

def render_technicals_page():
    st.subheader("🧭 Technicals")
    st.info("v1 Technicals uses **Yahoo** price/vol proxies (no user-entered API keys).")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Technicals Controls")
    lookback = st.sidebar.radio("Lookback", ["1Y", "2Y", "5Y", "Max"], index=1, horizontal=True)
    start_date = _start_date_from_lookback(lookback)

    slope_window = st.sidebar.slider("Breadth slope window (days)", 10, 60, 20, 5)
    rsi_len = st.sidebar.slider("RSI length", 7, 28, 14, 1)
    rv_window = st.sidebar.slider("Realized vol window (days)", 10, 60, 20, 5)

    vix_pct_window = st.sidebar.radio("VIX percentile window", ["1Y", "3Y", "5Y"], index=0, horizontal=True)
    vix_pct_days = {"1Y": 252, "3Y": 252 * 3, "5Y": 252 * 5}[vix_pct_window]

    tabs = st.tabs(["Breadth Proxies", "Sentiment & Volatility", "Trend & Momentum", "Volatility Regime", "Composite Risk Gauge"])

    with tabs[0]:
        st.markdown("### Breadth Proxies")
        syms = ["SPY", "RSP", "QQQ", "IWM"]
        close = fetch_close_df(syms, start_date)
        if close.empty:
            st.error("No data returned from Yahoo for breadth proxies.")
        else:
            rsp_spy = (close["RSP"] / close["SPY"]).dropna()
            qqq_iwm = (close["QQQ"] / close["IWM"]).dropna()

            s1 = slope_of_series(rsp_spy, slope_window)
            s2 = slope_of_series(qqq_iwm, slope_window)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("RSP/SPY slope", f"{s1:.6f}" if pd.notna(s1) else "—",
                          "Broadening" if pd.notna(s1) and s1 > 0 else ("Narrowing" if pd.notna(s1) else ""))
            with c2:
                st.metric("QQQ/IWM slope", f"{s2:.6f}" if pd.notna(s2) else "—",
                          "Large-cap leading" if pd.notna(s2) and s2 > 0 else ("Small-cap catching up" if pd.notna(s2) else ""))

            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(rsp_spy.index, rsp_spy.values, linewidth=2, label="RSP/SPY")
            ax.plot(qqq_iwm.index, qqq_iwm.values, linewidth=2, label="QQQ/IWM")
            ax.set_title("Breadth Proxies", fontsize=13, fontweight="bold")
            ax.set_xlabel("Date"); ax.set_ylabel("Ratio")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    with tabs[1]:
        st.markdown("### Sentiment & Volatility")
        vix_syms = ["^VIX", "^VVIX"]
        vix_close = fetch_close_df(vix_syms, start_date)
        if vix_close.empty or "^VIX" not in vix_close.columns:
            st.warning("VIX data not available from Yahoo right now.")
        else:
            vix = vix_close["^VIX"].dropna()
            vix50 = sma(vix, 50)
            pct = percentile_rank(vix, vix_pct_days)

            k1, k2, k3 = st.columns(3)
            k1.metric("VIX", f"{vix.iloc[-1]:.2f}")
            k2.metric("VIX vs 50D MA", f"{(vix.iloc[-1] - vix50.iloc[-1]):+.2f}" if vix50.notna().any() else "—")
            k3.metric(f"VIX percentile ({vix_pct_window})", f"{pct:.0f}%" if pd.notna(pct) else "—")

            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(vix.index, vix.values, linewidth=2, label="^VIX")
            if vix50.notna().any():
                ax.plot(vix50.index, vix50.values, linestyle="--", linewidth=1.5, label="50D MA")
            ax.set_title("VIX (Fear Gauge)", fontsize=13, fontweight="bold")
            ax.set_xlabel("Date"); ax.set_ylabel("VIX")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            if "^VVIX" in vix_close.columns and vix_close["^VVIX"].dropna().size > 10:
                vvix = vix_close["^VVIX"].dropna()
                fig2, ax2 = plt.subplots(figsize=(12, 4.0))
                ax2.plot(vvix.index, vvix.values, linewidth=2, label="^VVIX")
                ax2.set_title("VVIX (Vol of VIX)", fontsize=13, fontweight="bold")
                ax2.set_xlabel("Date"); ax2.set_ylabel("VVIX")
                ax2.grid(True, linestyle="--", alpha=0.35)
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("VVIX not available from Yahoo for this lookback or ticker.")

    with tabs[2]:
        st.markdown("### Trend & Momentum (SPY / QQQ / IWM)")
        idx_syms = ["SPY", "QQQ", "IWM"]
        c = fetch_close_df(idx_syms, start_date)

        if c.empty:
            st.warning("No index data returned.")
        else:
            for sym in idx_syms:
                if sym not in c.columns or c[sym].dropna().empty:
                    continue
                close = c[sym].dropna()
                ma50 = sma(close, 50)
                ma200 = sma(close, 200)
                r = rsi(close, rsi_len)

                last = close.iloc[-1]
                above_200 = (last > ma200.iloc[-1]) if ma200.notna().any() else None

                colA, colB, colC = st.columns([1.2, 1, 1])
                with colA:
                    st.markdown(f"#### {sym}")
                with colB:
                    st.metric("Trend (vs 200DMA)", "Bullish" if above_200 else ("Bearish" if above_200 is False else "—"))
                with colC:
                    if not r.dropna().empty:
                        rv = float(r.dropna().iloc[-1])
                        zone = "Overbought" if rv >= 70 else ("Oversold" if rv <= 30 else "Neutral")
                        st.metric("RSI", f"{rv:.1f}", zone)
                    else:
                        st.metric("RSI", "—")

                fig, ax = plt.subplots(figsize=(12, 4.0))
                ax.plot(close.index, close.values, linewidth=2, label=f"{sym} Close")
                if ma50.notna().any():
                    ax.plot(ma50.index, ma50.values, linestyle="--", linewidth=1.2, label="50DMA")
                if ma200.notna().any():
                    ax.plot(ma200.index, ma200.values, linestyle="--", linewidth=1.2, label="200DMA")
                ax.set_title(f"{sym} Price + Moving Averages", fontsize=12, fontweight="bold")
                ax.set_xlabel("Date"); ax.set_ylabel("Price")
                ax.grid(True, linestyle="--", alpha=0.35)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(12, 2.7))
                ax2.plot(r.index, r.values, linewidth=2, label=f"RSI({rsi_len})")
                ax2.axhline(70, linestyle="--", linewidth=1)
                ax2.axhline(30, linestyle="--", linewidth=1)
                ax2.set_title(f"{sym} RSI", fontsize=11, fontweight="bold")
                ax2.set_xlabel("Date"); ax2.set_ylabel("RSI")
                ax2.grid(True, linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig2)

                st.markdown("---")

    with tabs[3]:
        st.markdown("### Volatility Regime (Realized vs Implied)")
        c = fetch_close_df(["SPY", "^VIX"], start_date)
        if c.empty or "SPY" not in c.columns:
            st.warning("Could not load SPY/VIX data.")
        else:
            spy = c["SPY"].dropna()
            rv = realized_vol(spy, rv_window)
            vix = c["^VIX"].dropna() if "^VIX" in c.columns else pd.Series(dtype=float)
            vix_pct = percentile_rank(vix, vix_pct_days) if not vix.empty else np.nan

            a1, a2, a3 = st.columns(3)
            a1.metric(f"Realized Vol ({rv_window}D, ann.)", f"{rv.dropna().iloc[-1]:.1f}%" if not rv.dropna().empty else "—")
            a2.metric("VIX", f"{vix.iloc[-1]:.1f}" if not vix.empty else "—")
            a3.metric(f"VIX percentile ({vix_pct_window})", f"{vix_pct:.0f}%" if pd.notna(vix_pct) else "—")

            fig, ax = plt.subplots(figsize=(12, 4.5))
            if not rv.dropna().empty:
                ax.plot(rv.index, rv.values, linewidth=2, label=f"SPY Realized Vol ({rv_window}D)")
            if not vix.empty:
                ax.plot(vix.index, vix.values, linewidth=2, label="^VIX (Implied Vol Proxy)")
            ax.set_title("Realized vs Implied Volatility", fontsize=13, fontweight="bold")
            ax.set_xlabel("Date"); ax.set_ylabel("Vol (%)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    with tabs[4]:
        st.markdown("### Composite Risk Gauge (simple + transparent)")
        c = fetch_close_df(["SPY", "RSP", "^VIX"], start_date)
        if c.empty or "SPY" not in c.columns or "RSP" not in c.columns:
            st.warning("Could not load SPY/RSP (and/or VIX).")
        else:
            spy = c["SPY"].dropna()
            rsp = c["RSP"].dropna()
            vix = c["^VIX"].dropna() if "^VIX" in c.columns else pd.Series(dtype=float)

            ma200 = sma(spy, 200)
            trend_ok = (spy.iloc[-1] > ma200.iloc[-1]) if ma200.notna().any() else None

            ratio = (rsp / spy).dropna()
            breadth_slope = slope_of_series(ratio, slope_window)
            breadth_ok = (breadth_slope > 0) if pd.notna(breadth_slope) else None

            vix50 = sma(vix, 50) if not vix.empty else pd.Series(dtype=float)
            vol_ok = (vix.iloc[-1] < vix50.iloc[-1]) if (not vix.empty and vix50.notna().any()) else None

            score = 0
            weights = {"Trend": 34, "Breadth": 33, "Volatility": 33}
            detail_rows = []

            def comp(name, ok):
                nonlocal score
                if ok is None:
                    status = "N/A"; pts = 0
                elif ok:
                    status = "Positive"; pts = weights[name]; score += pts
                else:
                    status = "Negative"; pts = 0
                detail_rows.append({"Component": name, "Status": status, "Points": pts})

            comp("Trend", trend_ok)
            comp("Breadth", breadth_ok)
            comp("Volatility", vol_ok)

            if score >= 67:
                label = "Risk-On"
            elif score >= 34:
                label = "Neutral"
            else:
                label = "Risk-Off"

            m1, m2 = st.columns([1, 2])
            with m1:
                st.metric("Risk Gauge", f"{score}/100", label)
            with m2:
                st.caption("Trend = SPY above 200DMA. Breadth = RSP/SPY slope > 0. Volatility = VIX below 50D MA.")

            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

            fig, ax = plt.subplots(figsize=(12, 4.0))
            ax.plot(ratio.index, ratio.values, linewidth=2, label="RSP/SPY")
            ax.set_title("Breadth Proxy Used in Risk Gauge (RSP/SPY)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Date"); ax.set_ylabel("Ratio")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================
# FRED HELPERS (Economic v1)
# ============================================================
FRED_SERIES = {
    "Growth": {
        "INDPRO": "Industrial Production Index (INDPRO)",
        "PAYEMS": "Total Nonfarm Payrolls (PAYEMS)",
    },
    "Inflation": {
        "CPIAUCSL": "CPI All Items (CPIAUCSL)",
        "CPILFESL": "Core CPI (CPILFESL)",
    },
    "Rates": {
        "DGS2": "2Y Treasury (DGS2)",
        "DGS10": "10Y Treasury (DGS10)",
    },
    "Stress": {
        "STLFSI4": "St. Louis Fed Financial Stress Index (STLFSI4)",
        "NFCI": "Chicago Fed National Financial Conditions Index (NFCI)",
    },
}

def _get_fred_key() -> str:
    # Prefer Streamlit secrets, fallback to env var
    try:
        k = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        k = ""
    if not k:
        k = os.getenv("FRED_API_KEY", "")
    return k

def econ_start_date_from_preset(preset: str) -> str:
    today = dt.today()
    if preset == "1Y":
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")
    if preset == "3Y":
        return (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    if preset == "5Y":
        return (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    if preset == "10Y":
        return (today - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    return "1900-01-01"  # Max

@st.cache_data(ttl=60 * 60)
def fetch_fred_series(series_id: str, start_date: str) -> pd.Series:
    api_key = _get_fred_key()
    if not api_key:
        return pd.Series(dtype=float, name=series_id)

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        if not obs:
            return pd.Series(dtype=float, name=series_id)

        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # FRED often uses "." as missing value
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.dropna(subset=["date"]).set_index("date")["value"].dropna()
        s.name = series_id
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=series_id)

def transform_series(s: pd.Series, transform: str) -> pd.Series:
    s = s.dropna().sort_index()
    if s.empty:
        return s

    if transform == "Level":
        return s
    if transform == "YoY %":
        # approx YoY using a 365D shift (works across frequencies)
        return (s / s.shift(365, freq="D") - 1.0) * 100
    if transform == "MoM %":
        return s.pct_change() * 100
    return s

def plot_econ_series(s: pd.Series, title: str, y_label: str):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(s.index, s.values, linewidth=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)

def render_economic_page():
    st.subheader("📉 Economic (FRED) — v1")
    st.info("Growth • Inflation • Rates • Stress — pulled from FRED. Requires a free FRED API key.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Economic Controls")

    date_preset = st.sidebar.radio("Date range", ["1Y", "3Y", "5Y", "10Y", "Max"], index=2, horizontal=True)
    start_date = econ_start_date_from_preset(date_preset)

    transform = st.sidebar.radio("Transform", ["Level", "YoY %", "MoM %"], index=0, horizontal=True)

    if not _get_fred_key():
        st.error("FRED API key not found. Add it to Streamlit secrets as FRED_API_KEY (recommended) or set env var FRED_API_KEY.")
        st.stop()

    tab_growth, tab_infl, tab_rates, tab_stress = st.tabs(["📈 Growth", "🔥 Inflation", "🏦 Rates", "⚠️ Stress"])

    with tab_growth:
        sid = st.selectbox("Series", list(FRED_SERIES["Growth"].keys()), index=0, key="econ_growth_sid")
        label = FRED_SERIES["Growth"][sid]
        s = fetch_fred_series(sid, start_date)
        s2 = transform_series(s, transform)

        st.caption(label)
        if s2.empty:
            st.warning("No data returned for this series/date range.")
        else:
            ylab = f"{sid} ({transform})"
            plot_econ_series(s2, f"{label} — {transform}", ylab)

            df_out = pd.DataFrame({sid: s2})
            st.dataframe(df_out.tail(20), use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                data=df_out.to_csv().encode("utf-8"),
                file_name=f"econ_{sid}_{transform.replace(' ','').replace('%','pct')}_{date_preset}.csv",
                mime="text/csv",
            )

    with tab_infl:
        sid = st.selectbox("Series", list(FRED_SERIES["Inflation"].keys()), index=0, key="econ_infl_sid")
        label = FRED_SERIES["Inflation"][sid]
        s = fetch_fred_series(sid, start_date)
        s2 = transform_series(s, transform)

        st.caption(label)
        if s2.empty:
            st.warning("No data returned for this series/date range.")
        else:
            ylab = f"{sid} ({transform})"
            plot_econ_series(s2, f"{label} — {transform}", ylab)

            df_out = pd.DataFrame({sid: s2})
            st.dataframe(df_out.tail(20), use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                data=df_out.to_csv().encode("utf-8"),
                file_name=f"econ_{sid}_{transform.replace(' ','').replace('%','pct')}_{date_preset}.csv",
                mime="text/csv",
            )

    with tab_rates:
        st.caption("Treasury yields from FRED. Also shows the 10Y–2Y curve.")
        s2y = fetch_fred_series("DGS2", start_date)
        s10y = fetch_fred_series("DGS10", start_date)

        df = pd.concat([s2y, s10y], axis=1).dropna()
        if df.empty:
            st.warning("No rate data returned for this date range.")
        else:
            df["Curve_10Y_2Y"] = df["DGS10"] - df["DGS2"]

            which = st.radio("Plot", ["2Y", "10Y", "10Y–2Y Curve"], index=2, horizontal=True)
            if which == "2Y":
                s = transform_series(df["DGS2"], transform)
                plot_econ_series(s, "2Y Treasury — " + transform, "Percent")
                out = pd.DataFrame({"DGS2": s})
            elif which == "10Y":
                s = transform_series(df["DGS10"], transform)
                plot_econ_series(s, "10Y Treasury — " + transform, "Percent")
                out = pd.DataFrame({"DGS10": s})
            else:
                s = transform_series(df["Curve_10Y_2Y"], transform)
                plot_econ_series(s, "10Y–2Y Curve — " + transform, "Percentage Points")
                out = pd.DataFrame({"Curve_10Y_2Y": s})

            st.dataframe(out.tail(20), use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                data=out.to_csv().encode("utf-8"),
                file_name=f"econ_rates_{which.replace('–','_').replace(' ','_')}_{transform.replace(' ','').replace('%','pct')}_{date_preset}.csv",
                mime="text/csv",
            )

    with tab_stress:
        sid = st.selectbox("Series", list(FRED_SERIES["Stress"].keys()), index=0, key="econ_stress_sid")
        label = FRED_SERIES["Stress"][sid]
        s = fetch_fred_series(sid, start_date)
        s2 = transform_series(s, transform)

        st.caption(label)
        if s2.empty:
            st.warning("No data returned for this series/date range.")
        else:
            ylab = f"{sid} ({transform})"
            plot_econ_series(s2, f"{label} — {transform}", ylab)

            df_out = pd.DataFrame({sid: s2})
            st.dataframe(df_out.tail(20), use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                data=df_out.to_csv().encode("utf-8"),
                file_name=f"econ_{sid}_{transform.replace(' ','').replace('%','pct')}_{date_preset}.csv",
                mime="text/csv",
            )

# ============================================================
# APP HEADER + NAV
# ============================================================
st.title("📊 Charts to Watch")
st.caption(
    "Ratio Dashboard = 2 tickers (A/B). Performance = up to 20 tickers. "
    "Fundamentals = pinned dashboard + any metric dropdown + TTM + valuation multiples. "
    "Technicals = breadth/vol/trend dashboard (v1)."
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Ratio Dashboard", "📈 Performance", "📑 Fundamentals", "🧭 Technicals", "📉 Economic (FRED)", "📋 Cheat Sheet"],
    index=0
)

# ============================================================
# PAGE 1: RATIO DASHBOARD
# ============================================================
if page == "📊 Ratio Dashboard":
    st.subheader("📊 Ratio Dashboard")
    st.info("Charts a **ratio of two symbols** (A/B) and shows each symbol’s latest price.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ratio Controls")

    group_choice = st.sidebar.selectbox("Category", list(RATIO_GROUPS.keys()))
    label_choice = st.sidebar.radio("Preset ratios", list(RATIO_GROUPS[group_choice].keys()))
    preset_a, preset_b = RATIO_GROUPS[group_choice][label_choice]

    date_preset_ratio = st.sidebar.radio("Date range", ["YTD", "1Y", "3Y", "5Y", "Max"], horizontal=True)
    start_date_ratio = ratio_start_date_from_preset(date_preset_ratio)

    use_custom = st.sidebar.checkbox("Use custom tickers", value=False)
    custom_a = st.sidebar.text_input("Custom ticker 1", value="SPY").upper().strip()
    custom_b = st.sidebar.text_input("Custom ticker 2", value="TLT").upper().strip()

    with st.sidebar.expander("Advanced chart options"):
        show_ma = st.checkbox("Show moving averages", value=True)
        log_scale = st.checkbox("Log scale", value=False)
        show_extremes = st.checkbox("Label high/low/latest", value=False)

    if use_custom:
        sym_a, sym_b = custom_a, custom_b
        title_text = f"Custom Ratio: {sym_a}/{sym_b}"
        desc_text = "User-defined custom ratio."
        comm_text = "Interpretation depends on the relationship between the two chosen assets."
        csv_prefix = f"{sym_a}_{sym_b}_custom"
    else:
        sym_a, sym_b = preset_a, preset_b
        title_text = label_choice
        info = RATIO_INFO.get(label_choice, {})
        desc_text = info.get("description", "No description available.")
        comm_text = info.get("commentary", "No commentary available.")
        csv_prefix = f"{sym_a}_{sym_b}_preset"

    st.markdown(f"### {title_text}")
    st.write(f"**Date Range:** {date_preset_ratio} (start: `{start_date_ratio}`)")
    st.markdown(f"**Description:** {desc_text}")
    st.markdown(f"**Commentary:** {comm_text}")

    if "ratio_df" not in st.session_state:
        st.session_state["ratio_df"] = pd.DataFrame()
        st.session_state["ratio_name"] = ""

    if st.button("Plot ratio", type="primary"):
        with st.spinner("Downloading data..."):
            df_ratio = build_ratio_dataframe(sym_a, sym_b, start_date_ratio)
        if df_ratio.empty:
            st.error("No valid data returned. Try Max or check tickers.")
        else:
            st.session_state["ratio_df"] = df_ratio
            st.session_state["ratio_name"] = f"{sym_a}/{sym_b}"

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

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(f"{sym_a} Price", f"{a_last:.2f}", f"{a_delta_pct:+.2f}%")
        m2.metric(f"{sym_b} Price", f"{b_last:.2f}", f"{b_delta_pct:+.2f}%")
        m3.metric("Ratio (A/B)", f"{latest_ratio:.3f}", f"{ratio_delta_pct:+.2f}%")
        m4.metric("High", f"{high_ratio:.3f}", high_date.strftime("%Y-%m-%d"))
        m5.metric("Low", f"{low_ratio:.3f}", low_date.strftime("%Y-%m-%d"))
        m6.metric("Drawdown", f"{drawdown_pct:.2f}%", "")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df_saved.index, df_saved["ratio"], label=f"{sym_a}/{sym_b} Ratio", linewidth=1.7)

        if show_ma:
            if df_saved["ma50"].notna().any():
                ax.plot(df_saved.index, df_saved["ma50"], label="50-day MA", linestyle="--", linewidth=1.2)
            if df_saved["ma200"].notna().any():
                ax.plot(df_saved.index, df_saved["ma200"], label="200-day MA", linestyle="--", linewidth=1.2)

        if show_extremes:
            ax.scatter([high_date], [high_ratio], s=45)
            ax.scatter([low_date], [low_ratio], s=45)
            ax.scatter([r.index[-1]], [latest_ratio], s=45)
            ax.annotate(f"High {high_ratio:.3f}", xy=(high_date, high_ratio), xytext=(10, 10), textcoords="offset points")
            ax.annotate(f"Low {low_ratio:.3f}", xy=(low_date, low_ratio), xytext=(10, -15), textcoords="offset points")
            ax.annotate(f"Latest {latest_ratio:.3f}", xy=(r.index[-1], latest_ratio), xytext=(10, 0), textcoords="offset points")

        ax.set_title(title_text, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Ratio")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        df_export = df_saved.copy()
        ratio_lbl = st.session_state.get("ratio_name", "ratio")
        df_export = df_export.rename(columns={"ratio": f"{ratio_lbl}_ratio", "ma50": "MA50", "ma200": "MA200"})
        st.download_button(
            "📥 Download ratio series as CSV",
            data=df_export.to_csv().encode("utf-8"),
            file_name=f"{csv_prefix}_{date_preset_ratio}_ratio_series.csv",
            mime="text/csv"
        )

# ============================================================
# PAGE 2: PERFORMANCE
# ============================================================
elif page == "📈 Performance":
    st.subheader("📈 Performance")
    st.info("Compares **up to 20 tickers**. Lines are indexed to **100 at the start of the selected period**.")

    perf_preset = st.radio("Performance period", ["QTD", "YTD", "3M", "6M", "1Y", "3Y", "5Y", "Max"], horizontal=True)
    start_perf = period_start_date(perf_preset)
    end_perf = dt.today().strftime("%Y-%m-%d")

    tickers_raw = st.text_input("Enter up to 20 tickers (comma-separated)", value="SPY, QQQ, IWM, TLT")
    ticker_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if len(ticker_list) > 20:
        st.warning("⚠️ More than 20 tickers entered. Only the first 20 will be used.")
        ticker_list = ticker_list[:20]

    want_names = st.checkbox("🔎 Fetch company names (slower)", value=False)

    if "perf_export" not in st.session_state:
        st.session_state["perf_export"] = pd.DataFrame()

    if st.button("Plot Performance", type="primary"):
        if len(ticker_list) == 0:
            st.error("❌ No tickers entered.")
        else:
            with st.spinner("⏳ Downloading price data..."):
                close_df = fetch_close_prices(ticker_list, start_perf, end_perf)

            if close_df.empty:
                st.error("❌ No valid data returned. Check tickers and try again.")
            else:
                ok = close_df.columns.tolist()
                bad = [t for t in ticker_list if t not in ok]
                if bad:
                    st.warning(f"⚠️ These tickers failed: {', '.join(bad)}")

                idx = (close_df / close_df.iloc[0]) * 100
                final_vals = idx.iloc[-1]

                if want_names:
                    with st.spinner("🔎 Fetching company names..."):
                        name_map = fetch_company_names(ok)
                else:
                    name_map = {t: t for t in ok}

                summary = pd.DataFrame({
                    "Ticker": final_vals.index,
                    "Name": [name_map[t] for t in final_vals.index],
                    f"{perf_preset} % Return": final_vals.values - 100
                }).sort_values(by=f"{perf_preset} % Return", ascending=False)

                summary_display = summary.copy()
                summary_display[f"{perf_preset} % Return"] = summary_display[f"{perf_preset} % Return"].map(lambda x: f"{x:.1f}%")

                st.write(f"**Date Range:** {start_perf} → {end_perf}")
                st.subheader("Summary")
                st.dataframe(summary_display, use_container_width=True)

                fig, ax = plt.subplots(figsize=(14, 8))
                line_colors = {}
                for t in idx.columns:
                    line, = ax.plot(idx.index, idx[t], linewidth=2)
                    line_colors[t] = line.get_color()

                ax.axhline(y=100, color="#888888", linestyle="--", linewidth=1.5)

                sorted_tickers = final_vals.sort_values(ascending=False).index.tolist()
                spacing_offset = 0.55
                for rank, t in enumerate(sorted_tickers):
                    last_date = idx.index[-1]
                    last_value = idx[t].iloc[-1]
                    offset = spacing_offset * (len(sorted_tickers) - rank)
                    ax.text(
                        last_date, last_value + offset,
                        f"{t} ({last_value - 100:.1f}%)",
                        fontsize=8, ha="left", va="center",
                        color=line_colors[t],
                        bbox=dict(facecolor="white", edgecolor=line_colors[t], boxstyle="round,pad=0.25")
                    )

                ax.set_title(f"Performance ({perf_preset}) — Indexed to 100 at Period Start", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Performance (Indexed to 100)")
                ax.grid(True, linestyle="--", alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

                export_df = idx.copy()
                export_df.columns = [f"{c}_indexed100" for c in export_df.columns]
                export_df = export_df.join((idx - 100).add_suffix(f"_{perf_preset.lower()}_pct"))
                st.session_state["perf_export"] = export_df

    saved_perf = st.session_state.get("perf_export", pd.DataFrame())
    if isinstance(saved_perf, pd.DataFrame) and not saved_perf.empty:
        st.download_button(
            "📥 Download performance series as CSV",
            data=saved_perf.to_csv().encode("utf-8"),
            file_name=f"performance_{perf_preset}_{dt.now().year}.csv",
            mime="text/csv"
        )
    else:
        st.info("Enter tickers, choose a period, then click **Plot Performance**.")

# ============================================================
# PAGE 3: FUNDAMENTALS
# ============================================================
elif page == "📑 Fundamentals":
    st.subheader("📑 Fundamentals")
    st.info("Pinned dashboard + dropdown metrics + **TTM fundamentals** + **valuation multiples** + aligned data labels for Line/Bar charts.")

    left, _ = st.columns([1, 2])
    with left:
        fund_ticker = st.text_input("Ticker (stocks work best)", value="AAPL").upper().strip()
        frequency = st.radio("Frequency", ["Annual", "Quarterly"], horizontal=True)
        mode = st.radio("Mode", ["Pinned Dashboard", "Single Metric"], horizontal=True)

        scale = st.selectbox("Scale ($ items)", ["Raw", "Millions", "Billions"], index=2)
        chart_type = st.radio("Chart type", ["Line", "Bar"], horizontal=True)

        capex_positive = st.checkbox("Show CAPEX as positive spend", value=True)
        show_other = st.checkbox("Show 'Other Fundamentals' dropdown", value=True)

        show_data_labels = st.checkbox("Show data labels on charts", value=True)

    with st.spinner("Loading statements..."):
        stmts = fetch_statements_raw(fund_ticker, frequency=frequency)
        info = fetch_ticker_info(fund_ticker)

    income_raw = stmts.get("income_raw", pd.DataFrame())
    balance_raw = stmts.get("balance_raw", pd.DataFrame())
    cash_raw = stmts.get("cash_raw", pd.DataFrame())

    if (income_raw is None or income_raw.empty) and (balance_raw is None or balance_raw.empty) and (cash_raw is None or cash_raw.empty):
        st.error("No fundamentals data returned. Try a different ticker or switch Annual/Quarterly.")
    else:
        income_t = income_raw.T if income_raw is not None and not income_raw.empty else pd.DataFrame()
        balance_t = balance_raw.T if balance_raw is not None and not balance_raw.empty else pd.DataFrame()
        cash_t = cash_raw.T if cash_raw is not None and not cash_raw.empty else pd.DataFrame()

        ratios_t = compute_ratios_over_time(income_t, balance_t)

        revenue_row = find_line_item_row(income_raw, ["Total Revenue", "TotalRevenue", "Revenue"])
        eps_row = find_line_item_row(income_raw, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS", "EPS"])
        net_income_row = find_line_item_row(income_raw, ["Net Income", "NetIncome"])
        ebitda_row = find_line_item_row(income_raw, ["EBITDA", "Ebitda"])

        capex_row = find_line_item_row(
            cash_raw,
            ["Capital Expenditures", "CapitalExpenditures", "Purchase Of PPE", "Purchase of PPE",
             "Purchases of Property, Plant & Equipment", "InvestmentsInPropertyPlantAndEquipment"]
        )

        shares_outstanding = info.get("sharesOutstanding", None)

        def plot_series_with_table(series: pd.Series, title: str, is_money: bool,
                                   as_percent: bool = False, as_ratio: bool = False):
            if series is None or series.dropna().empty:
                st.warning("No data available.")
                return

            s = series.dropna().sort_index().copy()

            if is_money:
                s_plot = scaled_money(s, scale)
                ylab = f"{title} ({'$B' if scale=='Billions' else '$M' if scale=='Millions' else '$'})"
            else:
                s_plot = s.copy()
                ylab = title

            fig, ax = plt.subplots(figsize=(12, 4.8))

            label_texts = [
                _format_for_labels(v, is_money=is_money, scale=scale, as_percent=as_percent, as_ratio=as_ratio)
                for v in s_plot.values
            ]

            if chart_type == "Bar":
                x_str = pd.to_datetime(s_plot.index).strftime("%Y-%m-%d").tolist()
                ax.bar(x_str, s_plot.values)
                ax.set_xticklabels(x_str, rotation=45, ha="right")
                if show_data_labels:
                    add_data_labels(ax, x_str, s_plot.values, "Bar", label_texts=label_texts)
            else:
                ax.plot(s_plot.index, s_plot.values, linewidth=2, marker="o")
                ax.grid(True, linestyle="--", alpha=0.35)
                if show_data_labels:
                    add_data_labels(ax, s_plot.index, s_plot.values, "Line", label_texts=label_texts)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Period End")
            ax.set_ylabel(ylab)
            plt.tight_layout()
            st.pyplot(fig)

            tbl = pd.DataFrame(index=pd.to_datetime(s.index).strftime("%Y-%m-%d"))
            if as_percent:
                tbl[title] = pd.to_numeric(s, errors="coerce").map(fmt_percent)
            elif as_ratio:
                tbl[title] = pd.to_numeric(s, errors="coerce").map(fmt_ratio)
            else:
                if is_money:
                    tmp = scaled_money(s, scale)
                    tbl[title] = pd.to_numeric(tmp, errors="coerce").map(fmt_number)
                else:
                    tbl[title] = pd.to_numeric(s, errors="coerce").map(fmt_number)

            st.dataframe(tbl, use_container_width=True)
            st.download_button(
                "📥 Download CSV",
                data=tbl.to_csv().encode("utf-8"),
                file_name=f"{fund_ticker}_{title.replace(' ', '_')}_{frequency}.csv",
                mime="text/csv"
            )

        st.markdown("### 🧾 TTM Fundamentals & Valuation Multiples")
        if frequency != "Quarterly":
            st.info("TTM and valuation multiples are best from **Quarterly** statements. Switch frequency to Quarterly to enable.")
        else:
            rev_q = row_to_time_series(income_raw, revenue_row) if revenue_row else pd.Series(dtype=float)
            eps_q = row_to_time_series(income_raw, eps_row) if eps_row else pd.Series(dtype=float)
            ni_q = row_to_time_series(income_raw, net_income_row) if net_income_row else pd.Series(dtype=float)
            ebitda_q = row_to_time_series(income_raw, ebitda_row) if ebitda_row else pd.Series(dtype=float)

            rev_ttm = compute_ttm_from_quarterly_series(rev_q)
            eps_ttm = compute_ttm_from_quarterly_series(eps_q)
            ni_ttm = compute_ttm_from_quarterly_series(ni_q)
            ebitda_ttm = compute_ttm_from_quarterly_series(ebitda_q)

            start_px = (rev_ttm.index.min() - pd.Timedelta(days=20)).strftime("%Y-%m-%d") if not rev_ttm.empty else "2000-01-01"
            px = fetch_close_series(fund_ticker, start_px)
            px_at = align_price_to_period_ends(px, rev_ttm.index) if not rev_ttm.empty else pd.Series(dtype=float)

            pe_ttm = pd.Series(dtype=float)
            ev_ebitda_ttm = pd.Series(dtype=float)

            if not px_at.empty and not eps_ttm.empty:
                pe_ttm = (px_at / eps_ttm.reindex(px_at.index)).replace([np.inf, -np.inf], np.nan).dropna()

            if not px_at.empty and shares_outstanding and not balance_t.empty and not ebitda_ttm.empty:
                bal_aligned = balance_t.reindex(px_at.index)
                ev_series = compute_enterprise_value_series(px_at, shares_outstanding, bal_aligned)
                ev_ebitda_ttm = (ev_series / ebitda_ttm.reindex(ev_series.index)).replace([np.inf, -np.inf], np.nan).dropna()

            def latest_val(s: pd.Series) -> Tuple[str, str]:
                if s is None or s.dropna().empty:
                    return "—", ""
                v = float(s.dropna().iloc[-1])
                d = s.dropna().index[-1].strftime("%Y-%m-%d")
                return f"{v:,.2f}", d

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                v, d = latest_val(scaled_money(rev_ttm, "Billions"))
                st.metric("Revenue TTM ($B)", v, d)
            with m2:
                v, d = latest_val(eps_ttm)
                st.metric("EPS TTM", v, d)
            with m3:
                v, d = latest_val(pe_ttm)
                st.metric("P/E (TTM)", v, d)
            with m4:
                v, d = latest_val(ev_ebitda_ttm)
                st.metric("EV/EBITDA (TTM)", v, d)

            st.caption("EV is approximate: (Price × sharesOutstanding) + Debt − Cash. sharesOutstanding is latest from Yahoo info.")

            c1, c2 = st.columns(2)
            with c1:
                if not rev_ttm.empty:
                    plot_series_with_table(rev_ttm, "Revenue (TTM)", is_money=True)
                else:
                    st.warning("Revenue TTM unavailable.")
            with c2:
                if not eps_ttm.empty:
                    plot_series_with_table(eps_ttm, "EPS (TTM)", is_money=False)
                else:
                    st.warning("EPS TTM unavailable.")

            c3, c4 = st.columns(2)
            with c3:
                if not pe_ttm.empty:
                    plot_series_with_table(pe_ttm, "P/E (TTM)", is_money=False, as_ratio=True)
                else:
                    st.warning("P/E (TTM) unavailable.")
            with c4:
                if not ev_ebitda_ttm.empty:
                    plot_series_with_table(ev_ebitda_ttm, "EV/EBITDA (TTM)", is_money=False, as_ratio=True)
                else:
                    st.warning("EV/EBITDA (TTM) unavailable.")

        st.markdown("---")

        if show_other:
            st.markdown("### 🔎 Other Fundamentals (Pick any metric)")
            stmt_choice = st.selectbox("Statement", ["Income Statement", "Balance Sheet", "Cash Flow", "Computed Ratios"])

            if stmt_choice == "Income Statement":
                df_pick = income_raw; pick_from_index = True
            elif stmt_choice == "Balance Sheet":
                df_pick = balance_raw; pick_from_index = True
            elif stmt_choice == "Cash Flow":
                df_pick = cash_raw; pick_from_index = True
            else:
                df_pick = ratios_t; pick_from_index = False

            if df_pick is None or df_pick.empty:
                st.warning("No data available for that selection.")
            else:
                if pick_from_index:
                    metric_choice = st.selectbox("Metric", list(df_pick.index))
                    s = row_to_time_series(df_pick, metric_choice)
                    plot_series_with_table(s, title=f"{metric_choice}", is_money=True)
                else:
                    metric_choice = st.selectbox("Metric", list(df_pick.columns))
                    s = df_pick[metric_choice].dropna().copy()
                    if metric_choice in ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"]:
                        s = (s * 100).dropna()
                        plot_series_with_table(s, title=f"{metric_choice} (%)", is_money=False, as_percent=True)
                    else:
                        plot_series_with_table(s, title=metric_choice, is_money=False, as_ratio=True)

        st.markdown("---")
        if mode == "Pinned Dashboard":
            st.markdown(f"### {fund_ticker} — Pinned Dashboard ({frequency})")
            st.caption("Revenue • EPS • CAPEX • Net Margin")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            with c1:
                st.markdown("#### Revenue")
                if revenue_row and income_raw is not None and not income_raw.empty:
                    rev = row_to_time_series(income_raw, revenue_row)
                    plot_series_with_table(rev, title="Revenue", is_money=True)
                    st.markdown("**Revenue % Growth**")
                    if not rev.empty:
                        plot_series_with_table(pct_growth(rev).dropna(), title="Revenue Growth (%)", is_money=False, as_percent=True)
                else:
                    st.warning("Revenue not available for this ticker via Yahoo.")

            with c2:
                st.markdown("#### EPS")
                if eps_row and income_raw is not None and not income_raw.empty:
                    eps = row_to_time_series(income_raw, eps_row)
                    plot_series_with_table(eps, title="EPS", is_money=False)
                    st.markdown("**EPS % Growth**")
                    if not eps.empty:
                        plot_series_with_table(pct_growth(eps).dropna(), title="EPS Growth (%)", is_money=False, as_percent=True)
                else:
                    st.warning("EPS not available for this ticker via Yahoo.")

            with c3:
                st.markdown("#### CAPEX")
                if capex_row and cash_raw is not None and not cash_raw.empty:
                    cap = row_to_time_series(cash_raw, capex_row)
                    if capex_positive:
                        cap = -cap
                    plot_series_with_table(cap, title="CAPEX", is_money=True)
                    st.markdown("**CAPEX % Growth**")
                    if not cap.empty:
                        plot_series_with_table(pct_growth(cap).dropna(), title="CAPEX Growth (%)", is_money=False, as_percent=True)
                else:
                    st.warning("CAPEX not available from Yahoo for this ticker/frequency.")

            with c4:
                st.markdown("#### Net Margin")
                if ratios_t is not None and not ratios_t.empty and "Net Margin" in ratios_t.columns:
                    nm = (ratios_t["Net Margin"] * 100).dropna()
                    plot_series_with_table(nm, title="Net Margin (%)", is_money=False, as_percent=True)
                else:
                    st.warning("Net Margin could not be computed (missing revenue/net income).")

# ============================================================
# PAGE 4: TECHNICALS
# ============================================================
elif page == "🧭 Technicals":
    render_technicals_page()

# ============================================================
# PAGE 5: ECONOMIC (FRED)
# ============================================================
elif page == "📉 Economic (FRED)":
    render_economic_page()

# ============================================================
# PAGE 6: CHEAT SHEET
# ============================================================
else:
    st.subheader("📋 Cheat Sheet")
    st.info("Lists all preset ratios with descriptions and commentary (no chart).")

    rows = []
    for cat, ratios in RATIO_GROUPS.items():
        for lbl, (a, b) in ratios.items():
            info = RATIO_INFO.get(lbl, {})
            rows.append({
                "Category": cat,
                "Label": lbl,
                "Symbol 1": a,
                "Symbol 2": b,
                "Description": info.get("description", ""),
                "Commentary": info.get("commentary", "")
            })

    cheat_df = pd.DataFrame(rows)
    st.dataframe(cheat_df, use_container_width=True)

    st.download_button(
        "📥 Download cheat sheet as CSV",
        data=cheat_df.to_csv(index=False).encode("utf-8"),
        file_name="ratio_cheat_sheet.csv",
        mime="text/csv"
    )
