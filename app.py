import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import time as pytime

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

# ============================================================
# DESCRIPTIONS + COMMENTARY
# ============================================================
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
    return "2000-01-01"  # Max (practical)

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
def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if df is not None and not df.empty and c in df.columns:
            return c
    return None

def pct_growth(series: pd.Series) -> pd.Series:
    return series.pct_change() * 100

def _fmt_percent(x):
    return f"{x:.2f}%" if pd.notna(x) else ""

def _fmt_number(x):
    return f"{x:,.2f}" if pd.notna(x) else ""

# ============================================================
# DATA HELPERS
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

@st.cache_data(ttl=60*60)
def fetch_statements(ticker: str, frequency: str = "Annual") -> dict:
    t = yf.Ticker(ticker)

    if frequency == "Quarterly":
        inc = t.quarterly_financials
        bal = t.quarterly_balance_sheet
        cfs = t.quarterly_cashflow
    else:
        inc = t.financials
        bal = t.balance_sheet
        cfs = t.cashflow

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.T.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out = out.apply(pd.to_numeric, errors="coerce")
        return out

    return {"income": _prep(inc), "balance": _prep(bal), "cash": _prep(cfs)}

def compute_ratios_over_time(income_df: pd.DataFrame, balance_df: pd.DataFrame) -> pd.DataFrame:
    if income_df is None or income_df.empty or balance_df is None or balance_df.empty:
        return pd.DataFrame()

    df = income_df.join(balance_df, how="inner").copy()
    ratios = pd.DataFrame(index=df.index)

    def safe_div(a, b):
        return a / b.replace({0: pd.NA})

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

# ============================================================
# APP HEADER
# ============================================================
st.title("📊 Charts to Watch")
st.caption("Ratio Dashboard = 2 tickers (A/B). Performance = up to 20 tickers. Fundamentals = pinned dashboard + growth + ratios.")

# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Ratio Dashboard", "📈 Performance", "📑 Fundamentals", "📋 Cheat Sheet"], index=0)

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
# PAGE 3: FUNDAMENTALS (PINNED DASHBOARD + SINGLE METRIC)
# ============================================================
elif page == "📑 Fundamentals":
    st.subheader("📑 Fundamentals")
    st.info("Pinned dashboard (Revenue, EPS, CAPEX, Net Margin) plus growth metrics and common ratios.")

    left, right = st.columns([1, 2])
    with left:
        fund_ticker = st.text_input("Ticker", value="AAPL").upper().strip()
        frequency = st.radio("Frequency", ["Annual", "Quarterly"], horizontal=True)

        mode = st.radio("Mode", ["Pinned Dashboard", "Single Metric"], horizontal=True)

        scale = st.selectbox("Scale ($ items)", ["Raw", "Millions", "Billions"], index=2)
        chart_type = st.radio("Chart type", ["Line", "Bar"], horizontal=True)
        capex_positive = st.checkbox("Show CAPEX as positive spend", value=True)

    with st.spinner("Loading fundamentals..."):
        stmts = fetch_statements(fund_ticker, frequency=frequency)
    income_df = stmts.get("income", pd.DataFrame())
    balance_df = stmts.get("balance", pd.DataFrame())
    cash_df = stmts.get("cash", pd.DataFrame())

    if (income_df is None or income_df.empty) and (balance_df is None or balance_df.empty) and (cash_df is None or cash_df.empty):
        st.error("No fundamentals data returned for this ticker/frequency. Try another ticker or switch Annual/Quarterly.")
    else:
        # Column mapping
        revenue_col = pick_first_existing_col(income_df, ["Total Revenue", "TotalRevenue"])
        eps_col = pick_first_existing_col(income_df, ["Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"])
        capex_col = pick_first_existing_col(cash_df, ["Capital Expenditures", "CapitalExpenditures"])

        ratios_df = compute_ratios_over_time(income_df, balance_df)

        def scaled(series: pd.Series, is_money: bool) -> pd.Series:
            s = series.copy()
            if not is_money:
                return s
            if scale == "Millions":
                return s / 1_000_000
            if scale == "Billions":
                return s / 1_000_000_000
            return s

        def chart_and_table(series: pd.Series, title: str, y_label: str, fmt: str, is_money: bool, download_name: str):
            # Chart
            plot_series = scaled(series, is_money=is_money).dropna()
            if plot_series.empty:
                st.warning(f"No data for {title}.")
                return

            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            if chart_type == "Bar":
                ax.bar(plot_series.index.astype(str), plot_series.values)
                ax.set_xticklabels(plot_series.index.strftime("%Y-%m-%d"), rotation=45, ha="right", fontsize=8)
            else:
                ax.plot(plot_series.index, plot_series.values, linewidth=2)
                ax.grid(True, linestyle="--", alpha=0.35)

            ax.set_title(title, fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(y_label, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

            # Table
            tbl = series.to_frame(name=title).copy()
            tbl.index = pd.to_datetime(tbl.index).strftime("%Y-%m-%d")

            if fmt == "percent":
                tbl[title] = tbl[title].map(_fmt_percent)
            elif fmt == "ratio":
                tbl[title] = pd.to_numeric(tbl[title], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
            else:
                # numbers/money shown in chosen scale for consistency
                tmp = scaled(series, is_money=is_money)
                tbl[title] = tmp.values
                tbl[title] = pd.to_numeric(tbl[title], errors="coerce").map(_fmt_number)

            st.dataframe(tbl, use_container_width=True, height=210)

            st.download_button(
                "📥 CSV",
                data=tbl.to_csv().encode("utf-8"),
                file_name=download_name,
                mime="text/csv"
            )

        if mode == "Pinned Dashboard":
            st.markdown(f"### {fund_ticker} — Pinned Fundamentals Dashboard ({frequency})")
            st.caption("Four core charts + mini tables. Use Annual for longer history; Quarterly for more detail.")

            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            # Revenue
            with c1:
                st.markdown("#### Revenue")
                if revenue_col:
                    chart_and_table(
                        income_df[revenue_col].dropna(),
                        title="Revenue",
                        y_label=f"Revenue ({'$B' if scale=='Billions' else '$M' if scale=='Millions' else '$'})",
                        fmt="number",
                        is_money=True,
                        download_name=f"{fund_ticker}_revenue_{frequency}.csv"
                    )
                else:
                    st.warning("Revenue not available for this ticker.")

            # EPS
            with c2:
                st.markdown("#### EPS")
                if eps_col:
                    chart_and_table(
                        income_df[eps_col].dropna(),
                        title="EPS",
                        y_label="EPS",
                        fmt="number",
                        is_money=False,
                        download_name=f"{fund_ticker}_eps_{frequency}.csv"
                    )
                else:
                    st.warning("EPS not available for this ticker.")

            # CAPEX
            with c3:
                st.markdown("#### CAPEX")
                if capex_col:
                    raw = cash_df[capex_col].dropna()
                    capex_series = (-raw if capex_positive else raw)
                    chart_and_table(
                        capex_series,
                        title="CAPEX",
                        y_label=f"CAPEX ({'$B' if scale=='Billions' else '$M' if scale=='Millions' else '$'})",
                        fmt="number",
                        is_money=True,
                        download_name=f"{fund_ticker}_capex_{frequency}.csv"
                    )
                else:
                    st.warning("CAPEX not available for this ticker.")

            # Net Margin
            with c4:
                st.markdown("#### Net Margin")
                if ratios_df is not None and not ratios_df.empty and "Net Margin" in ratios_df.columns:
                    chart_and_table(
                        (ratios_df["Net Margin"] * 100).dropna(),
                        title="Net Margin (%)",
                        y_label="Percent",
                        fmt="percent",
                        is_money=False,
                        download_name=f"{fund_ticker}_net_margin_{frequency}.csv"
                    )
                else:
                    st.warning("Net Margin could not be computed for this ticker.")

        else:
            # Single Metric mode: includes growth options + ratios
            st.markdown(f"### {fund_ticker} — Single Fundamental Metric ({frequency})")

            pinned_options = []
            if revenue_col:
                pinned_options += ["Revenue", "Revenue % Growth"]
            if eps_col:
                pinned_options += ["EPS", "EPS % Growth"]
            if capex_col:
                pinned_options += ["CAPEX", "CAPEX % Growth"]

            if ratios_df is not None and not ratios_df.empty:
                for r in ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA", "Debt / Equity", "Current Ratio"]:
                    if r in ratios_df.columns:
                        pinned_options.append(r)

            if not pinned_options:
                st.error("Could not find Revenue/EPS/CAPEX or computed ratios for this ticker. Try another ticker.")
            else:
                metric = st.selectbox("Choose metric", pinned_options)

                series = None
                fmt = "number"
                is_money = False
                y_label = metric

                if metric == "Revenue":
                    series = income_df[revenue_col].dropna()
                    is_money = True
                elif metric == "Revenue % Growth":
                    base = income_df[revenue_col].dropna()
                    series = pct_growth(base).dropna()
                    fmt = "percent"
                elif metric == "EPS":
                    series = income_df[eps_col].dropna()
                elif metric == "EPS % Growth":
                    base = income_df[eps_col].dropna()
                    series = pct_growth(base).dropna()
                    fmt = "percent"
                elif metric == "CAPEX":
                    raw = cash_df[capex_col].dropna()
                    series = (-raw if capex_positive else raw)
                    is_money = True
                elif metric == "CAPEX % Growth":
                    raw = cash_df[capex_col].dropna()
                    base = (-raw if capex_positive else raw)
                    series = pct_growth(base).dropna()
                    fmt = "percent"
                else:
                    # computed ratios
                    series = ratios_df[metric].dropna()
                    if metric in ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"]:
                        series = (series * 100).dropna()
                        fmt = "percent"
                    else:
                        fmt = "ratio"

                if series is None or series.empty:
                    st.warning("No data for that metric.")
                else:
                    # Full-width chart + table
                    fig, ax = plt.subplots(figsize=(12, 5))
                    plot_series = series.copy()

                    if fmt == "number" and is_money:
                        if scale == "Millions":
                            plot_series = plot_series / 1_000_000
                            y_label = f"{metric} ($M)"
                        elif scale == "Billions":
                            plot_series = plot_series / 1_000_000_000
                            y_label = f"{metric} ($B)"
                        else:
                            y_label = f"{metric} ($)"

                    if chart_type == "Bar":
                        ax.bar(plot_series.index.astype(str), plot_series.values)
                        ax.set_xticklabels(plot_series.index.strftime("%Y-%m-%d"), rotation=45, ha="right")
                    else:
                        ax.plot(plot_series.index, plot_series.values, linewidth=2)
                        ax.grid(True, linestyle="--", alpha=0.35)

                    ax.set_title(metric, fontsize=13, fontweight="bold")
                    ax.set_xlabel("Period End")
                    ax.set_ylabel(y_label)
                    plt.tight_layout()
                    st.pyplot(fig)

                    tbl = series.to_frame(name=metric).copy()
                    tbl.index = pd.to_datetime(tbl.index).strftime("%Y-%m-%d")

                    if fmt == "percent":
                        tbl[metric] = tbl[metric].map(_fmt_percent)
                    elif fmt == "ratio":
                        tbl[metric] = pd.to_numeric(tbl[metric], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
                    else:
                        tmp = series.copy()
                        if is_money:
                            if scale == "Millions":
                                tmp = tmp / 1_000_000
                            elif scale == "Billions":
                                tmp = tmp / 1_000_000_000
                        tbl[metric] = tmp.values
                        tbl[metric] = pd.to_numeric(tbl[metric], errors="coerce").map(_fmt_number)

                    st.markdown("#### Data Table")
                    st.dataframe(tbl, use_container_width=True)

                    st.download_button(
                        "📥 Download as CSV",
                        data=tbl.to_csv().encode("utf-8"),
                        file_name=f"{fund_ticker}_fundamentals_{metric.replace(' ', '_')}_{frequency}.csv",
                        mime="text/csv"
                    )

# ============================================================
# PAGE 4: CHEAT SHEET
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
