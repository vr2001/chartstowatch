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
# DATE RANGE
# ============================================================
def start_date_from_preset(preset: str) -> str:
    today = dt.today()
    if preset == "YTD":
        return f"{today.year}-01-01"
    if preset == "1Y":
        return (today - timedelta(days=365)).strftime("%Y-%m-%d")
    if preset == "3Y":
        return (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    if preset == "5Y":
        return (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    return "2000-01-01"  # Max (practical)

# ============================================================
# DATA
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

    return pd.DataFrame({
        sym_a: df[sym_a],
        sym_b: df[sym_b],
        "ratio": ratio,
        "ma50": ma50,
        "ma200": ma200
    })

@st.cache_data(ttl=60 * 60)
def fetch_ytd_close_prices(ticker_list, start_date_str, end_date_str) -> pd.DataFrame:
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

def fetch_company_names(ticker_list, sleep_seconds=0.35) -> dict:
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
# APP HEADER
# ============================================================
st.title("📊 Charts to Watch")
st.caption("Two tools: (1) Ratio charts (two tickers) and (2) YTD performance (up to 10 tickers).")

# ============================================================
# NAVIGATION (THIS FIXES THE CONFUSION)
# ============================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Ratio Dashboard", "📈 YTD Performance", "📋 Cheat Sheet"],
    index=0
)

# ============================================================
# PAGE 1: RATIO DASHBOARD
# ============================================================
if page == "📊 Ratio Dashboard":
    st.subheader("📊 Ratio Dashboard")
    st.info("This page charts a **ratio of two symbols** (Symbol A / Symbol B). Use preset ratios or enter your own.")

    # Sidebar controls specific to Ratio Dashboard
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ratio Controls")

    group_choice = st.sidebar.selectbox("Category", list(RATIO_GROUPS.keys()))
    label_choice = st.sidebar.radio("Preset ratios", list(RATIO_GROUPS[group_choice].keys()))
    preset_a, preset_b = RATIO_GROUPS[group_choice][label_choice]

    date_preset = st.sidebar.radio("Date range", ["YTD", "1Y", "3Y", "5Y", "Max"], horizontal=True)
    start_date_ratio = start_date_from_preset(date_preset)

    use_custom = st.sidebar.checkbox("Use custom tickers", value=False)
    custom_a = st.sidebar.text_input("Custom ticker 1", value="SPY").upper().strip()
    custom_b = st.sidebar.text_input("Custom ticker 2", value="TLT").upper().strip()

    with st.sidebar.expander("Advanced chart options"):
        show_ma = st.checkbox("Show moving averages", value=True)
        log_scale = st.checkbox("Log scale", value=False)
        show_extremes = st.checkbox("Label high/low/latest", value=False)

    # Choose symbols + copy
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
    st.write(f"**Date Range:** {date_preset} (start: `{start_date_ratio}`)")
    st.markdown(f"**Description:** {desc_text}")
    st.markdown(f"**Commentary:** {comm_text}")

    if "ratio_df" not in st.session_state:
        st.session_state["ratio_df"] = pd.DataFrame()
        st.session_state["ratio_name"] = ""

    if st.button("Plot ratio", type="primary"):
        try:
            with st.spinner("Downloading data..."):
                df_ratio = build_ratio_dataframe(sym_a, sym_b, start_date_ratio)
            if df_ratio.empty:
                st.error("No valid data returned. Try Max or check tickers.")
            else:
                st.session_state["ratio_df"] = df_ratio
                st.session_state["ratio_name"] = f"{sym_a}/{sym_b}"
        except Exception as e:
            st.exception(e)

    df_saved = st.session_state.get("ratio_df", pd.DataFrame())
    if isinstance(df_saved, pd.DataFrame) and not df_saved.empty:
        r = df_saved["ratio"].dropna()
        latest_val = float(r.iloc[-1])
        prev_val = float(r.iloc[-2]) if len(r) > 1 else latest_val
        delta_pct = (latest_val / prev_val - 1) * 100 if prev_val != 0 else 0.0
        high_val = float(r.max()); high_date = r.idxmax()
        low_val = float(r.min()); low_date = r.idxmin()
        drawdown_pct = (latest_val / high_val - 1) * 100 if high_val != 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest", f"{latest_val:.3f}", f"{delta_pct:+.2f}%")
        m2.metric("High", f"{high_val:.3f}", high_date.strftime("%Y-%m-%d"))
        m3.metric("Low", f"{low_val:.3f}", low_date.strftime("%Y-%m-%d"))
        m4.metric("Drawdown from High", f"{drawdown_pct:.2f}%", "")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(df_saved.index, df_saved["ratio"], label=f"{sym_a}/{sym_b} Ratio", linewidth=1.7)

        if show_ma:
            if df_saved["ma50"].notna().any():
                ax.plot(df_saved.index, df_saved["ma50"], label="50-day MA", linestyle="--", linewidth=1.2)
            if df_saved["ma200"].notna().any():
                ax.plot(df_saved.index, df_saved["ma200"], label="200-day MA", linestyle="--", linewidth=1.2)

        if show_extremes:
            ax.scatter([high_date], [high_val], s=45)
            ax.scatter([low_date], [low_val], s=45)
            ax.scatter([r.index[-1]], [latest_val], s=45)
            ax.annotate(f"High {high_val:.3f}", xy=(high_date, high_val), xytext=(10, 10), textcoords="offset points")
            ax.annotate(f"Low {low_val:.3f}", xy=(low_date, low_val), xytext=(10, -15), textcoords="offset points")
            ax.annotate(f"Latest {latest_val:.3f}", xy=(r.index[-1], latest_val), xytext=(10, 0), textcoords="offset points")

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
            file_name=f"{csv_prefix}_{date_preset}_ratio_series.csv",
            mime="text/csv"
        )
    else:
        st.info("Tip: This page is for **two-ticker ratios**. Click **Plot ratio** to generate the chart.")

# ============================================================
# PAGE 2: YTD PERFORMANCE
# ============================================================
elif page == "📈 YTD Performance":
    st.subheader("📈 YTD Performance")
    st.info("This page compares **up to 10 tickers** indexed to 100 on Jan 1 (YTD performance).")

    tickers_raw = st.text_input("Enter up to 10 tickers (comma-separated)", value="SPY, QQQ, IWM")
    ticker_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if len(ticker_list) > 10:
        st.warning("⚠️ More than 10 tickers entered. Only the first 10 will be used.")
        ticker_list = ticker_list[:10]

    start_ytd = f"{dt.now().year}-01-01"
    end_ytd = dt.today().strftime("%Y-%m-%d")
    want_names = st.checkbox("🔎 Fetch company names (slower)", value=False)

    if "ytd_export" not in st.session_state:
        st.session_state["ytd_export"] = pd.DataFrame()

    if st.button("Plot YTD Performance", type="primary"):
        if len(ticker_list) == 0:
            st.error("❌ No tickers entered.")
        else:
            with st.spinner("⏳ Downloading price data..."):
                close_df = fetch_ytd_close_prices(ticker_list, start_ytd, end_ytd)

            if close_df.empty:
                st.error("❌ No valid data returned. Check tickers and try again.")
            else:
                ok = close_df.columns.tolist()
                bad = [t for t in ticker_list if t not in ok]
                if bad:
                    st.warning(f"⚠️ These tickers failed: {', '.join(bad)}")

                ytd_idx = (close_df / close_df.iloc[0]) * 100
                final_vals = ytd_idx.iloc[-1]

                if want_names:
                    with st.spinner("🔎 Fetching company names..."):
                        name_map = fetch_company_names(ok)
                else:
                    name_map = {t: t for t in ok}

                summary = pd.DataFrame({
                    "Ticker": final_vals.index,
                    "Name": [name_map[t] for t in final_vals.index],
                    "YTD % Return": final_vals.values - 100
                }).sort_values(by="YTD % Return", ascending=False)

                summary_display = summary.copy()
                summary_display["YTD % Return"] = summary_display["YTD % Return"].map(lambda x: f"{x:.1f}%")

                st.subheader("Summary")
                st.dataframe(summary_display, use_container_width=True)

                fig, ax = plt.subplots(figsize=(14, 9))
                line_colors = {}
                for t in ytd_idx.columns:
                    line, = ax.plot(ytd_idx.index, ytd_idx[t], linewidth=2)
                    line_colors[t] = line.get_color()

                ax.axhline(y=100, color="#888888", linestyle="--", linewidth=1.5)

                sorted_tickers = final_vals.sort_values(ascending=False).index.tolist()
                spacing_offset = 0.8
                for rank, t in enumerate(sorted_tickers):
                    last_date = ytd_idx.index[-1]
                    last_value = ytd_idx[t].iloc[-1]
                    offset = spacing_offset * (len(sorted_tickers) - rank)
                    adjusted_y = last_value + offset
                    label_text = f"{t} ({last_value - 100:.1f}%)"
                    ax.text(
                        last_date, adjusted_y, label_text,
                        fontsize=9, ha="left", va="center",
                        color=line_colors[t],
                        bbox=dict(facecolor="white", edgecolor=line_colors[t], boxstyle="round,pad=0.3")
                    )

                ax.set_title("YTD Performance (% Change from Jan 1)", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Performance (Indexed to 100)")
                ax.grid(True, linestyle="--", alpha=0.7)

                table_ax = fig.add_axes([0.15, -0.27, 0.7, 0.15])
                table_ax.axis("off")
                table_ax.table(
                    cellText=summary_display.values,
                    colLabels=summary_display.columns,
                    colLoc="center",
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.12, 0.50, 0.18]
                )

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.32)
                st.pyplot(fig)

                export_df = ytd_idx.copy()
                export_df.columns = [f"{c}_indexed100" for c in export_df.columns]
                export_df = export_df.join((ytd_idx - 100).add_suffix("_ytd_pct"))
                st.session_state["ytd_export"] = export_df

    saved_ytd = st.session_state.get("ytd_export", pd.DataFrame())
    if isinstance(saved_ytd, pd.DataFrame) and not saved_ytd.empty:
        st.download_button(
            "📥 Download YTD series as CSV",
            data=saved_ytd.to_csv().encode("utf-8"),
            file_name=f"ytd_performance_{dt.now().year}.csv",
            mime="text/csv"
        )
    else:
        st.info("Tip: This page is for **multi-ticker YTD**. Click **Plot YTD Performance** to generate the chart.")

# ============================================================
# PAGE 3: CHEAT SHEET
# ============================================================
else:
    st.subheader("📋 Cheat Sheet")
    st.info("This page lists all preset ratios with descriptions and commentary (no chart).")

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
