import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

plt.switch_backend("Agg")
st.set_page_config(page_title="Market Dashboard", layout="wide")

st.title("📊 Market Dashboard")
st.write(
    "Use the sidebar to choose a **preset ratio** (grouped by category), or use **custom tickers** in the Ratio Chart tab. "
    "Use the tabs for the Ratio Chart, Cheat Sheet, and YTD Performance."
)

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
# RATIO DESCRIPTIONS + COMMENTARY (PRESETS)
# ============================================================
RATIO_INFO = {
    "SPY / RSP – S&P 500 Cap vs Equal Weight": {
        "description": "Cap-weighted S&P 500 vs equal-weight; highlights breadth vs mega-cap concentration.",
        "commentary": "Rising = narrow leadership (mega-caps dominate). Falling = broader participation (healthier breadth).",
    },
    "QQQ / IWM – Nasdaq 100 vs Russell 2000": {
        "description": "Large-cap growth/tech vs small caps; a growth leadership and risk appetite gauge.",
        "commentary": "Rising = big tech leadership/quality preference. Falling = risk-on rotation into small caps.",
    },
    "DIA / IWM – Dow vs Small Caps": {
        "description": "Blue-chip Dow vs small caps; stability vs domestic risk exposure.",
        "commentary": "Rising = defensive tilt to established names. Falling = higher risk appetite / cyclicality.",
    },
    "MGK / SPY – Mega Cap Growth vs S&P 500": {
        "description": "Mega-cap growth vs broad market; measures concentration in growth leaders.",
        "commentary": "Rising = growth crowding. Falling = rotation into broader market/value/cyclicals.",
    },

    "SPY / TLT – Stocks vs Long-Term Bonds": {
        "description": "Classic risk-on/risk-off ratio: equities vs long-duration Treasuries.",
        "commentary": "Rising = risk-on. Falling = flight to safety / growth concerns.",
    },
    "HYG / IEF – High Yield vs Treasuries": {
        "description": "Credit risk appetite: high yield vs intermediate Treasuries.",
        "commentary": "Rising = healthy credit. Falling = spreads widening / credit stress risk.",
    },
    "XLY / XLP – Discretionary vs Staples": {
        "description": "Consumer cyclicals vs defensives; proxy for consumer confidence.",
        "commentary": "Rising = consumer/risk-on. Falling = defensive posture / growth caution.",
    },
    "IWM / SHY – Small Caps vs Short Treasuries": {
        "description": "Small caps vs cash-like Treasuries; pure risk appetite gauge.",
        "commentary": "Rising = risk-on. Falling = capital preservation / liquidity preference.",
    },
    "SPHB / SPLV – High Beta vs Low Vol": {
        "description": "High beta vs low volatility; aggression vs defense indicator.",
        "commentary": "Rising = speculation/risk-taking. Falling = preference for stability.",
    },

    "XLF / SPY – Financials vs Market": {
        "description": "Financial sector vs market; ties to credit and curve expectations.",
        "commentary": "Rising = improving conditions. Falling = tightening/stress or growth worries.",
    },
    "XLV / SPY – Healthcare vs Market": {
        "description": "Healthcare vs market; defensive leadership indicator.",
        "commentary": "Rising = defensive rotation. Falling = more risk-on market posture.",
    },
    "XLE / SPY – Energy vs Market": {
        "description": "Energy vs market; sensitive to oil/inflation dynamics.",
        "commentary": "Rising = energy/inflation strength. Falling = disinflation or weaker demand.",
    },
    "XLK / SPY – Tech vs Market": {
        "description": "Tech vs market; measures growth leadership and rate sensitivity.",
        "commentary": "Rising = tech leadership. Falling = rotation away from long-duration growth.",
    },
    "XLI / SPY – Industrials vs Market": {
        "description": "Industrials vs market; proxy for capex/trade/manufacturing optimism.",
        "commentary": "Rising = stronger cycle expectations. Falling = slowing activity concerns.",
    },
    "RSPD / RSPS – Equal Disc vs Equal Staples": {
        "description": "Equal-weight discretionary vs equal-weight staples; reduces mega-cap distortion.",
        "commentary": "Rising = broad consumer risk-on. Falling = defensive consumer stance.",
    },

    "DBC / SPY – Commodities vs Stocks": {
        "description": "Broad commodities vs equities; real-asset/inflation sensitivity.",
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
        "commentary": "Rising = energy/inflation risk. Falling = weaker demand or disinflation.",
    },
    "TIP / TLT – TIPS vs Treasuries": {
        "description": "Inflation-protected vs nominal Treasuries; inflation expectations proxy.",
        "commentary": "Rising = inflation expectations rising. Falling = disinflation expectations.",
    },
    "CPER / GLD – Copper vs Gold": {
        "description": "Cyclical copper vs defensive gold; growth vs fear signal.",
        "commentary": "Rising = growth optimism. Falling = risk-off / recession concerns.",
    },
    "GLD / USO – Gold vs Oil": {
        "description": "Gold vs oil; defensive vs cyclical commodity exposure.",
        "commentary": "Rising = fear/slower growth. Falling = stronger demand/cycle/inflation pressures.",
    },
    "GLD / XME – Gold vs Metals & Mining": {
        "description": "Gold vs industrial metals/mining; safety vs industrial cycle exposure.",
        "commentary": "Rising = defensive preference. Falling = pro-growth industrial demand theme.",
    },

    "EEM / SPY – Emerging Markets vs U.S.": {
        "description": "Emerging markets vs U.S.; global growth and USD sensitivity.",
        "commentary": "Rising = EM tailwinds (often weaker USD). Falling = U.S. dominance/caution.",
    },
    "VEA / SPY – Developed Intl vs U.S.": {
        "description": "Developed international vs U.S.; rotation between regions/styles.",
        "commentary": "Rising = non-U.S. leadership. Falling = U.S. dominance (often growth-led).",
    },
    "FXI / SPY – China vs U.S.": {
        "description": "China large caps vs U.S.; policy/growth and geopolitics sensitivity.",
        "commentary": "Rising = improving China sentiment. Falling = elevated risk/policy or growth concerns.",
    },

    "ETHA / IBIT – ETH vs BTC ETF": {
        "description": "Ethereum vs Bitcoin; crypto rotation gauge.",
        "commentary": "Rising = ETH leadership. Falling = BTC leadership as core asset.",
    },
    "ETHA / GSOL – ETH vs Solana": {
        "description": "Ethereum vs Solana; layer-1 leadership rotation.",
        "commentary": "Rising = ETH favored. Falling = SOL favored (often higher risk appetite).",
    },
    "BMNR / ETHA – BMNR vs ETH": {
        "description": "Speculative mining-related equity vs ETH; proxy for leveraged/speculative exposure.",
        "commentary": "Rising = higher speculation/leverage appetite. Falling = preference for underlying exposure.",
    },
    "MSTR / IBIT – MicroStrategy vs BTC ETF": {
        "description": "MSTR vs spot BTC ETF; equity optionality vs pure BTC exposure.",
        "commentary": "Rising = leverage/optionality rewarded. Falling = preference for pure BTC exposure.",
    },
}

# ============================================================
# SIDEBAR SELECTION
# ============================================================
st.sidebar.header("📌 Preset Ratios")
selected_group = st.sidebar.selectbox("Select category:", list(RATIO_GROUPS.keys()))
selected_label = st.sidebar.radio("Select a ratio:", list(RATIO_GROUPS[selected_group].keys()))
preset_sym1, preset_sym2 = RATIO_GROUPS[selected_group][selected_label]

# ============================================================
# HELPERS
# ============================================================
@st.cache_data(ttl=60 * 30)
def download_close_one(ticker: str, start_date: str) -> pd.Series:
    data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)
    return data["Close"].rename(ticker)

def compute_ratio(sym1: str, sym2: str, start_date: str):
    s1 = download_close_one(sym1, start_date)
    s2 = download_close_one(sym2, start_date)
    if s1.empty or s2.empty:
        return None

    df = pd.concat([s1, s2], axis=1).dropna()
    if df.empty:
        return None

    df.columns = [sym1, sym2]
    ratio = df[sym1] / df[sym2]
    ma50 = ratio.rolling(50).mean()
    ma200 = ratio.rolling(200).mean()

    out = pd.DataFrame({
        sym1: df[sym1],
        sym2: df[sym2],
        "ratio": ratio,
        "ma50": ma50,
        "ma200": ma200
    })
    return out

# YTD helper: returns Close price dataframe
@st.cache_data(ttl=60 * 60)
def download_ytd_close(tickers, start_date, end_date):
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        # Multi-ticker
        if "Close" in raw_data.columns.get_level_values(0):
            data = raw_data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        # Single ticker
        if "Close" in raw_data.columns:
            data = raw_data[["Close"]].copy()
            data.columns = [tickers[0]]
        else:
            return pd.DataFrame()

    data = data.dropna(axis=1, how="all")
    return data

def get_company_names(tickers, sleep_s=0.4):
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("shortName", t)
        except Exception:
            names[t] = t
        time.sleep(sleep_s)
    return names

# ============================================================
# TABS
# ============================================================
tab_chart, tab_cheat, tab_ytd = st.tabs(["📊 Ratio Chart", "📋 Cheat Sheet", "📈 YTD Performance"])

# ============================================================
# TAB 1: RATIO CHART
# ============================================================
with tab_chart:
    st.subheader("Custom Ratio (Optional) – at the Top")

    c1, c2, c3 = st.columns(3)
    with c1:
        use_custom = st.checkbox("Use custom tickers", value=False)
    with c2:
        custom_sym1 = st.text_input("Custom first ticker", value="SPY").upper().strip()
    with c3:
        custom_sym2 = st.text_input("Custom second ticker", value="TLT").upper().strip()

    start_date = st.text_input("Start date (YYYY-MM-DD)", value="2015-01-01").strip()
    st.markdown("---")

    if use_custom:
        sym1 = custom_sym1
        sym2 = custom_sym2
        title_text = f"Custom Ratio: {sym1}/{sym2}"
        description_text = "User-defined custom ratio."
        commentary_text = "Interpretation depends on the relationship between the two chosen assets."
        csv_name_prefix = f"{sym1}_{sym2}_custom"
    else:
        sym1 = preset_sym1
        sym2 = preset_sym2
        title_text = selected_label
        info = RATIO_INFO.get(selected_label, {})
        description_text = info.get("description", "No description available.")
        commentary_text = info.get("commentary", "No commentary available.")
        csv_name_prefix = f"{sym1}_{sym2}_preset"

    st.subheader(title_text)
    st.markdown(f"**Description:** {description_text}")
    st.markdown(f"**Commentary:** {commentary_text}")

    if "last_ratio_df" not in st.session_state:
        st.session_state.last_ratio_df = None
        st.session_state.last_ratio_name = None

    if st.button("Plot ratio"):
        try:
            with st.spinner("Downloading data..."):
                df = compute_ratio(sym1, sym2, start_date)

            if df is None or df.empty:
                st.error("No valid data returned. Check tickers and start date.")
            else:
                st.session_state.last_ratio_df = df.copy()
                st.session_state.last_ratio_name = f"{sym1}/{sym2}"

                latest = float(df["ratio"].dropna().iloc[-1])
                if df["ratio"].dropna().shape[0] > 1:
                    prev = float(df["ratio"].dropna().iloc[-2])
                    delta_pct = (latest / prev - 1) * 100 if prev != 0 else 0
                else:
                    delta_pct = 0.0

                st.metric(
                    label=f"Latest {sym1}/{sym2}",
                    value=f"{latest:.3f}",
                    delta=f"{delta_pct:+.2f}% vs prev close"
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df["ratio"], label=f"{sym1}/{sym2} Ratio", linewidth=1.5)
                if df["ma50"].notna().any():
                    ax.plot(df.index, df["ma50"], label="50-day MA", linestyle="--", linewidth=1.2)
                if df["ma200"].notna().any():
                    ax.plot(df.index, df["ma200"], label="200-day MA", linestyle="--", linewidth=1.2)

                ax.set_title(title_text, fontsize=13, fontweight="bold")
                ax.set_xlabel("Date")
                ax.set_ylabel("Ratio")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    if st.session_state.last_ratio_df is not None and not st.session_state.last_ratio_df.empty:
        export_df = st.session_state.last_ratio_df.copy()
        export_df = export_df.rename(columns={
            "ratio": f"{st.session_state.last_ratio_name}_ratio",
            "ma50": "MA50",
            "ma200": "MA200"
        })
        csv_bytes = export_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download ratio series as CSV",
            data=csv_bytes,
            file_name=f"{csv_name_prefix}_ratio_series.csv",
            mime="text/csv"
        )
    else:
        st.info("Click **Plot ratio** to view the chart and enable CSV download.")

# ============================================================
# TAB 2: CHEAT SHEET
# ============================================================
with tab_cheat:
    st.subheader("📋 Ratio Cheat Sheet (All Presets)")

    rows = []
    for group_name, ratios in RATIO_GROUPS.items():
        for label, (s1, s2) in ratios.items():
            info = RATIO_INFO.get(label, {})
            rows.append({
                "Category": group_name,
                "Label": label,
                "Symbol 1": s1,
                "Symbol 2": s2,
                "Description": info.get("description", ""),
                "Commentary": info.get("commentary", "")
            })

    cheat_df = pd.DataFrame(rows)
    st.dataframe(cheat_df, use_container_width=True)

    csv_cheat = cheat_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download cheat sheet as CSV",
        data=csv_cheat,
        file_name="ratio_cheat_sheet.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 3: YTD PERFORMANCE (MATCHES YOUR ORIGINAL STYLE)
# ============================================================
with tab_ytd:
    st.subheader("📈 YTD Performance (% Change from Jan 1)")
    st.write("Enter up to 10 tickers (comma-separated). Example: `SPY, QQQ, IWM, TLT`")

    tickers_input = st.text_input("Tickers", value="SPY, QQQ, IWM")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if len(tickers) > 10:
        st.warning("⚠️ More than 10 tickers entered. Only the first 10 will be used.")
        tickers = tickers[:10]

    start_date = f"{datetime.now().year}-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    fetch_names = st.checkbox("🔎 Fetch company names (slower; can hit rate limits)", value=False)

    if "last_ytd_export" not in st.session_state:
        st.session_state.last_ytd_export = None

    if st.button("Plot YTD Performance"):
        if len(tickers) == 0:
            st.error("❌ No tickers entered.")
        else:
            with st.spinner("⏳ Downloading price data..."):
                data = download_ytd_close(tickers, start_date, end_date)

            if data.empty:
                st.error("❌ No valid data returned. Check tickers and try again.")
            else:
                successful_tickers = data.columns.tolist()
                failed = [t for t in tickers if t not in successful_tickers]
                if failed:
                    st.warning(f"⚠️ These tickers failed: {', '.join(failed)}")

                ytd_returns = (data / data.iloc[0]) * 100
                final_values = ytd_returns.iloc[-1]

                # Names
                if fetch_names:
                    with st.spinner("🔎 Fetching company names..."):
                        names = get_company_names(successful_tickers)
                else:
                    names = {t: t for t in successful_tickers}

                summary_df = pd.DataFrame({
                    "Ticker": final_values.index,
                    "Name": [names[t] for t in final_values.index],
                    "YTD % Return": final_values.values - 100
                }).sort_values(by="YTD % Return", ascending=False)

                summary_show = summary_df.copy()
                summary_show["YTD % Return"] = summary_show["YTD % Return"].map(lambda x: f"{x:.1f}%")

                st.subheader("Summary")
                st.dataframe(summary_show, use_container_width=True)

                # Plot (with end labels + table under chart)
                fig, ax = plt.subplots(figsize=(14, 9))
                line_colors = {}

                for t in ytd_returns.columns:
                    line, = ax.plot(ytd_returns.index, ytd_returns[t], linewidth=2)
                    line_colors[t] = line.get_color()

                ax.axhline(y=100, color="#888888", linestyle="--", linewidth=1.5)

                sorted_tickers = final_values.sort_values(ascending=False).index.tolist()
                spacing_offset = 0.8

                for rank, t in enumerate(sorted_tickers):
                    last_date = ytd_returns.index[-1]
                    last_value = ytd_returns[t].iloc[-1]
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

                # Embed table below chart
                table_ax = fig.add_axes([0.15, -0.27, 0.7, 0.15])
                table_ax.axis("off")
                table_ax.table(
                    cellText=summary_show.values,
                    colLabels=summary_show.columns,
                    colLoc="center",
                    cellLoc="center",
                    loc="center",
                    colWidths=[0.12, 0.50, 0.18]
                )

                plt.tight_layout()
                plt.subplots_adjust(bottom=0.32)

                st.pyplot(fig)

                # CSV export
                export_df = ytd_returns.copy()
                export_df.columns = [f"{c}_indexed100" for c in export_df.columns]
                export_df = export_df.join((ytd_returns - 100).add_suffix("_ytd_pct"))
                st.session_state.last_ytd_export = export_df

    if st.session_state.last_ytd_export is not None and not st.session_state.last_ytd_export.empty:
        csv_bytes = st.session_state.last_ytd_export.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download YTD series as CSV",
            data=csv_bytes,
            file_name=f"ytd_performance_{datetime.now().year}.csv",
            mime="text/csv"
        )
    else:
        st.info("Click **Plot YTD Performance** to generate the chart and enable CSV download.")
