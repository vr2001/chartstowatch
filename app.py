import time as pytime
from datetime import datetime as dt, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

plt.switch_backend("Agg")

st.set_page_config(
    page_title="Charts to Watch",
    layout="wide"
)

# ============================================================
# PRESET RATIOS
# ============================================================

RATIO_GROUPS = {
    "Stock Market Breadth & Strength": {
        "SPY / RSP - Cap vs Equal Weight": ("SPY", "RSP"),
        "QQQ / IWM - Nasdaq vs Small Caps": ("QQQ", "IWM"),
        "DIA / IWM - Dow vs Small Caps": ("DIA", "IWM"),
        "MGK / SPY - Mega Cap Growth vs Market": ("MGK", "SPY"),
    },
    "Risk-On vs Risk-Off Sentiment": {
        "SPY / TLT - Stocks vs Bonds": ("SPY", "TLT"),
        "HYG / IEF - High Yield vs Treasuries": ("HYG", "IEF"),
        "XLY / XLP - Discretionary vs Staples": ("XLY", "XLP"),
        "IWM / SHY - Small Caps vs Treasuries": ("IWM", "SHY"),
        "SPHB / SPLV - High Beta vs Low Volatility": ("SPHB", "SPLV"),
    },
    "Sector Relative Strength": {
        "XLF / SPY - Financials": ("XLF", "SPY"),
        "XLV / SPY - Healthcare": ("XLV", "SPY"),
        "XLE / SPY - Energy": ("XLE", "SPY"),
        "XLK / SPY - Technology": ("XLK", "SPY"),
        "XLI / SPY - Industrials": ("XLI", "SPY"),
        "RSPD / RSPS - Equal Weight Consumer": ("RSPD", "RSPS"),
    },
    "Commodities & Inflation": {
        "DBC / SPY - Commodities vs Stocks": ("DBC", "SPY"),
        "GDX / SPY - Gold Miners vs Stocks": ("GDX", "SPY"),
        "GLD / TLT - Gold vs Bonds": ("GLD", "TLT"),
        "USO / SPY - Oil vs Stocks": ("USO", "SPY"),
        "TIP / TLT - TIPS vs Treasuries": ("TIP", "TLT"),
        "CPER / GLD - Copper vs Gold": ("CPER", "GLD"),
        "GLD / USO - Gold vs Oil": ("GLD", "USO"),
        "GLD / XME - Gold vs Metals": ("GLD", "XME"),
    },
    "Global vs U.S. Markets": {
        "EEM / SPY - Emerging Markets": ("EEM", "SPY"),
        "VEA / SPY - Developed Markets": ("VEA", "SPY"),
        "FXI / SPY - China vs U.S.": ("FXI", "SPY"),
    },
    "Crypto Relative Strength": {
        "ETHA / IBIT - Ethereum vs Bitcoin": ("ETHA", "IBIT"),
        "ETHA / GSOL - Ethereum vs Solana": ("ETHA", "GSOL"),
        "BMNR / ETHA - BMNR vs Ethereum": ("BMNR", "ETHA"),
        "MSTR / IBIT - MSTR vs Bitcoin": ("MSTR", "IBIT"),
    }
}


# ============================================================
# RATIO DESCRIPTIONS
# ============================================================

RATIO_INFO = {
    "SPY/RSP": (
        "Capital-weighted vs equal-weight S&P 500.",
        "Rising = narrow leadership. Falling = broader participation."
    ),
    "QQQ/IWM": (
        "Large-cap growth vs small-cap stocks.",
        "Rising = large-cap growth leadership."
    ),
    "DIA/IWM": (
        "Blue-chip stocks vs small caps.",
        "Rising may indicate preference for established companies."
    ),
    "MGK/SPY": (
        "Mega-cap growth vs the broader market.",
        "Rising = mega-cap growth leadership."
    ),
    "SPY/TLT": (
        "Stocks vs long-term Treasury bonds.",
        "Rising may indicate risk-on sentiment."
    ),
    "HYG/IEF": (
        "High-yield corporate bonds vs Treasuries.",
        "Falling may indicate increasing credit stress."
    ),
    "XLY/XLP": (
        "Consumer discretionary vs consumer staples.",
        "Rising may indicate stronger consumer confidence."
    ),
    "IWM/SHY": (
        "Small-cap stocks vs short-term Treasuries.",
        "Rising may indicate increasing risk appetite."
    ),
    "SPHB/SPLV": (
        "High-beta vs low-volatility stocks.",
        "Rising = increasing demand for aggressive equities."
    ),
    "XLF/SPY": (
        "Financial stocks vs the broader market.",
        "Rising = financial sector leadership."
    ),
    "XLV/SPY": (
        "Healthcare vs the broader market.",
        "Rising may indicate defensive rotation."
    ),
    "XLE/SPY": (
        "Energy vs the broader market.",
        "Rising = energy sector leadership."
    ),
    "XLK/SPY": (
        "Technology vs the broader market.",
        "Rising = technology sector leadership."
    ),
    "XLI/SPY": (
        "Industrials vs the broader market.",
        "Rising may indicate stronger industrial expectations."
    ),
    "RSPD/RSPS": (
        "Equal-weight discretionary vs staples.",
        "Rising may indicate broad consumer risk appetite."
    ),
    "DBC/SPY": (
        "Commodities vs equities.",
        "Rising = relative commodity strength."
    ),
    "GDX/SPY": (
        "Gold miners vs the broader market.",
        "Rising = gold miner relative strength."
    ),
    "GLD/TLT": (
        "Gold vs long-term Treasury bonds.",
        "Rising = preference for gold relative to bonds."
    ),
    "USO/SPY": (
        "Oil vs equities.",
        "Rising may indicate stronger oil prices or inflation pressure."
    ),
    "TIP/TLT": (
        "Inflation-protected vs long-term nominal Treasuries.",
        "Sensitive to inflation expectations and duration differences."
    ),
    "CPER/GLD": (
        "Copper vs gold.",
        "Rising may indicate improving growth expectations."
    ),
    "GLD/USO": (
        "Gold vs oil.",
        "Rising may indicate defensive commodity positioning."
    ),
    "GLD/XME": (
        "Gold vs metals and mining equities.",
        "Rising may indicate preference for defensive assets."
    ),
    "EEM/SPY": (
        "Emerging markets vs U.S. equities.",
        "Rising = emerging market leadership."
    ),
    "VEA/SPY": (
        "Developed international markets vs U.S.",
        "Rising = international market leadership."
    ),
    "FXI/SPY": (
        "Chinese equities vs U.S. equities.",
        "Rising = Chinese equity relative strength."
    ),
    "ETHA/IBIT": (
        "Ethereum vs Bitcoin.",
        "Rising = Ethereum leadership."
    ),
    "ETHA/GSOL": (
        "Ethereum vs Solana exposure.",
        "Rising = Ethereum relative strength."
    ),
    "BMNR/ETHA": (
        "BMNR equity vs Ethereum exposure.",
        "Rising = BMNR equity outperformance."
    ),
    "MSTR/IBIT": (
        "Strategy equity vs Bitcoin exposure.",
        "Rising = equity outperformance relative to Bitcoin."
    ),
}


# ============================================================
# DATE HELPERS
# ============================================================

def period_start_date(preset):

    today = dt.today()

    if preset == "QTD":
        month = ((today.month - 1) // 3) * 3 + 1
        return dt(today.year, month, 1).strftime("%Y-%m-%d")

    if preset == "YTD":
        return f"{today.year}-01-01"

    periods = {
        "3M": 92,
        "6M": 183,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
        "5Y": 1825,
        "10Y": 3650
    }

    if preset in periods:
        return (
            today - timedelta(days=periods[preset])
        ).strftime("%Y-%m-%d")

    return "2000-01-01"


# ============================================================
# FORMATTING HELPERS
# ============================================================

def fmt(value, style="number"):

    try:
        value = float(value)

        if not np.isfinite(value):
            return "N/A"

    except (TypeError, ValueError):
        return "N/A"

    if style == "percent":
        return f"{value:,.2f}%"

    if style == "multiple":
        return f"{value:,.2f}x"

    if style == "money":

        for divisor, suffix in [
            (1e12, "T"),
            (1e9, "B"),
            (1e6, "M"),
            (1e3, "K")
        ]:

            if abs(value) >= divisor:
                return f"{value / divisor:,.2f}{suffix}"

    return f"{value:,.2f}"


def metric_grid(metrics, columns=4):

    for start in range(0, len(metrics), columns):

        cols = st.columns(columns)

        for col, item in zip(
            cols, metrics[start:start + columns]
        ):

            label, value, style = item

            col.metric(label, fmt(value, style))


def safe_divide(a, b):

    return a.div(
        b.where(b > 0)
    ).replace([np.inf, -np.inf], np.nan)


def growth_rate(series):

    prior = series.shift(1)

    return (
        (series - prior) /
        prior.where(prior > 0)
    ) * 100


def scale_money(series, scale):

    if scale == "Billions":
        return series / 1e9

    if scale == "Millions":
        return series / 1e6

    return series


def csv_button(df, filename, key):

    st.download_button(
        "Download CSV",
        data=df.to_csv().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


# ============================================================
# DATA DOWNLOAD HELPERS
# ============================================================

@st.cache_data(ttl=1800)
def fetch_prices(symbols, start_date, adjusted=False):

    try:

        raw = yf.download(
            symbols,
            start=start_date,
            progress=False,
            auto_adjust=adjusted
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):

            if "Close" not in raw.columns.get_level_values(0):
                return pd.DataFrame()

            close = raw["Close"].copy()

        else:

            close = raw[["Close"]].copy()

            symbol = (
                symbols[0]
                if isinstance(symbols, list)
                else symbols
            )

            close.columns = [symbol]

        if isinstance(close, pd.Series):
            close = close.to_frame()

        close.index = pd.to_datetime(close.index)

        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)

        return close.dropna(
            axis=1, how="all"
        ).dropna(how="all")

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_ticker_info(ticker):

    try:
        return yf.Ticker(ticker).info or {}

    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_statements(ticker, frequency="Annual"):

    company = yf.Ticker(ticker)

    quarterly = frequency == "Quarterly"

    def clean(frame):

        if frame is None or frame.empty:
            return pd.DataFrame()

        frame = frame.copy()

        frame.index = frame.index.map(str)

        frame.columns = pd.to_datetime(
            frame.columns,
            errors="coerce"
        )

        frame = frame.loc[
            :, frame.columns.notna()
        ]

        frame = frame.loc[
            :, ~frame.columns.duplicated()
        ]

        frame = frame.sort_index(axis=1)

        return frame.apply(
            pd.to_numeric,
            errors="coerce"
        )

    statements = {}

    attributes = {
        "income": (
            "quarterly_income_stmt"
            if quarterly else "income_stmt"
        ),
        "balance": (
            "quarterly_balance_sheet"
            if quarterly else "balance_sheet"
        ),
        "cash": (
            "quarterly_cashflow"
            if quarterly else "cashflow"
        )
    }

    for name, attribute in attributes.items():

        try:

            statements[name] = clean(
                getattr(company, attribute)
            )

        except Exception:

            statements[name] = pd.DataFrame()

    return statements


# ============================================================
# FINANCIAL STATEMENT HELPERS
# ============================================================

def normalize_label(label):

    return "".join(
        char for char in str(label).lower()
        if char.isalnum()
    )


def statement_row(frame, *names):

    if frame is None or frame.empty:
        return pd.Series(dtype=float)

    lookup = {
        normalize_label(label): label
        for label in frame.index
    }

    for name in names:

        key = normalize_label(name)

        if key in lookup:

            series = frame.loc[lookup[key]]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[0]

            series = pd.to_numeric(
                series,
                errors="coerce"
            )

            return series.sort_index()

    return pd.Series(dtype=float)


def get_financial_history(statements, annual=True):

    income = statements["income"]
    balance = statements["balance"]
    cash = statements["cash"]

    dates = income.columns.union(
        balance.columns
    ).union(cash.columns).sort_values()

    history = pd.DataFrame(index=dates)

    income_fields = {
        "Revenue": (
            "Total Revenue",
            "Revenue"
        ),
        "Gross Profit": (
            "Gross Profit",
        ),
        "Operating Income": (
            "Operating Income",
        ),
        "Net Income": (
            "Net Income",
        ),
        "Diluted EPS": (
            "Diluted EPS",
        ),
        "EBITDA": (
            "EBITDA",
        ),
        "Pretax Income": (
            "Pretax Income",
        ),
        "Tax Provision": (
            "Tax Provision",
        ),
        "Interest Expense": (
            "Interest Expense",
        )
    }

    balance_fields = {
        "Total Assets": (
            "Total Assets",
        ),
        "Equity": (
            "Stockholders Equity",
            "Total Stockholder Equity"
        ),
        "Total Debt": (
            "Total Debt",
        ),
        "Cash": (
            "Cash And Cash Equivalents",
        ),
        "Invested Capital": (
            "Invested Capital",
        ),
        "Current Assets": (
            "Current Assets",
            "Total Current Assets"
        ),
        "Current Liabilities": (
            "Current Liabilities",
            "Total Current Liabilities"
        )
    }

    cash_fields = {
        "Operating Cash Flow": (
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ),
        "Free Cash Flow": (
            "Free Cash Flow",
        ),
        "CAPEX": (
            "Capital Expenditure",
            "Capital Expenditures"
        )
    }

    for fields, frame in [
        (income_fields, income),
        (balance_fields, balance),
        (cash_fields, cash)
    ]:

        for label, aliases in fields.items():

            history[label] = statement_row(
                frame, *aliases
            ).reindex(dates)

    history["CAPEX"] = history["CAPEX"].abs()

    history["Free Cash Flow"] = (
        history["Free Cash Flow"].combine_first(
            history["Operating Cash Flow"]
            - history["CAPEX"]
        )
    )

    for field in [
        "Revenue",
        "Net Income",
        "Operating Income",
        "Diluted EPS",
        "Free Cash Flow"
    ]:

        history[field + " Growth (%)"] = growth_rate(
            history[field]
        )

        if annual:

            consecutive = (
                history.index.to_series()
                .diff()
                .dt.days
                .between(300, 430)
            )

            history[field + " Growth (%)"] = (
                history[field + " Growth (%)"]
                .where(consecutive)
            )

    for field, label in [
        ("Gross Profit", "Gross Margin (%)"),
        ("Operating Income", "Operating Margin (%)"),
        ("Net Income", "Net Margin (%)"),
        ("Free Cash Flow", "FCF Margin (%)"),
        ("EBITDA", "EBITDA Margin (%)")
    ]:

        history[label] = safe_divide(
            history[field],
            history["Revenue"]
        ) * 100

    history["Debt / Equity"] = safe_divide(
        history["Total Debt"],
        history["Equity"]
    )

    history["Current Ratio"] = safe_divide(
        history["Current Assets"],
        history["Current Liabilities"]
    )

    history["Interest Coverage"] = safe_divide(
        history["Operating Income"],
        history["Interest Expense"].abs()
    )

    history["Debt / EBITDA"] = safe_divide(
        history["Total Debt"],
        history["EBITDA"]
    )

    if annual:

        consecutive = (
            history.index.to_series()
            .diff()
            .dt.days
            .between(300, 430)
        )

        avg_equity = (
            history["Equity"]
            + history["Equity"].shift(1)
        ) / 2

        avg_equity = avg_equity.where(
            consecutive
            & (history["Equity"] > 0)
            & (history["Equity"].shift(1) > 0)
        )

        avg_assets = (
            history["Total Assets"]
            + history["Total Assets"].shift(1)
        ) / 2

        avg_assets = avg_assets.where(consecutive)

        history["ROE (%)"] = safe_divide(
            history["Net Income"],
            avg_equity
        ) * 100

        history["ROA (%)"] = safe_divide(
            history["Net Income"],
            avg_assets
        ) * 100

        capital = history[
            "Invested Capital"
        ].combine_first(
            history["Total Debt"]
            + history["Equity"]
            - history["Cash"]
        )

        average_capital = (
            capital + capital.shift(1)
        ) / 2

        average_capital = average_capital.where(
            consecutive
            & (capital > 0)
            & (capital.shift(1) > 0)
        )

        tax_rate = safe_divide(
            history["Tax Provision"],
            history["Pretax Income"]
        )

        tax_rate = tax_rate.where(
            tax_rate.between(0, 1)
        )

        nopat = (
            history["Operating Income"]
            * (1 - tax_rate)
        )

        history["ROIC (%)"] = safe_divide(
            nopat,
            average_capital
        ) * 100

    return history.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# CHART HELPERS
# ============================================================

def plot_chart(
    data,
    title,
    ylabel="",
    chart_type="Line",
    labels=False
):

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(
        axis=1, how="all"
    ).dropna(how="all")

    if data.empty:
        st.info("No data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))

    if chart_type == "Bar":

        plot_data = data.copy()

        if isinstance(plot_data.index, pd.DatetimeIndex):

            plot_data.index = (
                plot_data.index.strftime("%Y-%m-%d")
            )

        plot_data.plot.bar(
            ax=ax,
            rot=30
        )

        if labels:

            for container in ax.containers:

                ax.bar_label(
                    container,
                    fmt="%.2f",
                    fontsize=8,
                    padding=3
                )

    else:

        for column in data.columns:

            ax.plot(
                data.index,
                data[column],
                linewidth=1.8,
                label=column
            )

            if labels:

                for x, y in data[column].dropna().items():

                    ax.annotate(
                        f"{y:,.2f}",
                        xy=(x, y),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8
                    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

    if len(data.columns) > 1:
        ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


def financial_table(history, columns):

    selected = history[
        [c for c in columns if c in history.columns]
    ].copy()

    selected.index = selected.index.strftime("%Y-%m-%d")

    table = selected.T.astype(object)

    for label in table.index:

        if "(%)" in label:
            style = "percent"

        elif label in [
            "Debt / Equity",
            "Debt / EBITDA",
            "Current Ratio",
            "Interest Coverage"
        ]:
            style = "multiple"

        elif "EPS" in label:
            style = "number"

        else:
            style = "money"

        table.loc[label] = table.loc[label].map(
            lambda x: fmt(x, style)
        )

    st.dataframe(
        table,
        use_container_width=True
    )


# ============================================================
# PAGE 1: RATIO DASHBOARD
# ============================================================

def render_ratio_dashboard():

    st.subheader("📊 Ratio Dashboard")

    st.info(
        "Compare two securities using relative strength ratios."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Ratio Controls")

    category = st.sidebar.selectbox(
        "Category",
        list(RATIO_GROUPS.keys())
    )

    preset = st.sidebar.radio(
        "Preset Ratios",
        list(RATIO_GROUPS[category].keys())
    )

    symbol_a, symbol_b = RATIO_GROUPS[category][preset]

    date_range = st.sidebar.radio(
        "Date Range",
        ["YTD", "1Y", "3Y", "5Y", "Max"],
        index=1,
        horizontal=True
    )

    custom = st.sidebar.checkbox(
        "Use custom tickers"
    )

    if custom:

        symbol_a = st.sidebar.text_input(
            "First Ticker",
            value="SPY"
        ).upper().strip()

        symbol_b = st.sidebar.text_input(
            "Second Ticker",
            value="TLT"
        ).upper().strip()

    with st.sidebar.expander("Advanced Chart Options"):

        show_ma = st.checkbox(
            "Show Moving Averages",
            value=True
        )

        log_scale = st.checkbox(
            "Log Scale",
            value=False
        )

        show_extremes = st.checkbox(
            "Label High, Low and Latest",
            value=False
        )

    ratio_name = f"{symbol_a}/{symbol_b}"

    st.markdown(f"### {ratio_name}")

    description, commentary = RATIO_INFO.get(
        ratio_name,
        (
            "Custom relative strength ratio.",
            "Rising means the first asset is outperforming the second."
        )
    )

    st.write("**Description:**", description)

    st.write("**Interpretation:**", commentary)

    start = period_start_date(date_range)

    st.caption(f"Start date: {start}")

    if not symbol_a or not symbol_b:
        st.warning("Enter two ticker symbols.")
        return

    if symbol_a == symbol_b:
        st.warning("Please enter two different symbols.")
        return

    selection = (
        symbol_a,
        symbol_b,
        date_range
    )

    if st.button("Plot Ratio", type="primary"):

        with st.spinner("Downloading market data..."):

            prices = fetch_prices(
                [symbol_a, symbol_b],
                start
            )

        if not all(
            s in prices.columns
            for s in [symbol_a, symbol_b]
        ):

            st.error("Unable to download both tickers.")
            return

        prices = prices[
            [symbol_a, symbol_b]
        ].dropna()

        prices["Ratio"] = (
            prices[symbol_a] /
            prices[symbol_b].replace(0, np.nan)
        )

        prices["Ratio"] = prices[
            "Ratio"
        ].replace([np.inf, -np.inf], np.nan)

        prices["50 DMA"] = (
            prices["Ratio"].rolling(50).mean()
        )

        prices["200 DMA"] = (
            prices["Ratio"].rolling(200).mean()
        )

        st.session_state["ratio_result"] = prices

        st.session_state["ratio_selection"] = selection

    if st.session_state.get("ratio_selection") != selection:

        st.info("Click Plot Ratio to update the chart.")
        return

    data = st.session_state.get(
        "ratio_result",
        pd.DataFrame()
    )

    if data.empty:
        return

    ratio = data["Ratio"].dropna()

    if ratio.empty:
        st.warning("No valid ratio data.")
        return

    def change(series):

        series = series.dropna()

        if len(series) < 2 or series.iloc[-2] == 0:
            return "N/A"

        return f"{(series.iloc[-1] / series.iloc[-2] - 1) * 100:+.2f}%"

    high = ratio.max()

    low = ratio.min()

    latest = ratio.iloc[-1]

    drawdown = (
        (latest / high - 1) * 100
        if high != 0 else np.nan
    )

    cols = st.columns(6)

    cols[0].metric(
        f"{symbol_a} Price",
        fmt(data[symbol_a].iloc[-1]),
        change(data[symbol_a])
    )

    cols[1].metric(
        f"{symbol_b} Price",
        fmt(data[symbol_b].iloc[-1]),
        change(data[symbol_b])
    )

    cols[2].metric(
        "Current Ratio",
        f"{latest:.3f}",
        change(ratio)
    )

    cols[3].metric(
        "Period High",
        f"{high:.3f}"
    )

    cols[4].metric(
        "Period Low",
        f"{low:.3f}"
    )

    cols[5].metric(
        "Drawdown",
        fmt(drawdown, "percent")
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        ratio.index,
        ratio,
        label=ratio_name,
        linewidth=2
    )

    if show_ma:

        for name in ["50 DMA", "200 DMA"]:

            ax.plot(
                data.index,
                data[name],
                linestyle="--",
                label=name
            )

    if show_extremes:

        points = [
            (ratio.idxmax(), high, "High"),
            (ratio.idxmin(), low, "Low"),
            (ratio.index[-1], latest, "Latest")
        ]

        for date, value, label in points:

            ax.scatter(date, value)

            ax.annotate(
                f"{label}: {value:.3f}",
                (date, value),
                xytext=(8, 8),
                textcoords="offset points"
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(ratio_name)
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    csv_button(
        data,
        f"{symbol_a}_{symbol_b}_ratio.csv",
        "ratio_csv"
    )


# ============================================================
# PAGE 2: PERFORMANCE
# ============================================================

def render_performance():

    st.subheader("📈 Performance")

    st.info(
        "Compare up to 20 securities. "
        "Performance is indexed to 100 at the beginning "
        "of the common available period."
    )

    date_range = st.radio(
        "Performance Period",
        ["QTD", "YTD", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
        horizontal=True
    )

    raw_tickers = st.text_input(
        "Enter up to 20 tickers, separated by commas",
        value="SPY, QQQ, IWM, TLT"
    )

    tickers = list(dict.fromkeys(
        t.strip().upper()
        for t in raw_tickers.split(",")
        if t.strip()
    ))

    if len(tickers) > 20:

        st.warning("Only the first 20 tickers will be used.")

        tickers = tickers[:20]

    include_names = st.checkbox(
        "Fetch Company Names",
        value=False
    )

    adjusted = st.checkbox(
        "Use Dividend and Split Adjusted Prices",
        value=True
    )

    selection = (
        tuple(tickers),
        date_range,
        include_names,
        adjusted
    )

    if st.button("Plot Performance", type="primary"):

        if not tickers:

            st.warning("Enter at least one ticker.")
            return

        with st.spinner("Downloading prices..."):

            prices = fetch_prices(
                tickers,
                period_start_date(date_range),
                adjusted=adjusted
            )

        if prices.empty:

            st.error("No price data returned.")
            return

        missing = [
            t for t in tickers
            if t not in prices.columns
        ]

        if missing:

            st.warning(
                "Unavailable tickers: "
                + ", ".join(missing)
            )

        prices = prices.dropna(how="any")

        if prices.empty:

            st.error(
                "No common dates are available."
            )
            return

        prices = prices.loc[
            :, prices.iloc[0] != 0
        ]

        indexed = (
            prices / prices.iloc[0]
        ) * 100

        st.session_state[
            "performance_result"
        ] = indexed

        st.session_state[
            "performance_selection"
        ] = selection

    if st.session_state.get(
        "performance_selection"
    ) != selection:

        st.info("Click Plot Performance to display results.")
        return

    indexed = st.session_state.get(
        "performance_result",
        pd.DataFrame()
    )

    if indexed.empty:
        return

    returns = (
        indexed.iloc[-1] - 100
    ).sort_values(ascending=False)

    summary = pd.DataFrame({
        "Ticker": returns.index,
        "Return (%)": returns.values
    })

    if include_names:

        summary["Company"] = [
            fetch_ticker_info(ticker).get(
                "shortName", ticker
            )
            for ticker in summary["Ticker"]
        ]

    st.caption(
        f"Actual period: "
        f"{indexed.index[0].date()} through "
        f"{indexed.index[-1].date()}"
    )

    st.dataframe(
        summary.style.format({
            "Return (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for ticker in returns.index:

        line, = ax.plot(
            indexed.index,
            indexed[ticker],
            linewidth=2,
            label=f"{ticker}: {returns[ticker]:+.1f}%"
        )

        ax.annotate(
            f"{ticker} ({returns[ticker]:+.1f}%)",
            (
                indexed.index[-1],
                indexed[ticker].iloc[-1]
            ),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=line.get_color()
        )

    ax.axhline(
        100,
        linestyle="--",
        color="gray"
    )

    ax.set_title(
        f"Performance Comparison - {date_range}"
    )

    ax.set_ylabel("Indexed Performance")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    csv_button(
        indexed.join(
            (indexed - 100).add_suffix("_Return_pct")
        ),
        "performance_comparison.csv",
        "performance_csv"
    )


# ============================================================
# PAGE 3: FUNDAMENTALS
# ============================================================

def render_fundamentals():

    st.subheader("📑 Fundamentals")

    st.info(
        "Explore financial statements, historical financial "
        "metrics, growth rates and valuation multiples."
    )

    left, right = st.columns(2)

    with left:

        ticker = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            key="fundamentals_ticker"
        ).upper().strip()

        frequency = st.radio(
            "Frequency",
            ["Annual", "Quarterly"],
            horizontal=True
        )

        mode = st.radio(
            "Mode",
            ["Pinned Dashboard", "Single Metric"],
            horizontal=True
        )

    with right:

        scale = st.selectbox(
            "Financial Statement Scale",
            ["Raw", "Millions", "Billions"],
            index=2
        )

        chart_type = st.radio(
            "Chart Type",
            ["Line", "Bar"],
            horizontal=True
        )

        labels = st.checkbox(
            "Show Data Labels",
            value=True
        )

        capex_positive = st.checkbox(
            "Show CAPEX as Positive Spending",
            value=True
        )

        show_other = st.checkbox(
            "Show Other Fundamentals",
            value=True
        )

    if not ticker:
        return

    with st.spinner("Loading financial statements..."):

        statements = fetch_statements(
            ticker,
            frequency
        )

        info = fetch_ticker_info(ticker)

        history = get_financial_history(
            statements,
            annual=frequency == "Annual"
        )

    if history.empty:

        st.error("No financial statements are available.")
        return

    st.markdown(
        f"### {info.get('longName', ticker)}"
    )

    st.caption(
        "Statement currency: "
        + str(info.get("financialCurrency", "Unknown"))
    )

    def show_metric(series, name, is_money=False):

        series = series.dropna()

        if series.empty:

            st.warning(f"{name} is unavailable.")
            return

        if is_money:

            series = scale_money(series, scale)

            ylabel = f"{name} ({scale})"

        else:

            ylabel = name

        series = series.rename(name)

        plot_chart(
            series,
            name,
            ylabel=ylabel,
            chart_type=chart_type,
            labels=labels
        )

        table = series.to_frame()

        table.index = table.index.strftime("%Y-%m-%d")

        st.dataframe(
            table.style.format("{:,.2f}"),
            use_container_width=True
        )

        csv_button(
            table,
            f"{ticker}_{name.replace(' ', '_')}.csv",
            f"fund_{name}"
        )

    # --------------------------------------------------------
    # TTM Financials and Valuation
    # --------------------------------------------------------

    st.markdown("### TTM Fundamentals & Valuation")

    if frequency == "Quarterly":

        quarterly = history.copy()

        consecutive = (
            quarterly.index.to_series()
            .diff()
            .dt.days
            .between(60, 120)
        )

        valid_ttm = (
            consecutive.rolling(3).sum() == 3
        )

        ttm = pd.DataFrame(index=quarterly.index)

        for field in [
            "Revenue",
            "Net Income",
            "Diluted EPS",
            "EBITDA"
        ]:

            ttm[field] = (
                quarterly[field]
                .rolling(4, min_periods=4)
                .sum()
                .where(valid_ttm)
            )

        prices = fetch_prices(
            [ticker],
            (
                ttm.index.min()
                - pd.Timedelta(days=30)
            ).strftime("%Y-%m-%d")
        )

        if ticker in prices.columns:

            price = prices[ticker].dropna()

            period_price = price.reindex(
                ttm.index,
                method="ffill"
            )

            ttm["P/E"] = safe_divide(
                period_price,
                ttm["Diluted EPS"]
            )

            shares = info.get(
                "sharesOutstanding",
                np.nan
            )

            try:
                shares = float(shares)

            except (TypeError, ValueError):
                shares = np.nan

            enterprise_value = (
                period_price * shares
                + quarterly["Total Debt"]
                - quarterly["Cash"]
            )

            ttm["EV/EBITDA"] = safe_divide(
                enterprise_value,
                ttm["EBITDA"]
            )

        latest = ttm.iloc[-1]

        metric_grid([
            (
                "Revenue TTM",
                latest.get("Revenue"),
                "money"
            ),
            (
                "EPS TTM",
                latest.get("Diluted EPS"),
                "number"
            ),
            (
                "P/E TTM",
                latest.get("P/E"),
                "multiple"
            ),
            (
                "EV/EBITDA TTM",
                latest.get("EV/EBITDA"),
                "multiple"
            )
        ])

        st.caption(
            "Historical EV is approximate and uses current "
            "shares outstanding with historical debt and cash."
        )

        c1, c2 = st.columns(2)

        with c1:

            show_metric(
                ttm["Revenue"],
                "Revenue TTM",
                True
            )

            if "P/E" in ttm:

                show_metric(
                    ttm["P/E"],
                    "P/E TTM"
                )

        with c2:

            show_metric(
                ttm["Diluted EPS"],
                "EPS TTM"
            )

            if "EV/EBITDA" in ttm:

                show_metric(
                    ttm["EV/EBITDA"],
                    "EV/EBITDA TTM"
                )

    else:

        st.info(
            "Select Quarterly to view trailing "
            "twelve-month financial metrics."
        )

    st.divider()

    # --------------------------------------------------------
    # Other Fundamentals
    # --------------------------------------------------------

    if show_other or mode == "Single Metric":

        st.markdown("### Other Fundamentals")

        statement = st.selectbox(
            "Statement",
            [
                "Income Statement",
                "Balance Sheet",
                "Cash Flow",
                "Computed Ratios"
            ]
        )

        lookup = {
            "Income Statement": "income",
            "Balance Sheet": "balance",
            "Cash Flow": "cash"
        }

        if statement == "Computed Ratios":

            options = [
                c for c in history.columns
                if (
                    "(%)" in c
                    or c in [
                        "Debt / Equity",
                        "Debt / EBITDA",
                        "Current Ratio",
                        "Interest Coverage"
                    ]
                )
            ]

            if options:

                metric = st.selectbox(
                    "Financial Metric",
                    options
                )

                show_metric(
                    history[metric],
                    metric
                )

        else:

            frame = statements[lookup[statement]]

            if not frame.empty:

                metric = st.selectbox(
                    "Financial Metric",
                    list(frame.index)
                )

                series = statement_row(
                    frame,
                    metric
                )

                normalized = normalize_label(metric)

                is_money = not any(
                    item in normalized
                    for item in [
                        "eps",
                        "shares",
                        "ratio",
                        "rate",
                        "pershare"
                    ]
                )

                show_metric(
                    series,
                    metric,
                    is_money
                )

            else:

                st.warning(
                    "Statement data is unavailable."
                )

    # --------------------------------------------------------
    # Pinned Fundamentals Dashboard
    # --------------------------------------------------------

    if mode == "Pinned Dashboard":

        st.divider()

        st.markdown(
            "### Pinned Fundamentals Dashboard"
        )

        c1, c2 = st.columns(2)

        with c1:

            show_metric(
                history["Revenue"],
                "Revenue",
                True
            )

            show_metric(
                history["Revenue Growth (%)"],
                "Revenue Growth (%)"
            )

            capex = history["CAPEX"]

            if not capex_positive:
                capex = -capex

            show_metric(
                capex,
                "CAPEX",
                True
            )

            show_metric(
                growth_rate(history["CAPEX"]),
                "CAPEX Growth (%)"
            )

        with c2:

            show_metric(
                history["Diluted EPS"],
                "Diluted EPS"
            )

            show_metric(
                history["Diluted EPS Growth (%)"],
                "EPS Growth (%)"
            )

            show_metric(
                history["Net Margin (%)"],
                "Net Margin (%)"
            )

        csv_button(
            history,
            f"{ticker}_{frequency}_fundamentals.csv",
            "fundamentals_all_csv"
        )


# ============================================================
# TECHNICAL INDICATOR HELPERS
# ============================================================

def sma(series, window):

    return series.rolling(window).mean()


def rsi(series, length=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = (-delta).clip(lower=0)

    average_gain = gain.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()

    relative_strength = (
        average_gain /
        average_loss.replace(0, np.nan)
    )

    result = 100 - (
        100 / (1 + relative_strength)
    )

    result = result.mask(
        (average_loss == 0)
        & (average_gain > 0),
        100
    )

    result = result.mask(
        (average_loss == 0)
        & (average_gain == 0),
        50
    )

    return result


def realized_volatility(series, window=20):

    returns = np.log(
        series / series.shift(1)
    )

    return (
        returns.rolling(window).std()
        * np.sqrt(252)
        * 100
    )


def percentile_rank(series, window=252):

    series = series.dropna().tail(window)

    if len(series) < 5:
        return np.nan

    return (
        series.rank(pct=True).iloc[-1]
        * 100
    )


def series_slope(series, window=20):

    series = series.dropna().tail(window)

    if len(series) < window:
        return np.nan

    x = np.arange(len(series))

    return float(
        np.polyfit(x, series.values, 1)[0]
    )


# ============================================================
# PAGE 4: TECHNICALS
# ============================================================

def render_technicals():

    st.subheader("🧭 Technicals")

    st.info(
        "Market breadth, sentiment, volatility, momentum "
        "and composite market risk indicators."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Technical Controls")

    lookback = st.sidebar.radio(
        "Lookback Period",
        ["1Y", "2Y", "5Y", "Max"],
        index=1
    )

    slope_window = st.sidebar.slider(
        "Breadth Slope Window",
        10, 60, 20, 5
    )

    rsi_length = st.sidebar.slider(
        "RSI Length",
        7, 28, 14
    )

    volatility_window = st.sidebar.slider(
        "Realized Volatility Window",
        10, 60, 20, 5
    )

    percentile_window = st.sidebar.radio(
        "VIX Percentile Window",
        ["1Y", "3Y", "5Y"]
    )

    percentile_days = {
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260
    }[percentile_window]

    start_date = period_start_date(lookback)

    prices = fetch_prices(
        ["SPY", "RSP", "QQQ", "IWM", "^VIX", "^VVIX"],
        start_date
    )

    if prices.empty:

        st.error("Unable to download market data.")
        return

    def get_series(symbol):

        if symbol in prices.columns:
            return prices[symbol].dropna()

        return pd.Series(dtype=float)

    spy = get_series("SPY")
    rsp = get_series("RSP")
    qqq = get_series("QQQ")
    iwm = get_series("IWM")
    vix = get_series("^VIX")
    vvix = get_series("^VVIX")

    tabs = st.tabs([
        "Breadth Proxies",
        "Sentiment & Volatility",
        "Trend & Momentum",
        "Volatility Regime",
        "Composite Risk Gauge"
    ])

    # --------------------------------------------------------
    # Breadth
    # --------------------------------------------------------

    with tabs[0]:

        st.markdown("### Market Breadth Proxies")

        rsp_spy = (rsp / spy).dropna()

        qqq_iwm = (qqq / iwm).dropna()

        breadth_slope = series_slope(
            rsp_spy,
            slope_window
        )

        growth_slope = series_slope(
            qqq_iwm,
            slope_window
        )

        a, b = st.columns(2)

        a.metric(
            "RSP/SPY Slope",
            (
                f"{breadth_slope:.6f}"
                if pd.notna(breadth_slope)
                else "N/A"
            ),
            (
                "Broadening"
                if breadth_slope > 0
                else "Narrowing"
            ) if pd.notna(breadth_slope) else None
        )

        b.metric(
            "QQQ/IWM Slope",
            (
                f"{growth_slope:.6f}"
                if pd.notna(growth_slope)
                else "N/A"
            ),
            (
                "Large Caps Leading"
                if growth_slope > 0
                else "Small Caps Improving"
            ) if pd.notna(growth_slope) else None
        )

        plot_chart(
            rsp_spy.rename("RSP/SPY"),
            "Equal Weight vs Cap Weighted"
        )

        plot_chart(
            qqq_iwm.rename("QQQ/IWM"),
            "Large Cap Growth vs Small Caps"
        )

    # --------------------------------------------------------
    # Sentiment and Volatility
    # --------------------------------------------------------

    with tabs[1]:

        st.markdown("### Sentiment & Volatility")

        if vix.empty:

            st.warning("VIX data is unavailable.")

        else:

            vix_ma = sma(vix, 50)

            long_vix = fetch_prices(
                ["^VIX"],
                period_start_date(percentile_window)
            )

            percentile_series = (
                long_vix["^VIX"]
                if "^VIX" in long_vix.columns
                else vix
            )

            metric_grid([
                (
                    "VIX",
                    vix.iloc[-1],
                    "number"
                ),
                (
                    "VIX vs 50 DMA",
                    vix.iloc[-1] - vix_ma.iloc[-1],
                    "number"
                ),
                (
                    "VIX Percentile",
                    percentile_rank(
                        percentile_series,
                        percentile_days
                    ),
                    "percent"
                )
            ])

            plot_chart(
                pd.concat([
                    vix.rename("VIX"),
                    vix_ma.rename("50 DMA")
                ], axis=1),
                "VIX Fear Gauge"
            )

        if not vvix.empty:

            plot_chart(
                vvix.rename("VVIX"),
                "VVIX - Volatility of Volatility"
            )

    # --------------------------------------------------------
    # Trend and Momentum
    # --------------------------------------------------------

    with tabs[2]:

        st.markdown("### Trend & Momentum")

        for symbol in ["SPY", "QQQ", "IWM"]:

            close = get_series(symbol)

            if close.empty:
                continue

            st.markdown(f"#### {symbol}")

            ma50 = sma(close, 50)

            ma200 = sma(close, 200)

            strength = rsi(
                close,
                rsi_length
            )

            last_price = close.iloc[-1]

            trend = "N/A"

            if pd.notna(ma200.iloc[-1]):

                trend = (
                    "Bullish"
                    if last_price > ma200.iloc[-1]
                    else "Bearish"
                )

            a, b, c = st.columns(3)

            a.metric(
                "Price",
                fmt(last_price)
            )

            b.metric(
                "200 DMA Trend",
                trend
            )

            last_rsi = strength.iloc[-1]

            zone = None

            if pd.notna(last_rsi):

                if last_rsi >= 70:
                    zone = "Overbought"

                elif last_rsi <= 30:
                    zone = "Oversold"

                else:
                    zone = "Neutral"

            c.metric(
                "RSI",
                fmt(last_rsi),
                zone
            )

            chart_data = pd.concat([
                close.rename(symbol),
                ma50.rename("50 DMA"),
                ma200.rename("200 DMA")
            ], axis=1)

            plot_chart(
                chart_data,
                f"{symbol} Price & Moving Averages"
            )

            fig, ax = plt.subplots(figsize=(12, 3))

            ax.plot(
                strength.index,
                strength,
                label="RSI"
            )

            ax.axhline(
                70,
                linestyle="--",
                color="red"
            )

            ax.axhline(
                30,
                linestyle="--",
                color="green"
            )

            ax.set_ylim(0, 100)

            ax.set_title(f"{symbol} RSI")

            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.divider()

    # --------------------------------------------------------
    # Volatility Regime
    # --------------------------------------------------------

    with tabs[3]:

        st.markdown(
            "### Realized vs Implied Volatility"
        )

        if spy.empty:

            st.warning("SPY prices are unavailable.")

        else:

            realized = realized_volatility(
                spy,
                volatility_window
            )

            latest_vix = (
                vix.iloc[-1]
                if not vix.empty
                else np.nan
            )

            metric_grid([
                (
                    "Realized Volatility",
                    realized.iloc[-1],
                    "percent"
                ),
                (
                    "Implied Volatility (VIX)",
                    latest_vix,
                    "percent"
                )
            ])

            plot_chart(
                pd.concat([
                    realized.rename("Realized Volatility"),
                    vix.rename("VIX")
                ], axis=1),
                "Realized vs Implied Volatility",
                ylabel="Volatility (%)"
            )

    # --------------------------------------------------------
    # Composite Risk Gauge
    # --------------------------------------------------------

    with tabs[4]:

        st.markdown("### Composite Market Risk Gauge")

        market_ma = sma(spy, 200)

        breadth = (rsp / spy).dropna()

        slope = series_slope(
            breadth,
            slope_window
        )

        vix_average = sma(vix, 50)

        trend_ok = None

        breadth_ok = None

        volatility_ok = None

        if not spy.empty and pd.notna(
            market_ma.iloc[-1]
        ):

            trend_ok = bool(
                spy.iloc[-1] > market_ma.iloc[-1]
            )

        if pd.notna(slope):

            breadth_ok = bool(slope > 0)

        if not vix.empty and pd.notna(
            vix_average.iloc[-1]
        ):

            volatility_ok = bool(
                vix.iloc[-1] < vix_average.iloc[-1]
            )

        components = [
            ("Trend", trend_ok, 34),
            ("Breadth", breadth_ok, 33),
            ("Volatility", volatility_ok, 33)
        ]

        score = 0

        available = 0

        rows = []

        for name, status, weight in components:

            if status is None:

                label = "N/A"

                points = np.nan

            else:

                available += weight

                label = (
                    "Positive"
                    if status
                    else "Negative"
                )

                points = weight if status else 0

                score += points

            rows.append({
                "Component": name,
                "Status": label,
                "Points": points
            })

        if available == 100:

            if score >= 67:
                regime = "Risk-On"

            elif score >= 34:
                regime = "Neutral"

            else:
                regime = "Risk-Off"

            st.metric(
                "Market Risk Gauge",
                f"{score}/100",
                regime
            )

            st.progress(score / 100)

        else:

            st.warning(
                "Insufficient data for a complete risk score."
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

        st.caption(
            "Trend: SPY above its 200-day average. "
            "Breadth: RSP/SPY slope is positive. "
            "Volatility: VIX below its 50-day average."
        )

        plot_chart(
            breadth.rename("RSP/SPY"),
            "Market Breadth Proxy"
        )


# ============================================================
# PAGE 5: COMPANY OVERVIEW
# ============================================================

def render_company_overview():

    st.subheader("🏢 Company Overview")

    st.info(
        "Enter a stock ticker to view three years of financial "
        "performance, growth, valuation, profitability, "
        "capital efficiency and financial health."
    )

    # --------------------------------------------------------
    # Ticker Input
    # --------------------------------------------------------

    with st.form("company_overview_form"):

        ticker_input = st.text_input(
            "Enter Stock Symbol",
            value="AAPL"
        ).upper().strip()

        submitted = st.form_submit_button(
            "Analyze Company",
            type="primary"
        )

    if submitted or "overview_ticker" not in st.session_state:

        st.session_state[
            "overview_ticker"
        ] = ticker_input

    ticker = st.session_state["overview_ticker"]

    if not ticker or len(ticker.split()) != 1:

        st.warning("Please enter one valid stock symbol.")
        return

    # --------------------------------------------------------
    # Download Company Data
    # --------------------------------------------------------

    with st.spinner(f"Loading {ticker} financial data..."):

        info = fetch_ticker_info(ticker)

        statements = fetch_statements(
            ticker,
            "Annual"
        )

        history = get_financial_history(
            statements,
            annual=True
        )

    if not info and history.empty:

        st.error(
            "No company data returned. "
            "Check the ticker symbol and try again."
        )

        return

    # --------------------------------------------------------
    # Company Profile
    # --------------------------------------------------------

    company_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker
    )

    st.markdown(
        f"## {company_name} ({ticker})"
    )

    sector = info.get("sector", "N/A")

    industry = info.get("industry", "N/A")

    exchange = info.get("exchange", "N/A")

    currency = info.get("currency", "Unknown")

    financial_currency = info.get(
        "financialCurrency",
        "Unknown"
    )

    st.caption(
        f"Sector: {sector} | "
        f"Industry: {industry} | "
        f"Exchange: {exchange}"
    )

    st.caption(
        f"Quote Currency: {currency} | "
        f"Financial Statement Currency: {financial_currency}"
    )

    def get_value(key, multiplier=1):

        try:

            value = float(info.get(key)) * multiplier

            return (
                value
                if np.isfinite(value)
                else np.nan
            )

        except (TypeError, ValueError):

            return np.nan

    price = get_value("currentPrice")

    if pd.isna(price):

        price = get_value("regularMarketPrice")

    # --------------------------------------------------------
    # Company Market Snapshot
    # --------------------------------------------------------

    st.markdown("### Market Snapshot")

    metric_grid([
        (
            "Current Share Price",
            price,
            "number"
        ),
        (
            "Market Capitalization",
            get_value("marketCap"),
            "money"
        ),
        (
            "52-Week High",
            get_value("fiftyTwoWeekHigh"),
            "number"
        ),
        (
            "52-Week Low",
            get_value("fiftyTwoWeekLow"),
            "number"
        ),
        (
            "Enterprise Value",
            get_value("enterpriseValue"),
            "money"
        ),
        (
            "Beta",
            get_value("beta"),
            "number"
        ),
        (
            "50-Day Moving Average",
            get_value("fiftyDayAverage"),
            "number"
        ),
        (
            "200-Day Moving Average",
            get_value("twoHundredDayAverage"),
            "number"
        )
    ])

    if info.get("longBusinessSummary"):

        with st.expander("About the Company"):

            st.write(
                info["longBusinessSummary"]
            )

    st.divider()

    # --------------------------------------------------------
    # Three-Year Financial Summary
    # --------------------------------------------------------

    st.markdown("### Three-Year Financial Summary")

    if history.empty:

        st.warning(
            "Annual financial statements are unavailable. "
            "Current valuation metrics may still be available."
        )

        recent = pd.DataFrame()

        latest = pd.Series(dtype=float)

    else:

        # Annual income statement dates are used to identify
        # completed reporting periods.

        income_dates = statements["income"].columns

        history = history.loc[
            history.index.isin(income_dates)
        ]

        recent = history.tail(3)

        latest = (
            recent.iloc[-1]
            if not recent.empty
            else pd.Series(dtype=float)
        )

        if len(recent) < 3:

            st.warning(
                "Fewer than three fiscal years are available."
            )

        summary_fields = [
            "Revenue",
            "Revenue Growth (%)",
            "Gross Profit",
            "Operating Income",
            "Operating Income Growth (%)",
            "Net Income",
            "Net Income Growth (%)",
            "Diluted EPS",
            "Diluted EPS Growth (%)",
            "EBITDA",
            "Operating Cash Flow",
            "CAPEX",
            "Free Cash Flow",
            "Free Cash Flow Growth (%)"
        ]

        financial_table(
            recent,
            summary_fields
        )

        st.caption(
            "Financial periods are based on company fiscal "
            "year-end dates. Growth is calculated year over year. "
            "Growth from a nonpositive prior-year value is N/A."
        )

    # --------------------------------------------------------
    # Financial Charts
    # --------------------------------------------------------

    if not recent.empty:

        st.markdown("### Financial Performance Charts")

        c1, c2 = st.columns(2)

        with c1:

            plot_chart(
                recent["Revenue"] / 1e9,
                "Annual Revenue",
                ylabel=f"Billions ({financial_currency})",
                chart_type="Bar",
                labels=True
            )

            plot_chart(
                recent[[
                    "Revenue Growth (%)",
                    "Net Income Growth (%)"
                ]],
                "Revenue & Net Income Growth",
                ylabel="Growth (%)",
                chart_type="Bar",
                labels=True
            )

        with c2:

            plot_chart(
                recent[[
                    "Gross Profit",
                    "Operating Income",
                    "Net Income"
                ]] / 1e9,
                "Annual Profits",
                ylabel=f"Billions ({financial_currency})",
                chart_type="Bar",
                labels=True
            )

            plot_chart(
                recent[[
                    "Operating Cash Flow",
                    "Free Cash Flow"
                ]] / 1e9,
                "Cash Flow",
                ylabel=f"Billions ({financial_currency})",
                chart_type="Bar",
                labels=True
            )

    st.divider()

    # --------------------------------------------------------
    # Valuation Ratios
    # --------------------------------------------------------

    st.markdown("### Valuation Ratios")

    peg = get_value("trailingPegRatio")

    if pd.isna(peg):

        peg = get_value("pegRatio")

    metric_grid([
        (
            "Trailing P/E",
            get_value("trailingPE"),
            "multiple"
        ),
        (
            "Forward P/E",
            get_value("forwardPE"),
            "multiple"
        ),
        (
            "Price / Sales",
            get_value("priceToSalesTrailing12Months"),
            "multiple"
        ),
        (
            "PEG Ratio",
            peg,
            "multiple"
        ),
        (
            "Price / Book",
            get_value("priceToBook"),
            "multiple"
        ),
        (
            "EV / EBITDA",
            get_value("enterpriseToEbitda"),
            "multiple"
        ),
        (
            "EV / Revenue",
            get_value("enterpriseToRevenue"),
            "multiple"
        ),
        (
            "Earnings Yield",
            (
                100 / get_value("trailingPE")
                if get_value("trailingPE") > 0
                else np.nan
            ),
            "percent"
        )
    ])

    market_cap = get_value("marketCap")

    free_cash_flow = get_value("freeCashflow")

    same_currency = (
        currency == financial_currency
        and currency != "Unknown"
    )

    fcf_yield = np.nan

    price_fcf = np.nan

    if same_currency and market_cap > 0:

        fcf_yield = (
            free_cash_flow / market_cap
        ) * 100

        if free_cash_flow > 0:

            price_fcf = (
                market_cap / free_cash_flow
            )

    metric_grid([
        (
            "Free Cash Flow Yield",
            fcf_yield,
            "percent"
        ),
        (
            "Price / Free Cash Flow",
            price_fcf,
            "multiple"
        )
    ])

    st.caption(
        "Valuation ratios use the latest available Yahoo "
        "Finance quote metrics. Forward P/E and PEG depend "
        "on analyst estimates."
    )

    st.divider()

    # --------------------------------------------------------
    # Profitability and Capital Efficiency
    # --------------------------------------------------------

    st.markdown("### Profitability & Capital Efficiency")

    metric_grid([
        (
            "Return on Invested Capital (ROIC)",
            latest.get("ROIC (%)"),
            "percent"
        ),
        (
            "Return on Equity (ROE)",
            latest.get("ROE (%)"),
            "percent"
        ),
        (
            "Return on Assets (ROA)",
            latest.get("ROA (%)"),
            "percent"
        ),
        (
            "Gross Margin",
            latest.get("Gross Margin (%)"),
            "percent"
        ),
        (
            "Operating Margin",
            latest.get("Operating Margin (%)"),
            "percent"
        ),
        (
            "Net Profit Margin",
            latest.get("Net Margin (%)"),
            "percent"
        ),
        (
            "EBITDA Margin",
            latest.get("EBITDA Margin (%)"),
            "percent"
        ),
        (
            "Free Cash Flow Margin",
            latest.get("FCF Margin (%)"),
            "percent"
        )
    ])

    st.caption(
        "Profitability ratios are calculated from the "
        "latest completed fiscal year."
    )

    st.divider()

    # --------------------------------------------------------
    # Financial Health and Liquidity
    # --------------------------------------------------------

    st.markdown("### Financial Health & Liquidity")

    # Yahoo expresses debt-to-equity in percentage points.
    # 100 is equivalent to a debt-to-equity ratio of 1.0.

    debt_equity = get_value(
        "debtToEquity",
        0.01
    )

    metric_grid([
        (
            "Debt / Equity",
            debt_equity,
            "multiple"
        ),
        (
            "Current Ratio",
            get_value("currentRatio"),
            "multiple"
        ),
        (
            "Quick Ratio",
            get_value("quickRatio"),
            "multiple"
        ),
        (
            "Total Debt",
            get_value("totalDebt"),
            "money"
        ),
        (
            "Total Cash",
            get_value("totalCash"),
            "money"
        ),
        (
            "Net Debt",
            get_value("totalDebt") - get_value("totalCash"),
            "money"
        ),
        (
            "Debt / EBITDA (Annual)",
            latest.get("Debt / EBITDA"),
            "multiple"
        ),
        (
            "Interest Coverage (Annual)",
            latest.get("Interest Coverage"),
            "multiple"
        )
    ])

    st.divider()

    # --------------------------------------------------------
    # Growth Metrics
    # --------------------------------------------------------

    st.markdown("### Growth Metrics")

    metric_grid([
        (
            "Annual Revenue Growth",
            latest.get("Revenue Growth (%)"),
            "percent"
        ),
        (
            "Annual Net Income Growth",
            latest.get("Net Income Growth (%)"),
            "percent"
        ),
        (
            "Annual Operating Income Growth",
            latest.get("Operating Income Growth (%)"),
            "percent"
        ),
        (
            "Annual EPS Growth",
            latest.get("Diluted EPS Growth (%)"),
            "percent"
        ),
        (
            "Annual Free Cash Flow Growth",
            latest.get("Free Cash Flow Growth (%)"),
            "percent"
        ),
        (
            "Latest Reported Revenue Growth",
            get_value("revenueGrowth", 100),
            "percent"
        ),
        (
            "Latest Quarterly Earnings Growth",
            get_value("earningsQuarterlyGrowth", 100),
            "percent"
        )
    ])

    st.divider()

    # --------------------------------------------------------
    # Historical Profitability and Financial Ratios
    # --------------------------------------------------------

    st.markdown(
        "### Three-Year Historical Financial Ratios"
    )

    if not recent.empty:

        historical_fields = [
            "ROIC (%)",
            "ROE (%)",
            "ROA (%)",
            "Gross Margin (%)",
            "Operating Margin (%)",
            "Net Margin (%)",
            "EBITDA Margin (%)",
            "FCF Margin (%)",
            "Debt / Equity",
            "Debt / EBITDA",
            "Current Ratio",
            "Interest Coverage"
        ]

        financial_table(
            recent,
            historical_fields
        )

        c1, c2 = st.columns(2)

        with c1:

            plot_chart(
                recent[[
                    "ROIC (%)",
                    "ROE (%)",
                    "ROA (%)"
                ]],
                "Returns on Capital",
                ylabel="Percent",
                chart_type="Bar",
                labels=True
            )

        with c2:

            plot_chart(
                recent[[
                    "Gross Margin (%)",
                    "Operating Margin (%)",
                    "Net Margin (%)"
                ]],
                "Profit Margin Trends",
                ylabel="Percent",
                chart_type="Bar",
                labels=True
            )

    st.divider()

    # --------------------------------------------------------
    # Cash Flow Summary
    # --------------------------------------------------------

    st.markdown("### Cash Flow Summary")

    metric_grid([
        (
            "Operating Cash Flow",
            latest.get("Operating Cash Flow"),
            "money"
        ),
        (
            "Capital Expenditures",
            latest.get("CAPEX"),
            "money"
        ),
        (
            "Free Cash Flow",
            latest.get("Free Cash Flow"),
            "money"
        ),
        (
            "Free Cash Flow Margin",
            latest.get("FCF Margin (%)"),
            "percent"
        )
    ])

    st.divider()

    # --------------------------------------------------------
    # Balance Sheet Summary
    # --------------------------------------------------------

    st.markdown("### Balance Sheet Summary")

    if not recent.empty:

        financial_table(
            recent,
            [
                "Total Assets",
                "Equity",
                "Total Debt",
                "Cash",
                "Current Assets",
                "Current Liabilities"
            ]
        )

    # --------------------------------------------------------
    # Ratio Definitions
    # --------------------------------------------------------

    with st.expander(
        "Financial Ratio Definitions & Data Limitations"
    ):

        st.markdown("""
**Valuation**

- **Trailing P/E:** Share price divided by trailing earnings per share.
- **Forward P/E:** Share price divided by estimated future earnings per share.
- **P/S:** Market capitalization divided by revenue.
- **PEG:** P/E adjusted for expected earnings growth.
- **P/B:** Market capitalization divided by book equity.
- **EV/EBITDA:** Enterprise value divided by EBITDA.
- **EV/Revenue:** Enterprise value divided by revenue.
- **Earnings Yield:** Earnings divided by market price.
- **Free Cash Flow Yield:** Free cash flow divided by market capitalization.

**Profitability**

- **ROIC:** After-tax operating income divided by average invested capital.
- **ROE:** Net income divided by average shareholder equity.
- **ROA:** Net income divided by average total assets.
- **Gross Margin:** Gross profit divided by revenue.
- **Operating Margin:** Operating income divided by revenue.
- **Net Margin:** Net income divided by revenue.
- **FCF Margin:** Free cash flow divided by revenue.

**Financial Health**

- **Debt/Equity:** Total debt divided by shareholder equity.
- **Current Ratio:** Current assets divided by current liabilities.
- **Quick Ratio:** Liquid current assets divided by current liabilities.
- **Debt/EBITDA:** Total debt divided by annual EBITDA.
- **Interest Coverage:** Operating income divided by interest expense.
- **Net Debt:** Total debt minus cash.

**ROIC Calculation**

ROIC = NOPAT / Average Invested Capital

NOPAT = Operating Income × (1 − Effective Tax Rate)

Reported invested capital is preferred when available.

Otherwise, invested capital is estimated as debt plus
equity minus cash.

**Data Limitations**

- All financial data comes from Yahoo Finance through yfinance.
- Missing financial data is displayed as N/A.
- Annual and trailing financial metrics may have different
  reporting dates.
- Growth rates are not calculated when the prior-period value
  is zero or negative.
- Some ratios are unsuitable for banks, insurance companies,
  ETFs and companies with negative earnings or negative equity.
- ROIC requires adequate tax and invested capital information.
- Quotes may be delayed.
        """)

    # --------------------------------------------------------
    # Download Company Financial Data
    # --------------------------------------------------------

    if not recent.empty:

        csv_button(
            recent,
            f"{ticker}_company_financial_summary.csv",
            "company_summary_csv"
        )


# ============================================================
# PAGE 6: CHEAT SHEET
# ============================================================

def render_cheat_sheet():

    st.subheader("📋 Cheat Sheet")

    st.info(
        "Reference guide for all preset market ratios."
    )

    rows = []

    for category, ratios in RATIO_GROUPS.items():

        for label, symbols in ratios.items():

            symbol_a, symbol_b = symbols

            ratio = f"{symbol_a}/{symbol_b}"

            description, commentary = RATIO_INFO.get(
                ratio,
                ("", "")
            )

            rows.append({
                "Category": category,
                "Ratio": ratio,
                "Symbol 1": symbol_a,
                "Symbol 2": symbol_b,
                "Description": description,
                "Interpretation": commentary
            })

    cheat_sheet = pd.DataFrame(rows)

    st.dataframe(
        cheat_sheet,
        use_container_width=True,
        hide_index=True
    )

    csv_button(
        cheat_sheet,
        "ratio_cheat_sheet.csv",
        "cheat_sheet_csv"
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📊 Charts to Watch")

st.caption(
    "Market Ratios | Performance Comparisons | "
    "Financial Statements | Technical Indicators | "
    "Company Financial Analysis"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📊 Ratio Dashboard",
        "📈 Performance",
        "📑 Fundamentals",
        "🧭 Technicals",
        "🏢 Company Overview",
        "📋 Cheat Sheet"
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Market and financial data provided by Yahoo Finance."
)


# ============================================================
# APPLICATION PAGE ROUTING
# ============================================================

if page == "📊 Ratio Dashboard":

    render_ratio_dashboard()

elif page == "📈 Performance":

    render_performance()

elif page == "📑 Fundamentals":

    render_fundamentals()

elif page == "🧭 Technicals":

    render_technicals()

elif page == "🏢 Company Overview":

    render_company_overview()

elif page == "📋 Cheat Sheet":

    render_cheat_sheet()
