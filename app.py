import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from datetime import datetime, timedelta

plt.switch_backend("Agg")

st.set_page_config(
    page_title="Charts to Watch",
    layout="wide"
)

# ============================================================
# MARKET RATIO PRESETS
# ============================================================

RATIO_GROUPS = {
    "Market Breadth & Strength": {
        "SPY / RSP - Cap vs Equal Weight": ("SPY", "RSP"),
        "QQQ / IWM - Nasdaq vs Small Caps": ("QQQ", "IWM"),
        "DIA / IWM - Dow vs Small Caps": ("DIA", "IWM"),
        "MGK / SPY - Mega Cap Growth": ("MGK", "SPY"),
    },
    "Risk-On vs Risk-Off": {
        "SPY / TLT - Stocks vs Bonds": ("SPY", "TLT"),
        "HYG / IEF - Credit Risk": ("HYG", "IEF"),
        "XLY / XLP - Consumer Sentiment": ("XLY", "XLP"),
        "IWM / SHY - Small Cap Risk": ("IWM", "SHY"),
        "SPHB / SPLV - High Beta vs Low Vol": ("SPHB", "SPLV"),
    },
    "Sector Relative Strength": {
        "XLF / SPY - Financials": ("XLF", "SPY"),
        "XLV / SPY - Healthcare": ("XLV", "SPY"),
        "XLE / SPY - Energy": ("XLE", "SPY"),
        "XLK / SPY - Technology": ("XLK", "SPY"),
        "XLI / SPY - Industrials": ("XLI", "SPY"),
        "XLB / SPY - Materials": ("XLB", "SPY"),
        "XLU / SPY - Utilities": ("XLU", "SPY"),
        "XLRE / SPY - Real Estate": ("XLRE", "SPY"),
        "XLC / SPY - Communication": ("XLC", "SPY"),
        "RSPD / RSPS - Equal Weight Consumer": ("RSPD", "RSPS"),
    },
    "Commodities & Inflation": {
        "DBC / SPY - Commodities": ("DBC", "SPY"),
        "GDX / SPY - Gold Miners": ("GDX", "SPY"),
        "GLD / TLT - Gold vs Bonds": ("GLD", "TLT"),
        "USO / SPY - Oil vs Stocks": ("USO", "SPY"),
        "TIP / TLT - Inflation": ("TIP", "TLT"),
        "CPER / GLD - Copper vs Gold": ("CPER", "GLD"),
        "GLD / USO - Gold vs Oil": ("GLD", "USO"),
        "GLD / XME - Gold vs Metals": ("GLD", "XME"),
    },
    "International Markets": {
        "EEM / SPY - Emerging Markets": ("EEM", "SPY"),
        "VEA / SPY - Developed Markets": ("VEA", "SPY"),
        "FXI / SPY - China vs US": ("FXI", "SPY"),
    },
    "Cryptocurrency": {
        "ETHA / IBIT - Ethereum vs Bitcoin": ("ETHA", "IBIT"),
        "ETHA / GSOL - Ethereum vs Solana": ("ETHA", "GSOL"),
        "BMNR / ETHA - BMNR vs Ethereum": ("BMNR", "ETHA"),
        "MSTR / IBIT - MSTR vs Bitcoin": ("MSTR", "IBIT"),
    }
}

RATIO_INFO = {
    "SPY/RSP": "Rising = narrow mega-cap leadership. Falling = broader participation.",
    "QQQ/IWM": "Rising = large-cap growth leadership. Falling = small-cap relative strength.",
    "DIA/IWM": "Rising = established blue-chip leadership.",
    "MGK/SPY": "Rising = mega-cap growth outperforming the market.",
    "SPY/TLT": "Rising may indicate stronger equity risk appetite.",
    "HYG/IEF": "Falling may indicate increasing credit stress.",
    "XLY/XLP": "Rising may indicate stronger consumer confidence.",
    "IWM/SHY": "Rising may indicate increasing risk appetite.",
    "SPHB/SPLV": "Rising = aggressive stocks outperforming defensive stocks.",
    "XLF/SPY": "Rising = financial sector leadership.",
    "XLV/SPY": "Rising may indicate defensive rotation.",
    "XLE/SPY": "Rising = energy sector leadership.",
    "XLK/SPY": "Rising = technology sector leadership.",
    "XLI/SPY": "Rising = industrial sector leadership.",
    "RSPD/RSPS": "Rising = broad consumer discretionary leadership.",
    "DBC/SPY": "Rising = commodities outperforming equities.",
    "GDX/SPY": "Rising = gold miner relative strength.",
    "GLD/TLT": "Rising = gold outperforming long-term bonds.",
    "USO/SPY": "Rising may indicate stronger oil prices.",
    "TIP/TLT": "Sensitive to inflation expectations and duration differences.",
    "CPER/GLD": "Rising may indicate stronger economic growth expectations.",
    "GLD/USO": "Rising may indicate defensive commodity positioning.",
    "GLD/XME": "Rising may indicate defensive positioning.",
    "EEM/SPY": "Rising = emerging market leadership.",
    "VEA/SPY": "Rising = international market leadership.",
    "FXI/SPY": "Rising = Chinese equity leadership.",
    "ETHA/IBIT": "Rising = Ethereum outperforming Bitcoin.",
    "ETHA/GSOL": "Rising = Ethereum outperforming Solana.",
    "BMNR/ETHA": "Rising = BMNR outperforming Ethereum exposure.",
    "MSTR/IBIT": "Rising = MSTR outperforming Bitcoin exposure.",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def start_date(period):

    today = datetime.today()

    if period == "YTD":
        return f"{today.year}-01-01"

    if period == "QTD":
        month = ((today.month - 1) // 3) * 3 + 1
        return datetime(today.year, month, 1).strftime("%Y-%m-%d")

    days = {
        "3M": 92,
        "6M": 183,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
        "5Y": 1825,
        "10Y": 3650
    }

    if period in days:
        return (
            today - timedelta(days=days[period])
        ).strftime("%Y-%m-%d")

    return "1900-01-01"


def number(value, default=np.nan):

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def fmt(value, style="number"):

    value = number(value)

    if pd.isna(value):
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


def divide(a, b):

    a = number(a)
    b = number(b)

    if pd.isna(a) or pd.isna(b) or b <= 0:
        return np.nan

    return a / b


def safe_divide(a, b):

    return (
        a / b.where(b > 0)
    ).replace([np.inf, -np.inf], np.nan)


def growth_rate(series):

    previous = series.shift(1)

    return (
        (series - previous) /
        previous.where(previous > 0)
    ) * 100


def metric_grid(metrics, columns=4):

    for i in range(0, len(metrics), columns):

        cols = st.columns(columns)

        for col, item in zip(cols, metrics[i:i + columns]):

            label, value, style = item

            col.metric(label, fmt(value, style))


def download_csv(df, filename, key):

    st.download_button(
        "Download CSV",
        data=df.to_csv().encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


# ============================================================
# YAHOO FINANCE DATA
# ============================================================

@st.cache_data(ttl=1800)
def fetch_prices(tickers, start, adjusted=False):

    try:

        raw = yf.download(
            list(tickers),
            start=start,
            auto_adjust=adjusted,
            progress=False
        )

        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):

            close = raw["Close"].copy()

        else:

            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]

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
def fetch_info(ticker):

    try:
        return yf.Ticker(ticker).info or {}

    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_statements(ticker, frequency="Annual"):

    company = yf.Ticker(ticker)

    quarterly = frequency == "Quarterly"

    attributes = {
        "Income Statement": (
            "quarterly_income_stmt"
            if quarterly else "income_stmt"
        ),
        "Balance Sheet": (
            "quarterly_balance_sheet"
            if quarterly else "balance_sheet"
        ),
        "Cash Flow": (
            "quarterly_cashflow"
            if quarterly else "cashflow"
        )
    }

    result = {}

    for name, attribute in attributes.items():

        try:

            df = getattr(company, attribute).copy()

            df.columns = pd.to_datetime(
                df.columns,
                errors="coerce"
            )

            df = df.loc[:, df.columns.notna()]

            df = df.loc[:, ~df.columns.duplicated()]

            df = df.sort_index(axis=1)

            result[name] = df.apply(
                pd.to_numeric,
                errors="coerce"
            )

        except Exception:

            result[name] = pd.DataFrame()

    return result


def normalize_label(label):

    return "".join(
        c for c in str(label).lower()
        if c.isalnum()
    )


def statement_row(df, *names):

    if df is None or df.empty:
        return pd.Series(dtype=float)

    lookup = {
        normalize_label(label): label
        for label in df.index
    }

    for name in names:

        key = normalize_label(name)

        if key in lookup:

            series = df.loc[lookup[key]]

            if isinstance(series, pd.DataFrame):
                series = series.iloc[0]

            return pd.to_numeric(
                series,
                errors="coerce"
            ).sort_index()

    return pd.Series(dtype=float)


# ============================================================
# FINANCIAL HISTORY
# ============================================================

def financial_history(statements, annual=True):

    income = statements["Income Statement"]
    balance = statements["Balance Sheet"]
    cash = statements["Cash Flow"]

    dates = income.columns

    if len(dates) == 0:
        dates = balance.columns.union(cash.columns)

    dates = dates.sort_values()

    history = pd.DataFrame(index=dates)

    income_fields = {
        "Revenue": ("Total Revenue", "Revenue"),
        "Gross Profit": ("Gross Profit",),
        "Operating Income": ("Operating Income",),
        "Net Income": ("Net Income",),
        "Diluted EPS": ("Diluted EPS",),
        "EBITDA": ("EBITDA",),
        "Pretax Income": ("Pretax Income",),
        "Tax Provision": ("Tax Provision",),
        "Interest Expense": ("Interest Expense",),
    }

    balance_fields = {
        "Total Assets": ("Total Assets",),
        "Equity": ("Stockholders Equity", "Total Stockholder Equity"),
        "Total Debt": ("Total Debt",),
        "Cash": ("Cash And Cash Equivalents",),
        "Invested Capital": ("Invested Capital",),
        "Current Assets": ("Current Assets", "Total Current Assets"),
        "Current Liabilities": (
            "Current Liabilities",
            "Total Current Liabilities"
        ),
        "Inventory": ("Inventory",),
        "Minority Interest": ("Minority Interest",),
        "Preferred Stock": ("Preferred Stock",),
    }

    cash_fields = {
        "Operating Cash Flow": (
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ),
        "Free Cash Flow": ("Free Cash Flow",),
        "CAPEX": ("Capital Expenditure", "Capital Expenditures"),
        "Depreciation & Amortization": (
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation"
        ),
        "Change in Working Capital": (
            "Change In Working Capital",
        ),
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
            history["Operating Cash Flow"] -
            history["CAPEX"]
        )
    )

    # Yahoo cash flow working capital changes use
    # cash flow signs. Convert to investment signs.

    history["Working Capital Investment"] = (
        -history["Change in Working Capital"]
    )

    growth_period = 1 if annual else 4

    for field in [
        "Revenue",
        "Operating Income",
        "Net Income",
        "Diluted EPS",
        "Free Cash Flow",
        "CAPEX"
    ]:

        prior = history[field].shift(growth_period)

        history[field + " Growth (%)"] = (
            (history[field] - prior) /
            prior.where(prior > 0)
        ) * 100

        date_difference = (
            history.index.to_series()
            .diff(growth_period)
            .dt.days
        )

        history[field + " Growth (%)"] = (
            history[field + " Growth (%)"]
            .where(date_difference.between(300, 430))
        )

    for field, label in [
        ("Gross Profit", "Gross Margin (%)"),
        ("Operating Income", "Operating Margin (%)"),
        ("Net Income", "Net Margin (%)"),
        ("Free Cash Flow", "FCF Margin (%)"),
        ("EBITDA", "EBITDA Margin (%)"),
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

    history["Quick Ratio"] = safe_divide(
        history["Current Assets"] -
        history["Inventory"],
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

    history["Effective Tax Rate (%)"] = (
        safe_divide(
            history["Tax Provision"],
            history["Pretax Income"]
        ) * 100
    )

    if annual:

        consecutive = (
            history.index.to_series()
            .diff()
            .dt.days
            .between(300, 430)
        )

        average_equity = (
            history["Equity"] +
            history["Equity"].shift(1)
        ) / 2

        average_equity = average_equity.where(
            consecutive &
            (history["Equity"] > 0) &
            (history["Equity"].shift(1) > 0)
        )

        average_assets = (
            history["Total Assets"] +
            history["Total Assets"].shift(1)
        ) / 2

        average_assets = average_assets.where(consecutive)

        history["ROE (%)"] = safe_divide(
            history["Net Income"],
            average_equity
        ) * 100

        history["ROA (%)"] = safe_divide(
            history["Net Income"],
            average_assets
        ) * 100

        capital = history["Invested Capital"].combine_first(
            history["Total Debt"] +
            history["Equity"] -
            history["Cash"]
        )

        average_capital = (
            capital + capital.shift(1)
        ) / 2

        average_capital = average_capital.where(
            consecutive &
            (capital > 0) &
            (capital.shift(1) > 0)
        )

        tax = (
            history["Effective Tax Rate (%)"] / 100
        )

        tax = tax.where(tax.between(0, 1))

        nopat = (
            history["Operating Income"] * (1 - tax)
        )

        history["ROIC (%)"] = safe_divide(
            nopat,
            average_capital
        ) * 100

    return history.replace(
        [np.inf, -np.inf],
        np.nan
    )


def trailing_financials(history):

    if len(history) < 4:
        return pd.Series(dtype=float)

    dates = history.index[-4:]

    intervals = dates.to_series().diff().dropna().dt.days

    if not intervals.between(60, 120).all():
        return pd.Series(dtype=float)

    recent = history.tail(4)

    result = recent.iloc[-1].copy()

    flow_fields = [
        "Revenue",
        "Gross Profit",
        "Operating Income",
        "Net Income",
        "Diluted EPS",
        "EBITDA",
        "Pretax Income",
        "Tax Provision",
        "Interest Expense",
        "Operating Cash Flow",
        "Free Cash Flow",
        "CAPEX",
        "Depreciation & Amortization",
        "Working Capital Investment"
    ]

    for field in flow_fields:

        result[field] = recent[field].sum(min_count=4)

    return result


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

        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(12, 4.5))

    if chart_type == "Bar":

        display = data.copy()

        if isinstance(display.index, pd.DatetimeIndex):

            display.index = display.index.strftime("%Y-%m-%d")

        display.plot.bar(ax=ax, rot=25)

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
                        (x, y),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8
                    )

    ax.set_title(title, fontweight="bold")

    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.3)

    if len(data.columns) > 1:
        ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


def financial_table(history, fields):

    fields = [f for f in fields if f in history.columns]

    if history.empty or not fields:

        st.info("Financial data unavailable.")
        return

    table = history[fields].T.astype(object)

    table.columns = [
        d.strftime("%Y-%m-%d")
        for d in table.columns
    ]

    multiples = [
        "Debt / Equity",
        "Debt / EBITDA",
        "Current Ratio",
        "Quick Ratio",
        "Interest Coverage"
    ]

    for field in table.index:

        if "(%)" in field:
            style = "percent"

        elif field in multiples:
            style = "multiple"

        elif "EPS" in field:
            style = "number"

        else:
            style = "money"

        table.loc[field] = table.loc[field].map(
            lambda x: fmt(x, style)
        )

    st.dataframe(
        table,
        use_container_width=True
    )


# ============================================================
# PAGE 1: RATIO DASHBOARD
# ============================================================

def render_ratios():

    st.subheader("📊 Ratio Dashboard")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Ratio Controls")

    group = st.sidebar.selectbox(
        "Category",
        list(RATIO_GROUPS.keys())
    )

    preset = st.sidebar.radio(
        "Preset Ratio",
        list(RATIO_GROUPS[group].keys())
    )

    a, b = RATIO_GROUPS[group][preset]

    if st.sidebar.checkbox("Use Custom Tickers"):

        a = st.sidebar.text_input(
            "Ticker A", a
        ).upper().strip()

        b = st.sidebar.text_input(
            "Ticker B", b
        ).upper().strip()

    period = st.sidebar.radio(
        "Date Range",
        ["YTD", "1Y", "3Y", "5Y", "Max"],
        index=1
    )

    with st.sidebar.expander("Chart Options"):

        moving_averages = st.checkbox(
            "Show Moving Averages",
            value=True
        )

        log_scale = st.checkbox("Log Scale")

        extremes = st.checkbox("Show High / Low / Latest")

    st.markdown(f"### {a} / {b}")

    st.write(
        RATIO_INFO.get(
            f"{a}/{b}",
            "Rising means the numerator is outperforming the denominator."
        )
    )

    if not a or not b or a == b:

        st.warning("Enter two different ticker symbols.")
        return

    selection = (a, b, period)

    if st.button("Plot Ratio", type="primary"):

        requested_start = pd.Timestamp(start_date(period))

        download_start = max(
            requested_start - pd.Timedelta(days=400),
            pd.Timestamp("1900-01-01")
        )

        with st.spinner("Downloading data..."):

            prices = fetch_prices(
                (a, b),
                download_start.strftime("%Y-%m-%d")
            )

        if a not in prices or b not in prices:

            st.error("Unable to download both securities.")
            return

        data = prices[[a, b]].dropna()

        data["Ratio"] = (
            data[a] / data[b].replace(0, np.nan)
        )

        data["50 DMA"] = data["Ratio"].rolling(50).mean()

        data["200 DMA"] = data["Ratio"].rolling(200).mean()

        data = data.loc[requested_start:]

        st.session_state["ratio_data"] = data

        st.session_state["ratio_selection"] = selection

    if st.session_state.get("ratio_selection") != selection:

        st.info("Click Plot Ratio to display the chart.")
        return

    data = st.session_state["ratio_data"]

    if data.empty:

        st.warning("No data available.")
        return

    ratio = data["Ratio"].dropna()

    if ratio.empty:
        return

    def change(series):

        series = series.dropna()

        if len(series) < 2 or series.iloc[-2] == 0:
            return None

        return f"{(series.iloc[-1] / series.iloc[-2] - 1) * 100:+.2f}%"

    cols = st.columns(6)

    cols[0].metric(
        f"{a} Price",
        fmt(data[a].iloc[-1]),
        change(data[a])
    )

    cols[1].metric(
        f"{b} Price",
        fmt(data[b].iloc[-1]),
        change(data[b])
    )

    cols[2].metric(
        "Current Ratio",
        f"{ratio.iloc[-1]:.3f}",
        change(ratio)
    )

    cols[3].metric(
        "High",
        f"{ratio.max():.3f}",
        ratio.idxmax().strftime("%Y-%m-%d")
    )

    cols[4].metric(
        "Low",
        f"{ratio.min():.3f}",
        ratio.idxmin().strftime("%Y-%m-%d")
    )

    cols[5].metric(
        "Drawdown",
        fmt(
            (ratio.iloc[-1] / ratio.max() - 1) * 100,
            "percent"
        )
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        ratio.index,
        ratio,
        label=f"{a}/{b}",
        linewidth=2
    )

    if moving_averages:

        for column in ["50 DMA", "200 DMA"]:

            ax.plot(
                data.index,
                data[column],
                linestyle="--",
                label=column
            )

    if extremes:

        points = [
            (ratio.idxmax(), ratio.max(), "High"),
            (ratio.idxmin(), ratio.min(), "Low"),
            (ratio.index[-1], ratio.iloc[-1], "Latest")
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

    ax.set_title(f"{a}/{b} Relative Strength")

    ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    download_csv(
        data,
        f"{a}_{b}_ratio.csv",
        "ratio_csv"
    )


# ============================================================
# PAGE 2: PERFORMANCE
# ============================================================

def render_performance():

    st.subheader("📈 Performance")

    st.info(
        "Compare up to 20 securities using normalized performance."
    )

    text = st.text_input(
        "Enter Tickers Separated by Commas",
        "SPY, QQQ, IWM, TLT"
    )

    tickers = tuple(dict.fromkeys(
        t.strip().upper()
        for t in text.split(",")
        if t.strip()
    ))[:20]

    period = st.radio(
        "Performance Period",
        ["QTD", "YTD", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
        horizontal=True
    )

    adjusted = st.checkbox(
        "Use Dividend and Split Adjusted Prices",
        value=True
    )

    names = st.checkbox("Include Company Names")

    selection = (tickers, period, adjusted)

    if st.button("Plot Performance", type="primary"):

        if not tickers:

            st.warning("Enter at least one ticker.")
            return

        with st.spinner("Downloading data..."):

            data = fetch_prices(
                tickers,
                start_date(period),
                adjusted
            )

        missing = [t for t in tickers if t not in data]

        if missing:

            st.warning(
                "Unavailable tickers: " + ", ".join(missing)
            )

        data = data.dropna(how="any")

        if data.empty:

            st.error("No common price history available.")
            return

        data = data.loc[:, data.iloc[0] != 0]

        indexed = data / data.iloc[0] * 100

        st.session_state["performance"] = indexed

        st.session_state["performance_selection"] = selection

    if st.session_state.get("performance_selection") != selection:

        st.info("Click Plot Performance.")
        return

    indexed = st.session_state["performance"]

    returns = (
        indexed.iloc[-1] - 100
    ).sort_values(ascending=False)

    summary = pd.DataFrame({
        "Ticker": returns.index,
        "Return (%)": returns.values
    })

    if names:

        summary["Company"] = [
            fetch_info(t).get("shortName", t)
            for t in summary["Ticker"]
        ]

    st.caption(
        f"Actual comparison period: "
        f"{indexed.index[0].date()} through "
        f"{indexed.index[-1].date()}"
    )

    st.dataframe(
        summary.style.format({
            "Return (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

    plot_chart(
        indexed,
        "Performance Comparison",
        "Indexed to 100"
    )

    plot_chart(
        returns.rename("Total Return (%)"),
        "Period Returns",
        "Percent",
        "Bar",
        True
    )

    download_csv(
        indexed,
        "performance.csv",
        "performance_csv"
    )


# ============================================================
# PAGE 3: FUNDAMENTALS
# ============================================================

def render_fundamentals():

    st.subheader("📑 Fundamentals")

    ticker = st.text_input(
        "Stock Symbol",
        "AAPL",
        key="fund_ticker"
    ).upper().strip()

    c1, c2, c3 = st.columns(3)

    frequency = c1.radio(
        "Frequency",
        ["Annual", "Quarterly"]
    )

    chart_type = c2.radio(
        "Chart Type",
        ["Line", "Bar"]
    )

    scale = c3.selectbox(
        "Financial Scale",
        ["Raw", "Millions", "Billions"],
        index=2
    )

    labels = st.checkbox(
        "Show Data Labels",
        value=True,
        key="fund_labels"
    )

    if not ticker:
        return

    statements = fetch_statements(ticker, frequency)

    history = financial_history(
        statements,
        annual=frequency == "Annual"
    )

    info = fetch_info(ticker)

    if history.empty:

        st.warning("No financial statements available.")
        return

    st.markdown(
        f"### {info.get('longName', ticker)}"
    )

    st.caption(
        "Financial currency: " +
        str(info.get("financialCurrency", "Unknown"))
    )

    divisor = {
        "Raw": 1,
        "Millions": 1e6,
        "Billions": 1e9
    }[scale]

    tabs = st.tabs([
        "Pinned Dashboard",
        "Financial Statements",
        "Financial Ratios",
        "TTM & Valuation"
    ])

    with tabs[0]:

        c1, c2 = st.columns(2)

        with c1:

            plot_chart(
                history["Revenue"] / divisor,
                "Revenue",
                scale,
                chart_type,
                labels
            )

            plot_chart(
                history["Revenue Growth (%)"],
                "Revenue Growth",
                "Percent",
                chart_type,
                labels
            )

            plot_chart(
                history["CAPEX"] / divisor,
                "Capital Expenditures",
                scale,
                chart_type,
                labels
            )

        with c2:

            plot_chart(
                history["Net Income"] / divisor,
                "Net Income",
                scale,
                chart_type,
                labels
            )

            plot_chart(
                history["Diluted EPS"],
                "Diluted EPS",
                "Per Share",
                chart_type,
                labels
            )

            plot_chart(
                history["Free Cash Flow"] / divisor,
                "Free Cash Flow",
                scale,
                chart_type,
                labels
            )

        financial_table(
            history,
            [
                "Revenue",
                "Revenue Growth (%)",
                "Net Income",
                "Net Income Growth (%)",
                "Diluted EPS",
                "Diluted EPS Growth (%)",
                "CAPEX",
                "Free Cash Flow"
            ]
        )

    with tabs[1]:

        statement = st.selectbox(
            "Select Financial Statement",
            list(statements.keys())
        )

        frame = statements[statement]

        if not frame.empty:

            selected = st.selectbox(
                "Select Financial Metric",
                list(frame.index)
            )

            series = statement_row(frame, selected)

            normalized = normalize_label(selected)

            is_money = not any(
                word in normalized
                for word in ["eps", "shares", "pershare", "ratio", "rate"]
            )

            plot_chart(
                series / divisor if is_money else series,
                selected,
                scale if is_money else "",
                chart_type,
                labels
            )

            st.dataframe(
                frame,
                use_container_width=True
            )

            download_csv(
                frame,
                f"{ticker}_{statement}.csv",
                "statement_csv"
            )

        else:

            st.warning("Statement unavailable.")

    with tabs[2]:

        fields = [
            column for column in history.columns
            if "(%)" in column or column in [
                "Debt / Equity",
                "Current Ratio",
                "Quick Ratio",
                "Debt / EBITDA",
                "Interest Coverage"
            ]
        ]

        selected = st.selectbox(
            "Select Financial Ratio",
            fields
        )

        plot_chart(
            history[selected],
            selected,
            "Percent" if "(%)" in selected else "Multiple",
            chart_type,
            labels
        )

        financial_table(history, fields)

    with tabs[3]:

        quarterly = financial_history(
            fetch_statements(ticker, "Quarterly"),
            annual=False
        )

        if len(quarterly) >= 4:

            ttm = pd.DataFrame(index=quarterly.index)

            intervals = (
                quarterly.index.to_series()
                .diff()
                .dt.days
                .between(60, 120)
            )

            valid = intervals.rolling(3).sum() == 3

            for field in [
                "Revenue",
                "Net Income",
                "Diluted EPS",
                "EBITDA",
                "Free Cash Flow"
            ]:

                ttm[field] = (
                    quarterly[field]
                    .rolling(4, min_periods=4)
                    .sum()
                    .where(valid)
                )

            latest = ttm.iloc[-1]

            metric_grid([
                ("Revenue TTM", latest["Revenue"], "money"),
                ("Net Income TTM", latest["Net Income"], "money"),
                ("EPS TTM", latest["Diluted EPS"], "number"),
                ("FCF TTM", latest["Free Cash Flow"], "money"),
                ("Trailing P/E", info.get("trailingPE"), "multiple"),
                ("Forward P/E", info.get("forwardPE"), "multiple"),
                ("EV/EBITDA", info.get("enterpriseToEbitda"), "multiple")
            ])

            plot_chart(
                ttm["Revenue"] / divisor,
                "Trailing Twelve-Month Revenue",
                scale,
                chart_type,
                labels
            )

            plot_chart(
                ttm["Diluted EPS"],
                "Trailing Twelve-Month EPS",
                "EPS",
                chart_type,
                labels
            )

            quote_currency = info.get("currency")

            financial_currency = info.get("financialCurrency")

            if quote_currency and quote_currency == financial_currency:

                prices = fetch_prices(
                    (ticker,),
                    (
                        ttm.index.min() -
                        pd.Timedelta(days=30)
                    ).strftime("%Y-%m-%d")
                )

                if ticker in prices.columns:

                    price = prices[ticker].dropna().sort_index()

                    period_price = price.reindex(
                        ttm.index,
                        method="ffill"
                    )

                    ttm["Historical P/E"] = safe_divide(
                        period_price,
                        ttm["Diluted EPS"]
                    )

                    plot_chart(
                        ttm["Historical P/E"],
                        "Fiscal Period-End P/E",
                        "Multiple",
                        chart_type,
                        labels
                    )

                    shares = number(info.get("sharesOutstanding"))

                    enterprise = (
                        period_price * shares +
                        quarterly["Total Debt"] -
                        quarterly["Cash"]
                    )

                    ttm["Approximate EV/EBITDA"] = safe_divide(
                        enterprise,
                        ttm["EBITDA"]
                    )

                    plot_chart(
                        ttm["Approximate EV/EBITDA"],
                        "Approximate Historical EV/EBITDA",
                        "Multiple",
                        chart_type,
                        labels
                    )

                    st.caption(
                        "Historical EV uses current shares outstanding. "
                        "Fiscal period-end ratios are not point-in-time "
                        "backtesting data."
                    )

            st.dataframe(
                ttm,
                use_container_width=True
            )

        else:

            st.warning(
                "At least four quarters are needed for TTM calculations."
            )

    download_csv(
        history,
        f"{ticker}_fundamentals.csv",
        "fund_all_csv"
    )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def rsi(close, length=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

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

    rs = average_gain / average_loss.replace(0, np.nan)

    result = 100 - 100 / (1 + rs)

    result = result.mask(
        (average_loss == 0) & (average_gain > 0),
        100
    )

    result = result.mask(
        (average_loss == 0) & (average_gain == 0),
        50
    )

    return result


def slope(series, window=20):

    series = series.dropna().tail(window)

    if len(series) < window:
        return np.nan

    return float(
        np.polyfit(
            np.arange(len(series)),
            series.values,
            1
        )[0]
    )


def realized_volatility(series, window=20):

    returns = np.log(series / series.shift(1))

    return (
        returns.rolling(window).std() *
        np.sqrt(252) * 100
    )


def percentile_rank(series, window):

    series = series.dropna().tail(window)

    if len(series) < window:
        return np.nan

    return series.rank(pct=True).iloc[-1] * 100


# ============================================================
# PAGE 4: TECHNICALS
# ============================================================

def render_technicals():

    st.subheader("🧭 Technicals")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Technical Controls")

    period = st.sidebar.radio(
        "Lookback Period",
        ["1Y", "2Y", "5Y", "Max"],
        index=1
    )

    slope_window = st.sidebar.slider(
        "Breadth Slope Window",
        10, 60, 20
    )

    rsi_length = st.sidebar.slider(
        "RSI Length",
        7, 28, 14
    )

    vol_window = st.sidebar.slider(
        "Volatility Window",
        10, 60, 20
    )

    pct_window = st.sidebar.radio(
        "VIX Percentile Period",
        ["1Y", "3Y", "5Y"]
    )

    pct_days = {
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260
    }[pct_window]

    prices = fetch_prices(
        ("SPY", "RSP", "QQQ", "IWM", "^VIX", "^VVIX"),
        start_date(period)
    )

    if prices.empty:

        st.warning("Unable to download market data.")
        return

    def series(symbol):

        if symbol in prices:
            return prices[symbol].dropna()

        return pd.Series(dtype=float)

    spy = series("SPY")
    rsp = series("RSP")
    qqq = series("QQQ")
    iwm = series("IWM")
    vix = series("^VIX")
    vvix = series("^VVIX")

    breadth = (rsp / spy).dropna()

    growth = (qqq / iwm).dropna()

    tabs = st.tabs([
        "Breadth Proxies",
        "Sentiment & Volatility",
        "Trend & Momentum",
        "Volatility Regime",
        "Composite Risk Gauge"
    ])

    with tabs[0]:

        metric_grid([
            ("RSP/SPY Slope", slope(breadth, slope_window), "number"),
            ("QQQ/IWM Slope", slope(growth, slope_window), "number")
        ], columns=2)

        plot_chart(
            breadth.rename("RSP/SPY"),
            "Equal Weight vs Cap Weighted"
        )

        plot_chart(
            growth.rename("QQQ/IWM"),
            "Large Caps vs Small Caps"
        )

    with tabs[1]:

        if not vix.empty:

            vix_ma = vix.rolling(50).mean()

            longer = fetch_prices(
                ("^VIX",),
                start_date("10Y")
            )

            longer_vix = (
                longer["^VIX"]
                if "^VIX" in longer else vix
            )

            metric_grid([
                ("VIX", vix.iloc[-1], "number"),
                (
                    "VIX vs 50 DMA",
                    vix.iloc[-1] - vix_ma.iloc[-1],
                    "number"
                ),
                (
                    "VIX Percentile",
                    percentile_rank(longer_vix, pct_days),
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
                "Volatility of Volatility"
            )

    with tabs[2]:

        for symbol in ["SPY", "QQQ", "IWM"]:

            close = series(symbol)

            if close.empty:
                continue

            st.markdown(f"### {symbol}")

            ma50 = close.rolling(50).mean()

            ma100 = close.rolling(100).mean()

            ma200 = close.rolling(200).mean()

            strength = rsi(close, rsi_length)

            trend = "N/A"

            if pd.notna(ma200.iloc[-1]):

                trend = (
                    "Bullish"
                    if close.iloc[-1] > ma200.iloc[-1]
                    else "Bearish"
                )

            c1, c2, c3 = st.columns(3)

            c1.metric("Price", fmt(close.iloc[-1]))

            c2.metric("200 DMA Trend", trend)

            c3.metric("RSI", fmt(strength.iloc[-1]))

            plot_chart(
                pd.concat([
                    close.rename(symbol),
                    ma50.rename("50 DMA"),
                    ma100.rename("100 DMA"),
                    ma200.rename("200 DMA")
                ], axis=1),
                f"{symbol} Moving Averages"
            )

            fig, ax = plt.subplots(figsize=(12, 3))

            ax.plot(strength.index, strength)

            ax.axhline(70, linestyle="--", color="red")

            ax.axhline(30, linestyle="--", color="green")

            ax.set_ylim(0, 100)

            ax.set_title(f"{symbol} RSI")

            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            plt.close(fig)

    with tabs[3]:

        if not spy.empty:

            realized = realized_volatility(
                spy,
                vol_window
            )

            plot_chart(
                pd.concat([
                    realized.rename("Realized Volatility"),
                    vix.rename("VIX")
                ], axis=1),
                "Realized vs Implied Volatility",
                "Percent"
            )

    with tabs[4]:

        st.markdown("### Composite Risk Gauge")

        ma200 = spy.rolling(200).mean()

        vix50 = vix.rolling(50).mean()

        breadth_slope = slope(breadth, slope_window)

        trend_ok = None

        breadth_ok = None

        vol_ok = None

        if not spy.empty and pd.notna(ma200.iloc[-1]):

            trend_ok = bool(spy.iloc[-1] > ma200.iloc[-1])

        if pd.notna(breadth_slope):

            breadth_ok = bool(breadth_slope > 0)

        if not vix.empty and pd.notna(vix50.iloc[-1]):

            vol_ok = bool(vix.iloc[-1] < vix50.iloc[-1])

        components = [
            ("Trend", trend_ok, 34),
            ("Breadth", breadth_ok, 33),
            ("Volatility", vol_ok, 33)
        ]

        rows = []

        score = 0

        available = 0

        for name, condition, weight in components:

            if condition is None:

                status = "N/A"

                points = np.nan

            else:

                available += weight

                status = "Positive" if condition else "Negative"

                points = weight if condition else 0

                score += points

            rows.append({
                "Component": name,
                "Status": status,
                "Points": points
            })

        if available == 100:

            regime = (
                "Risk-On" if score >= 67
                else "Neutral" if score >= 34
                else "Risk-Off"
            )

            st.metric(
                "Market Risk Gauge",
                f"{score}/100",
                regime
            )

            st.progress(score / 100)

        else:

            st.warning("Insufficient data for a complete score.")

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True
        )

        st.caption(
            "Trend = SPY above 200 DMA. "
            "Breadth = positive RSP/SPY slope. "
            "Volatility = VIX below 50 DMA."
        )


# ============================================================
# PAGE 5: COMPANY OVERVIEW
# ============================================================

def render_company():

    st.subheader("🏢 Company Overview")

    with st.form("company_form"):

        symbol = st.text_input(
            "Enter Stock Symbol",
            "AAPL"
        ).upper().strip()

        submitted = st.form_submit_button(
            "Analyze Company",
            type="primary"
        )

    if submitted or "company_symbol" not in st.session_state:

        st.session_state["company_symbol"] = symbol

    ticker = st.session_state["company_symbol"]

    if not ticker:
        return

    with st.spinner("Loading Company Financials..."):

        info = fetch_info(ticker)

        history = financial_history(
            fetch_statements(ticker, "Annual")
        )

    if not info and history.empty:

        st.error("No company data available.")
        return

    st.markdown(
        f"## {info.get('longName', ticker)} ({ticker})"
    )

    st.caption(
        f"Sector: {info.get('sector', 'N/A')} | "
        f"Industry: {info.get('industry', 'N/A')}"
    )

    st.caption(
        f"Quote Currency: {info.get('currency', 'N/A')} | "
        f"Financial Currency: {info.get('financialCurrency', 'N/A')}"
    )

    def v(key, multiplier=1):

        return number(info.get(key)) * multiplier

    price = number(
        info.get("currentPrice"),
        number(info.get("regularMarketPrice"))
    )

    st.markdown("### Market Snapshot")

    metric_grid([
        ("Share Price", price, "number"),
        ("Market Cap", v("marketCap"), "money"),
        ("Enterprise Value", v("enterpriseValue"), "money"),
        ("Beta", v("beta"), "number"),
        ("52-Week High", v("fiftyTwoWeekHigh"), "number"),
        ("52-Week Low", v("fiftyTwoWeekLow"), "number"),
        ("50 DMA", v("fiftyDayAverage"), "number"),
        ("200 DMA", v("twoHundredDayAverage"), "number")
    ])

    with st.expander("Company Description"):

        st.write(
            info.get(
                "longBusinessSummary",
                "Description unavailable."
            )
        )

    st.divider()

    # --------------------------------------------------------
    # Historical Financial Statements
    # --------------------------------------------------------

    st.markdown("### Three-Year Financial Summary")

    recent = history.tail(3)

    latest = (
        recent.iloc[-1]
        if not recent.empty
        else pd.Series(dtype=float)
    )

    fields = [
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

    financial_table(recent, fields)

    st.caption(
        "Annual fiscal-year data. Growth rates are calculated "
        "from financial statements. Growth from nonpositive "
        "prior values is displayed as N/A."
    )

    if not recent.empty:

        c1, c2 = st.columns(2)

        with c1:

            plot_chart(
                recent["Revenue"] / 1e9,
                "Annual Revenue",
                "Billions",
                "Bar",
                True
            )

            plot_chart(
                recent[[
                    "Revenue Growth (%)",
                    "Net Income Growth (%)"
                ]],
                "Revenue and Earnings Growth",
                "Percent",
                "Bar",
                True
            )

        with c2:

            plot_chart(
                recent[[
                    "Gross Profit",
                    "Operating Income",
                    "Net Income"
                ]] / 1e9,
                "Annual Profits",
                "Billions",
                "Bar",
                True
            )

            plot_chart(
                recent[[
                    "Operating Cash Flow",
                    "Free Cash Flow"
                ]] / 1e9,
                "Cash Flow",
                "Billions",
                "Bar",
                True
            )

    st.divider()

    # --------------------------------------------------------
    # Yahoo Finance Valuation Ratios
    # --------------------------------------------------------

    st.markdown("### Current Valuation Ratios")

    st.caption(
        "Source: Yahoo Finance quote statistics. "
        "Reporting periods follow the provider's definitions."
    )

    peg = number(
        info.get("trailingPegRatio"),
        number(info.get("pegRatio"))
    )

    metric_grid([
        ("Trailing P/E", v("trailingPE"), "multiple"),
        ("Forward P/E", v("forwardPE"), "multiple"),
        ("Price / Sales", v("priceToSalesTrailing12Months"), "multiple"),
        ("PEG Ratio", peg, "multiple"),
        ("Price / Book", v("priceToBook"), "multiple"),
        ("EV / EBITDA", v("enterpriseToEbitda"), "multiple"),
        ("EV / Revenue", v("enterpriseToRevenue"), "multiple"),
        ("Earnings Yield", divide(100, v("trailingPE")), "percent")
    ])

    st.divider()

    # --------------------------------------------------------
    # Current Profitability
    # --------------------------------------------------------

    st.markdown("### Current Profitability")

    st.caption("Source: Yahoo Finance reported profitability ratios.")

    metric_grid([
        ("ROE", v("returnOnEquity", 100), "percent"),
        ("ROA", v("returnOnAssets", 100), "percent"),
        ("Gross Margin", v("grossMargins", 100), "percent"),
        ("Operating Margin", v("operatingMargins", 100), "percent"),
        ("Net Margin", v("profitMargins", 100), "percent"),
        ("EBITDA Margin", v("ebitdaMargins", 100), "percent"),
        ("Revenue Growth", v("revenueGrowth", 100), "percent"),
        ("Quarterly Earnings Growth", v("earningsQuarterlyGrowth", 100), "percent")
    ])

    st.divider()

    # --------------------------------------------------------
    # Financial Health
    # --------------------------------------------------------

    st.markdown("### Current Financial Health")

    st.caption("Source: Yahoo Finance quote statistics.")

    metric_grid([
        ("Debt / Equity", v("debtToEquity", 0.01), "multiple"),
        ("Current Ratio", v("currentRatio"), "multiple"),
        ("Quick Ratio", v("quickRatio"), "multiple"),
        ("Total Debt", v("totalDebt"), "money"),
        ("Total Cash", v("totalCash"), "money"),
        ("Net Debt", v("totalDebt") - v("totalCash"), "money"),
        ("Operating Cash Flow", v("operatingCashflow"), "money"),
        ("Free Cash Flow", v("freeCashflow"), "money")
    ])

    st.divider()

    # --------------------------------------------------------
    # Independently Calculated Ratios
    # --------------------------------------------------------

    st.markdown("### Advanced Calculated Metrics")

    same_currency = (
        info.get("currency") is not None and
        info.get("currency") == info.get("financialCurrency")
    )

    fcf_yield = (
        divide(v("freeCashflow"), v("marketCap")) * 100
        if same_currency else np.nan
    )

    price_fcf = (
        divide(v("marketCap"), v("freeCashflow"))
        if same_currency else np.nan
    )

    metric_grid([
        ("ROIC - Annual", latest.get("ROIC (%)"), "percent"),
        ("FCF Yield", fcf_yield, "percent"),
        ("Price / FCF", price_fcf, "multiple"),
        ("Interest Coverage - Annual", latest.get("Interest Coverage"), "multiple"),
        ("Debt / EBITDA - Annual", latest.get("Debt / EBITDA"), "multiple"),
        ("FCF Margin - Annual", latest.get("FCF Margin (%)"), "percent")
    ])

    st.divider()

    st.markdown("### Three-Year Historical Ratios")

    ratios = [
        "ROIC (%)",
        "ROE (%)",
        "ROA (%)",
        "Gross Margin (%)",
        "Operating Margin (%)",
        "Net Margin (%)",
        "EBITDA Margin (%)",
        "FCF Margin (%)",
        "Debt / Equity",
        "Current Ratio",
        "Quick Ratio",
        "Debt / EBITDA",
        "Interest Coverage"
    ]

    financial_table(recent, ratios)

    st.caption(
        "Source: independently calculated annual financial ratios. "
        "ROE, ROA and ROIC use average annual balance sheet values."
    )

    if not recent.empty:

        c1, c2 = st.columns(2)

        with c1:

            plot_chart(
                recent[["ROIC (%)", "ROE (%)", "ROA (%)"]],
                "Capital Efficiency",
                "Percent",
                "Bar",
                True
            )

        with c2:

            plot_chart(
                recent[[
                    "Gross Margin (%)",
                    "Operating Margin (%)",
                    "Net Margin (%)"
                ]],
                "Profitability Trends",
                "Percent",
                "Bar",
                True
            )

    st.markdown("### Balance Sheet Summary")

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

    with st.expander("Financial Ratio Definitions"):

        st.markdown("""
- **P/E:** Price divided by earnings per share.
- **P/S:** Market capitalization divided by revenue.
- **PEG:** P/E relative to expected earnings growth.
- **EV/EBITDA:** Enterprise value divided by EBITDA.
- **ROE:** Net income divided by average equity.
- **ROA:** Net income divided by average assets.
- **ROIC:** After-tax operating income divided by average invested capital.
- **Debt/Equity:** Total debt divided by shareholder equity.
- **FCF Yield:** Free cash flow divided by market capitalization.
- **Interest Coverage:** Operating income divided by interest expense.

Current ratios are retrieved from Yahoo Finance.
Historical ratios are independently calculated.
Annual and trailing reporting periods may differ.
        """)

    if not recent.empty:

        download_csv(
            recent,
            f"{ticker}_company_overview.csv",
            "company_csv"
        )


# ============================================================
# DCF VALUATION ENGINE
# ============================================================

def calculate_dcf(
    revenue,
    forecast,
    wacc,
    terminal_growth,
    exit_multiple,
    method,
    cash,
    debt,
    shares,
    other_adjustments=0
):

    if revenue <= 0 or shares <= 0 or wacc <= 0:
        raise ValueError("Revenue, shares and WACC must be positive.")

    if forecast.empty:
        raise ValueError("Forecast cannot be empty.")

    if method == "Perpetual Growth":

        if terminal_growth >= wacc:
            raise ValueError(
                "WACC must exceed terminal growth."
            )

        if terminal_growth <= -1:
            raise ValueError("Invalid terminal growth.")

    if method == "Exit Multiple" and exit_multiple <= 0:
        raise ValueError("Exit multiple must be positive.")

    rows = []

    current_revenue = revenue

    for i, row in forecast.reset_index(drop=True).iterrows():

        year = i + 1

        growth = float(row["Revenue Growth %"]) / 100

        margin = float(row["EBIT Margin %"]) / 100

        tax = float(row["Tax Rate %"]) / 100

        da_rate = float(row["D&A % Revenue"]) / 100

        capex_rate = float(row["CAPEX % Revenue"]) / 100

        nwc_rate = float(row["NWC Investment % Revenue"]) / 100

        inputs = [
            growth, margin, tax,
            da_rate, capex_rate, nwc_rate
        ]

        if not np.isfinite(inputs).all():
            raise ValueError("All forecast inputs must be valid numbers.")

        if growth <= -1:
            raise ValueError("Revenue growth must exceed -100%.")

        if not 0 <= tax <= 1:
            raise ValueError("Tax rate must be between 0% and 100%.")

        if da_rate < 0 or capex_rate < 0:
            raise ValueError("D&A and CAPEX cannot be negative.")

        current_revenue *= 1 + growth

        ebit = current_revenue * margin

        # Do not automatically assume tax benefits
        # when operating income is negative.

        cash_taxes = max(ebit, 0) * tax

        nopat = ebit - cash_taxes

        da = current_revenue * da_rate

        capex = current_revenue * capex_rate

        change_nwc = current_revenue * nwc_rate

        ebitda = ebit + da

        fcff = nopat + da - capex - change_nwc

        discount_factor = 1 / (1 + wacc) ** year

        present_value = fcff * discount_factor

        rows.append({
            "Year": year,
            "Revenue": current_revenue,
            "EBIT": ebit,
            "Cash Taxes": cash_taxes,
            "NOPAT": nopat,
            "D&A": da,
            "EBITDA": ebitda,
            "CAPEX": capex,
            "NWC Investment": change_nwc,
            "FCFF": fcff,
            "Discount Factor": discount_factor,
            "PV of FCFF": present_value
        })

    projection = pd.DataFrame(rows).set_index("Year")

    last = projection.iloc[-1]

    if method == "Perpetual Growth":

        terminal_fcff = last["FCFF"] * (1 + terminal_growth)

        if terminal_fcff <= 0:
            raise ValueError(
                "Perpetual growth requires positive terminal cash flow."
            )

        terminal_value = (
            terminal_fcff /
            (wacc - terminal_growth)
        )

    else:

        if last["EBITDA"] <= 0:
            raise ValueError(
                "Exit multiple method requires positive terminal EBITDA."
            )

        terminal_value = (
            last["EBITDA"] * exit_multiple
        )

    pv_terminal = (
        terminal_value /
        (1 + wacc) ** len(projection)
    )

    pv_cash_flows = projection["PV of FCFF"].sum()

    enterprise_value = pv_cash_flows + pv_terminal

    equity_value = (
        enterprise_value +
        cash -
        debt +
        other_adjustments
    )

    intrinsic_value = equity_value / shares

    return {
        "Projection": projection,
        "PV Cash Flows": pv_cash_flows,
        "Terminal Value": terminal_value,
        "PV Terminal Value": pv_terminal,
        "Enterprise Value": enterprise_value,
        "Equity Value": equity_value,
        "Intrinsic Value": intrinsic_value,
        "Terminal Contribution": (
            pv_terminal / enterprise_value * 100
            if enterprise_value > 0 else np.nan
        )
    }


# ============================================================
# PAGE 6: DCF VALUATION
# ============================================================

def render_dcf():

    st.subheader("💰 Discounted Cash Flow Valuation")

    st.info(
        "Enter a stock symbol. Yahoo Finance supplies available "
        "historical financial data. All valuation assumptions "
        "can be edited."
    )

    with st.form("dcf_load_form"):

        symbol = st.text_input(
            "Stock Symbol",
            "AAPL",
            key="dcf_symbol_input"
        ).upper().strip()

        reporting_period = st.radio(
            "Base Financial Period",
            ["TTM", "Annual"],
            horizontal=True
        )

        load = st.form_submit_button(
            "Load Company Financials",
            type="primary"
        )

    if load or "dcf_selection" not in st.session_state:

        st.session_state["dcf_selection"] = (
            symbol,
            reporting_period
        )

        st.session_state["dcf_version"] = (
            st.session_state.get("dcf_version", 0) + 1
        )

    ticker, period = st.session_state["dcf_selection"]

    version = st.session_state["dcf_version"]

    prefix = f"dcf_{version}"

    if not ticker:

        st.warning("Enter a stock symbol.")
        return

    with st.spinner("Loading Financial Statements..."):

        info = fetch_info(ticker)

        annual = financial_history(
            fetch_statements(ticker, "Annual")
        )

        quarterly = financial_history(
            fetch_statements(ticker, "Quarterly"),
            annual=False
        )

    if period == "TTM":

        base = trailing_financials(quarterly)

        base_date = (
            quarterly.index[-1]
            if not base.empty else None
        )

        if base.empty:

            st.warning(
                "TTM data unavailable. Select Annual or "
                "enter the financial inputs manually."
            )

    else:

        base = (
            annual.iloc[-1]
            if not annual.empty
            else pd.Series(dtype=float)
        )

        base_date = (
            annual.index[-1]
            if not annual.empty else None
        )

    st.markdown(
        f"## {info.get('longName', ticker)} ({ticker})"
    )

    currency = info.get("financialCurrency") or "USD"

    quote_currency = info.get("currency")

    st.caption(
        f"Selected period: {period} | "
        f"Financial currency: {currency} | "
        f"Quote currency: {quote_currency or 'Unknown'}"
    )

    if base_date is not None:

        st.caption(
            f"Financial period ending: {base_date.date()}"
        )

    if info.get("sector") == "Financial Services":

        st.warning(
            "FCFF valuation is generally less suitable for banks "
            "and insurers. An equity cash flow or excess return "
            "model may be more appropriate."
        )

    st.caption(
        "Financial amounts and share counts are expressed in "
        "billions. Share price and intrinsic value are per share."
    )

    def edit(
        label,
        value,
        name,
        step=0.1,
        minimum=None,
        maximum=None,
        help_text=None
    ):

        value = number(value)

        kwargs = {}

        if minimum is not None:

            kwargs["min_value"] = float(minimum)

            if pd.notna(value) and value < minimum:
                value = np.nan

        if maximum is not None:

            kwargs["max_value"] = float(maximum)

            if pd.notna(value) and value > maximum:
                value = np.nan

        return st.number_input(
            label,
            value=float(value) if pd.notna(value) else None,
            step=float(step),
            format="%.6f",
            key=f"{prefix}_{name}",
            help=help_text,
            **kwargs
        )

    def annual_latest(field):

        if annual.empty:
            return np.nan

        return number(annual.iloc[-1].get(field))

    # --------------------------------------------------------
    # SECTION 1: BASE FINANCIAL INPUTS
    # --------------------------------------------------------

    st.markdown("### 1. Base Financial Inputs")

    st.caption(
        "Defaults are retrieved from the selected financial "
        "statement period. Blank values require manual entry."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        revenue = edit(
            "Base Revenue (Billions)",
            number(base.get("Revenue")) / 1e9,
            "revenue",
            minimum=0.0
        )

        ebit = edit(
            "Operating Income / EBIT (Billions)",
            number(base.get("Operating Income")) / 1e9,
            "ebit"
        )

        pretax = edit(
            "Pretax Income (Billions)",
            number(base.get("Pretax Income")) / 1e9,
            "pretax"
        )

    with c2:

        da = edit(
            "Depreciation & Amortization (Billions)",
            number(base.get("Depreciation & Amortization")) / 1e9,
            "da",
            minimum=0
        )

        capex = edit(
            "Capital Expenditures (Billions)",
            number(base.get("CAPEX")) / 1e9,
            "capex",
            minimum=0
        )

        nwc = edit(
            "Working Capital Investment (Billions)",
            number(base.get("Working Capital Investment")) / 1e9,
            "nwc",
            help_text=(
                "Positive means investment in working capital "
                "and reduces free cash flow. Negative means "
                "working capital is released."
            )
        )

    with c3:

        tax_expense = edit(
            "Tax Expense (Billions)",
            number(base.get("Tax Provision")) / 1e9,
            "tax_expense"
        )

        historical_growth = edit(
            "Historical Annual Revenue Growth (%)",
            annual_latest("Revenue Growth (%)"),
            "historical_growth"
        )

        interest_expense = edit(
            "Interest Expense (Billions)",
            abs(number(base.get("Interest Expense"))) / 1e9,
            "interest_expense",
            minimum=0
        )

    st.divider()

    # --------------------------------------------------------
    # SECTION 2: CAPITAL STRUCTURE
    # --------------------------------------------------------

    st.markdown("### 2. Capital Structure & Market Data")

    price_default = number(
        info.get("currentPrice"),
        number(info.get("regularMarketPrice"))
    )

    market_cap_default = number(
        info.get("marketCap")
    ) / 1e9

    cash_default = number(info.get("totalCash")) / 1e9

    debt_default = number(info.get("totalDebt")) / 1e9

    if pd.isna(cash_default):

        cash_default = number(base.get("Cash")) / 1e9

    if pd.isna(debt_default):

        debt_default = number(base.get("Total Debt")) / 1e9

    if quote_currency != currency:

        st.warning(
            "Quote and financial currencies differ or are unknown. "
            "Enter share price and market capitalization in the "
            "financial statement currency. Verify ADR share counts "
            "and currency conversion before valuing the company."
        )

        price_default = np.nan

        market_cap_default = np.nan

    c1, c2, c3 = st.columns(3)

    with c1:

        share_price = edit(
            "Current Share Price",
            price_default,
            "price",
            minimum=0
        )

        market_cap = edit(
            "Market Capitalization (Billions)",
            market_cap_default,
            "market_cap",
            minimum=0
        )

        shares = edit(
            "Shares Outstanding (Billions)",
            number(info.get("sharesOutstanding")) / 1e9,
            "shares",
            step=0.001,
            minimum=0
        )

    with c2:

        total_cash = edit(
            "Cash & Cash Equivalents (Billions)",
            cash_default,
            "cash",
            minimum=0,
            help_text="Use excess cash when estimating operating enterprise value."
        )

        total_debt = edit(
            "Total Debt (Billions)",
            debt_default,
            "debt",
            minimum=0
        )

        beta = edit(
            "Beta",
            number(info.get("beta")),
            "beta"
        )

    with c3:

        minority = edit(
            "Minority Interest (Billions)",
            number(base.get("Minority Interest")),
            "minority_raw",
            minimum=0
        ) if False else edit(
            "Minority Interest (Billions)",
            number(base.get("Minority Interest"), 0) / 1e9,
            "minority",
            minimum=0,
            help_text="Defaults to zero if unavailable. Verify manually."
        )

        preferred = edit(
            "Preferred Stock (Billions)",
            number(base.get("Preferred Stock"), 0) / 1e9,
            "preferred",
            minimum=0,
            help_text="Defaults to zero if unavailable. Verify manually."
        )

        other_assets = edit(
            "Other Nonoperating Adjustments (Billions)",
            0.0,
            "other_assets",
            help_text=(
                "Positive for additional assets, negative for "
                "additional nonoperating liabilities."
            )
        )

    st.divider()

    # --------------------------------------------------------
    # SECTION 3: WACC
    # --------------------------------------------------------

    st.markdown("### 3. Weighted Average Cost of Capital")

    st.caption(
        "Risk-free rate and equity risk premium defaults are "
        "illustrative assumptions, not live market estimates."
    )

    observed_tax = divide(tax_expense, pretax) * 100

    if not np.isfinite(observed_tax) or not 0 <= observed_tax <= 100:
        observed_tax = 21.0

    estimated_debt_cost = (
        divide(interest_expense, total_debt) * 100
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        risk_free = edit(
            "Risk-Free Rate (%)",
            4.0,
            "risk_free"
        )

        equity_premium = edit(
            "Equity Risk Premium (%)",
            5.0,
            "erp"
        )

    with c2:

        cost_debt = edit(
            "Pretax Cost of Debt (%)",
            estimated_debt_cost,
            "cost_debt",
            minimum=0,
            help_text=(
                "Estimated using interest expense divided "
                "by total debt. Replace with marginal borrowing cost."
            )
        )

        tax_rate = edit(
            "Marginal Tax Rate (%)",
            observed_tax,
            "tax_rate",
            minimum=0,
            maximum=100,
            help_text=(
                "Uses the observed effective tax rate if valid, "
                "otherwise an illustrative 21%."
            )
        )

    with c3:

        wacc_method = st.radio(
            "Discount Rate Method",
            ["Calculated WACC", "Manual WACC"],
            key=f"{prefix}_wacc_method"
        )

        manual_wacc = edit(
            "Manual WACC (%)",
            9.0,
            "manual_wacc",
            minimum=0.01
        )

    cost_equity = np.nan

    calculated_wacc = np.nan

    equity_weight = np.nan

    debt_weight = np.nan

    capm_inputs = [risk_free, beta, equity_premium]

    if all(x is not None for x in capm_inputs):

        cost_equity = (
            risk_free + beta * equity_premium
        )

    if (
        market_cap is not None and
        total_debt is not None and
        market_cap + total_debt > 0
    ):

        equity_weight = market_cap / (market_cap + total_debt)

        debt_weight = total_debt / (market_cap + total_debt)

        if np.isfinite(cost_equity) and tax_rate is not None:

            if total_debt == 0:

                calculated_wacc = cost_equity

            elif cost_debt is not None:

                calculated_wacc = (
                    equity_weight * cost_equity +
                    debt_weight * cost_debt *
                    (1 - tax_rate / 100)
                )

    wacc_percent = (
        calculated_wacc
        if wacc_method == "Calculated WACC"
        else number(manual_wacc)
    )

    metric_grid([
        ("Cost of Equity", cost_equity, "percent"),
        ("Equity Weight", equity_weight * 100, "percent"),
        ("Debt Weight", debt_weight * 100, "percent"),
        ("Selected WACC", wacc_percent, "percent")
    ])

    st.divider()

    # --------------------------------------------------------
    # SECTION 4: FORECAST ASSUMPTIONS
    # --------------------------------------------------------

    st.markdown("### 4. Operating Forecast Assumptions")

    st.caption(
        "Historical ratios are starting points, not forecasts. "
        "Review and edit every assumption."
    )

    years = st.slider(
        "Projection Period (Years)",
        3, 10, 5,
        key=f"{prefix}_years"
    )

    margin_default = divide(ebit, revenue) * 100

    da_default = divide(da, revenue) * 100

    capex_default = divide(capex, revenue) * 100

    nwc_default = divide(nwc, revenue) * 100

    c1, c2, c3 = st.columns(3)

    with c1:

        growth = edit(
            "Projected Annual Revenue Growth (%)",
            historical_growth,
            "growth"
        )

        margin = edit(
            "Projected EBIT Margin (%)",
            margin_default,
            "margin"
        )

    with c2:

        forecast_tax = edit(
            "Projected Tax Rate (%)",
            tax_rate,
            "forecast_tax",
            minimum=0,
            maximum=100
        )

        forecast_da = edit(
            "D&A as % of Revenue",
            da_default,
            "forecast_da",
            minimum=0
        )

    with c3:

        forecast_capex = edit(
            "CAPEX as % of Revenue",
            capex_default,
            "forecast_capex",
            minimum=0
        )

        forecast_nwc = edit(
            "NWC Investment as % of Revenue",
            nwc_default,
            "forecast_nwc"
        )

    assumption_values = [
        growth,
        margin,
        forecast_tax,
        forecast_da,
        forecast_capex,
        forecast_nwc
    ]

    if any(x is None for x in assumption_values):

        st.warning(
            "Complete all operating forecast assumptions."
        )

        return

    forecast_defaults = pd.DataFrame({
        "Year": list(range(1, years + 1)),
        "Revenue Growth %": [growth] * years,
        "EBIT Margin %": [margin] * years,
        "Tax Rate %": [forecast_tax] * years,
        "D&A % Revenue": [forecast_da] * years,
        "CAPEX % Revenue": [forecast_capex] * years,
        "NWC Investment % Revenue": [forecast_nwc] * years
    })

    st.markdown("#### Editable Year-by-Year Forecast")

    st.caption(
        "Edit individual table cells to model different growth "
        "rates, margins and reinvestment assumptions each year."
    )

    forecast_signature = (
        years,
        tuple(assumption_values)
    )

    signature_key = f"{prefix}_forecast_signature"

    revision_key = f"{prefix}_forecast_revision"

    if st.session_state.get(signature_key) != forecast_signature:

        st.session_state[signature_key] = forecast_signature

        st.session_state[revision_key] = (
            st.session_state.get(revision_key, 0) + 1
        )

    if st.button(
        "Reset Annual Forecast to Default Assumptions",
        key=f"{prefix}_reset_forecast"
    ):

        st.session_state[revision_key] += 1

    revision = st.session_state[revision_key]

    forecast = st.data_editor(
        forecast_defaults,
        hide_index=True,
        use_container_width=True,
        disabled=["Year"],
        num_rows="fixed",
        key=f"{prefix}_forecast_{revision}"
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 5: TERMINAL VALUE
    # --------------------------------------------------------

    st.markdown("### 5. Terminal Value Assumptions")

    method = st.radio(
        "Terminal Valuation Method",
        ["Perpetual Growth", "Exit Multiple"],
        horizontal=True,
        key=f"{prefix}_terminal_method"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        terminal_growth = edit(
            "Terminal Growth Rate (%)",
            2.5,
            "terminal_growth"
        )

    with c2:

        exit_multiple = edit(
            "Terminal EV / EBITDA Multiple",
            12.0,
            "exit_multiple",
            minimum=0.01
        )

    with c3:

        margin_safety = edit(
            "Margin of Safety (%)",
            20.0,
            "margin_safety",
            minimum=0,
            maximum=100
        )

    st.caption(
        "Terminal assumptions are illustrative and editable. "
        "Perpetual growth assumes the final-year operating "
        "economics and reinvestment remain sustainable."
    )

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    required = {
        "Revenue": revenue,
        "Share Price": share_price,
        "Shares Outstanding": shares,
        "Cash": total_cash,
        "Debt": total_debt,
        "Minority Interest": minority,
        "Preferred Stock": preferred,
        "Other Adjustments": other_assets,
        "WACC": wacc_percent,
        "Terminal Growth": terminal_growth,
        "Exit Multiple": exit_multiple,
        "Margin of Safety": margin_safety
    }

    missing = [
        name for name, value in required.items()
        if value is None or not np.isfinite(number(value))
    ]

    if missing:

        st.warning(
            "Complete the following inputs: " +
            ", ".join(missing)
        )

        return

    if revenue <= 0 or shares <= 0 or share_price <= 0:

        st.error(
            "Revenue, share price and shares outstanding "
            "must be positive."
        )

        return

    if wacc_percent <= 0:

        st.error("WACC must be greater than zero.")

        return

    other_adjustments = (
        other_assets - minority - preferred
    )

    wacc = wacc_percent / 100

    terminal_rate = terminal_growth / 100

    try:

        result = calculate_dcf(
            revenue,
            forecast,
            wacc,
            terminal_rate,
            exit_multiple,
            method,
            total_cash,
            total_debt,
            shares,
            other_adjustments
        )

    except (ValueError, TypeError) as error:

        st.error(str(error))

        return

    intrinsic = result["Intrinsic Value"]

    upside = (
        intrinsic / share_price - 1
    ) * 100

    safety_price = (
        intrinsic * (1 - margin_safety / 100)
        if intrinsic > 0 else np.nan
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 6: DCF RESULTS
    # --------------------------------------------------------

    st.markdown("### 6. DCF Valuation Results")

    a, b, c, d = st.columns(4)

    a.metric(
        "Current Share Price",
        f"{currency} {share_price:,.2f}"
    )

    b.metric(
        "Estimated Intrinsic Value",
        f"{currency} {intrinsic:,.2f}"
    )

    c.metric(
        "Implied Upside / Downside",
        f"{upside:+.2f}%"
    )

    d.metric(
        "Margin of Safety Price",
        fmt(safety_price)
    )

    if intrinsic <= 0:

        st.warning(
            "The modeled equity value is nonpositive. "
            "This is not a literal negative tradable share price."
        )

    elif intrinsic > share_price:

        st.success(
            "The estimated intrinsic value exceeds the "
            "current share price under these assumptions."
        )

    else:

        st.warning(
            "The estimated intrinsic value is below the "
            "current share price under these assumptions."
        )

    metric_grid([
        (
            "Enterprise Value (B)",
            result["Enterprise Value"],
            "number"
        ),
        (
            "Equity Value (B)",
            result["Equity Value"],
            "number"
        ),
        (
            "Terminal Value Contribution",
            result["Terminal Contribution"],
            "percent"
        ),
        (
            "WACC",
            wacc_percent,
            "percent"
        )
    ])

    if result["Terminal Contribution"] > 80:

        st.warning(
            "More than 80% of modeled enterprise value comes "
            "from terminal value. Valuation is highly sensitive "
            "to long-term assumptions."
        )

    # --------------------------------------------------------
    # SECTION 7: PROJECTED FINANCIAL STATEMENTS
    # --------------------------------------------------------

    st.markdown("### 7. Projected Financial Statements")

    projection = result["Projection"]

    st.caption(
        f"Financial amounts in billions of {currency}."
    )

    st.dataframe(
        projection.T.style.format("{:,.3f}"),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:

        plot_chart(
            projection["Revenue"],
            "Projected Revenue",
            "Billions",
            "Bar",
            True
        )

        plot_chart(
            projection[["EBIT", "EBITDA", "NOPAT"]],
            "Projected Operating Profits",
            "Billions",
            "Bar",
            True
        )

    with c2:

        plot_chart(
            projection[["FCFF", "PV of FCFF"]],
            "Projected Free Cash Flow to the Firm",
            "Billions",
            "Bar",
            True
        )

        plot_chart(
            projection[["D&A", "CAPEX", "NWC Investment"]],
            "Reinvestment Requirements",
            "Billions",
            "Bar",
            True
        )

    # --------------------------------------------------------
    # SECTION 8: VALUATION BRIDGE
    # --------------------------------------------------------

    st.markdown("### 8. Enterprise to Equity Value Bridge")

    bridge = pd.DataFrame({
        "Component": [
            "PV of Projected Free Cash Flows",
            "Undiscounted Terminal Value",
            "PV of Terminal Value",
            "Enterprise Value",
            "Add Cash",
            "Less Debt",
            "Less Minority Interest",
            "Less Preferred Stock",
            "Other Nonoperating Adjustments",
            "Equity Value",
            "Shares Outstanding (Billions)",
            "Intrinsic Value Per Share"
        ],
        "Value": [
            result["PV Cash Flows"],
            result["Terminal Value"],
            result["PV Terminal Value"],
            result["Enterprise Value"],
            total_cash,
            -total_debt,
            -minority,
            -preferred,
            other_assets,
            result["Equity Value"],
            shares,
            intrinsic
        ]
    })

    st.dataframe(
        bridge.style.format({"Value": "{:,.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    values = [
        share_price,
        intrinsic,
        safety_price
    ]

    bars = ax.bar(
        ["Market Price", "Intrinsic Value", "Safety Price"],
        values,
        color=["gray", "steelblue", "seagreen"]
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=4
    )

    ax.set_ylabel(f"{currency} per Share")

    ax.set_title("Market Price vs Estimated Intrinsic Value")

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # --------------------------------------------------------
    # SECTION 9: SENSITIVITY ANALYSIS
    # --------------------------------------------------------

    st.markdown("### 9. DCF Sensitivity Analysis")

    st.caption(
        "Intrinsic value per share under different WACC "
        "and terminal valuation assumptions."
    )

    c1, c2 = st.columns(2)

    with c1:

        wacc_step = st.number_input(
            "WACC Increment (Percentage Points)",
            min_value=0.1,
            value=1.0,
            step=0.25,
            key=f"{prefix}_wacc_step"
        )

    with c2:

        terminal_step = st.number_input(
            "Terminal Growth Increment (Percentage Points)"
            if method == "Perpetual Growth"
            else "Exit Multiple Increment",
            min_value=0.1,
            value=0.5 if method == "Perpetual Growth" else 2.0,
            step=0.1,
            key=f"{prefix}_terminal_step_{method}"
        )

    wacc_values = [
        wacc_percent + i * wacc_step
        for i in [-2, -1, 0, 1, 2]
    ]

    if method == "Perpetual Growth":

        terminal_values = [
            terminal_growth + i * terminal_step
            for i in [-2, -1, 0, 1, 2]
        ]

        column_names = [
            f"{x:.2f}%"
            for x in terminal_values
        ]

    else:

        terminal_values = [
            exit_multiple + i * terminal_step
            for i in [-2, -1, 0, 1, 2]
        ]

        column_names = [
            f"{x:.2f}x"
            for x in terminal_values
        ]

    sensitivity = pd.DataFrame(
        index=[f"{x:.2f}%" for x in wacc_values],
        columns=column_names,
        dtype=float
    )

    sensitivity.index.name = "WACC"

    for w in wacc_values:

        for j, terminal in enumerate(terminal_values):

            try:

                test = calculate_dcf(
                    revenue,
                    forecast,
                    w / 100,
                    (
                        terminal / 100
                        if method == "Perpetual Growth"
                        else terminal_rate
                    ),
                    (
                        terminal
                        if method == "Exit Multiple"
                        else exit_multiple
                    ),
                    method,
                    total_cash,
                    total_debt,
                    shares,
                    other_adjustments
                )

                value = test["Intrinsic Value"]

            except (ValueError, TypeError):

                value = np.nan

            sensitivity.loc[
                f"{w:.2f}%",
                column_names[j]
            ] = value

    def sensitivity_color(value):

        if pd.isna(value):
            return ""

        if value >= share_price:

            return (
                "background-color: #d7edda; "
                "color: #16451d"
            )

        return (
            "background-color: #f3dada; "
            "color: #701c1c"
        )

    styled = sensitivity.style.format(
        "{:,.2f}",
        na_rep="N/A"
    )

    styled = styled.map(sensitivity_color)

    st.dataframe(
        styled,
        use_container_width=True
    )

    st.caption(
        "Green indicates value at or above the current share price. "
        "Red indicates value below the current share price. "
        "Invalid valuation combinations display N/A."
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 10: SCENARIO ANALYSIS
    # --------------------------------------------------------

    st.markdown("### 10. Bear, Base & Bull Case Analysis")

    st.caption(
        "Adjustments are percentage-point changes applied "
        "to each year of the base-case forecast."
    )

    scenarios = pd.DataFrame({
        "Scenario": ["Bear Case", "Base Case", "Bull Case"],
        "Revenue Growth Adjustment": [-3.0, 0.0, 3.0],
        "EBIT Margin Adjustment": [-2.0, 0.0, 2.0],
        "WACC Adjustment": [1.0, 0.0, -1.0],
        "Terminal Growth Adjustment": [-0.5, 0.0, 0.5],
        "Exit Multiple Adjustment": [-2.0, 0.0, 2.0]
    })

    scenarios = st.data_editor(
        scenarios,
        disabled=["Scenario"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        key=f"{prefix}_scenarios"
    )

    scenario_results = []

    for _, row in scenarios.iterrows():

        scenario_forecast = forecast.copy()

        scenario_forecast["Revenue Growth %"] += (
            row["Revenue Growth Adjustment"]
        )

        scenario_forecast["EBIT Margin %"] += (
            row["EBIT Margin Adjustment"]
        )

        scenario_wacc = (
            wacc_percent +
            row["WACC Adjustment"]
        ) / 100

        scenario_terminal = (
            terminal_growth +
            row["Terminal Growth Adjustment"]
        ) / 100

        scenario_multiple = (
            exit_multiple +
            row["Exit Multiple Adjustment"]
        )

        try:

            scenario = calculate_dcf(
                revenue,
                scenario_forecast,
                scenario_wacc,
                scenario_terminal,
                scenario_multiple,
                method,
                total_cash,
                total_debt,
                shares,
                other_adjustments
            )

            value = scenario["Intrinsic Value"]

        except (ValueError, TypeError):

            value = np.nan

        scenario_results.append({
            "Scenario": row["Scenario"],
            "Intrinsic Value": value,
            "Upside / Downside (%)": (
                value / share_price - 1
            ) * 100,
            "Margin of Safety Price": (
                value * (1 - margin_safety / 100)
                if value > 0 else np.nan
            )
        })

    scenario_results = pd.DataFrame(scenario_results)

    st.dataframe(
        scenario_results.style.format({
            "Intrinsic Value": "{:,.2f}",
            "Upside / Downside (%)": "{:+.2f}%",
            "Margin of Safety Price": "{:,.2f}"
        }, na_rep="N/A"),
        use_container_width=True,
        hide_index=True
    )

    plot_chart(
        scenario_results.set_index("Scenario")["Intrinsic Value"],
        "Scenario Valuations",
        f"{currency} per Share",
        "Bar",
        True
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 11: MODEL DEFINITIONS
    # --------------------------------------------------------

    with st.expander("DCF Formulas & Model Assumptions"):

        st.markdown("""
### Free Cash Flow to the Firm

FCFF = NOPAT + D&A - CAPEX - Change in NWC

NOPAT = EBIT - Estimated Cash Taxes

### Cost of Equity

Cost of Equity = Risk-Free Rate + Beta × Equity Risk Premium

### Weighted Average Cost of Capital

WACC = Equity Weight × Cost of Equity
+ Debt Weight × Cost of Debt × (1 - Tax Rate)

### Perpetual Growth Terminal Value

Terminal Value = Final Year FCFF × (1 + Terminal Growth)
÷ (WACC - Terminal Growth)

### Exit Multiple Terminal Value

Terminal Value = Final Year EBITDA × Exit Multiple

### Enterprise Value

Enterprise Value = PV of Forecast Cash Flows
+ PV of Terminal Value

### Equity Value

Equity Value = Enterprise Value + Cash - Debt
- Minority Interest - Preferred Stock
+ Other Nonoperating Adjustments

### Intrinsic Value

Intrinsic Value = Equity Value ÷ Shares Outstanding

### Margin of Safety

Margin of Safety Price =
Intrinsic Value × (1 - Margin of Safety)

### Model Limitations

- Yahoo Finance data may be incomplete or delayed.
- All forecast assumptions are editable estimates.
- Historical revenue growth is not necessarily sustainable.
- Working capital investment is forecast as a percentage of revenue.
- Terminal growth must be lower than WACC.
- Terminal reinvestment must support long-term growth.
- Debt, cash and share counts may have different reporting dates.
- Debt book value is used as a proxy for market value.
- Current shares outstanding may not capture future dilution.
- Minority interests and preferred stock default to zero
  when unavailable and should be verified.
- The model does not separately forecast tax loss carryforwards.
- CAPEX and depreciation assumptions should reflect the
  company's long-term capital requirements.
- This model is generally unsuitable for valuing banks,
  insurers and other financial institutions.
- Intrinsic value is an estimate, not a guaranteed price target.
        """)

    # --------------------------------------------------------
    # SECTION 12: EXPORT DATA
    # --------------------------------------------------------

    st.markdown("### 11. Download DCF Analysis")

    c1, c2 = st.columns(2)

    with c1:

        download_csv(
            projection,
            f"{ticker}_DCF_Projections.csv",
            f"{prefix}_projection_csv"
        )

        download_csv(
            sensitivity,
            f"{ticker}_DCF_Sensitivity.csv",
            f"{prefix}_sensitivity_csv"
        )

        download_csv(
            forecast,
            f"{ticker}_DCF_Assumptions.csv",
            f"{prefix}_forecast_csv"
        )

    with c2:

        download_csv(
            bridge,
            f"{ticker}_DCF_Valuation.csv",
            f"{prefix}_bridge_csv"
        )

        download_csv(
            scenario_results,
            f"{ticker}_DCF_Scenarios.csv",
            f"{prefix}_scenario_csv"
        )

        model_inputs = pd.DataFrame({
            "Input": [
                "Ticker",
                "Reporting Period",
                "Financial Currency",
                "Revenue",
                "Share Price",
                "Market Capitalization",
                "Shares Outstanding",
                "Cash",
                "Debt",
                "Minority Interest",
                "Preferred Stock",
                "Other Adjustments",
                "Beta",
                "Risk-Free Rate (%)",
                "Equity Risk Premium (%)",
                "Cost of Debt (%)",
                "Tax Rate (%)",
                "WACC (%)",
                "Terminal Growth (%)",
                "Exit Multiple",
                "Terminal Method",
                "Margin of Safety (%)"
            ],
            "Value": [
                ticker,
                period,
                currency,
                revenue,
                share_price,
                market_cap,
                shares,
                total_cash,
                total_debt,
                minority,
                preferred,
                other_assets,
                beta,
                risk_free,
                equity_premium,
                cost_debt,
                tax_rate,
                wacc_percent,
                terminal_growth,
                exit_multiple,
                method,
                margin_safety
            ]
        })

        download_csv(
            model_inputs,
            f"{ticker}_DCF_Model_Inputs.csv",
            f"{prefix}_inputs_csv"
        )


# ============================================================
# PAGE 7: CHEAT SHEET
# ============================================================

def render_cheat_sheet():

    st.subheader("📋 Market Ratio Cheat Sheet")

    rows = []

    for category, ratios in RATIO_GROUPS.items():

        for name, symbols in ratios.items():

            a, b = symbols

            ratio = f"{a}/{b}"

            rows.append({
                "Category": category,
                "Ratio": ratio,
                "Description": name,
                "Interpretation": RATIO_INFO.get(
                    ratio,
                    "Rising indicates numerator outperformance."
                )
            })

    table = pd.DataFrame(rows)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    download_csv(
        table,
        "market_ratio_cheat_sheet.csv",
        "cheat_sheet_csv"
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📊 Charts to Watch")

st.caption(
    "Market Ratios | Performance | Fundamentals | "
    "Technicals | Company Research | DCF Valuation"
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
        "💰 DCF Valuation",
        "📋 Cheat Sheet"
    ],
    index=0
)

st.sidebar.markdown("---")

if st.sidebar.button("Refresh Yahoo Finance Data"):

    st.cache_data.clear()

    st.rerun()

st.sidebar.caption(
    "Financial and market data provided by Yahoo Finance."
)

st.sidebar.caption(
    "For investment research and educational purposes."
)


# ============================================================
# APPLICATION ROUTING
# ============================================================

if page == "📊 Ratio Dashboard":

    render_ratios()

elif page == "📈 Performance":

    render_performance()

elif page == "📑 Fundamentals":

    render_fundamentals()

elif page == "🧭 Technicals":

    render_technicals()

elif page == "🏢 Company Overview":

    render_company()

elif page == "💰 DCF Valuation":

    render_dcf()

elif page == "📋 Cheat Sheet":

    render_cheat_sheet()
