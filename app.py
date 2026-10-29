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
st.set_page_config(page_title="Charts to Watch", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container { max-width: 1440px; padding-top: 2rem; }
h1 { letter-spacing: -0.04em; }
h2, h3 { letter-spacing: -0.02em; }
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #dce5eb;
    border-radius: 12px; padding: 1rem;
}
[data-testid="stSidebar"] { border-right: 1px solid #dce5eb; }
[data-testid="stSidebar"] h1 { font-size: 1.5rem; }
div.stButton > button[kind="primary"] { border-radius: 8px; }
@media (max-width: 640px) {
    .block-container { padding: 1rem; }
    [data-testid="stMetric"] { padding: 0.65rem; }
}
</style>
""", unsafe_allow_html=True)
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#475569", "text.color": "#172b3a",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.prop_cycle": plt.cycler(color=["#0f766e", "#2563eb", "#d97706", "#9333ea", "#e11d48"]),
})

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
