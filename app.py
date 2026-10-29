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
