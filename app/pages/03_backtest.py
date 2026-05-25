"""Backtest page: single-pair OOS backtest with KPI row and money chart."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import disclaimer, footer, header, methodology
from app.state import init_state

init_state()

PAGE_TITLE = "Backtest"


def _form() -> dict:
    with st.form("backtest_form"):
        pair_default = st.session_state.selected_pair or ("KO", "PEP")
        c1, c2 = st.columns(2)
        leg_a = c1.text_input("Leg A", value=pair_default[0])
        leg_b = c2.text_input("Leg B", value=pair_default[1])
        c3, c4 = st.columns(2)
        oos_start = c3.date_input("OOS start", key="oos_start")
        oos_end = c4.date_input("OOS end", key="oos_end")
        cost_profile = st.selectbox(
            "Cost profile",
            [
                "large_cap_optimistic",
                "large_cap_realistic",
                "mid_cap_realistic",
                "small_cap_conservative",
            ],
            index=1,
            key="cost_profile",
        )
        slippage = st.slider(
            "Slippage (bps)",
            min_value=0,
            max_value=25,
            value=int(st.session_state.slippage_bps),
            step=1,
            key="slippage_bps",
        )
        sizing = st.radio(
            "Position sizing",
            ["fixed_notional", "vol_target", "kelly_capped"],
            index=0,
            horizontal=True,
        )
        submitted = st.form_submit_button("Run backtest", use_container_width=True)
    return {
        "pair": (leg_a.strip().upper(), leg_b.strip().upper()),
        "oos_start": oos_start,
        "oos_end": oos_end,
        "cost_profile": cost_profile,
        "slippage": slippage,
        "sizing": sizing,
        "submitted": submitted,
    }


def _extract_metric(result, *names, default=float("nan")) -> float:
    for name in names:
        if hasattr(result, name):
            val = getattr(result, name)
            if val is not None:
                return float(val)
        if isinstance(result, dict) and name in result:
            return float(result[name])
    return default


def _kpis(result) -> None:
    is_sharpe = _extract_metric(result, "is_sharpe", "in_sample_sharpe")
    oos_sharpe = _extract_metric(result, "oos_sharpe", "out_of_sample_sharpe", "sharpe")
    max_dd = _extract_metric(result, "max_drawdown", "mdd")
    turnover = _extract_metric(result, "turnover", "annual_turnover")
    cols = st.columns(4)
    cols[0].metric("OOS Sharpe", f"{oos_sharpe:.2f}")
    delta = (
        f"{oos_sharpe - is_sharpe:+.2f} vs IS"
        if pd.notna(is_sharpe) and pd.notna(oos_sharpe)
        else None
    )
    cols[1].metric("IS Sharpe", f"{is_sharpe:.2f}", delta=delta, delta_color="inverse")
    cols[2].metric("Max drawdown", f"{max_dd:.1%}" if pd.notna(max_dd) else "n/a")
    cols[3].metric("Turnover", f"{turnover:.1f}x" if pd.notna(turnover) else "n/a")


def main() -> None:
    header.render(PAGE_TITLE)
    disclaimer.render()
    params = _form()

    result = st.session_state.last_backtest_result
    if params["submitted"]:
        try:
            from app.cache import run_backtest_cached

            with st.spinner("Running backtest..."):
                result = run_backtest_cached(
                    params["pair"],
                    params["oos_start"],
                    params["oos_end"],
                    params["cost_profile"],
                    params["sizing"],
                )
            st.session_state.last_backtest_result = result
            st.session_state.selected_pair = params["pair"]
        except ImportError:
            st.error(
                "Module pairs.backtest not available -- install [app] extras and run "
                "`pip install -e .[app,dev]`"
            )
            return
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            return

    if result is None:
        st.info("Configure the form and click Run backtest.")
        footer.render()
        return

    _kpis(result)

    is_sharpe = _extract_metric(result, "is_sharpe", "in_sample_sharpe")
    oos_sharpe = _extract_metric(result, "oos_sharpe", "out_of_sample_sharpe", "sharpe")
    if pd.notna(is_sharpe) and pd.notna(oos_sharpe):
        gap = is_sharpe - oos_sharpe
        if gap > 1.0:
            st.warning(f"IS-OOS Sharpe gap is {gap:+.2f}. This is a strong overfitting signal.")
        elif gap > 0.5:
            st.info(f"IS-OOS Sharpe gap is {gap:+.2f}. Mild degradation observed.")

    st.subheader("In-sample vs out-of-sample Sharpe")
    try:
        from app.plots import is_vs_oos_bars

        pair_label = f"{params['pair'][0]}/{params['pair'][1]}"
        df = pd.DataFrame(
            {
                "pair": [pair_label],
                "is_sharpe": [is_sharpe],
                "oos_sharpe": [oos_sharpe],
            }
        )
        st.plotly_chart(is_vs_oos_bars.render(df), use_container_width=True)
    except ImportError:
        st.warning("IS/OOS plot unavailable.")

    st.subheader("Equity and drawdown")
    equity = getattr(result, "equity_curve", None)
    if equity is None and isinstance(result, dict):
        equity = result.get("equity_curve")
    if equity is not None:
        try:
            from app.plots import equity_drawdown

            st.plotly_chart(equity_drawdown.render(equity), use_container_width=True)
        except ImportError:
            st.warning("Equity plot unavailable.")
    else:
        st.caption("Equity curve not returned by backtest engine.")

    st.subheader("Trades")
    trades = getattr(result, "trades", None)
    if trades is None and isinstance(result, dict):
        trades = result.get("trades")
    if trades is not None:
        df_trades = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder

            gb = GridOptionsBuilder.from_dataframe(df_trades)
            gb.configure_default_column(filter=True, sortable=True, resizable=True)
            AgGrid(df_trades, gridOptions=gb.build(), height=320, theme="streamlit")
        except Exception:
            st.dataframe(df_trades, use_container_width=True, height=320)
    else:
        st.caption("No trade log returned.")

    methodology.render("is_vs_oos")
    methodology.render("drawdown")
    footer.render()


main()
