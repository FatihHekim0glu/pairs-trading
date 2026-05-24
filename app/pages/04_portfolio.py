"""Portfolio page: multi-pair aggregation, weighting, and correlation heatmap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import disclaimer, footer, header, methodology
from app.state import init_state

init_state()

PAGE_TITLE = "Portfolio"


def _surviving_pairs() -> list[tuple[str, str]]:
    survivors: list[tuple[str, str]] = []
    result = st.session_state.get("last_scan_result")
    if result is not None:
        items = getattr(result, "survivors", None) or getattr(result, "pairs", None)
        if items is not None:
            for item in items:
                if isinstance(item, tuple) and len(item) == 2:
                    survivors.append((str(item[0]), str(item[1])))
                elif hasattr(item, "a") and hasattr(item, "b"):
                    survivors.append((str(item.a), str(item.b)))
    if not survivors:
        survivors = [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("MSFT", "ORCL")]
    return survivors


def _aggregate_kpis(result) -> None:
    metrics = {
        "Portfolio Sharpe": _safe_get(result, ["sharpe", "portfolio_sharpe"]),
        "Annual return": _safe_get(result, ["annual_return", "cagr"], pct=True),
        "Max drawdown": _safe_get(result, ["max_drawdown", "mdd"], pct=True),
        "Annual vol": _safe_get(result, ["annual_vol", "volatility"], pct=True),
    }
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items(), strict=False):
        col.metric(label, value)


def _safe_get(result, names, pct: bool = False) -> str:
    for name in names:
        val = None
        if hasattr(result, name):
            val = getattr(result, name)
        elif isinstance(result, dict) and name in result:
            val = result[name]
        if val is not None:
            try:
                fval = float(val)
                return f"{fval:.1%}" if pct else f"{fval:.2f}"
            except (TypeError, ValueError):
                continue
    return "n/a"


def main() -> None:
    header.render(PAGE_TITLE)
    disclaimer.render()

    candidates = _surviving_pairs()
    labels = [f"{a}/{b}" for a, b in candidates]
    selected = st.multiselect(
        "Pairs to include",
        options=labels,
        default=labels,
    )
    chosen_pairs = [candidates[labels.index(lbl)] for lbl in selected]

    c1, c2 = st.columns(2)
    weighting = c1.radio(
        "Weighting scheme",
        ["equal", "inverse_vol", "hrp"],
        index=0,
        horizontal=True,
    )
    rebalance = c2.selectbox(
        "Rebalance frequency",
        ["monthly", "quarterly", "annual"],
        index=0,
    )

    run = st.button("Run portfolio backtest", type="primary")
    result = None
    if run and chosen_pairs:
        try:
            from app.cache import run_portfolio_cached

            result = run_portfolio_cached(
                tuple(chosen_pairs),
                weighting,
                rebalance,
                st.session_state.oos_start,
                st.session_state.oos_end,
                st.session_state.cost_profile,
            )
        except ImportError:
            st.error(
                "Module pairs.portfolio not available -- install [app] extras and run "
                "`pip install -e .[app,dev]`"
            )
        except Exception as exc:
            st.error(f"Portfolio backtest failed: {exc}")

    if result is None:
        st.info("Select pairs and run the portfolio backtest to view aggregate results.")
        footer.render()
        return

    _aggregate_kpis(result)

    equity = getattr(result, "equity_curve", None)
    if equity is None and isinstance(result, dict):
        equity = result.get("equity_curve")
    if equity is not None:
        st.subheader("Portfolio equity and drawdown")
        try:
            from app.plots import equity_drawdown

            st.plotly_chart(equity_drawdown.render(equity), use_container_width=True)
        except ImportError:
            st.warning("Equity plot unavailable.")

    per_pair_pnl = getattr(result, "per_pair_pnl", None) or {}
    oos_sharpe = getattr(result, "per_pair_sharpe", None) or {}
    if per_pair_pnl:
        st.subheader("Pair contribution")
        try:
            from app.plots import pair_contribution

            st.plotly_chart(
                pair_contribution.render(per_pair_pnl, oos_sharpe),
                use_container_width=True,
            )
        except ImportError:
            st.warning("Pair contribution plot unavailable.")

    returns_df = getattr(result, "pair_returns", None)
    if isinstance(returns_df, pd.DataFrame) and not returns_df.empty:
        st.subheader("Pair return correlation")
        corr = returns_df.corr().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        heatmap = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            origin="lower",
            aspect="auto",
        )
        heatmap.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=40, r=40, t=30, b=30),
            coloraxis_colorbar=dict(title="rho"),
        )
        st.plotly_chart(heatmap, use_container_width=True)

    methodology.render("portfolio_equity")
    footer.render()


main()
