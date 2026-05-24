"""Methodology expander with per-chart explanations."""

from __future__ import annotations

import streamlit as st

TEMPLATES: dict[str, str] = {
    "pair_finder": (
        "Pairs are screened with the Engle-Granger and Johansen tests on the training "
        "window. The Engle-Granger residual is also tested with the Phillips-Ouliaris "
        "approach for robustness. P-values are adjusted across the universe using the "
        "Benjamini-Hochberg false discovery rate to produce q-values. Pairs are "
        "filtered by half-life bounds and a minimum mean-reversion strength."
    ),
    "spread_zscore": (
        "The spread is constructed as log(A) - beta * log(B) with beta estimated via "
        "total least squares on the training window to avoid OLS attenuation bias. "
        "The rolling z-score uses the in-sample mean and standard deviation. Dashed "
        "horizontal lines mark entry, exit, and stop thresholds. The shaded band is "
        "the no-trade zone where the spread is considered statistically uninformative."
    ),
    "is_vs_oos": (
        "Each pair is fit and trained in-sample, then evaluated on a held-out "
        "out-of-sample window with no parameter refitting. A large gap between IS and "
        "OOS Sharpe is the canonical sign of overfitting. The dashed reference line at "
        "Sharpe = 1.0 is a common institutional threshold for systematic strategies."
    ),
    "drawdown": (
        "Drawdown is the percentage decline from the running peak of the equity curve. "
        "It is reported on a per-strategy basis with no benchmark deduction. The chart "
        "highlights the worst trough and the recovery duration, both critical inputs "
        "into Kelly sizing and risk-of-ruin calculations."
    ),
    "portfolio_equity": (
        "Portfolio equity aggregates per-pair PnL under the selected weighting scheme "
        "(equal-dollar, inverse-volatility, or hierarchical risk parity). Costs and "
        "slippage are applied per pair before aggregation. Rebalancing is performed at "
        "the configured frequency without lookahead. The drawdown panel reflects "
        "compounded portfolio returns net of all frictions."
    ),
}


def render(chart_id: str) -> None:
    """Render the methodology expander for a given chart identifier."""
    body = TEMPLATES.get(chart_id, "Methodology notes not yet available for this view.")
    with st.expander("How this is computed"):
        st.markdown(body)
