"""Equity curve and underwater drawdown plot."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak) - 1.0


def render(equity: pd.Series, *, benchmark: pd.Series | None = None) -> go.Figure:
    """Return a 2-row figure: equity on top, drawdown below."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=("Equity", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Strategy",
            line=dict(color="#1f4e79", width=1.6),
        ),
        row=1,
        col=1,
    )
    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                name="Benchmark",
                line=dict(color="#999999", width=1.2, dash="dot"),
            ),
            row=1,
            col=1,
        )
    dd = _drawdown(equity)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#762a83", width=1),
            fillcolor="rgba(118, 42, 131, 0.35)",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=40, r=40, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig
