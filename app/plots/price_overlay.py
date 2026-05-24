"""Dual-axis price overlay for the two legs of a pair."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(prices_a: pd.Series, prices_b: pd.Series) -> go.Figure:
    """Return a dual-y, shared-x price overlay with weekend rangebreaks."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    name_a = prices_a.name or "A"
    name_b = prices_b.name or "B"
    fig.add_trace(
        go.Scatter(
            x=prices_a.index,
            y=prices_a.values,
            name=str(name_a),
            line=dict(color="#1f4e79", width=1.4),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=prices_b.index,
            y=prices_b.values,
            name=str(name_b),
            line=dict(color="#1b7837", width=1.4),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text=str(name_a), secondary_y=False)
    fig.update_yaxes(title_text=str(name_b), secondary_y=True)
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=40, t=30, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig
