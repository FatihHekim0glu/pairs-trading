"""Two-row spread and z-score plot with threshold lines and no-trade band."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render(
    spread: pd.Series,
    zscore: pd.Series,
    *,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    half_life: float | None = None,
    hedge_ratio: float | None = None,
) -> go.Figure:
    """Return a 2-row figure: spread on top, z-score with thresholds below."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.5],
        subplot_titles=("Spread", "Z-score"),
    )
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            name="Spread",
            line=dict(color="#1f4e79", width=1.4),
        ),
        row=1,
        col=1,
    )
    if hedge_ratio is not None:
        annotation = f"Hedge ratio beta = {hedge_ratio:.4f}"
        if half_life is not None:
            annotation += f" | Half-life = {half_life:.1f} days"
        fig.add_annotation(
            text=annotation,
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.02,
            showarrow=False,
            font=dict(size=11, color="#444"),
        )

    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            name="Z-score",
            line=dict(color="#762a83", width=1.4),
        ),
        row=2,
        col=1,
    )
    for level, label, color in [
        (z_entry, "+entry", "#b35806"),
        (-z_entry, "-entry", "#b35806"),
        (z_exit, "+exit", "#1b7837"),
        (-z_exit, "-exit", "#1b7837"),
        (z_stop, "+stop", "#a50026"),
        (-z_stop, "-stop", "#a50026"),
    ]:
        fig.add_hline(
            y=level,
            line=dict(color=color, width=1, dash="dash"),
            row=2,
            col=1,
            annotation_text=label,
            annotation_position="right",
        )
    fig.add_hrect(
        y0=-z_exit,
        y1=z_exit,
        fillcolor="#e0e0e0",
        opacity=0.35,
        line_width=0,
        row=2,
        col=1,
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=40, r=40, t=40, b=30),
        hovermode="x unified",
        showlegend=False,
    )
    return fig
