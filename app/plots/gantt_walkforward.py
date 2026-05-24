"""Horizontal bar Gantt chart for walk-forward train/test windows."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _coerce_to_frame(walk_forward_result: Any) -> pd.DataFrame:
    if isinstance(walk_forward_result, pd.DataFrame):
        return walk_forward_result
    if hasattr(walk_forward_result, "to_frame"):
        return walk_forward_result.to_frame()
    if hasattr(walk_forward_result, "windows"):
        return pd.DataFrame(walk_forward_result.windows)
    return pd.DataFrame(walk_forward_result)


def render(walk_forward_result: Any) -> go.Figure:
    """Return a horizontal bar chart of train/test windows over time."""
    df = _coerce_to_frame(walk_forward_result).copy()
    required = {"start", "end", "split", "fold"}
    missing = required - set(df.columns)
    if missing:
        for col in missing:
            df[col] = None
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="fold",
        color="split",
        color_discrete_map={"train": "#1f4e79", "test": "#1b7837"},
    )
    fig.update_yaxes(autorange="reversed", title="Fold")
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig
