"""In-sample vs out-of-sample Sharpe bar chart (the money chart)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _build_figure(is_oos_df: pd.DataFrame) -> go.Figure:
    pair_col = "pair" if "pair" in is_oos_df.columns else is_oos_df.columns[0]
    is_col = "is_sharpe" if "is_sharpe" in is_oos_df.columns else "IS"
    oos_col = "oos_sharpe" if "oos_sharpe" in is_oos_df.columns else "OOS"
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=is_oos_df[pair_col],
            y=is_oos_df[is_col],
            name="IS Sharpe",
            marker_color="#1f4e79",
        )
    )
    fig.add_trace(
        go.Bar(
            x=is_oos_df[pair_col],
            y=is_oos_df[oos_col],
            name="OOS Sharpe",
            marker_color="#1b7837",
        )
    )
    fig.add_hline(
        y=1.0,
        line=dict(color="#444", width=1, dash="dash"),
        annotation_text="institutional threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        barmode="group",
        title="In-Sample vs Out-of-Sample Sharpe - robustness check",
        xaxis_title="Pair",
        yaxis_title="Sharpe",
        margin=dict(l=40, r=40, t=60, b=60),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def render(is_oos_df: pd.DataFrame) -> go.Figure:
    """Return the IS vs OOS Sharpe bar chart for the supplied dataframe."""
    return _build_figure(is_oos_df)


def export_png(
    is_oos_df: pd.DataFrame,
    path: str | Path,
    width: int = 1280,
    height: int = 640,
) -> Path:
    """Export the figure to a PNG using kaleido and return the path."""
    fig = _build_figure(is_oos_df)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(target), width=width, height=height, format="png")
    return target
