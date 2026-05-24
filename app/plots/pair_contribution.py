"""Per-pair PnL contribution bar with viridis colouring by OOS Sharpe."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def render(
    per_pair_pnl: dict[str, float],
    oos_sharpe: dict[str, float],
) -> go.Figure:
    """Return a horizontal bar chart of per-pair PnL coloured by OOS Sharpe."""
    df = pd.DataFrame(
        {
            "pair": list(per_pair_pnl.keys()),
            "pnl": list(per_pair_pnl.values()),
            "sharpe": [oos_sharpe.get(p, 0.0) for p in per_pair_pnl],
        }
    ).sort_values("pnl")
    fig = go.Figure(
        go.Bar(
            x=df["pnl"],
            y=df["pair"],
            orientation="h",
            marker=dict(
                color=df["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="OOS Sharpe"),
                cmin=float(df["sharpe"].min()) if not df.empty else 0.0,
                cmax=float(df["sharpe"].max()) if not df.empty else 1.0,
            ),
            hovertemplate="%{y}<br>PnL: %{x:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(320, 22 * len(df) + 80),
        margin=dict(l=80, r=40, t=30, b=30),
        xaxis_title="PnL contribution",
        yaxis_title="Pair",
    )
    return fig
