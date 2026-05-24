"""Scan-result table renderer (AgGrid preferred, dataframe fallback)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _style_qvalue(df: pd.DataFrame) -> Any:
    if "q_value" not in df.columns:
        return df
    styler = df.style.format(dict.fromkeys(df.select_dtypes("number").columns, "{:.4f}"))

    def _color(val: float) -> str:
        if pd.isna(val):
            return ""
        if val < 0.01:
            return "background-color: #c7e9c0; color: #00441b"
        if val < 0.05:
            return "background-color: #f7fcb9"
        return "background-color: #fee0d2"

    # `Styler.applymap` was renamed to `Styler.map` in pandas 2.1 and removed
    # in pandas 3.0. Use the modern name unconditionally.
    return styler.map(_color, subset=["q_value"])


def _try_aggrid(df: pd.DataFrame) -> str | None:
    try:
        from st_aggrid import AgGrid, GridOptionsBuilder
    except Exception:
        return None
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_selection("single", use_checkbox=False)
    for col in df.select_dtypes("number").columns:
        gb.configure_column(col, type=["numericColumn"], valueFormatter="(value).toFixed(4)")
    grid_options = gb.build()
    response = AgGrid(
        df,
        gridOptions=grid_options,
        height=440,
        theme="streamlit",
        update_mode="SELECTION_CHANGED",
        allow_unsafe_jscode=False,
    )
    selected = response.get("selected_rows") if isinstance(response, dict) else None
    if selected is not None and len(selected) > 0:
        row = selected[0] if isinstance(selected, list) else selected.iloc[0].to_dict()
        ticker_a = row.get("ticker_a") or row.get("a") or row.get("leg_a")
        ticker_b = row.get("ticker_b") or row.get("b") or row.get("leg_b")
        if ticker_a and ticker_b:
            return f"{ticker_a}|{ticker_b}"
    return None


def render(diagnostics: pd.DataFrame) -> None:
    """Render the diagnostics table and update selected_pair on row click."""
    if diagnostics is None or diagnostics.empty:
        st.info("No scan results yet. Run the pair finder from the sidebar.")
        return
    selection = _try_aggrid(diagnostics)
    if selection is None:
        st.dataframe(_style_qvalue(diagnostics), use_container_width=True, height=440)
    else:
        a, b = selection.split("|", 1)
        st.session_state.selected_pair = (a, b)
        st.success(f"Selected pair: {a} / {b}")
