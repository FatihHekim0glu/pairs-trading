"""Pair Finder page: cointegration screen with filters and result table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import disclaimer, footer, header, methodology
from app.state import init_state

init_state()

PAGE_TITLE = "Pair Finder"


def _kpi_row(diagnostics: pd.DataFrame) -> None:
    cols = st.columns(4)
    total = 0 if diagnostics is None else len(diagnostics)
    survivors = 0
    median_hl = float("nan")
    median_q = float("nan")
    if diagnostics is not None and not diagnostics.empty:
        if "q_value" in diagnostics.columns:
            survivors = int((diagnostics["q_value"] < 0.05).sum())
            median_q = float(diagnostics["q_value"].median())
        if "half_life" in diagnostics.columns:
            median_hl = float(diagnostics["half_life"].median())
    cols[0].metric("Pairs tested", f"{total}")
    cols[1].metric("q < 0.05 survivors", f"{survivors}")
    cols[2].metric("Median half-life", f"{median_hl:.1f}" if pd.notna(median_hl) else "n/a")
    cols[3].metric("Median q-value", f"{median_q:.4f}" if pd.notna(median_q) else "n/a")


def _sidebar() -> dict:
    with st.sidebar:
        st.subheader("Screen parameters")
        with st.form("pair_finder_form"):
            universe = st.selectbox(
                "Universe",
                ["curated_25_v1", "xlk_v1"],
                index=0,
                key="universe_name",
            )
            train_start = st.date_input("Training start", key="train_start")
            train_end = st.date_input("Training end", key="train_end")
            hl_min, hl_max = st.slider(
                "Half-life bounds (days)",
                min_value=1,
                max_value=120,
                value=(5, 60),
                step=1,
            )
            p_threshold = st.number_input(
                "p-value threshold",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.005,
                format="%.3f",
            )
            fdr_method = st.radio(
                "Multiple-testing correction",
                ["benjamini_hochberg", "bonferroni", "none"],
                index=0,
            )
            submitted = st.form_submit_button("Run scan", use_container_width=True)
    return {
        "universe": universe,
        "train_start": train_start,
        "train_end": train_end,
        "hl_min": hl_min,
        "hl_max": hl_max,
        "p_threshold": p_threshold,
        "fdr_method": fdr_method,
        "submitted": submitted,
    }


def main() -> None:
    header.render(PAGE_TITLE)
    disclaimer.render()
    params = _sidebar()

    diagnostics: pd.DataFrame | None = None
    if params["submitted"]:
        try:
            from app.cache import run_screen
            from pairs.selection import ScreenResult  # noqa: F401

            params_hash = (
                f"hl{params['hl_min']}-{params['hl_max']}|"
                f"p{params['p_threshold']:.3f}|fdr{params['fdr_method']}"
            )
            with st.spinner("Running cointegration scan..."):
                result = run_screen(
                    params["universe"],
                    params["train_start"],
                    params["train_end"],
                    params_hash,
                )
            st.session_state.last_scan_result = result
            diagnostics = getattr(result, "diagnostics", None)
            if diagnostics is None and isinstance(result, pd.DataFrame):
                diagnostics = result
        except ImportError:
            st.error(
                "Module pairs.selection not available -- install [app] extras and run "
                "`pip install -e .[app,dev]`"
            )
        except Exception as exc:
            st.error(f"Scan failed: {exc}")

    if diagnostics is None and st.session_state.last_scan_result is not None:
        diagnostics = getattr(st.session_state.last_scan_result, "diagnostics", None)
        if diagnostics is None and isinstance(st.session_state.last_scan_result, pd.DataFrame):
            diagnostics = st.session_state.last_scan_result

    _kpi_row(diagnostics)
    st.subheader("Scan results")
    try:
        from app.plots import pair_scan_table

        pair_scan_table.render(diagnostics if diagnostics is not None else pd.DataFrame())
    except ImportError:
        st.error("Plot module unavailable.")

    methodology.render("pair_finder")
    footer.render()


main()
