"""Pair Finder page: cointegration screen with filters and result table.

Honest-backtest infrastructure
------------------------------
The sidebar exposes a "Pair selection universe" choice (Custom vs S&P 500 PIT).
The PIT option is only useful when a Polygon API key is configured -- it uses
:class:`pairs.data_providers.SP500UniverseBuilder` to intersect the modern S&P
500 list with names that actually traded on the training-window start, so
delisted/acquired tickers are not silently overrepresented (a well-known source
of survivorship bias in pairs-trading research). The result panel surfaces two
badges -- ``data:`` and ``universe:`` -- so the source of every number is
visible.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.components import disclaimer, footer, header, methodology
from app.state import init_state

init_state()

PAGE_TITLE = "Pair Finder"


def _polygon_key_present() -> bool:
    """Return True iff a usable Polygon key is configured in the environment."""
    return bool(os.environ.get("POLYGON_API_KEY", "").strip())


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
    polygon_available = _polygon_key_present()
    with st.sidebar:
        st.subheader("Screen parameters")
        with st.form("pair_finder_form"):
            universe_mode_options = ["Custom", "S&P 500 PIT"]
            universe_mode = st.radio(
                "Pair selection universe",
                universe_mode_options,
                index=0,
                help=(
                    "S&P 500 PIT (point-in-time) requires POLYGON_API_KEY. It "
                    "intersects the modern S&P 500 list with names that actually "
                    "traded on the training-window start so delisted/acquired "
                    "tickers are not silently overrepresented."
                ),
                key="universe_mode",
            )
            if universe_mode == "S&P 500 PIT" and not polygon_available:
                st.warning(
                    "POLYGON_API_KEY not set -- the S&P 500 PIT universe will fall "
                    "back to the Custom universe for this run."
                )
            universe = st.selectbox(
                "Custom universe",
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
    effective_universe_mode = (
        "S&P 500 PIT" if universe_mode == "S&P 500 PIT" and polygon_available else "Custom"
    )
    return {
        "universe": universe,
        "universe_mode": effective_universe_mode,
        "requested_universe_mode": universe_mode,
        "polygon_available": polygon_available,
        "train_start": train_start,
        "train_end": train_end,
        "hl_min": hl_min,
        "hl_max": hl_max,
        "p_threshold": p_threshold,
        "fdr_method": fdr_method,
        "submitted": submitted,
    }


def _provenance_badges(params: dict) -> None:
    """Render the two-line ``data:`` / ``universe:`` badges in the result panel."""
    data_source = "polygon" if params.get("polygon_available") else "yfinance"
    universe_kind = (
        "PIT" if params.get("universe_mode") == "S&P 500 PIT" else "custom"
    )
    cols = st.columns(2)
    cols[0].markdown(f"**data:** `{data_source}`")
    cols[1].markdown(f"**universe:** `{universe_kind}`")


def _maybe_resolve_pit_universe(params: dict) -> list[str] | None:
    """Return the as-of S&P 500 list when PIT is selected and a key is set; else None."""
    if params.get("universe_mode") != "S&P 500 PIT":
        return None
    try:
        from datetime import date

        from pairs.data_providers import SP500UniverseBuilder, make_provider

        provider = make_provider()
        # The factory always returns *something*; only PolygonProvider supports PIT.
        if not params.get("polygon_available"):
            return None
        builder = SP500UniverseBuilder(provider=provider)  # type: ignore[arg-type]
        anchor = params["train_start"]
        if not isinstance(anchor, date):
            anchor = date.today()
        return builder.get_membership_as_of(anchor)
    except Exception as exc:  # noqa: BLE001 -- demo UX must never crash on PIT lookup
        st.warning(f"PIT universe lookup failed; falling back to Custom. ({exc})")
        return None


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

    _provenance_badges(params)
    # PIT universe is resolved on-demand so the lookup cost is paid only when
    # the user picks that mode. The returned list is stashed on session state
    # for downstream consumers (Spread Explorer, Backtest) to discover.
    pit_members = _maybe_resolve_pit_universe(params)
    if pit_members is not None:
        st.session_state["sp500_pit_members"] = pit_members
        st.caption(f"S&P 500 PIT members at training start: {len(pit_members)} names")
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
