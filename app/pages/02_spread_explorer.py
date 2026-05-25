"""Spread Explorer page: interactive z-score panel inside a Streamlit fragment."""

from __future__ import annotations

import streamlit as st

from app.components import disclaimer, footer, header, methodology
from app.state import init_state

init_state()

PAGE_TITLE = "Spread Explorer"

FALLBACK_PAIR = ("KO", "PEP")


def _pair_options() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    result = st.session_state.get("last_scan_result")
    if result is not None:
        survivors = getattr(result, "survivors", None)
        if survivors is not None:
            for item in survivors:
                if isinstance(item, tuple) and len(item) == 2:
                    options.append((str(item[0]), str(item[1])))
                elif hasattr(item, "a") and hasattr(item, "b"):
                    options.append((str(item.a), str(item.b)))
    if not options:
        options.append(FALLBACK_PAIR)
    return options


@st.fragment
def _spread_panel(pair: tuple[str, str]) -> None:
    c1, c2, c3 = st.columns(3)
    z_entry = c1.slider(
        "Entry threshold", 1.0, 4.0, st.session_state.z_entry, 0.1, key="z_entry_slider"
    )
    z_exit = c2.slider(
        "Exit threshold", 0.0, 2.0, st.session_state.z_exit, 0.1, key="z_exit_slider"
    )
    lookback = c3.slider(
        "Lookback (days)", 60, 504, st.session_state.lookback_days, 21, key="lookback_slider"
    )
    st.session_state.z_entry = z_entry
    st.session_state.z_exit = z_exit
    st.session_state.lookback_days = lookback

    try:
        from app.cache import compute_spread_cached
        from app.plots import spread_zscore

        bundle = compute_spread_cached(pair, lookback)
        fig = spread_zscore.render(
            bundle["spread"],
            bundle["zscore"],
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=st.session_state.z_stop,
            half_life=bundle.get("half_life"),
            hedge_ratio=bundle.get("hedge_ratio"),
        )
        st.plotly_chart(fig, use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Hedge ratio (beta)", f"{bundle.get('hedge_ratio', float('nan')):.4f}")
        m2.metric("Half-life (days)", f"{bundle.get('half_life', float('nan')):.1f}")
        latest_z = (
            float(bundle["zscore"].dropna().iloc[-1])
            if len(bundle["zscore"].dropna())
            else float("nan")
        )
        m3.metric("Current z-score", f"{latest_z:.2f}")
    except ImportError:
        st.error(
            "Module pairs.spread not available -- install [app] extras and run "
            "`pip install -e .[app,dev]`"
        )
    except Exception as exc:
        st.error(f"Failed to compute spread: {exc}")


def main() -> None:
    header.render(PAGE_TITLE)
    disclaimer.render()
    options = _pair_options()
    labels = [f"{a} / {b}" for a, b in options]
    default_idx = 0
    if st.session_state.selected_pair in options:
        default_idx = options.index(st.session_state.selected_pair)
    choice = st.selectbox("Pair", labels, index=default_idx)
    pair = options[labels.index(choice)]
    st.session_state.selected_pair = pair

    _spread_panel(pair)

    methodology.render("spread_zscore")

    st.subheader("Underlying prices")
    try:
        from app.cache import fetch_prices
        from app.plots import price_overlay

        end_date = st.session_state.oos_end
        start_date = st.session_state.train_start
        prices = fetch_prices(pair, start_date, end_date)
        if hasattr(prices, "columns") and pair[0] in prices.columns and pair[1] in prices.columns:
            fig = price_overlay.render(prices[pair[0]], prices[pair[1]])
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Price overlay unavailable until pairs.data is installed.")
    except Exception as exc:
        st.warning(f"Could not render price overlay: {exc}")

    footer.render()


main()
