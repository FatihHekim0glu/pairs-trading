# pairs-trading

> **Now live as an interactive web tool at https://fatihhekimoglu-platform.vercel.app/tools/pairs-trading** — part of the fatihhekimoglu.com quantitative-tools platform. The Streamlit app here remains usable for local development; the hosted version uses the same compute library wrapped in a FastAPI backend.

Cointegration-based statistical arbitrage with walk-forward validation, Deflated Sharpe ratios, and an honest in-sample-vs-out-of-sample comparison. The Out-of-Sample number is the headline because that is what survives contact with the future.

![CI](https://github.com/FatihHekim0glu/pairs-trading/actions/workflows/ci.yml/badge.svg) ![Coverage](https://codecov.io/gh/FatihHekim0glu/pairs-trading/branch/main/graph/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![Ruff](https://img.shields.io/badge/code%20style-ruff-blue.svg) ![Mypy](https://img.shields.io/badge/types-mypy%20strict-blue.svg)

![In-sample vs Out-of-Sample Sharpe — pending generation by app/plots/is_vs_oos_bars.py](assets/money_chart.png)

## Why this project

Most public pairs-trading repositories report an in-sample Sharpe above 2 and stop there. This project tests hundreds of pairs with rolling cointegration, walk-forward validation, and a Deflated Sharpe Ratio that accounts for the multiple-testing problem; the median surviving pair drops sharply from in-sample to out-of-sample. The repository exists to make that decay visible, reproducible, and honest.

## 60-second Quickstart

```bash
git clone https://github.com/FatihHekim0glu/pairs-trading && cd pairs-trading
pip install -e ".[dev,app]"
pytest -q
streamlit run app/streamlit_app.py
```

## Live demo

- Streamlit Cloud: <pending deploy URL>
- Hugging Face Spaces (mirror): <pending>

## Methodology summary

- Data layer normalizes prices, dividends, and splits from a vendor-neutral interface; survivorship caveats are documented.
- Pre-screen pairs by sector membership, correlation floor, and minimum overlapping history before any cointegration test runs.
- Primary cointegration test is Engle-Granger; Johansen provides a confirming rank check; KPSS stationarity on the residual is the fourth-cell consistency check.
- Multiple-testing correction uses Benjamini-Hochberg FDR across all candidate pairs in each training window.
- Spread dynamics modelled as an Ornstein-Uhlenbeck process; half-life estimated via AR(1) regression on the residual.
- Signal rules use z-score entry and exit thresholds calibrated on training data only; a hard stop fires on half-life blowout.
- Backtest uses next-bar execution, configurable commission and slippage profiles, and explicit short-borrow cost ledgers.
- Portfolio overlay sizes positions by equal-vol target with a per-pair capital cap and a portfolio-level gross exposure cap.
- Out-of-sample evaluation uses anchored walk-forward windows plus Combinatorial Purged Cross-Validation.
- Headline metric is the Deflated Sharpe Ratio with Probability of Backtest Overfitting (PBO) reported beside it.

## Data sources & universe (survivorship-bias-aware)

Pairs trading is the strategy class most sensitive to survivorship bias because the trade thesis is *relative*: any pair containing a name that was later acquired or delisted at a wide spread looks like a clean mean-reversion winner in hindsight. Empirically, roughly one in ten pairs in any five-year window contains such a name, which biases reported Sharpes upward by a non-trivial margin and silently inflates win rates in the long-tail months.

The repo now ships with a two-tier data layer that lets you opt into an honest backtest without changing call sites:

- `pairs.data_providers.PolygonProvider` -- REST client for Polygon.io with token-bucket rate limiting (100 rpm Starter tier), exponential backoff on 429/5xx, and three endpoints: `get_eod`, `get_ticker_meta`, and crucially `get_grouped_daily`. The grouped-daily endpoint returns every ticker that *actually* traded on a given historical calendar date -- delisted and acquired names included. That is the single piece of data the free yfinance feed cannot give you, and it is what makes a point-in-time universe possible at this price point.
- `pairs.data_providers.YFinanceProvider` -- mirrors the same surface and delegates to the existing parquet cache so local dev keeps working without a key. Grouped-daily and ticker-meta raise `ProviderError` because yfinance has no equivalent.
- `pairs.data_providers.make_provider()` -- environment-driven factory. Returns Polygon when `POLYGON_API_KEY` is set, yfinance otherwise. The rest of the codebase depends on the abstract surface, not the concrete class.
- `pairs.data_providers.SP500UniverseBuilder` -- intersects a snapshot of modern S&P 500 constituents with Polygon's grouped-daily list at each as-of date. Members who were not trading on that date (pre-IPO, post-delisting) are dropped; `get_membership_window(start, end, freq="ME")` returns one list per month-end so the request count scales with rebalances, not calendar days.

The Streamlit Pair Finder exposes a "Pair selection universe" toggle (Custom vs S&P 500 PIT) and prints two provenance badges in the result panel -- `data: polygon|yfinance` and `universe: PIT|custom` -- so the source of every number on screen is visible. PIT is the right default for any honest research run; Custom remains useful for narrow sector studies (`xlk_v1`, `curated_25_v1`).

To enable the Polygon path:

```bash
export POLYGON_API_KEY=...   # Starter tier is sufficient
streamlit run app/streamlit_app.py
```

## Honest limitations

- Survivorship bias is mitigated by adding delisted tickers where available, but the free `yfinance` tier excludes most delistings; results on the live universe overstate the truth for any backtest before 2010.
- Transaction costs are modelled as flat per-trade commission plus a square-root impact term; small-cap reality may be 2-3x worse and that gap is not captured.
- Capacity is untested above small-AUM ranges; spread crossing latency and queue position effects are out of scope.
- Regime decay is visible in 2020 (COVID volatility) and 2022 (rate shock); both are documented in the case study.
- Broken-pair tail risk is asymmetric: a kill switch caps the left tail but cannot eliminate it (see `docs/broken_pair_case_study.md`).
- Short-borrow rates are assumed flat per ticker; hard-to-borrow names are not separately modelled, so any pair involving a recall-prone short overstates returns.

## References

1. Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
2. Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4).
3. Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
4. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
5. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*, 19(3).
6. Engle, R. F., & Granger, C. W. J. (1987). Co-integration and Error Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2).
7. Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models. *Econometrica*, 59(6).
8. Krauss, C. (2017). Statistical Arbitrage Pairs Trading Strategies: Review and Outlook. *Journal of Economic Surveys*, 31(2).
9. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

Extended bibliography in [`docs/references.md`](docs/references.md).

## How to cite

If you use this work, please cite via [`CITATION.cff`](CITATION.cff).

## Contact

GitHub: [@FatihHekim0glu](https://github.com/FatihHekim0glu)
