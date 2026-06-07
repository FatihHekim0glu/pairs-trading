# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Polygon.io data provider** (`pairs.data_providers.polygon.PolygonProvider`):
  REST client with sliding-window token-bucket rate limiting (100 rpm Starter-tier
  ceiling), exponential backoff with jitter on 429/5xx/transient network errors,
  typed error hierarchy (`ProviderAuthError`, `ProviderRateLimitError`,
  `ProviderDataError`), and three endpoints: `get_eod`, `get_ticker_meta`, and
  `get_grouped_daily`. Always requests split- and dividend-adjusted bars.
- **yfinance fallback provider** (`pairs.data_providers.yfinance.YFinanceProvider`):
  mirrors the Polygon surface so the rest of the codebase can code against a single
  interface. Grouped-daily and ticker-meta raise `ProviderError` (no equivalent);
  `get_eod` delegates to the existing `pairs.data.load_prices` cache.
- **Environment-driven provider factory** (`pairs.data_providers.factory.make_provider`):
  returns Polygon when `POLYGON_API_KEY` is set (live or via environment),
  yfinance otherwise. Explicit `api_key` argument wins over the env var.
- **S&P 500 point-in-time universe** (`pairs.data_providers.sp500_universe.SP500UniverseBuilder`):
  intersects a snapshot of modern S&P 500 constituents with Polygon's grouped-daily
  list of tickers that actually traded on each as-of date. Drops names that were
  not yet trading; exposes `get_membership_as_of(date)` and
  `get_membership_window(start, end, freq="ME")` for month-end rebalance cadence.
- **Streamlit Pair Finder universe selector**: a new sidebar control
  ("Pair selection universe": Custom vs S&P 500 PIT) which uses the PIT builder
  when a Polygon key is configured, plus a result-panel header that shows two
  provenance badges -- `data: polygon|yfinance` and `universe: PIT|custom` --
  so every chart's data source is visible.
- 35 unit tests covering the provider, factory, yfinance fallback, and PIT
  universe builder; httpx is mocked throughout so the suite has no live network
  dependency.

### Changed

- `httpx>=0.27` added to the base runtime dependency list (required by the
  Polygon REST client).
- README: new "Data sources & universe (survivorship-bias-aware)" section
  explaining why pairs trading is the strategy class most sensitive to
  survivorship bias and how the Polygon + PIT universe combination mitigates it.

## [0.1.0] - 2026-05-23

### Added

- **Data layer**: vendor-agnostic price ingestion with point-in-time universe
  reconstruction, delisting-aware history, total-return adjustments for
  splits, dividends, and corporate actions, and a survivorship-safe symbol
  index keyed on the as-of date.
- **Cointegration tests**: Engle-Granger two-step procedure with MacKinnon
  (2010) critical values, Johansen trace and maximum-eigenvalue tests with
  Osterwald-Lenum tables, Phillips-Ouliaris residual test, and KPSS
  stationarity confirmation. All tests expose effective sample size,
  test statistic, and asymptotic p-value.
- **Spread construction, OU calibration, and Kalman filtering**: rolling
  hedge-ratio estimation via OLS and Kalman state-space, Ornstein-Uhlenbeck
  parameter fits returning mean, speed of reversion, diffusion, and
  half-life, plus residual normality and autocorrelation diagnostics.
- **Pair selection with multiple-testing correction**: candidate generation
  by sector and liquidity filters, family-wise error control via
  Benjamini-Hochberg FDR, optional Romano-Wolf stepdown, and a stability
  filter requiring cointegration in rolling sub-windows.
- **Strategy and backtest engine**: z-score entry and exit rules, stop-loss
  and time-stop overlays, transaction-cost model with commissions, slippage,
  spread, and borrow-fee accruals, mark-to-market position accounting, and
  event-driven order simulation with next-bar execution.
- **Portfolio overlay**: pair-level Kelly-fraction sizing capped by inverse
  variance, sector and gross-exposure constraints, and a daily turnover
  budget.
- **Out-of-sample evaluation protocol**: anchored walk-forward harness,
  Memmel-adjusted Sharpe ratios, Deflated Sharpe Ratio per Bailey and
  Lopez de Prado (2014), Newey-West standard errors, stationary bootstrap
  confidence intervals per Politis and Romano (1994), and Hansen (2005)
  superior predictive ability test.
- **Streamlit dashboard**: interactive pair browser, rolling cointegration
  panel, equity-curve overlay, drawdown visualizer, and exportable
  in-sample-versus-out-of-sample comparison chart.
- **Documentation**: methodology, limitations, references, interview
  preparation, and a worked broken-pair case study.

[Unreleased]: https://github.com/FatihHekim0glu/pairs-trading/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/pairs-trading/releases/tag/v0.1.0
