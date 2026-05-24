# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/noxire-dev/pairs-trading/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/noxire-dev/pairs-trading/releases/tag/v0.1.0
