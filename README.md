# QQQ Options System

Phase 0/1 tooling for local ORATS SMV strike files. The current focus is QQQ-only data audit and pilot parquet normalization for June 2026.

## Layout

```text
qqq_options_system/
  configs/qqq_pilot.json
  src/qqq_options_system/
    data/
    market/
  tests/
  outputs/
    data/raw_manifest/
    data/silver_options/
    reports/
```

Source CSVs stay on the external drive. All generated artifacts are written under this workspace by default.

## Setup

Use Python 3.9+ with pandas and pyarrow:

```bash
cd /Users/cyrus.dong/Documents/Codex/2026-07-02/qqq-options-implementation/qqq_options_system
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For ad hoc runs without installing the package, set `PYTHONPATH=src`.

## Inventory

Header/schema inventory scans file metadata and headers only unless `--row-count` is passed.

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system inventory \
  --start 2026-06-01 \
  --end 2026-06-30
```

Output:

```text
outputs/data/raw_manifest/orats_inventory.csv
```

## QQQ Daily Audit

The default config is intentionally scoped to the pilot range `2026-06-01` through `2026-06-30`; it does not scan the full 562 GB data store.

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system audit
```

Output:

```text
outputs/reports/qqq_daily_audit_20260601_20260630.csv
```

Each row reports QQQ row count, min/max expiry, min/max strike, schema version, and trade-date mismatches versus the source filename.

## Pilot QQQ Parquet Conversion

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system convert
```

Output partitions:

```text
outputs/data/silver_options/symbol=QQQ/year=YYYY/trade_date=YYYY-MM-DD/option_chain_eod.parquet
```

The normalized long table includes:

```text
trade_date, underlying, expiry, dte, right, strike, contract_id,
opra_alias, bid, mid, ask, volume, open_interest, bid_iv, mid_iv,
ask_iv, smv_vol, delta, gamma, theta, vega, rho, theo,
underlying_price, spot_px, source_file, schema_version
```

`contract_id` is constructed from underlying, expiry, right, and strike, so early files without `cOpra`/`pOpra` remain usable. `stkPx` is used as `underlying_price`; `spot_px` is retained only as a check field.

The parquet lake intentionally preserves normalized source quotes for auditability. Tradeability filtering is applied by the backtest selection layer through `QuoteFilterConfig`, rather than destructively removing rows from the lake.

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Current tests cover canonical contract IDs, schema drift when `cOpra`/`pOpra` are missing, and call/put wide-to-long conversion.

## Incremental Historical Conversion

Use date windows or one calendar year at a time. `--skip-existing` makes the command restartable.

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system convert \
  --year 2024 \
  --skip-existing
```

For the full research lake, run years independently:

```bash
for year in $(seq 2007 2024); do
  PYTHONPATH=src .venv/bin/python -m qqq_options_system convert \
    --year "$year" \
    --skip-existing
done
```

## Trigger Research Backtest

The sample research config validates the framework on a small 2024 Q1 window. It enforces `entry_lag_days = 1`, so EOD return/IV signals enter on the next available trading day.

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-experiment \
  --experiment-config configs/research_sample.json
```

Outputs:

```text
outputs/backtest_results/QQQ_20240110_20240320/
  daily_features.csv
  trades.parquet
  trades.csv
  summary.csv
  period_splits.csv
  report.md
```

The full-grid config is prepared for `2007-01-01` through `2024-12-31` after the parquet lake exists:

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-experiment \
  --experiment-config configs/research_full_grid.json
```

Current strategy engines cover short put, defined-risk put spread, long call, risk reversal, and `defined_risk_bullish_combo`.

The narrow three-leg bullish combo example uses a `-1%` return trigger, T+1 entry, approximately 45 DTE, a 21-trading-day hold, short-put delta `-0.50`, and a fixed put-wing width of 10% of spot:

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-experiment \
  --experiment-config configs/defined_risk_bullish_combo_sample.json
```

For this strategy, the long-call budget is always executable put-spread credit (`short put bid - long put ask`). The selected call is the most expensive tradeable 45-DTE call whose ask fits that budget. Capital/max loss is put-spread width less the remaining three-leg entry credit, plus modeled round-trip costs. Entry failures such as non-positive credit or no affordable call are retained with explicit status and `failure_reason`.

## Entry Signal Features

`research.entry_signals` provides the fixed, non-Cartesian signal ablation layer used before T+1 entry:

- `standardized_shock` divides the Day-T QQQ return by the prior 20 trading-day return volatility. The rolling denominator is shifted by one day.
- `skew25` uses the nearest 45-DTE expiry with tradeable puts and calls, then selects same-expiry `-0.25/+0.25` delta IVs. Missing IV remains missing; `skew25_5d_change` is never forward-filled.
- `close_location` is calculated from an external OHLC CSV/DataFrame; zero-range days remain missing.
- Training calibration matches the baseline `return <= -1%` event rate using only gated training dates where both return and standardized shock are valid, then applies that rate to the same shock sample's lower-tail quantile. Validation/test rows cannot alter this threshold.
- The close confirmation threshold is the frozen median among training-period standardized-plus-skew candidates.

The four emitted masks are `baseline`, `standardized`, `standardized_skew_nonaccelerating`, and `standardized_skew_close_confirmation`. They are Day-T signal masks only; execution remains T+1. This feature layer does not modify the fixed `defined_risk_bullish_combo` contract parameters.

## Walk-Forward Anti-Overfit Run

Use `run-walk-forward` for candidate selection. Unlike `run-experiment`, this ranks parameters only on the configured training period, then reports validation and test metrics separately.

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-walk-forward \
  --experiment-config configs/research_walk_forward.json
```

Full-history periods in the default walk-forward config:

```text
training:   2007-01-01 to 2016-12-31
validation: 2017-01-01 to 2020-12-31
test:       2021-01-01 to 2024-12-31
```

Outputs:

```text
outputs/backtest_results/walk_forward_QQQ_20070101_20241231/
  walk_forward_all_candidates.csv
  walk_forward_candidates.csv
  walk_forward_trades.parquet
  walk_forward_report.md
```

`walk_forward_candidates.csv` is the anti-overfit selection artifact. `period_splits.csv` from normal experiments is only diagnostic and must not be treated as train-only candidate selection.

## Five-Factor VIX/VXN Regime Study

Use the frozen regime specification in:

[`docs/vix_regime_backtest_spec.md`](docs/vix_regime_backtest_spec.md)

Pilot:

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-regime-study \
  --regime-config configs/regime_vix_pilot.json
```

Full history after annual parquet conversion:

```bash
PYTHONPATH=src .venv/bin/python -m qqq_options_system run-regime-study \
  --regime-config configs/regime_vix_walk_forward.json
```

The background helper runs annual restartable conversion first, then the full regime study:

```bash
nohup zsh scripts/run_full_regime_2007_2024.zsh \
  > outputs/logs/regime_full_2007_2024.log 2>&1 &
```

Regime outputs include `daily_regimes`, `conditional_trades`, `conditional_by_vix_regime`, `portfolio_daily`, `portfolio_metrics`, `walk_forward_regime_candidates`, and `report.md`. VIX/VXN are sourced from FRED `VIXCLS` and `VXNCLS`; ORATS VIX option rows are not used as VIX closes. The first term-structure implementation is QQQ option ATM IV30 versus IV90, not VIX futures.

`conditional_by_vix_regime.csv` is a full-sample diagnostic. The anti-overfit artifact is `walk_forward_regime_candidates.csv`: it filters to non-overlapping `bid_ask` trades by default, splits on `signal_date`, ranks/selects using training metrics only, and attaches validation/test metrics for independent evaluation. `portfolio_daily` and `portfolio_metrics` declare `portfolio_scope=equity_allocation_only`; this minimal NAV simulation contains regime-dependent QQQ allocation returns, not option sleeve PnL.

### Backtest Assumptions And Safeguards

- Same-day EOD signal fills are rejected by `run-experiment`; Day T EOD return/IV signals enter on T+1 or later.
- Quote selection uses `QuoteFilterConfig`: finite bid/mid/ask, `bid >= 0`, `ask >= bid`, `mid > 0`, configurable relative spread, OI/volume thresholds, and optional IV-positive checks.
- `fill_price()` rejects NaN, infinite, negative, zero-mid, and crossed quotes.
- Costs are configurable and default to `$0.65` commission plus `$0.02` exchange fee per contract per side.
- Trade output includes `gross_pnl`, `total_costs`, and net `pnl`.
- Short-put capital can be cash-secured or margin-based through `margin_requirement_pct`; sample/full configs currently use `0.35`.
- Missing or invalid exit quotes are retained in `trades` with non-closed `status`; metrics report `failed_trades` and compute PnL stats on closed trades only.
- Sharpe uses annualized `risk_free_rate`; Sortino uses configurable target downside return.
- IV percentile uses rolling lookback/min-period settings, not an expanding full-history percentile.
