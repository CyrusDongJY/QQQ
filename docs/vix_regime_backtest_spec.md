# QQQ Five-Factor Volatility Regime Backtest Specification

## Research Question

Measure QQQ and QQQ-option strategy outcomes inside fixed VIX price buckets while requiring a joint volatility regime built from QQQ IV level, VXN direction, QQQ option term structure, IV-minus-RV, and QQQ trend. VIX is an attribution variable and one regime input; it is not a standalone entry signal.

## Data And Timing

- Option data: local ORATS QQQ EOD chains, 2007-01-01 through 2024-12-31.
- VIX/VXN: FRED daily closes (`VIXCLS`, `VXNCLS`) sourced from Cboe. Snapshot source: `/private/tmp/vix_vxn_2007_2024.csv`.
- Source pages: https://fred.stlouisfed.org/series/VIXCLS and https://fred.stlouisfed.org/series/VXNCLS.
- Joint VIX/VXN coverage before QQQ-calendar merge: 4,530 valid days.
- All features use information available at the close of day T. Orders enter on the next QQQ trading day, T+1 (`entry_lag_days=1`).
- No backward fill, centered rolling windows, or use of validation/test data in training ranking.

## VIX Attribution Buckets

Fixed close buckets use left-closed/right-open boundaries:

| Bucket | Raw VIX observations, 2007-2024 |
|---|---:|
| `<15` | 1,528 |
| `[15,20)` | 1,347 |
| `[20,25)` | 822 |
| `[25,30)` | 416 |
| `>=30` | 436 |

Also report trailing 252-trading-day VIX percentile buckets: `0-20`, `20-40`, `40-60`, `60-80`, `80-100`. Percentile buckets are a threshold-stability sensitivity and do not influence holdout ranking.

## Five Factors

1. **QQQ IV level**: constant-maturity ATM 30-day IV (`iv30`) interpolated between adjacent expiries using same-day chains; trailing 252-day percentile with a configurable minimum history.
2. **VXN direction**: 1-day change, 5-day percentage change, and 5-day slope. Initial states are rising at `>= +5%`, falling at `<= -5%`, stable otherwise.
3. **QQQ IV term structure**: constant-maturity ATM 30-day and 90-day IV. Record `term_spread = iv30 - iv90` and `term_ratio = iv30 / iv90`. Initial labels: contango at spread `<= -0.01`, flat between `-0.01` and `+0.01`, inverted at `>= +0.01`.
4. **IV-RV**: QQQ close-to-close log-return RV21 and RV63 annualized with `sqrt(252)`, `iv_rv_spread = iv30 - rv21`, and RV21 five-day change.
5. **QQQ trend**: spot versus MA20/MA50/MA200 and MA ordering. Trend rules are configuration values, not optimized against the test period.

The first implementation uses QQQ option IV term structure because it can be reconstructed consistently from local chains. VIX futures or VIX/VIX3M term structure is a later sensitivity, not silently substituted here.

## Mutually Exclusive States

Evaluate in this priority order and store the booleans plus a human-readable `regime_reason`:

1. `high_rising`: VIX `>=25` and VXN rising, with stress confirmation from inverted term structure, elevated QQQ IV, or downtrend. No new short gamma.
2. `high_falling`: VIX `>=25` and VXN falling. Monetize part of existing protection and restore QQQ exposure in stages.
3. `vol_rising`: VXN rising with elevated/rising QQQ IV or term stress. Reduce or prohibit short gamma.
4. `low_calm`: VIX `<15`, low QQQ IV percentile, VXN not rising, and term structure not inverted. Permit staggered positive-gamma/vega purchases.
5. `normalized`: VXN not rising, term structure not inverted, RV21 falling or below RV63, IV-RV positive but not extreme, and QQQ trend recovered. Only this state permits new defined-risk short-premium trades.
6. `transition_no_trade`: all remaining observations.

Thresholds must live in JSON configuration. The initial theory thresholds are frozen before full-period results are inspected.

## Conditional Cohort Study

For each signal date, attribute T+1-entered outcomes by fixed VIX bucket, VIX percentile bucket, joint state, strategy, DTE, delta, hold period, and execution model.

Default candidate set:

- QQQ forward/equity baseline: 5, 10, and 20 trading days.
- Long call: delta `0.35/0.45/0.55`, DTE `30/60/90`.
- Long put: delta `-0.15/-0.25/-0.35`, DTE `30/60/90`.
- Long put spread: long delta `-0.25/-0.35`, lower wing `2.5%/5%` of spot.
- Defined-risk short put spread: short delta `-0.15/-0.25/-0.30`, 5% wing, DTE `14/30/45`; entries allowed only in `normalized`.

Use bid/ask fills plus per-leg, per-side fees as the primary case. Use half-spread only as sensitivity. Report overlapping and non-overlapping cohorts separately.

Every option trade must record `gross_pnl`, `total_costs`, `net_pnl`, `capital_required`, `max_loss`, `return_on_risk`, and `pnl_per_1000_risk`. Do not rank unlike strategies on raw per-contract dollars.

## Sequential Portfolio Study

Run a separate daily NAV simulation with initial NAV 100,000 and QQQ buy-and-hold as benchmark. Conditional trade sums are not portfolio NAV.

Initial theory allocation, configurable and not holdout-optimized:

| State | QQQ target | New option action |
|---|---:|---|
| `low_calm` | 100% | Stagger small positive-convexity premium budget |
| `vol_rising` | 75% | Stop short premium; retain/add defined protection |
| `high_rising` | 25-50% | Preserve protection; no short gamma |
| `high_falling` | 50-75% | Monetize part of hedge; restore QQQ in stages |
| `normalized` | 100% | Permit defined-risk put spread, max-loss budget <=1% NAV |
| `transition_no_trade` | Previous/neutral target | Do not add short gamma |

Compare three portfolios: QQQ buy-and-hold, VIX-only allocation rule, and five-factor joint-state rule.

## Anti-Overfit Protocol

- Training: 2007-2016.
- Validation: 2017-2020.
- Test: 2021-2024.
- Parameter ranking is training-only. Validation confirms stability. Test is reported once and never improves rank.
- Report crisis slices for 2008, 2011, 2018 Q4, 2020 Q1, and 2022.
- Keep the default cohort grid below several hundred combinations.
- Include trade counts, block-bootstrap 95% confidence intervals, and warnings for sparse cells.

## Required Outputs

- `daily_regimes.parquet` and `.csv`
- `conditional_trades.parquet`
- `conditional_by_vix_regime.csv`
- `portfolio_daily.parquet`
- `portfolio_metrics.csv`
- `walk_forward_regime_candidates.csv` when ranking is enabled
- `report.md`

The report must state that results are historical research under the specified data, candidate set, execution costs, and capital model, not an investment recommendation or return promise.
