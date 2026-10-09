# QQQ Cloud Email Integration

`multi_tenor_builder.py` adds research-only QQQ signal cards to the existing
plain-text email report. It does not submit orders or alter the builder's
maintenance and laddering rules.

## Authoritative source

The authoritative live modules are:

`src/qqq_options_system/live/qqq_strategy_email.py`

`src/qqq_options_system/live/qqq_contract_scanner.py`

Sync and hash-check the vendored cloud copy with:

```bash
cd /path/to/QQQ
.venv/bin/python scripts/sync_live_qqq_signal.py
.venv/bin/python scripts/sync_live_qqq_signal.py --check
```

Pass an explicit deployment path to update another vendored copy:

```bash
.venv/bin/python scripts/sync_live_qqq_signal.py \
  /path/to/qqq_strategy_email.py \
  --scanner-destination /path/to/qqq_contract_scanner.py
```

## Required live metadata

- Five-factor gating recognizes only `five_factor_regime` or `joint_regime`
  using the frozen six-state vocabulary. Missing or unknown values cannot
  release the downtrend trade from `WATCH`.
- A wall breakout is `TRIGGERED` only when the database provides
  `wall_expiry`/`option_wall_expiry`, or matching `put_wall_expiry` and
  `call_wall_expiry`, equal to the target standard monthly expiry. Existing
  rows without this metadata remain `WATCH`.
- The wall signal is evaluated six NYSE sessions before settlement. T+1 entry
  therefore has five sessions through settlement, matching the frozen study.
- `MISSED_WINDOW` requires an actual Put/Call Wall break within the 3% distance
  cap. Merely reaching the five-session point while QQQ remains between both
  walls is `NO_SIGNAL`. Missed and stale signals never receive contract scans.

QQQ OHLC uses unadjusted Yahoo prices (`auto_adjust=False`) to match the frozen
price-action definition. The email prints the full completed daily OHLC, gap
return, intraday high/low returns and close return. Small ex-dividend moves can
affect raw close returns.

## Executable contract candidates

Signal detection still runs before the IBKR connection, so a broker outage
cannot suppress the research signal. After the read-only connection succeeds,
the builder enriches current, non-stale `TRIGGERED` and `WATCH` signals with
specific QQQ contracts:

- uptrend: the closest liquid `-0.40` Delta put and the most expensive
  same-expiry call whose ask fits the put bid;
- downtrend: the closest liquid `-0.50` Delta put, the same-expiry 10%-spot
  lower protective put, and the most expensive call whose ask fits the
  executable put-spread credit;
- monthly wall: the long option nearest the validated wall and the same-expiry
  short option nearest the 5%-spot outer target.

Every leg prints expiry, strike, Delta, bid/ask/mid, IV, OI, volume, IBKR market
data type and the fill side. Combination credit/debit always uses leg-level
bid/ask. Invalid, crossed, overly wide or low-OI quotes are rejected, with the
failure reason retained in the email. Every displayed `TRIGGERED`, `WATCH` or
`BLOCKED` signal receives either a concrete candidate or an explicit broker/
quote failure result. `WATCH` and `BLOCKED` candidates are labeled as
non-executable. If live IBKR option quotes are unusable, the scanner retries
with delayed and delayed-frozen market data and labels the returned market-data
type. Quote collection uses the option chain's `tradingClass` and `multiplier`,
accepts IBKR `modelGreeks`/`bidGreeks`/`askGreeks`/`lastGreeks` in that order,
and reports per-attempt requested, qualified, bid/ask, Delta and OI counts when
selection fails. Set `QQQ_ALLOW_DELAYED_MARKET_DATA_FALLBACK=False` to disable
this fallback. The optional cloud setting `QQQ_OPTION_ENTRY_COST_PER_CONTRACT`
defaults to `$0.67` per opening leg. Collection tuning defaults are
`QQQ_OPTION_QUOTE_WAIT_SECONDS=8.0`, `QQQ_OPTION_QUOTE_BATCH_SIZE=20`,
`QQQ_OPTION_QUOTE_READY_RATIO=0.60`, and
`QQQ_OPTION_FALLBACK_MARKET_DATA_TYPES=(3, 4)`.

Put and call strikes are sampled separately. Put candidates cover spot and the
10%-spot protective-wing area; call candidates default to spot, 3%, 6% and 10%
above spot so the budget-constrained OTM call is not accidentally omitted. Tune
`QQQ_FUNDED_CALL_CENTER_MULTIPLIERS` and `QQQ_FUNDED_CALL_COUNT_PER_CENTER`
in the cloud config if quote load or OTM coverage needs adjustment.

The staged live schedule keeps the frozen `T+1 EOD` research rule unchanged:

- the existing post-close report calls `--mode qqq-section --stage post_close
  --wait-until-ready` after its normal collection and freezes unadjusted OHLC
  no earlier than the actual NYSE close plus ten minutes;
- the existing pre-market report calls `--mode qqq-section --stage pre_market`
  after writing the current wall snapshot, then reviews that same frozen signal;
- at 09:35 ET, `--mode qqq-live-scan` obtains the first executable IBKR option
  snapshot and sends mail only for an active signal or a scan failure;
- the original 15:35 ET full builder remains the EOD refresh and continues all
  existing reconciliation, maintenance, laddering and inventory work.

The 09:35 and 15:35 scans are candidate snapshots, not locked fills, and never
create an IBKR order. The email instructs the operator to recheck Delta,
bid/ask and liquidity near execution. All stages share the atomically written
`~/TradingRadar/state/qqq_frozen_signal.json` record (override with
`QQQ_SIGNAL_STATE_PATH`) and print the same signal ID. If that state is missing
or does not match the T+1 date, the scanner reconstructs it from completed bars
and labels the fallback instead of silently presenting it as the persisted
post-close snapshot.

## Current-position source

The report's current QQQ/TQQQ holdings come directly from the read-only IBKR
`positions()` snapshot, including signed quantity, contract identity and IB
average cost. `unified_active_legs` is no longer presented as current physical
inventory. It is compared with IBKR as a separate theoretical-ledger audit;
quantity mismatches are printed explicitly and are not auto-deleted or written
back to Supabase.

## Connection and delivery failures

Both the full builder and the 09:35 scan use a read-only IBKR connection with
a 20-second timeout for initialization requests, including execution-history
synchronization. Connection failures are retried up to three attempts with
3- and 6-second pauses and a disconnect between attempts. If cleanup fails,
the builder stops retrying. Reconciliation, maintenance and laddering run only
after connection succeeds.

Connection failures record their exception type even when the exception message
is empty. The full traceback and IBKR error-level messages remain in the logs.
Exhausted retries still produce the existing failure report. SMTP acceptance
and rejection are logged explicitly; SMTP acceptance does not confirm inbox
delivery. Fatal runs and email failures return exit status 1. A trading-day
skip or a 09:35 scan without an active signal still exits quietly with status 0.

## Verification

```bash
cd /path/to/QQQ
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Unit tests cover the pure rules and the cloud adapter with mocked IO/calendar
dependencies. They are not a production Supabase/yfinance/email end-to-end
test.
