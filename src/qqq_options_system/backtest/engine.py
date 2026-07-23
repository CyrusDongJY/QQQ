from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import pandas as pd

from qqq_options_system.backtest.execution import ExecutionCostConfig, fill_price
from qqq_options_system.backtest.selection import (
    select_by_delta_dte,
    select_call_within_premium_budget,
    select_put_wing_by_width,
)
from qqq_options_system.data.normalize import QuoteFilterConfig


MULTIPLIER = 100.0


class EntrySelectionError(ValueError):
    """Auditable entry failure for a signal that cannot form all required legs."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class StrategySpec:
    strategy: str
    dte_target: int
    hold_days: int
    price_mode: str
    short_put_delta: float | None = None
    long_call_delta: float | None = None
    long_put_delta: float | None = None
    put_spread_width_pct: float | None = None
    call_dte_target: int | None = None
    exit_dte_min: int = 1

    @property
    def label(self) -> str:
        parts = [self.strategy, f"dte{self.dte_target}", f"hold{self.hold_days}", self.price_mode]
        if self.short_put_delta is not None:
            parts.append(f"spd{self.short_put_delta:g}")
        if self.long_call_delta is not None:
            parts.append(f"lcd{self.long_call_delta:g}")
        if self.long_put_delta is not None:
            parts.append(f"lpd{self.long_put_delta:g}")
        if self.put_spread_width_pct is not None:
            parts.append(f"wing{self.put_spread_width_pct:g}")
        if self.call_dte_target is not None:
            parts.append(f"cdte{self.call_dte_target:g}")
        return "_".join(parts).replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class BacktestConfig:
    cost_config: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    margin_requirement_pct: float = 1.0
    quote_filter_config: QuoteFilterConfig = field(default_factory=QuoteFilterConfig)

    @property
    def capital_model(self) -> str:
        return f"short_put_margin_pct={self.margin_requirement_pct:g}"


def load_chain_partitions(data_root: Path, start: str, end: str, underlying: str = "QQQ") -> dict[str, pd.DataFrame]:
    base = data_root / "data" / "silver_options" / f"symbol={underlying.upper()}"
    frames: dict[str, pd.DataFrame] = {}
    if not base.exists():
        return frames
    for path in sorted(base.glob("year=*/trade_date=*/option_chain_eod.parquet")):
        date_text = path.parent.name.split("=", 1)[1]
        if date_text < start or date_text > end:
            continue
        frame = pd.read_parquet(path)
        for col in (
            "dte",
            "strike",
            "bid",
            "mid",
            "ask",
            "delta",
            "underlying_price",
            "open_interest",
            "volume",
            "bid_iv",
            "mid_iv",
            "ask_iv",
        ):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames[date_text] = frame
    return frames


def _quote_key(row: pd.Series) -> tuple[str, str, float]:
    return (str(row["expiry"]), str(row["right"]), float(row["strike"]))


def _find_quote(chain: pd.DataFrame, key: tuple[str, str, float]) -> pd.Series | None:
    expiry, right, strike = key
    candidates = chain.loc[
        (chain["expiry"] == expiry)
        & (chain["right"] == right)
        & ((chain["strike"].astype(float) - strike).abs() < 0.0001)
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _entry_fill(row: pd.Series, qty: int, price_mode: str) -> float:
    side = "buy" if qty > 0 else "sell"
    return fill_price(float(row["bid"]), float(row["mid"]), float(row["ask"]), side, price_mode)


def _exit_fill(row: pd.Series, qty: int, price_mode: str) -> float:
    side = "sell" if qty > 0 else "buy"
    return fill_price(float(row["bid"]), float(row["mid"]), float(row["ask"]), side, price_mode)


def _capital_required(strategy: str, legs: list[dict], config: BacktestConfig) -> float:
    if strategy in {"long_call", "long_put", "long_put_spread"}:
        return sum(max(0.0, leg["entry_cash_flow"] * -1.0) for leg in legs)
    if strategy == "short_put":
        short = legs[0]
        return float(short["entry_strike"]) * MULTIPLIER * float(config.margin_requirement_pct)
    if strategy == "put_spread":
        short = next(leg for leg in legs if leg["qty"] < 0)
        long = next(leg for leg in legs if leg["qty"] > 0)
        width = float(short["entry_strike"]) - float(long["entry_strike"])
        credit = sum(leg["entry_cash_flow"] for leg in legs)
        return max(0.0, width * MULTIPLIER - credit)
    if strategy == "defined_risk_bullish_combo":
        short = next(leg for leg in legs if leg["right"] == "P" and leg["qty"] < 0)
        long_put = next(leg for leg in legs if leg["right"] == "P" and leg["qty"] > 0)
        width = float(short["entry_strike"]) - float(long_put["entry_strike"])
        residual_credit = sum(leg["entry_cash_flow"] for leg in legs)
        round_trip_costs = sum(float(leg["leg_cost"]) for leg in legs)
        # The call consumes put-spread credit but adds no loss when QQQ finishes
        # below both put strikes. Defined risk is therefore spread width less
        # the remaining three-leg credit, including modeled round-trip costs.
        return max(0.0, width * MULTIPLIER - residual_credit + round_trip_costs)
    if strategy == "risk_reversal":
        # Short put drives the margin; the long call is financed by the put
        # premium (net credit >= 0 by construction) and adds no margin. Any
        # residual net debit (call cost above the credit) is added on top.
        short = next(leg for leg in legs if leg["qty"] < 0)
        put_margin = float(short["entry_strike"]) * MULTIPLIER * float(config.margin_requirement_pct)
        net_cash_flow = sum(leg["entry_cash_flow"] for leg in legs)
        residual_debit = max(0.0, -net_cash_flow)
        return put_margin + residual_debit
    return sum(abs(leg["entry_cash_flow"]) for leg in legs)


def _build_legs(
    entry_chain: pd.DataFrame,
    spec: StrategySpec,
    config: BacktestConfig,
) -> list[tuple[pd.Series, int]]:
    if spec.strategy == "short_put":
        short_put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.short_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        return [] if short_put is None else [(short_put, -1)]
    if spec.strategy == "long_call":
        call = select_by_delta_dte(
            entry_chain,
            "C",
            float(spec.long_call_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        return [] if call is None else [(call, 1)]
    if spec.strategy == "long_put":
        put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.long_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        return [] if put is None else [(put, 1)]
    if spec.strategy == "long_put_spread":
        long_put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.long_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        if long_put is None:
            return []
        wing = select_put_wing_by_width(
            entry_chain,
            long_put,
            float(spec.put_spread_width_pct),
            quote_filter_config=config.quote_filter_config,
        )
        if wing is None:
            return []
        return [(long_put, 1), (wing, -1)]
    if spec.strategy == "put_spread":
        short_put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.short_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        if short_put is None:
            return []
        wing = select_put_wing_by_width(
            entry_chain,
            short_put,
            float(spec.put_spread_width_pct),
            quote_filter_config=config.quote_filter_config,
        )
        if wing is None:
            return []
        return [(short_put, -1), (wing, 1)]
    if spec.strategy == "risk_reversal":
        short_put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.short_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        if short_put is None:
            return []
        # Premium collected from selling the put funds the long call. Budget is
        # the per-share credit at the configured price mode.
        put_credit = _entry_fill(short_put, -1, spec.price_mode)
        call_dte = int(spec.call_dte_target if spec.call_dte_target is not None else spec.dte_target)
        entry_cost_column = "mid" if spec.price_mode == "midpoint" else "ask"
        call = select_call_within_premium_budget(
            entry_chain,
            call_dte_target=call_dte,
            premium_budget=put_credit,
            entry_cost_column=entry_cost_column,
            min_dte=spec.exit_dte_min,
            required_expiry=str(short_put["expiry"]),
            quote_filter_config=config.quote_filter_config,
        )
        if call is None:
            return []
        return [(short_put, -1), (call, 1)]
    if spec.strategy == "defined_risk_bullish_combo":
        if spec.price_mode != "bid_ask":
            raise EntrySelectionError(
                "failed_unsupported_execution",
                "defined_risk_bullish_combo requires bid_ask execution",
            )
        short_put = select_by_delta_dte(
            entry_chain,
            "P",
            float(spec.short_put_delta),
            spec.dte_target,
            spec.exit_dte_min,
            quote_filter_config=config.quote_filter_config,
        )
        if short_put is None:
            raise EntrySelectionError(
                "failed_no_short_put",
                "No tradeable short put matched the target delta/DTE",
            )
        long_put = select_put_wing_by_width(
            entry_chain,
            short_put,
            float(spec.put_spread_width_pct),
            quote_filter_config=config.quote_filter_config,
        )
        if long_put is None:
            raise EntrySelectionError(
                "failed_no_put_wing",
                "No tradeable lower-strike put wing matched the configured width",
            )
        try:
            spread_credit = _entry_fill(short_put, -1, "bid_ask") - _entry_fill(
                long_put, 1, "bid_ask"
            )
        except ValueError as exc:
            raise EntrySelectionError("failed_invalid_entry_quote", str(exc)) from exc
        if spread_credit <= 0:
            raise EntrySelectionError(
                "failed_nonpositive_put_spread_credit",
                f"Executable put-spread credit must be positive, got {spread_credit:g}",
            )
        call_dte = int(
            spec.call_dte_target if spec.call_dte_target is not None else spec.dte_target
        )
        call = select_call_within_premium_budget(
            entry_chain,
            call_dte_target=call_dte,
            premium_budget=spread_credit,
            entry_cost_column="ask",
            min_dte=spec.exit_dte_min,
            required_expiry=str(short_put["expiry"]),
            quote_filter_config=config.quote_filter_config,
        )
        if call is None:
            raise EntrySelectionError(
                "failed_no_affordable_call",
                f"No tradeable call ask fits executable put-spread credit {spread_credit:g}",
            )
        return [(short_put, -1), (long_put, 1), (call, 1)]
    raise ValueError(f"Unsupported strategy: {spec.strategy}")


def _choose_exit_index(
    dates: list[str],
    entry_index: int,
    expiry: str,
    hold_days: int,
    exit_dte_min: int,
) -> int | None:
    scheduled = entry_index + hold_days
    last_date = pd.Timestamp(dates[-1])
    expiry_ts = pd.Timestamp(expiry)
    # Do not turn a right-censored trade into an artificial early exit at the
    # dataset boundary. An earlier exit remains valid when expiry forces it.
    if scheduled >= len(dates) and (expiry_ts - last_date).days >= exit_dte_min:
        return None
    scheduled = min(scheduled, len(dates) - 1)
    latest_allowed = None
    for index in range(entry_index + 1, len(dates)):
        dte = (expiry_ts - pd.Timestamp(dates[index])).days
        if dte >= exit_dte_min:
            latest_allowed = index
        else:
            break
    if latest_allowed is None:
        return None
    return min(scheduled, latest_allowed)


def run_strategy(
    chains: dict[str, pd.DataFrame],
    signal_dates: Iterable[str],
    spec: StrategySpec,
    entry_lag_days: int = 1,
    config: BacktestConfig | None = None,
    calendar_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    cfg = config or BacktestConfig()
    dates = sorted(set(str(value) for value in calendar_dates)) if calendar_dates is not None else sorted(chains)
    date_to_index = {date: index for index, date in enumerate(dates)}
    rows: list[dict] = []
    for signal_date in signal_dates:
        signal_index = date_to_index.get(signal_date)
        if signal_index is None:
            continue
        entry_index = signal_index + entry_lag_days
        if entry_index >= len(dates):
            continue
        entry_date = dates[entry_index]
        entry_chain = chains.get(entry_date)
        if entry_chain is None:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date="",
                    entry_lag_days=entry_lag_days,
                    status="failed_missing_entry_chain",
                    failure_reason="No option-chain partition on the scheduled market-calendar entry date",
                    config=cfg,
                )
            )
            continue
        try:
            legs = _build_legs(entry_chain, spec, cfg)
        except EntrySelectionError as exc:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date="",
                    entry_lag_days=entry_lag_days,
                    status=exc.status,
                    failure_reason=exc.reason,
                    config=cfg,
                )
            )
            continue
        if not legs:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date="",
                    entry_lag_days=entry_lag_days,
                    status="failed_no_entry_selection",
                    failure_reason=(
                        "No complete set of tradeable option legs matched the configured "
                        "delta, DTE, quote-quality, and premium-budget rules"
                    ),
                    config=cfg,
                )
            )
            continue
        earliest_expiry = min(str(row["expiry"]) for row, _ in legs)
        exit_index = _choose_exit_index(
            dates, entry_index, earliest_expiry, spec.hold_days, spec.exit_dte_min
        )
        if exit_index is None:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date="",
                    entry_lag_days=entry_lag_days,
                    status="failed_no_valid_exit_date",
                    failure_reason="No exit date before exit_dte_min",
                    config=cfg,
                )
            )
            continue
        exit_date = dates[exit_index]
        exit_chain = chains.get(exit_date)
        if exit_chain is None:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_lag_days=entry_lag_days,
                    status="failed_missing_exit_chain",
                    failure_reason="No option-chain partition on the scheduled market-calendar exit date",
                    config=cfg,
                )
            )
            continue
        leg_records: list[dict] = []
        gross_pnl = 0.0
        failed_reason = ""
        for leg_row, qty in legs:
            exit_quote = _find_quote(exit_chain, _quote_key(leg_row))
            if exit_quote is None:
                failed_reason = (
                    f"Missing exit quote for {leg_row['right']} {leg_row['expiry']} "
                    f"{float(leg_row['strike'])}"
                )
                break
            try:
                entry_price = _entry_fill(leg_row, qty, spec.price_mode)
                exit_price = _exit_fill(exit_quote, qty, spec.price_mode)
            except ValueError as exc:
                failed_reason = str(exc)
                break
            entry_cash_flow = -qty * entry_price * MULTIPLIER
            exit_cash_flow = qty * exit_price * MULTIPLIER
            leg_pnl = entry_cash_flow + exit_cash_flow
            gross_pnl += leg_pnl
            leg_cost = abs(qty) * cfg.cost_config.round_trip_cost_per_contract
            leg_records.append(
                {
                    "qty": qty,
                    "right": leg_row["right"],
                    "expiry": leg_row["expiry"],
                    "entry_strike": float(leg_row["strike"]),
                    "entry_delta": float(leg_row["delta"]),
                    "entry_dte": int(leg_row["dte"]),
                    "entry_bid": float(leg_row["bid"]),
                    "entry_mid": float(leg_row["mid"]),
                    "entry_ask": float(leg_row["ask"]),
                    "exit_bid": float(exit_quote["bid"]),
                    "exit_mid": float(exit_quote["mid"]),
                    "exit_ask": float(exit_quote["ask"]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "entry_cash_flow": entry_cash_flow,
                    "exit_cash_flow": exit_cash_flow,
                    "leg_cost": leg_cost,
                    "leg_pnl": leg_pnl,
                }
            )
        if failed_reason:
            rows.append(
                _trade_row(
                    spec=spec,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_lag_days=entry_lag_days,
                    status="failed_no_exit_quote"
                    if failed_reason.startswith("Missing exit quote")
                    else "failed_invalid_quote",
                    failure_reason=failed_reason,
                    config=cfg,
                    leg_records=leg_records,
                )
            )
            continue
        total_costs = sum(leg["leg_cost"] for leg in leg_records)
        pnl = gross_pnl - total_costs
        capital = _capital_required(spec.strategy, leg_records, cfg)
        max_loss = capital
        rows.append(
            _trade_row(
                spec=spec,
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_lag_days=entry_lag_days,
                status="closed",
                failure_reason="",
                config=cfg,
                leg_records=leg_records,
                capital_required=capital,
                max_loss=max_loss,
                gross_pnl=gross_pnl,
                total_costs=total_costs,
                pnl=pnl,
            )
        )
    return pd.DataFrame(rows)


def _trade_row(
    spec: StrategySpec,
    signal_date: str,
    entry_date: str,
    exit_date: str,
    entry_lag_days: int,
    status: str,
    failure_reason: str,
    config: BacktestConfig,
    leg_records: list[dict] | None = None,
    capital_required: float = 0.0,
    max_loss: float = 0.0,
    gross_pnl: float = 0.0,
    total_costs: float = 0.0,
    pnl: float = 0.0,
) -> dict:
    legs = leg_records or []
    return {
        "strategy_id": spec.label,
        "strategy": spec.strategy,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_lag_days": entry_lag_days,
        "hold_days": spec.hold_days,
        "dte_target": spec.dte_target,
        "price_mode": spec.price_mode,
        "short_put_delta": spec.short_put_delta,
        "long_call_delta": spec.long_call_delta,
        "long_put_delta": spec.long_put_delta,
        "put_spread_width_pct": spec.put_spread_width_pct,
        "call_dte_target": spec.call_dte_target,
        "status": status,
        "failure_reason": failure_reason,
        "capital_model": (
            "defined_risk_width_minus_residual_credit_plus_round_trip_costs"
            if spec.strategy == "defined_risk_bullish_combo"
            else config.capital_model
        ),
        "commission_per_contract": config.cost_config.commission_per_contract,
        "exchange_fee_per_contract": config.cost_config.exchange_fee_per_contract,
        "slippage_per_contract": config.cost_config.slippage_per_contract,
        "margin_requirement_pct": config.margin_requirement_pct,
        "legs": len(legs),
        "leg_details": repr(legs),
        "capital_required": capital_required,
        "max_loss": max_loss,
        "gross_pnl": gross_pnl,
        "total_costs": total_costs,
        "net_pnl": pnl,
        "pnl": pnl,
        "return_on_capital": pnl / capital_required if capital_required else 0.0,
        "return_on_risk": pnl / max_loss if max_loss else 0.0,
        "pnl_per_1000_risk": pnl / max_loss * 1000.0 if max_loss else 0.0,
    }
