from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable

import pandas as pd

from qqq_options_system.backtest.execution import ExecutionCostConfig, fill_price
from qqq_options_system.backtest.selection import select_by_delta_dte, select_put_wing_by_width
from qqq_options_system.data.normalize import QuoteFilterConfig


MULTIPLIER = 100.0


@dataclass(frozen=True)
class StrategySpec:
    strategy: str
    dte_target: int
    hold_days: int
    price_mode: str
    short_put_delta: float | None = None
    long_call_delta: float | None = None
    put_spread_width_pct: float | None = None
    exit_dte_min: int = 1

    @property
    def label(self) -> str:
        parts = [self.strategy, f"dte{self.dte_target}", f"hold{self.hold_days}", self.price_mode]
        if self.short_put_delta is not None:
            parts.append(f"spd{self.short_put_delta:g}")
        if self.long_call_delta is not None:
            parts.append(f"lcd{self.long_call_delta:g}")
        if self.put_spread_width_pct is not None:
            parts.append(f"wing{self.put_spread_width_pct:g}")
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
    if strategy == "long_call":
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
    raise ValueError(f"Unsupported strategy: {spec.strategy}")


def _choose_exit_index(
    dates: list[str],
    entry_index: int,
    expiry: str,
    hold_days: int,
    exit_dte_min: int,
) -> int | None:
    scheduled = min(entry_index + hold_days, len(dates) - 1)
    latest_allowed = None
    expiry_ts = pd.Timestamp(expiry)
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
) -> pd.DataFrame:
    cfg = config or BacktestConfig()
    dates = sorted(chains)
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
        entry_chain = chains[entry_date]
        legs = _build_legs(entry_chain, spec, cfg)
        if not legs:
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
        exit_chain = chains[exit_date]
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
        "put_spread_width_pct": spec.put_spread_width_pct,
        "status": status,
        "failure_reason": failure_reason,
        "capital_model": config.capital_model,
        "commission_per_contract": config.cost_config.commission_per_contract,
        "exchange_fee_per_contract": config.cost_config.exchange_fee_per_contract,
        "slippage_per_contract": config.cost_config.slippage_per_contract,
        "margin_requirement_pct": config.margin_requirement_pct,
        "legs": len(legs),
        "leg_details": repr(legs),
        "capital_required": capital_required,
        "gross_pnl": gross_pnl,
        "total_costs": total_costs,
        "pnl": pnl,
        "return_on_capital": pnl / capital_required if capital_required else 0.0,
    }
