from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_options_system.backtest.engine import (
    BacktestConfig,
    StrategySpec,
    load_chain_partitions,
    run_strategy,
)
from qqq_options_system.backtest.execution import ExecutionCostConfig
from qqq_options_system.backtest.metrics import split_summary, summarize_trades
from qqq_options_system.data.normalize import QuoteFilterConfig
from qqq_options_system.research.features import build_daily_features, signal_dates_for_trigger


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_strategy_specs(config: dict[str, Any]):
    price_modes = config.get("price_modes", ["bid_ask"])
    dte_targets = config.get("dte_targets", [30])
    hold_days = config.get("hold_days", [5])
    exit_dte_min = int(config.get("exit_dte_min", 1))
    for price_mode in price_modes:
        for dte in dte_targets:
            for hold in hold_days:
                if "short_put" in config.get("strategies", []):
                    for delta in config.get("short_put_deltas", [-0.25]):
                        yield StrategySpec(
                            strategy="short_put",
                            dte_target=int(dte),
                            hold_days=int(hold),
                            price_mode=price_mode,
                            short_put_delta=float(delta),
                            exit_dte_min=exit_dte_min,
                        )
                if "put_spread" in config.get("strategies", []):
                    for delta in config.get("short_put_deltas", [-0.25]):
                        for width in config.get("put_spread_width_pcts", [0.05]):
                            yield StrategySpec(
                                strategy="put_spread",
                                dte_target=int(dte),
                                hold_days=int(hold),
                                price_mode=price_mode,
                                short_put_delta=float(delta),
                                put_spread_width_pct=float(width),
                                exit_dte_min=exit_dte_min,
                            )
                if "long_call" in config.get("strategies", []):
                    for delta in config.get("long_call_deltas", [0.55]):
                        yield StrategySpec(
                            strategy="long_call",
                            dte_target=int(dte),
                            hold_days=int(hold),
                            price_mode=price_mode,
                            long_call_delta=float(delta),
                            exit_dte_min=exit_dte_min,
                        )
                if "risk_reversal" in config.get("strategies", []):
                    for put_delta in config.get("risk_reversal_put_deltas", [-0.50]):
                        for call_dte in config.get("call_dte_targets", [int(dte)]):
                            yield StrategySpec(
                                strategy="risk_reversal",
                                dte_target=int(dte),
                                hold_days=int(hold),
                                price_mode=price_mode,
                                short_put_delta=float(put_delta),
                                call_dte_target=int(call_dte),
                                exit_dte_min=exit_dte_min,
                            )
                if "defined_risk_bullish_combo" in config.get("strategies", []):
                    combo_deltas = config.get(
                        "defined_risk_bullish_combo_put_deltas",
                        config.get("short_put_deltas", [-0.50]),
                    )
                    for put_delta in combo_deltas:
                        for width in config.get("put_spread_width_pcts", [0.05, 0.10]):
                            for call_dte in config.get("call_dte_targets", [int(dte)]):
                                yield StrategySpec(
                                    strategy="defined_risk_bullish_combo",
                                    dte_target=int(dte),
                                    hold_days=int(hold),
                                    price_mode=price_mode,
                                    short_put_delta=float(put_delta),
                                    put_spread_width_pct=float(width),
                                    call_dte_target=int(call_dte),
                                    exit_dte_min=exit_dte_min,
                                )


def iter_triggers(config: dict[str, Any]):
    for threshold in config.get("return_thresholds", []):
        yield "return", float(threshold)
    for threshold in config.get("iv_percentile_thresholds", []):
        yield "iv_percentile", float(threshold)
    for threshold in config.get("iv_change_thresholds", []):
        yield "iv_change", float(threshold)


def run_experiment(config: dict[str, Any], output_root: Path) -> dict[str, Path]:
    underlying = config.get("underlying", "QQQ").upper()
    data_start = config["data_start"]
    data_end = config["data_end"]
    research_start = config.get("research_start", data_start)
    research_end = config.get("research_end", data_end)
    entry_lag_days = int(config.get("entry_lag_days", 1))
    if entry_lag_days < 1:
        raise ValueError("EOD trigger experiments require entry_lag_days >= 1")
    risk_free_rate = float(config.get("risk_free_rate", 0.0))
    sortino_target_return = float(config.get("sortino_target_return", 0.0))
    backtest_config = _backtest_config_from_experiment(config)

    chains = load_chain_partitions(output_root, data_start, data_end, underlying)
    if not chains:
        raise FileNotFoundError(
            f"No parquet partitions found under {output_root} for {underlying} {data_start}..{data_end}"
        )
    features = build_daily_features(
        chains,
        iv_percentile_lookback=int(config.get("iv_percentile_lookback", 252)),
        iv_percentile_min_periods=int(config.get("iv_percentile_min_periods", 60)),
    )
    features = features.loc[(features["date"] >= research_start) & (features["date"] <= research_end)].copy()
    all_dates = sorted(date for date in chains if research_start <= date <= research_end)

    trades_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    split_rows: list[pd.DataFrame] = []
    for trigger_kind, threshold in iter_triggers(config):
        signal_dates = signal_dates_for_trigger(features, trigger_kind, threshold)
        for spec in iter_strategy_specs(config):
            trades = run_strategy(
                chains,
                signal_dates,
                spec,
                entry_lag_days=entry_lag_days,
                config=backtest_config,
            )
            if trades.empty:
                continue
            trades["trigger_kind"] = trigger_kind
            trades["trigger_threshold"] = threshold
            trades_frames.append(trades)
            metrics = summarize_trades(
                trades,
                all_dates=all_dates,
                risk_free_rate=risk_free_rate,
                sortino_target_return=sortino_target_return,
            )
            metrics.update(
                {
                    "trigger_kind": trigger_kind,
                    "trigger_threshold": threshold,
                    "strategy_id": spec.label,
                    "strategy": spec.strategy,
                    "dte_target": spec.dte_target,
                    "hold_days": spec.hold_days,
                    "price_mode": spec.price_mode,
                    "short_put_delta": spec.short_put_delta,
                    "long_call_delta": spec.long_call_delta,
                    "put_spread_width_pct": spec.put_spread_width_pct,
                    "call_dte_target": spec.call_dte_target,
                    "entry_lag_days": entry_lag_days,
                    "capital_model": backtest_config.capital_model,
                }
            )
            summary_rows.append(metrics)
            splits = split_summary(
                trades,
                risk_free_rate=risk_free_rate,
                sortino_target_return=sortino_target_return,
            )
            if not splits.empty:
                splits["trigger_kind"] = trigger_kind
                splits["trigger_threshold"] = threshold
                splits["strategy_id"] = spec.label
                split_rows.append(splits)

    run_dir = (
        output_root
        / "backtest_results"
        / f"{underlying}_{research_start.replace('-', '')}_{research_end.replace('-', '')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    features_path = run_dir / "daily_features.csv"
    features.to_csv(features_path, index=False)
    trades_path = run_dir / "trades.parquet"
    trades_csv_path = run_dir / "trades.csv"
    summary_path = run_dir / "summary.csv"
    split_path = run_dir / "period_splits.csv"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        all_trades = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    splits = pd.concat(split_rows, ignore_index=True) if split_rows else pd.DataFrame()
    all_trades.to_parquet(trades_path, index=False)
    all_trades.to_csv(trades_csv_path, index=False)
    _sort_summary(summary).to_csv(summary_path, index=False)
    splits.to_csv(split_path, index=False)
    report_path = write_markdown_report(
        run_dir,
        config,
        features,
        all_trades,
        summary,
        splits,
    )
    return {
        "run_dir": run_dir,
        "features": features_path,
        "trades": trades_path,
        "trades_csv": trades_csv_path,
        "summary": summary_path,
        "period_splits": split_path,
        "report": report_path,
    }


def _backtest_config_from_experiment(config: dict[str, Any]) -> BacktestConfig:
    quote_filter = config.get("quote_filter", {})
    costs = config.get("costs", {})
    return BacktestConfig(
        cost_config=ExecutionCostConfig(
            commission_per_contract=float(costs.get("commission_per_contract", 0.65)),
            exchange_fee_per_contract=float(costs.get("exchange_fee_per_contract", 0.02)),
            slippage_per_contract=float(costs.get("slippage_per_contract", 0.0)),
        ),
        margin_requirement_pct=float(config.get("margin_requirement_pct", 1.0)),
        quote_filter_config=QuoteFilterConfig(
            max_relative_spread=quote_filter.get("max_relative_spread", 0.50),
            min_open_interest=int(quote_filter.get("min_open_interest", 1)),
            min_volume=int(quote_filter.get("min_volume", 0)),
            require_iv_positive=bool(quote_filter.get("require_iv_positive", False)),
        ),
    )


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    for col in ("sharpe", "capital_return_sum", "trades"):
        if col not in summary.columns:
            summary[col] = pd.NA
    return summary.sort_values(
        ["sharpe", "capital_return_sum", "trades"],
        ascending=[False, False, False],
        na_position="last",
    )


def write_markdown_report(
    run_dir: Path,
    config: dict[str, Any],
    features: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    splits: pd.DataFrame,
) -> Path:
    report_path = run_dir / "report.md"
    top = _sort_summary(summary).head(15)
    lines = [
        "# QQQ Options Trigger Research Report",
        "",
        "This is a framework validation run, not a final optimized strategy recommendation.",
        "",
        "## Assumptions",
        "",
        f"- Data window: `{config['data_start']}` to `{config['data_end']}`.",
        f"- Research signal window: `{config.get('research_start', config['data_start'])}` to `{config.get('research_end', config['data_end'])}`.",
        f"- Entry lag: `{config.get('entry_lag_days', 1)}` trading day. EOD return and IV signals enter on the next available trading day.",
        "- Entry convention: Day T EOD signals may only enter on T+1 or later; same-day EOD fills are rejected by `run_experiment`.",
        "- Default execution: bid/ask by leg; other price modes are sensitivity runs.",
        f"- Costs: commission `${config.get('costs', {}).get('commission_per_contract', 0.65)}` + exchange fee `${config.get('costs', {}).get('exchange_fee_per_contract', 0.02)}` per contract per side.",
        f"- Capital model: short put margin requirement `{config.get('margin_requirement_pct', 1.0)}` of strike notional; put spreads use max loss.",
        f"- Risk-free rate for Sharpe: `{config.get('risk_free_rate', 0.0)}` annualized.",
        "- Quote selection uses configurable bid/ask/mid, spread, OI, volume, and IV quality filters.",
        "- Failed exits are retained in `trades` with non-closed status and excluded from closed-trade PnL metrics.",
        "- Metrics are per-parameter trade-stream metrics using capital-required normalization, not a complete portfolio allocator.",
        "- IV/RV spread is reserved in the design but not active in this first run.",
        "",
        "## Dataset",
        "",
        f"- Feature rows: `{len(features)}`.",
        f"- Trade rows: `{len(trades)}`.",
        f"- Parameter summaries: `{len(summary)}`.",
        "",
        "## Top Rows By Sharpe",
        "",
    ]
    if top.empty:
        lines.append("No trades were generated.")
    else:
        display_cols = [
            "trigger_kind",
            "trigger_threshold",
            "strategy",
            "dte_target",
            "hold_days",
            "price_mode",
            "short_put_delta",
            "long_call_delta",
            "put_spread_width_pct",
            "call_dte_target",
            "trades",
            "closed_trades",
            "failed_trades",
            "total_pnl",
            "gross_pnl",
            "total_costs",
            "capital_return_sum",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "max_drawdown_duration_days",
            "win_rate",
            "max_single_loss",
            "max_capital_required",
        ]
        lines.extend(_markdown_table(top[display_cols]))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `daily_features.csv`",
            "- `trades.parquet` and `trades.csv`",
            "- `summary.csv`",
            "- `period_splits.csv`",
            "",
            "Holdout periods are emitted in `period_splits.csv`; for short sample windows they may be empty or not meaningful.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    headers = list(frame.columns)
    rows = []
    for _, row in frame.iterrows():
        rows.append([_format_cell(row[col]) for col in headers])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
