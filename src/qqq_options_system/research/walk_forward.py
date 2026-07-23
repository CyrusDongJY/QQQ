from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from qqq_options_system.backtest.engine import load_chain_partitions, run_strategy
from qqq_options_system.backtest.metrics import summarize_trades
from qqq_options_system.research.experiments import (
    _backtest_config_from_experiment,
    _format_cell,
    _markdown_table,
    iter_strategy_specs,
    iter_triggers,
)
from qqq_options_system.research.features import build_daily_features, signal_dates_for_trigger


ALLOWED_GRID_SEARCH_SCOPES = {"training_only"}
ALLOWED_SELECTION_METRICS = {"sharpe", "calmar", "total_pnl", "capital_return_sum"}


def validate_walk_forward_config(config: dict[str, Any]) -> None:
    scope = config.get("grid_search_scope")
    if scope not in ALLOWED_GRID_SEARCH_SCOPES:
        raise ValueError(
            f"grid_search_scope must be one of {sorted(ALLOWED_GRID_SEARCH_SCOPES)}, got {scope!r}"
        )
    metric = config.get("selection_metric", "sharpe")
    if metric not in ALLOWED_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {sorted(ALLOWED_SELECTION_METRICS)}, got {metric!r}"
        )
    for period_name in ("training_period", "validation_period", "test_period"):
        period = config.get(period_name)
        if not isinstance(period, dict) or "start" not in period or "end" not in period:
            raise ValueError(f"{period_name} with start/end is required for walk-forward")
    if int(config.get("entry_lag_days", 1)) < 1:
        raise ValueError("EOD trigger walk-forward requires entry_lag_days >= 1")


def run_walk_forward_experiment(config: dict[str, Any], output_root: Path) -> dict[str, Path]:
    validate_walk_forward_config(config)
    underlying = config.get("underlying", "QQQ").upper()
    data_start = config["data_start"]
    data_end = config["data_end"]
    entry_lag_days = int(config.get("entry_lag_days", 1))
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

    rows: list[dict[str, Any]] = []
    trades_frames: list[pd.DataFrame] = []
    periods = {
        "training": config["training_period"],
        "validation": config["validation_period"],
        "test": config["test_period"],
    }
    period_dates = {
        name: [date for date in sorted(chains) if period["start"] <= date <= period["end"]]
        for name, period in periods.items()
    }

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
            row: dict[str, Any] = {
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
            for name, period in periods.items():
                period_trades = filter_trades_by_entry_period(trades, period["start"], period["end"])
                metrics = summarize_trades(
                    period_trades,
                    all_dates=period_dates[name],
                    risk_free_rate=risk_free_rate,
                    sortino_target_return=sortino_target_return,
                )
                row.update(prefix_metrics(metrics, name))
            row.update(robustness_fields(row, config.get("max_holdout_sharpe_decay")))
            rows.append(row)

    candidates = rank_walk_forward_candidates(
        pd.DataFrame(rows),
        selection_metric=config.get("selection_metric", "sharpe"),
        top_n=int(config.get("top_n_candidates", 50)),
        min_training_trades=int(config.get("min_training_trades", 30)),
    )

    run_dir = (
        output_root
        / "backtest_results"
        / f"walk_forward_{underlying}_{data_start.replace('-', '')}_{data_end.replace('-', '')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    all_candidates_path = run_dir / "walk_forward_all_candidates.csv"
    candidates_path = run_dir / "walk_forward_candidates.csv"
    trades_path = run_dir / "walk_forward_trades.parquet"
    report_path = run_dir / "walk_forward_report.md"

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(all_candidates_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    if trades_frames:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
                category=FutureWarning,
            )
            pd.concat(trades_frames, ignore_index=True).to_parquet(trades_path, index=False)
    else:
        pd.DataFrame().to_parquet(trades_path, index=False)
    write_walk_forward_report(report_path, config, all_rows, candidates)
    return {
        "run_dir": run_dir,
        "all_candidates": all_candidates_path,
        "candidates": candidates_path,
        "trades": trades_path,
        "report": report_path,
    }


def filter_trades_by_entry_period(trades: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if trades.empty or "entry_date" not in trades.columns:
        return trades.iloc[0:0].copy()
    dated = trades.copy()
    entry_dates = pd.to_datetime(dated["entry_date"], errors="coerce")
    mask = (entry_dates >= pd.Timestamp(start)) & (entry_dates <= pd.Timestamp(end))
    return dated.loc[mask].copy()


def prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def robustness_fields(row: dict[str, Any], max_holdout_sharpe_decay: Any = None) -> dict[str, Any]:
    training = row.get("training_sharpe")
    validation = row.get("validation_sharpe")
    test = row.get("test_sharpe")
    fields = {
        "validation_sharpe_decay": sharpe_decay(training, validation),
        "test_sharpe_decay": sharpe_decay(training, test),
    }
    if max_holdout_sharpe_decay is not None:
        limit = float(max_holdout_sharpe_decay)
        fields["validation_decay_pass"] = (
            fields["validation_sharpe_decay"] is not None
            and fields["validation_sharpe_decay"] <= limit
        )
        fields["test_decay_pass"] = (
            fields["test_sharpe_decay"] is not None
            and fields["test_sharpe_decay"] <= limit
        )
    return fields


def sharpe_decay(training: Any, holdout: Any) -> float | None:
    if training is None or holdout is None or pd.isna(training) or pd.isna(holdout):
        return None
    training_value = float(training)
    if training_value == 0:
        return None
    return (training_value - float(holdout)) / abs(training_value)


def rank_walk_forward_candidates(
    candidates: pd.DataFrame,
    selection_metric: str = "sharpe",
    top_n: int = 50,
    min_training_trades: int = 30,
) -> pd.DataFrame:
    if selection_metric not in ALLOWED_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {sorted(ALLOWED_SELECTION_METRICS)}, got {selection_metric!r}"
        )
    if candidates.empty:
        return candidates.copy()
    metric_col = f"training_{selection_metric}"
    trades_col = "training_closed_trades"
    if metric_col not in candidates.columns:
        raise ValueError(f"Missing training metric column: {metric_col}")
    if trades_col not in candidates.columns:
        raise ValueError(f"Missing training trades column: {trades_col}")
    eligible = candidates.loc[
        (pd.to_numeric(candidates[trades_col], errors="coerce").fillna(0) >= min_training_trades)
        & candidates[metric_col].notna()
    ].copy()
    eligible["_training_rank_metric"] = pd.to_numeric(eligible[metric_col], errors="coerce")
    ranked = eligible.sort_values(
        ["_training_rank_metric", "training_closed_trades"],
        ascending=[False, False],
        na_position="last",
    ).head(top_n)
    ranked.insert(0, "training_rank", range(1, len(ranked) + 1))
    return ranked.drop(columns=["_training_rank_metric"])


def write_walk_forward_report(
    path: Path,
    config: dict[str, Any],
    all_candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    top_cols = [
        "training_rank",
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
        "training_closed_trades",
        "training_sharpe",
        "validation_sharpe",
        "test_sharpe",
        "validation_sharpe_decay",
        "test_sharpe_decay",
        "training_total_pnl",
        "validation_total_pnl",
        "test_total_pnl",
    ]
    display = selected[[col for col in top_cols if col in selected.columns]].head(20)
    lines = [
        "# QQQ Walk-Forward Research Report",
        "",
        "This report performs anti-overfit candidate selection. Ranking uses training metrics only; validation and test periods are evaluation-only holdouts.",
        "",
        "## Periods",
        "",
        f"- Training: `{config['training_period']['start']}` to `{config['training_period']['end']}`",
        f"- Validation: `{config['validation_period']['start']}` to `{config['validation_period']['end']}`",
        f"- Test: `{config['test_period']['start']}` to `{config['test_period']['end']}`",
        f"- Grid search scope: `{config.get('grid_search_scope')}`",
        f"- Selection metric: `training_{config.get('selection_metric', 'sharpe')}`",
        f"- Top N candidates: `{config.get('top_n_candidates', 50)}`",
        f"- Minimum training trades: `{config.get('min_training_trades', 30)}`",
        "",
        "## Candidate Counts",
        "",
        f"- Evaluated parameter rows: `{len(all_candidates)}`",
        f"- Selected candidates: `{len(selected)}`",
        "",
        "## Top Training-Ranked Candidates",
        "",
    ]
    if display.empty:
        lines.append("No candidates met the training-only filters.")
    else:
        lines.extend(_markdown_table(display))
    lines.extend(
        [
            "",
            "## Important Distinction",
            "",
            "`period_splits.csv` from normal experiments is diagnostic. `walk_forward_candidates.csv` is the anti-overfit selection artifact because it ranks only on the training period and reports validation/test separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
