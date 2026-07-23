from __future__ import annotations

import json
import math
import shutil
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_options_system.backtest.engine import BacktestConfig, StrategySpec, load_chain_partitions, run_strategy
from qqq_options_system.backtest.execution import ExecutionCostConfig
from qqq_options_system.backtest.metrics import summarize_trades
from qqq_options_system.backtest.portfolio import PORTFOLIO_SCOPE
from qqq_options_system.data.normalize import QuoteFilterConfig
from qqq_options_system.research.regimes import build_regime_features, load_vix_vxn_snapshot
from qqq_options_system.research.experiments import _markdown_table
from qqq_options_system.research.walk_forward import (
    prefix_metrics,
    rank_walk_forward_candidates,
    robustness_fields,
)


def load_regime_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_regime_study(config: dict[str, Any], output_root: Path) -> dict[str, Path]:
    if int(config.get("entry_lag_days", 1)) != 1:
        raise ValueError("Regime study requires entry_lag_days=1")
    underlying = config.get("underlying", "QQQ").upper()
    start = config["data_start"]
    end = config["data_end"]
    chains = load_chain_partitions(output_root, start, end, underlying)
    if not chains:
        raise FileNotFoundError(f"No QQQ parquet partitions found for {start}..{end}")
    vix_source = Path(config["vix_vxn_csv"])
    vix_vxn = load_vix_vxn_snapshot(vix_source, start, end)
    regimes = build_regime_features(chains, vix_vxn, config)
    run_dir = output_root / "backtest_results" / f"regime_{underlying}_{start.replace('-', '')}_{end.replace('-', '')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata" / "external"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = metadata_dir / "vix_vxn_2007_2024.csv"
    if vix_source.exists():
        shutil.copyfile(vix_source, snapshot_path)
    source_note = metadata_dir / "vix_vxn_sources.txt"
    source_note.write_text(
        "VIXCLS: https://fred.stlouisfed.org/series/VIXCLS\n"
        "VXNCLS: https://fred.stlouisfed.org/series/VXNCLS\n"
        f"Snapshot: {snapshot_path}\n",
        encoding="utf-8",
    )

    daily_csv = run_dir / "daily_regimes.csv"
    daily_parquet = run_dir / "daily_regimes.parquet"
    regimes.to_csv(daily_csv, index=False)
    regimes.to_parquet(daily_parquet, index=False)

    trades = build_conditional_trades(chains, regimes, config)
    trades_path = run_dir / "conditional_trades.parquet"
    trades_csv_path = run_dir / "conditional_trades.csv"
    trades.to_parquet(trades_path, index=False)
    trades.to_csv(trades_csv_path, index=False)
    summary = summarize_conditional(trades, config)
    summary_path = run_dir / "conditional_by_vix_regime.csv"
    summary.to_csv(summary_path, index=False)
    portfolio_daily = simulate_minimal_portfolio(regimes, config)
    portfolio_path = run_dir / "portfolio_daily.parquet"
    portfolio_csv_path = run_dir / "portfolio_daily.csv"
    portfolio_daily.to_parquet(portfolio_path, index=False)
    portfolio_daily.to_csv(portfolio_csv_path, index=False)
    portfolio_metrics = portfolio_summary(portfolio_daily)
    portfolio_metrics_path = run_dir / "portfolio_metrics.csv"
    portfolio_metrics.to_csv(portfolio_metrics_path, index=False)
    wf_path = run_dir / "walk_forward_regime_candidates.csv"
    walk_forward = walk_forward_regime_candidates(trades, config)
    walk_forward.to_csv(wf_path, index=False)
    report_path = run_dir / "report.md"
    write_regime_report(report_path, config, regimes, trades, summary, walk_forward)
    return {
        "run_dir": run_dir,
        "daily_regimes_csv": daily_csv,
        "daily_regimes": daily_parquet,
        "conditional_trades": trades_path,
        "conditional_by_vix_regime": summary_path,
        "portfolio_daily": portfolio_path,
        "portfolio_metrics": portfolio_metrics_path,
        "walk_forward_regime_candidates": wf_path,
        "report": report_path,
        "vix_vxn_snapshot": snapshot_path,
    }


def backtest_config(config: dict[str, Any]) -> BacktestConfig:
    costs = config.get("costs", {})
    quote = config.get("quote_filter", {})
    return BacktestConfig(
        cost_config=ExecutionCostConfig(
            commission_per_contract=float(costs.get("commission_per_contract", 0.65)),
            exchange_fee_per_contract=float(costs.get("exchange_fee_per_contract", 0.02)),
            slippage_per_contract=float(costs.get("slippage_per_contract", 0.0)),
        ),
        margin_requirement_pct=float(config.get("margin_requirement_pct", 0.35)),
        quote_filter_config=QuoteFilterConfig(
            max_relative_spread=quote.get("max_relative_spread", 0.5),
            min_open_interest=int(quote.get("min_open_interest", 1)),
            min_volume=int(quote.get("min_volume", 0)),
            require_iv_positive=bool(quote.get("require_iv_positive", False)),
        ),
    )


def iter_regime_strategy_specs(config: dict[str, Any], regime: str):
    price_modes = config.get("price_modes", ["bid_ask"])
    for price_mode in price_modes:
        for hold in config.get("baseline_hold_days", [5, 10, 20]):
            yield StrategySpec("qqq_baseline", 0, int(hold), price_mode)
        for dte in config.get("long_option_dtes", [30, 60, 90]):
            for delta in config.get("long_call_deltas", [0.35, 0.45, 0.55]):
                yield StrategySpec("long_call", int(dte), int(config.get("option_hold_days", 10)), price_mode, long_call_delta=float(delta))
            for delta in config.get("long_put_deltas", [-0.15, -0.25, -0.35]):
                yield StrategySpec("long_put", int(dte), int(config.get("option_hold_days", 10)), price_mode, long_put_delta=float(delta))
            for delta in config.get("long_put_spread_deltas", [-0.25, -0.35]):
                for wing in config.get("long_put_spread_width_pcts", [0.025, 0.05]):
                    yield StrategySpec("long_put_spread", int(dte), int(config.get("option_hold_days", 10)), price_mode, long_put_delta=float(delta), put_spread_width_pct=float(wing))
        if regime == "normalized":
            for dte in config.get("short_put_spread_dtes", [14, 30, 45]):
                for delta in config.get("short_put_spread_deltas", [-0.15, -0.25, -0.30]):
                    yield StrategySpec("put_spread", int(dte), int(config.get("short_premium_hold_days", 5)), price_mode, short_put_delta=float(delta), put_spread_width_pct=float(config.get("short_put_spread_width_pct", 0.05)))


def build_conditional_trades(
    chains: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    dates = sorted(chains)
    date_to_index = {date: index for index, date in enumerate(dates)}
    cfg = backtest_config(config)
    rows: list[pd.DataFrame] = []
    regime_by_date = regimes.set_index("date").to_dict("index")
    for date_text in regimes["date"].astype(str):
        if date_text not in date_to_index:
            continue
        regime_row = regime_by_date[date_text]
        for spec in iter_regime_strategy_specs(config, str(regime_row["regime"])):
            if spec.strategy == "qqq_baseline":
                baseline = baseline_trade(dates, chains, date_text, spec, config)
                if baseline:
                    rows.append(pd.DataFrame([attach_regime_fields(baseline, regime_row, spec)]))
                continue
            trades = run_strategy(chains, [date_text], spec, entry_lag_days=1, config=cfg)
            if trades.empty:
                continue
            for key, value in regime_fields(regime_row).items():
                trades[key] = value
            trades["net_pnl"] = trades["pnl"]
            rows.append(trades)
    if not rows:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        out = pd.concat(rows, ignore_index=True)
    out["non_overlap_eligible"] = non_overlap_flags(out)
    return out


def baseline_trade(
    dates: list[str],
    chains: dict[str, pd.DataFrame],
    signal_date: str,
    spec: StrategySpec,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    entry_lag = int(config.get("entry_lag_days", 1))
    signal_index = {date: index for index, date in enumerate(dates)}.get(signal_date)
    if signal_index is None:
        return None
    entry_index = signal_index + entry_lag
    exit_index = entry_index + int(spec.hold_days)
    if entry_index >= len(dates) or exit_index >= len(dates):
        return None
    entry_date = dates[entry_index]
    exit_date = dates[exit_index]
    entry = float(pd.to_numeric(chains[entry_date]["underlying_price"], errors="coerce").dropna().iloc[0])
    exit_px = float(pd.to_numeric(chains[exit_date]["underlying_price"], errors="coerce").dropna().iloc[0])
    gross = exit_px - entry
    risk = entry
    return {
        "strategy_id": spec.label,
        "strategy": "qqq_baseline",
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_lag_days": entry_lag,
        "hold_days": spec.hold_days,
        "dte_target": spec.dte_target,
        "price_mode": spec.price_mode,
        "status": "closed",
        "failure_reason": "",
        "legs": 1,
        "capital_required": risk,
        "max_loss": risk,
        "gross_pnl": gross,
        "total_costs": 0.0,
        "net_pnl": gross,
        "pnl": gross,
        "return_on_risk": gross / risk if risk else 0.0,
        "return_on_capital": gross / risk if risk else 0.0,
        "pnl_per_1000_risk": gross / risk * 1000.0 if risk else 0.0,
    }


def regime_fields(row: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "vix": row.get("vix"),
        "vxn": row.get("vxn"),
        "vix_bucket": row.get("vix_bucket"),
        "vix_percentile_bucket": row.get("vix_percentile_bucket"),
        "regime": row.get("regime"),
        "regime_reason": row.get("regime_reason"),
        "allow_new_short_gamma": row.get("allow_new_short_gamma"),
    }
    for key in (
        "iv30",
        "iv90",
        "term_spread",
        "term_ratio",
        "rv21",
        "rv63",
        "iv_rv_spread",
        "trend_state",
        "vxn_direction",
    ):
        fields[key] = row.get(key)
    return fields


def attach_regime_fields(trade: dict[str, Any], regime_row: dict[str, Any], spec: StrategySpec) -> dict[str, Any]:
    out = dict(trade)
    out.update(regime_fields(regime_row))
    return out


def non_overlap_flags(trades: pd.DataFrame) -> pd.Series:
    flags = []
    last_exit_by_strategy: dict[str, str] = {}
    for _, row in trades.sort_values(["strategy_id", "entry_date"]).iterrows():
        key = str(row["strategy_id"])
        entry = str(row["entry_date"])
        if key not in last_exit_by_strategy or entry > last_exit_by_strategy[key]:
            flags.append(True)
            last_exit_by_strategy[key] = str(row["exit_date"])
        else:
            flags.append(False)
    result = pd.Series(flags, index=trades.sort_values(["strategy_id", "entry_date"]).index)
    return result.reindex(trades.index).fillna(False).astype(bool)


def summarize_conditional(trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["vix_bucket", "vix_percentile_bucket", "regime", "strategy", "strategy_id"]
    for keys, group in trades.groupby(group_cols, dropna=False):
        metrics = summarize_trades(group, risk_free_rate=float(config.get("risk_free_rate", 0.0)))
        closed = group.loc[group.get("status", "closed") == "closed"]
        returns = pd.to_numeric(closed.get("return_on_risk", pd.Series(dtype=float)), errors="coerce").dropna()
        metrics["mean_return_on_risk"] = float(returns.mean()) if not returns.empty else 0.0
        metrics["median_return_on_risk"] = float(returns.median()) if not returns.empty else 0.0
        metrics["bootstrap_mean_return_low"], metrics["bootstrap_mean_return_high"] = bootstrap_ci(returns)
        metrics["sample_warning"] = "sparse" if int(metrics.get("closed_trades", 0)) < int(config.get("sparse_trade_warning_threshold", 30)) else ""
        row = dict(zip(group_cols, keys))
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: pd.Series) -> tuple[float | None, float | None]:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return None, None
    means = []
    for offset in range(200):
        sample = clean.sample(n=len(clean), replace=True, random_state=10_000 + offset)
        means.append(float(sample.mean()))
    return float(pd.Series(means).quantile(0.025)), float(pd.Series(means).quantile(0.975))


def simulate_minimal_portfolio(regimes: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    nav0 = float(config.get("initial_nav", 100000.0))
    entry_lag_days = int(config.get("entry_lag_days", 1))
    if entry_lag_days < 0:
        raise ValueError("entry_lag_days must be non-negative")
    out = regimes[["date", "qqq_close", "vix", "regime"]].dropna(subset=["qqq_close"]).copy()
    out["qqq_return"] = out["qqq_close"].pct_change().fillna(0.0)
    allocations = config.get("portfolio_targets", {})
    default_joint = float(allocations.get("transition_no_trade", 0.75))
    out["joint_target"] = out["regime"].map(lambda state: float(allocations.get(state, default_joint)))
    out["vix_only_target"] = out["vix"].map(lambda value: 0.5 if value >= 25 else (1.0 if value < 20 else 0.75))
    # A target observed at close T and entered at close T+lag first earns the
    # following close-to-close return, hence the additional one-row shift.
    effective_lag = entry_lag_days + 1
    out["joint_effective_target"] = out["joint_target"].shift(effective_lag).fillna(1.0)
    out["vix_only_effective_target"] = out["vix_only_target"].shift(effective_lag).fillna(1.0)
    out["target_effective_lag_rows"] = effective_lag
    out["buy_hold_nav"] = nav0 * (1.0 + out["qqq_return"]).cumprod()
    out["joint_nav"] = nav0 * (1.0 + out["joint_effective_target"] * out["qqq_return"]).cumprod()
    out["vix_only_nav"] = nav0 * (1.0 + out["vix_only_effective_target"] * out["qqq_return"]).cumprod()
    out["portfolio_scope"] = PORTFOLIO_SCOPE
    return out


def portfolio_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("buy_hold_nav", "joint_nav", "vix_only_nav"):
        returns = portfolio[col].pct_change().fillna(0.0)
        rows.append(
            {
                "portfolio": col.replace("_nav", ""),
                "portfolio_scope": PORTFOLIO_SCOPE,
                "start_nav": float(portfolio[col].iloc[0]) if not portfolio.empty else 0.0,
                "end_nav": float(portfolio[col].iloc[-1]) if not portfolio.empty else 0.0,
                "total_return": float(portfolio[col].iloc[-1] / portfolio[col].iloc[0] - 1.0) if len(portfolio) else 0.0,
                "annualized_vol": float(returns.std(ddof=0) * math.sqrt(252)) if len(returns) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _regime_walk_forward_periods(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    defaults = {
        "training": {"start": "2007-01-01", "end": "2016-12-31"},
        "validation": {"start": "2017-01-01", "end": "2020-12-31"},
        "test": {"start": "2021-01-01", "end": "2024-12-31"},
    }
    nested = config.get("periods", {})
    periods: dict[str, dict[str, str]] = {}
    for name in ("training", "validation", "test"):
        explicit = config.get(f"{name}_period")
        period = explicit if explicit is not None else nested.get(name, defaults[name])
        if not isinstance(period, dict) or "start" not in period or "end" not in period:
            raise ValueError(f"{name}_period with start/end is required for regime walk-forward")
        periods[name] = {"start": str(period["start"]), "end": str(period["end"])}
    return periods


def walk_forward_regime_candidates(
    conditional_trades: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Rank regime/strategy candidates using training trades only.

    Validation and test metrics are attached after each candidate's training
    metrics are computed. They are never inputs to eligibility or sorting.
    """
    if conditional_trades.empty:
        return pd.DataFrame()
    required = {"signal_date", "regime", "strategy", "strategy_id"}
    missing = required.difference(conditional_trades.columns)
    if missing:
        raise ValueError(f"conditional_trades missing required columns: {sorted(missing)}")

    trades = conditional_trades.copy()
    price_mode = str(config.get("walk_forward_price_mode", "bid_ask"))
    if "price_mode" in trades.columns:
        trades = trades.loc[trades["price_mode"].astype(str) == price_mode].copy()
    if bool(config.get("walk_forward_non_overlap_only", True)):
        if "non_overlap_eligible" not in trades.columns:
            raise ValueError("non_overlap_eligible is required for non-overlap walk-forward")
        trades = trades.loc[trades["non_overlap_eligible"].fillna(False).astype(bool)].copy()
    if trades.empty:
        return pd.DataFrame()

    trades["_signal_date"] = pd.to_datetime(trades["signal_date"], errors="coerce")
    trades = trades.loc[trades["_signal_date"].notna()].copy()
    periods = _regime_walk_forward_periods(config)
    risk_free_rate = float(config.get("risk_free_rate", 0.0))
    sortino_target_return = float(config.get("sortino_target_return", 0.0))
    candidate_cols = ["regime", "strategy", "strategy_id"]
    parameter_cols = [
        "dte_target",
        "hold_days",
        "price_mode",
        "short_put_delta",
        "long_call_delta",
        "long_put_delta",
        "put_spread_width_pct",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in trades.groupby(candidate_cols, dropna=False):
        row: dict[str, Any] = dict(zip(candidate_cols, keys))
        for column in parameter_cols:
            if column in group.columns:
                row[column] = group[column].iloc[0]
        for name, period in periods.items():
            start = pd.Timestamp(period["start"])
            end = pd.Timestamp(period["end"])
            period_trades = group.loc[
                (group["_signal_date"] >= start) & (group["_signal_date"] <= end)
            ].drop(columns=["_signal_date"])
            metrics = summarize_trades(
                period_trades,
                risk_free_rate=risk_free_rate,
                sortino_target_return=sortino_target_return,
            )
            row.update(prefix_metrics(metrics, name))
        row.update(robustness_fields(row, config.get("max_holdout_sharpe_decay")))
        rows.append(row)

    selection_metric = str(config.get("selection_metric", "sharpe"))
    candidates = pd.DataFrame(rows)
    training_metric_col = f"training_{selection_metric}"
    if training_metric_col not in candidates.columns:
        candidates[training_metric_col] = None
    ranked = rank_walk_forward_candidates(
        candidates,
        selection_metric=selection_metric,
        top_n=int(config.get("top_n_candidates", 50)),
        min_training_trades=int(config.get("min_training_trades", 30)),
    )
    ranked["grid_search_scope"] = "training_only"
    ranked["walk_forward_price_mode"] = price_mode
    ranked["walk_forward_non_overlap_only"] = bool(
        config.get("walk_forward_non_overlap_only", True)
    )
    for name, period in periods.items():
        ranked[f"{name}_period"] = f"{period['start']}..{period['end']}"
    return ranked


def write_regime_report(
    path: Path,
    config: dict[str, Any],
    regimes: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    walk_forward: pd.DataFrame | None = None,
) -> None:
    state_counts = regimes["regime"].value_counts(dropna=False).reset_index()
    state_counts.columns = ["regime", "days"]
    top = summary.sort_values("mean_return_on_risk", ascending=False).head(15) if not summary.empty else pd.DataFrame()
    lines = [
        "# QQQ Five-Factor VIX Regime Study",
        "",
        "Historical research only. Results are conditional on the candidate set, execution costs, capital model, and local ORATS/FRED data; they are not investment advice or a return promise.",
        "",
        "## Timing And Data",
        "",
        "- Features are observed at Day T close; trades enter on T+1 (`entry_lag_days=1`).",
        "- VIX/VXN use FRED `VIXCLS` and `VXNCLS`; ORATS VIX option rows are not used as VIX closes.",
        "- Term structure is QQQ option ATM IV30 versus IV90, not VIX futures.",
        "- Conditional cohort summaries are full-sample diagnostics. Only `walk_forward_regime_candidates.csv` performs training-only candidate ranking.",
        "",
        "## Portfolio Scope",
        "",
        f"- `portfolio_scope={PORTFOLIO_SCOPE}`: portfolio NAV applies regime-dependent QQQ equity allocations only.",
        "- With EOD-only prices, a Day T close signal enters at Day T+1 close and first earns the Day T+1-to-T+2 close return.",
        "- Option trade PnL is reported in the conditional trade/cohort artifacts and is not included in portfolio NAV.",
        "",
        "## Regime Counts",
        "",
        *_markdown_table(state_counts),
        "",
        f"Conditional trades: `{len(trades)}`",
        "",
        "## Top Conditional Cohorts By Mean Return On Risk",
        "",
    ]
    if top.empty:
        lines.append("No cohort rows were generated.")
    else:
        cols = [col for col in ["vix_bucket", "regime", "strategy", "strategy_id", "closed_trades", "mean_return_on_risk", "bootstrap_mean_return_low", "bootstrap_mean_return_high", "sample_warning"] if col in top.columns]
        lines.extend(_markdown_table(top[cols]))
    selected_count = 0 if walk_forward is None else len(walk_forward)
    lines.extend(
        [
            "",
            "## Walk-Forward Selection",
            "",
            "Candidates use `signal_date` to split training, validation, and test periods. Ranking and minimum-trade eligibility use training metrics only; validation and test are evaluation-only.",
            "",
            f"Training-selected candidates: `{selected_count}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
