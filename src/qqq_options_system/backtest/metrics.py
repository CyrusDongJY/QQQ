from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


SPLIT_PERIODS = {
    "research_2007_2016": ("2007-01-01", "2016-12-31"),
    "validation_2017_2020": ("2017-01-01", "2020-12-31"),
    "holdout_2021_2024": ("2021-01-01", "2024-12-31"),
    "covid_2020": ("2020-02-15", "2020-04-30"),
    "rate_shock_2022": ("2022-01-01", "2022-12-31"),
}


@dataclass(frozen=True)
class MetricRow:
    fields: dict


def _safe_ratio(num: float, den: float) -> float | None:
    if den == 0 or math.isnan(den):
        return None
    return num / den


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    return float(drawdown.min())


def _drawdown_duration_stats(cumulative: pd.Series) -> tuple[int, int]:
    if cumulative.empty:
        return 0, 0
    running_max = cumulative.cummax()
    durations: list[int] = []
    current = 0
    for value, peak in zip(cumulative, running_max):
        if value < peak:
            current += 1
        else:
            durations.append(current)
            current = 0
    durations.append(current)
    max_duration = max(durations) if durations else 0
    current_duration = current
    return int(max_duration), int(current_duration)


def _tail_stats(values: pd.Series, quantile: float) -> tuple[float | None, float | None]:
    clean = values.dropna()
    if clean.empty:
        return None, None
    var = float(clean.quantile(quantile))
    cvar_values = clean.loc[clean <= var]
    cvar = float(cvar_values.mean()) if not cvar_values.empty else var
    return var, cvar


def summarize_trades(
    trades: pd.DataFrame,
    all_dates: list[str] | None = None,
    risk_free_rate: float = 0.0,
    sortino_target_return: float = 0.0,
) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "closed_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "capital_return_sum": 0.0,
        }

    out: dict[str, float | int | str | None] = {}
    if "status" in trades.columns:
        closed = trades.loc[trades["status"] == "closed"].copy()
        failed_count = int((trades["status"] != "closed").sum())
    else:
        closed = trades.copy()
        failed_count = 0
    out["trades"] = int(len(trades))
    out["closed_trades"] = int(len(closed))
    out["failed_trades"] = failed_count
    if closed.empty:
        out.update(
            {
                "total_pnl": 0.0,
                "gross_pnl": 0.0,
                "total_costs": 0.0,
                "capital_return_sum": 0.0,
            }
        )
        return out

    pnl = closed["pnl"].astype(float)
    roc = closed["return_on_capital"].astype(float)
    out["total_pnl"] = float(pnl.sum())
    out["gross_pnl"] = float(closed.get("gross_pnl", pnl).astype(float).sum())
    out["total_costs"] = float(closed.get("total_costs", pd.Series(0.0, index=closed.index)).astype(float).sum())
    out["avg_pnl"] = float(pnl.mean())
    out["avg_return_on_capital"] = float(roc.mean())
    out["capital_return_sum"] = float(roc.sum())
    out["win_rate"] = float((pnl > 0).mean())
    out["avg_win"] = float(pnl.loc[pnl > 0].mean()) if (pnl > 0).any() else 0.0
    out["avg_loss"] = float(pnl.loc[pnl < 0].mean()) if (pnl < 0).any() else 0.0
    out["max_single_loss"] = float(pnl.min())
    out["max_capital_required"] = float(closed["capital_required"].max())
    out["avg_capital_required"] = float(closed["capital_required"].mean())
    out["return_per_capital_required"] = _safe_ratio(
        float(pnl.sum()), float(closed["capital_required"].sum())
    )

    exit_returns = closed.groupby("exit_date")["return_on_capital"].sum().sort_index()
    if all_dates:
        daily = pd.Series(0.0, index=pd.Index(all_dates, name="date"))
        daily.loc[exit_returns.index.intersection(daily.index)] = exit_returns
    else:
        daily = exit_returns
    cumulative = daily.cumsum()
    daily_std = float(daily.std(ddof=0)) if len(daily) > 1 else 0.0
    daily_mean = float(daily.mean()) if len(daily) else 0.0
    daily_rf = float(risk_free_rate) / 252.0
    target_daily = float(sortino_target_return) / 252.0
    excess_daily = daily - daily_rf
    downside_diff = daily - target_daily
    downside = downside_diff.loc[downside_diff < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    years = max(1.0 / 252.0, len(daily) / 252.0)
    out["annualized_return"] = float(daily.sum() / years)
    out["annualized_volatility"] = float(daily_std * math.sqrt(252))
    out["risk_free_rate"] = float(risk_free_rate)
    out["sortino_target_return"] = float(sortino_target_return)
    out["sharpe"] = _safe_ratio(float(excess_daily.mean()), daily_std)
    if out["sharpe"] is not None:
        out["sharpe"] = float(out["sharpe"] * math.sqrt(252))
    out["sortino"] = _safe_ratio(daily_mean - target_daily, downside_std)
    if out["sortino"] is not None:
        out["sortino"] = float(out["sortino"] * math.sqrt(252))
    out["max_drawdown"] = _max_drawdown(cumulative)
    out["max_drawdown_duration_days"], out["current_drawdown_duration_days"] = (
        _drawdown_duration_stats(cumulative)
    )
    out["calmar"] = _safe_ratio(float(out["annualized_return"]), abs(float(out["max_drawdown"])))
    out["var_1pct"], out["cvar_1pct"] = _tail_stats(daily, 0.01)
    out["var_5pct"], out["cvar_5pct"] = _tail_stats(daily, 0.05)
    return out


def split_summary(
    trades: pd.DataFrame,
    risk_free_rate: float = 0.0,
    sortino_target_return: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    if trades.empty:
        return pd.DataFrame()
    dated = trades.copy()
    dated["entry_date"] = pd.to_datetime(dated["entry_date"])
    for label, (start, end) in SPLIT_PERIODS.items():
        mask = (dated["entry_date"] >= pd.Timestamp(start)) & (dated["entry_date"] <= pd.Timestamp(end))
        metrics = summarize_trades(
            dated.loc[mask],
            risk_free_rate=risk_free_rate,
            sortino_target_return=sortino_target_return,
        )
        metrics["period"] = label
        rows.append(metrics)
    for year, group in dated.groupby(dated["entry_date"].dt.year):
        metrics = summarize_trades(
            group,
            risk_free_rate=risk_free_rate,
            sortino_target_return=sortino_target_return,
        )
        metrics["period"] = str(year)
        rows.append(metrics)
    return pd.DataFrame(rows)
