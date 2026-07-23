from __future__ import annotations

import pandas as pd

from qqq_options_system.data.normalize import QuoteFilterConfig, filter_quote_quality


def select_by_delta_dte(
    chain: pd.DataFrame,
    right: str,
    target_delta: float,
    target_dte: int,
    min_dte: int = 1,
    quote_filter_config: QuoteFilterConfig | None = None,
    dte_weight: float = 1.0,
    delta_weight: float = 1.0,
) -> pd.Series | None:
    tradeable = filter_quote_quality(chain, quote_filter_config)
    candidates = tradeable.loc[
        (tradeable["right"] == right)
        & tradeable["dte"].notna()
        & (tradeable["dte"].astype(float) >= min_dte)
        & tradeable["delta"].notna()
    ].copy()
    if candidates.empty:
        return None
    dte_denominator = max(1.0, float(target_dte))
    delta_denominator = 0.50
    candidates["dte_score"] = (
        (candidates["dte"].astype(float) - float(target_dte)).abs() / dte_denominator
    ) * float(dte_weight)
    candidates["delta_score"] = (
        (candidates["delta"].astype(float) - float(target_delta)).abs() / delta_denominator
    ) * float(delta_weight)
    candidates["selection_score"] = candidates["dte_score"] + candidates["delta_score"]
    candidates["delta_error"] = (candidates["delta"].astype(float) - float(target_delta)).abs()
    candidates["dte_error"] = (candidates["dte"].astype(float) - float(target_dte)).abs()
    return candidates.sort_values(
        ["selection_score", "delta_error", "dte_error", "strike"]
    ).iloc[0]


def select_put_wing_by_width(
    chain: pd.DataFrame,
    short_put: pd.Series,
    width_pct: float,
    quote_filter_config: QuoteFilterConfig | None = None,
) -> pd.Series | None:
    """Select a lower-strike put near ``short strike - spot * width_pct``."""
    target_strike = float(short_put["strike"]) - float(short_put["underlying_price"]) * width_pct
    tradeable = filter_quote_quality(chain, quote_filter_config)
    candidates = tradeable.loc[
        (tradeable["right"] == "P")
        & (tradeable["expiry"] == short_put["expiry"])
        & (tradeable["strike"].astype(float) < float(short_put["strike"]))
    ].copy()
    if candidates.empty:
        return None
    candidates["wing_score"] = (candidates["strike"].astype(float) - target_strike).abs()
    return candidates.sort_values(["wing_score", "strike"], ascending=[True, False]).iloc[0]


def select_call_within_premium_budget(
    chain: pd.DataFrame,
    call_dte_target: int,
    premium_budget: float,
    entry_cost_column: str = "ask",
    min_dte: int = 1,
    dte_tolerance: int | None = None,
    required_expiry: str | None = None,
    quote_filter_config: QuoteFilterConfig | None = None,
) -> pd.Series | None:
    """Pick the most expensive (highest-delta) call whose entry cost fits the budget.

    The call is financed by the short-put premium, so ``premium_budget`` is the
    per-share credit collected from the put. Only calls whose per-share entry
    cost (``entry_cost_column``) is at or below the budget are eligible. Among
    those closest to ``call_dte_target``, the highest-cost call is chosen because
    that maximizes long-call delta / upside participation within the budget.
    """
    if premium_budget <= 0:
        return None
    tradeable = filter_quote_quality(chain, quote_filter_config)
    candidates = tradeable.loc[
        (tradeable["right"] == "C")
        & tradeable["dte"].notna()
        & (tradeable["dte"].astype(float) >= min_dte)
        & tradeable["delta"].notna()
    ].copy()
    if candidates.empty:
        return None

    if required_expiry is not None:
        candidates = candidates.loc[
            candidates["expiry"].astype(str) == str(required_expiry)
        ].copy()
        if candidates.empty:
            return None

    if entry_cost_column not in candidates.columns:
        raise ValueError(f"entry_cost_column not in chain: {entry_cost_column!r}")
    entry_cost = pd.to_numeric(candidates[entry_cost_column], errors="coerce")
    candidates = candidates.loc[entry_cost.notna() & (entry_cost <= float(premium_budget))].copy()
    if candidates.empty:
        return None

    # Restrict to calls near the requested DTE so budget selection compares like
    # maturities. Without a tolerance the nearest available DTE bucket is used.
    dte_error = (candidates["dte"].astype(float) - float(call_dte_target)).abs()
    if dte_tolerance is not None:
        candidates = candidates.loc[dte_error <= float(dte_tolerance)].copy()
        if candidates.empty:
            return None
        dte_error = (candidates["dte"].astype(float) - float(call_dte_target)).abs()
    else:
        nearest = float(dte_error.min())
        candidates = candidates.loc[dte_error == nearest].copy()

    candidates["entry_cost"] = pd.to_numeric(candidates[entry_cost_column], errors="coerce")
    candidates["dte_error"] = (candidates["dte"].astype(float) - float(call_dte_target)).abs()
    # Highest cost within budget -> most ITM / highest delta; break ties by
    # closeness to the target DTE, then higher delta.
    return candidates.sort_values(
        ["entry_cost", "dte_error", "delta"], ascending=[False, True, False]
    ).iloc[0]
