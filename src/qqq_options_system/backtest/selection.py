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
