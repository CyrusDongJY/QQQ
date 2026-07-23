from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_options_system.data.normalize import QuoteFilterConfig, filter_quote_quality


@dataclass(frozen=True)
class EntrySignalCalibration:
    training_start: str
    training_end: str
    baseline_return_threshold: float
    baseline_event_rate_q: float
    standardized_shock_threshold: float
    close_location_threshold: float
    training_gate_days: int
    training_shock_observations: int
    training_close_candidates: int


def standardized_shock(
    returns: pd.Series,
    lookback: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Standardize each return using volatility known before that trading day."""
    periods = int(lookback if min_periods is None else min_periods)
    numeric = pd.to_numeric(returns, errors="coerce")
    prior_vol = numeric.rolling(int(lookback), min_periods=periods).std(ddof=1).shift(1)
    prior_vol = prior_vol.where(prior_vol > 0)
    shock = (numeric / prior_vol).replace([float("inf"), float("-inf")], pd.NA)
    return pd.DataFrame(
        {
            "prior_return_vol20": prior_vol,
            "standardized_shock": pd.to_numeric(shock, errors="coerce"),
        },
        index=returns.index,
    )


def _empty_skew_result() -> dict[str, Any]:
    return {
        "skew_expiry": None,
        "skew_dte": math.nan,
        "put25_delta": math.nan,
        "call25_delta": math.nan,
        "put25_mid_iv": math.nan,
        "call25_mid_iv": math.nan,
        "skew25": math.nan,
    }


def select_daily_skew25(
    chain: pd.DataFrame,
    target_dte: int = 45,
    quote_filter_config: QuoteFilterConfig | None = None,
) -> dict[str, Any]:
    """Select same-expiry 25-delta put/call IVs from the nearest 45-DTE expiry."""
    required = {"expiry", "dte", "right", "strike", "delta", "bid", "mid", "ask"}
    if chain.empty or required.difference(chain.columns):
        return _empty_skew_result()
    try:
        tradeable = filter_quote_quality(chain, quote_filter_config)
    except ValueError:
        return _empty_skew_result()
    usable = tradeable.loc[
        tradeable["expiry"].notna()
        & tradeable["dte"].notna()
        & tradeable["delta"].notna()
        & tradeable["right"].isin(["P", "C"])
    ].copy()
    if usable.empty:
        return _empty_skew_result()
    usable["dte_numeric"] = pd.to_numeric(usable["dte"], errors="coerce")
    usable["delta_numeric"] = pd.to_numeric(usable["delta"], errors="coerce")
    usable = usable.dropna(subset=["dte_numeric", "delta_numeric"])

    expiry_rows: list[dict[str, Any]] = []
    for expiry, group in usable.groupby("expiry"):
        rights = set(group["right"].astype(str))
        if not {"P", "C"}.issubset(rights):
            continue
        expiry_rows.append(
            {
                "expiry": str(expiry),
                "dte": float(group["dte_numeric"].median()),
            }
        )
    if not expiry_rows:
        return _empty_skew_result()
    expiries = pd.DataFrame(expiry_rows)
    expiries["dte_error"] = (expiries["dte"] - float(target_dte)).abs()
    selected_expiry = expiries.sort_values(["dte_error", "expiry"]).iloc[0]
    expiry = str(selected_expiry["expiry"])
    same_expiry = usable.loc[usable["expiry"].astype(str) == expiry].copy()

    put_candidates = same_expiry.loc[same_expiry["right"] == "P"].copy()
    call_candidates = same_expiry.loc[same_expiry["right"] == "C"].copy()
    if put_candidates.empty or call_candidates.empty:
        return _empty_skew_result()
    put_candidates["delta_error"] = (put_candidates["delta_numeric"] + 0.25).abs()
    call_candidates["delta_error"] = (call_candidates["delta_numeric"] - 0.25).abs()
    put = put_candidates.sort_values(["delta_error", "strike"]).iloc[0]
    call = call_candidates.sort_values(["delta_error", "strike"]).iloc[0]

    result = _empty_skew_result()
    result.update(
        {
            "skew_expiry": expiry,
            "skew_dte": float(selected_expiry["dte"]),
            "put25_delta": float(put["delta_numeric"]),
            "call25_delta": float(call["delta_numeric"]),
        }
    )
    if "mid_iv" not in same_expiry.columns:
        return result
    put_iv = pd.to_numeric(pd.Series([put.get("mid_iv")]), errors="coerce").iloc[0]
    call_iv = pd.to_numeric(pd.Series([call.get("mid_iv")]), errors="coerce").iloc[0]
    if (
        pd.isna(put_iv)
        or pd.isna(call_iv)
        or not math.isfinite(float(put_iv))
        or not math.isfinite(float(call_iv))
        or float(put_iv) <= 0
        or float(call_iv) <= 0
    ):
        return result
    result["put25_mid_iv"] = float(put_iv)
    result["call25_mid_iv"] = float(call_iv)
    result["skew25"] = float(put_iv) - float(call_iv)
    return result


def build_skew25_features(
    chains: dict[str, pd.DataFrame],
    target_dte: int = 45,
    quote_filter_config: QuoteFilterConfig | None = None,
) -> pd.DataFrame:
    rows = []
    for date_text, chain in sorted(chains.items()):
        row = {"date": str(date_text)}
        row.update(select_daily_skew25(chain, target_dte, quote_filter_config))
        rows.append(row)
    columns = ["date", *_empty_skew_result().keys(), "skew25_5d_change"]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["skew25_5d_change"] = pd.to_numeric(out["skew25"], errors="coerce").diff(5)
    return out


def close_location_features(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Normalize external OHLC and calculate close location without filling gaps."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(Path(source))
    lookup = {str(column).lower(): column for column in frame.columns}
    required = {"date", "high", "low", "close"}
    missing = required.difference(lookup)
    if missing:
        raise ValueError(f"OHLC source missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[lookup["date"]], errors="coerce"),
            "high": pd.to_numeric(frame[lookup["high"]], errors="coerce"),
            "low": pd.to_numeric(frame[lookup["low"]], errors="coerce"),
            "close": pd.to_numeric(frame[lookup["close"]], errors="coerce"),
        }
    )
    denominator = out["high"] - out["low"]
    out["close_location"] = ((out["close"] - out["low"]) / denominator).where(
        denominator > 0
    )
    out = out.loc[out["date"].notna()].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.sort_values("date").reset_index(drop=True)


def build_entry_signal_features(
    daily: pd.DataFrame,
    chains: dict[str, pd.DataFrame],
    ohlc: pd.DataFrame | str | Path,
    quote_filter_config: QuoteFilterConfig | None = None,
    return_column: str = "underlying_return",
    shock_lookback: int = 20,
    shock_min_periods: int | None = None,
) -> pd.DataFrame:
    """Combine same-day entry features; every row is a Day-T signal snapshot."""
    if "date" not in daily.columns or return_column not in daily.columns:
        raise ValueError(f"daily must contain date and {return_column!r}")
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    shock = standardized_shock(
        out[return_column], lookback=shock_lookback, min_periods=shock_min_periods
    )
    out[["prior_return_vol20", "standardized_shock"]] = shock
    out = out.merge(
        build_skew25_features(chains, quote_filter_config=quote_filter_config),
        on="date",
        how="left",
    )
    out = out.merge(close_location_features(ohlc), on="date", how="left")
    return out


def calibrate_training_thresholds(
    features: pd.DataFrame,
    training_start: str,
    training_end: str,
    baseline_return_threshold: float = -0.01,
    gate_column: str = "gate_no_stress",
    return_column: str = "underlying_return",
) -> EntrySignalCalibration:
    """Freeze shock and close thresholds using only gated training rows."""
    required = {
        "date",
        gate_column,
        return_column,
        "standardized_shock",
        "skew25_5d_change",
        "close_location",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing calibration columns: {sorted(missing)}")
    dates = pd.to_datetime(features["date"], errors="coerce")
    gate = features[gate_column].fillna(False).astype(bool)
    training = features.loc[
        (dates >= pd.Timestamp(training_start))
        & (dates <= pd.Timestamp(training_end))
        & gate
    ].copy()
    training["_calibration_return"] = pd.to_numeric(
        training[return_column], errors="coerce"
    )
    training["_calibration_shock"] = pd.to_numeric(
        training["standardized_shock"], errors="coerce"
    )
    calibration_rows = training.loc[
        training["_calibration_return"].notna()
        & training["_calibration_shock"].notna()
    ].copy()
    if calibration_rows.empty:
        raise ValueError(
            "No gated training dates with both valid returns and standardized shocks"
        )
    training_returns = calibration_rows["_calibration_return"]
    training_shocks = calibration_rows["_calibration_shock"]
    event_rate = float((training_returns <= float(baseline_return_threshold)).mean())
    if event_rate <= 0:
        shock_threshold = -math.inf
    elif event_rate >= 1:
        shock_threshold = math.inf
    else:
        shock_threshold = float(training_shocks.quantile(event_rate))

    layer3_candidates = calibration_rows.loc[
        (calibration_rows["_calibration_shock"] <= shock_threshold)
        & (pd.to_numeric(calibration_rows["skew25_5d_change"], errors="coerce") <= 0)
    ]
    close_values = pd.to_numeric(
        layer3_candidates["close_location"], errors="coerce"
    ).dropna()
    close_threshold = float(close_values.median()) if not close_values.empty else math.nan
    return EntrySignalCalibration(
        training_start=str(training_start),
        training_end=str(training_end),
        baseline_return_threshold=float(baseline_return_threshold),
        baseline_event_rate_q=event_rate,
        standardized_shock_threshold=shock_threshold,
        close_location_threshold=close_threshold,
        training_gate_days=int(len(training)),
        training_shock_observations=int(len(training_shocks)),
        training_close_candidates=int(len(close_values)),
    )


def build_ablation_masks(
    features: pd.DataFrame,
    calibration: EntrySignalCalibration,
    gate_column: str = "gate_no_stress",
    return_column: str = "underlying_return",
) -> pd.DataFrame:
    """Return four nested ablation masks without a Cartesian parameter grid."""
    required = {
        "date",
        gate_column,
        return_column,
        "standardized_shock",
        "skew25_5d_change",
        "close_location",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"features missing signal columns: {sorted(missing)}")
    gate = features[gate_column].fillna(False).astype(bool)
    returns = pd.to_numeric(features[return_column], errors="coerce")
    shocks = pd.to_numeric(features["standardized_shock"], errors="coerce")
    skew_change = pd.to_numeric(features["skew25_5d_change"], errors="coerce")
    close_location = pd.to_numeric(features["close_location"], errors="coerce")
    baseline = gate & (returns <= calibration.baseline_return_threshold)
    standardized = gate & (shocks <= calibration.standardized_shock_threshold)
    standardized_skew = standardized & (skew_change <= 0)
    if math.isnan(calibration.close_location_threshold):
        standardized_skew_close = pd.Series(False, index=features.index)
    else:
        standardized_skew_close = standardized_skew & (
            close_location >= calibration.close_location_threshold
        )
    return pd.DataFrame(
        {
            "date": features["date"].astype(str),
            "baseline": baseline.fillna(False).astype(bool),
            "standardized": standardized.fillna(False).astype(bool),
            "standardized_skew_nonaccelerating": standardized_skew.fillna(False).astype(bool),
            "standardized_skew_close_confirmation": standardized_skew_close.fillna(False).astype(bool),
        },
        index=features.index,
    )
