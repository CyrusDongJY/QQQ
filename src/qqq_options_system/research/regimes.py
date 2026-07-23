from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_options_system.research.features import rolling_percentile


VIX_BUCKETS = [
    ("<15", -math.inf, 15.0),
    ("[15,20)", 15.0, 20.0),
    ("[20,25)", 20.0, 25.0),
    ("[25,30)", 25.0, 30.0),
    (">=30", 30.0, math.inf),
]


def vix_bucket(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    for label, lower, upper in VIX_BUCKETS:
        if number >= lower and number < upper:
            return label
    return None


def percentile_bucket(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if number < 20:
        return "0-20"
    if number < 40:
        return "20-40"
    if number < 60:
        return "40-60"
    if number < 80:
        return "60-80"
    return "80-100"


def load_vix_vxn_snapshot(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={"observation_date": "date", "VIXCLS": "vix", "VXNCLS": "vxn"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["vix"] = pd.to_numeric(frame["vix"], errors="coerce")
    frame["vxn"] = pd.to_numeric(frame["vxn"], errors="coerce")
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    return frame.dropna(subset=["vix", "vxn"]).sort_values("date").reset_index(drop=True)


def constant_maturity_atm_iv(chain: pd.DataFrame, target_dte: int) -> float | None:
    if chain.empty:
        return None
    required = {"dte", "strike", "mid_iv", "underlying_price"}
    if required.difference(chain.columns):
        return None
    spot_series = pd.to_numeric(chain["underlying_price"], errors="coerce").dropna()
    if spot_series.empty:
        return None
    spot = float(spot_series.iloc[0])
    usable = chain.loc[
        chain["dte"].notna()
        & chain["strike"].notna()
        & chain["mid_iv"].notna()
        & (pd.to_numeric(chain["mid_iv"], errors="coerce") > 0)
    ].copy()
    if usable.empty:
        return None
    usable["dte_float"] = pd.to_numeric(usable["dte"], errors="coerce")
    usable["strike_float"] = pd.to_numeric(usable["strike"], errors="coerce")
    usable["iv_float"] = pd.to_numeric(usable["mid_iv"], errors="coerce")
    atm_by_dte: list[dict[str, float]] = []
    for dte, group in usable.groupby("dte_float"):
        if pd.isna(dte):
            continue
        group = group.copy()
        group["moneyness_error"] = (group["strike_float"] / spot - 1.0).abs()
        best_error = group["moneyness_error"].min()
        best = group.loc[group["moneyness_error"] == best_error]
        atm_by_dte.append({"dte": float(dte), "iv": float(best["iv_float"].mean())})
    term = pd.DataFrame(atm_by_dte).dropna().sort_values("dte")
    if term.empty:
        return None
    exact = term.loc[term["dte"] == float(target_dte)]
    if not exact.empty:
        return float(exact["iv"].iloc[0])
    lower = term.loc[term["dte"] < float(target_dte)].tail(1)
    upper = term.loc[term["dte"] > float(target_dte)].head(1)
    if lower.empty or upper.empty:
        return None
    lo_dte, lo_iv = float(lower["dte"].iloc[0]), float(lower["iv"].iloc[0])
    hi_dte, hi_iv = float(upper["dte"].iloc[0]), float(upper["iv"].iloc[0])
    if hi_dte == lo_dte:
        return lo_iv
    weight = (float(target_dte) - lo_dte) / (hi_dte - lo_dte)
    return lo_iv + weight * (hi_iv - lo_iv)


def build_regime_features(
    chains: dict[str, pd.DataFrame],
    vix_vxn: pd.DataFrame,
    config: dict[str, Any],
    underlying_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_text, chain in sorted(chains.items()):
        spot_series = pd.to_numeric(chain["underlying_price"], errors="coerce").dropna()
        if spot_series.empty:
            continue
        spot = float(spot_series.iloc[0])
        iv30 = constant_maturity_atm_iv(chain, 30)
        iv90 = constant_maturity_atm_iv(chain, 90)
        rows.append({"date": date_text, "qqq_close": spot, "iv30": iv30, "iv90": iv90})
    option_features = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if underlying_close is None:
        features = option_features
    else:
        close = underlying_close.copy()
        if {"date", "close"}.difference(close.columns):
            raise ValueError("underlying_close must contain date and close columns")
        close["date"] = pd.to_datetime(close["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        close["qqq_close"] = pd.to_numeric(close["close"], errors="coerce")
        close = close.dropna(subset=["date", "qqq_close"])
        close = close.sort_values("date").drop_duplicates("date", keep="last")
        features = close[["date", "qqq_close"]].merge(
            option_features[["date", "iv30", "iv90"]], on="date", how="left"
        )
    features = features.merge(vix_vxn, on="date", how="left")
    features["vix_bucket"] = features["vix"].map(vix_bucket)
    features["vix_percentile"] = rolling_percentile(
        features["vix"],
        lookback=int(config.get("vix_percentile_lookback", 252)),
        min_periods=int(config.get("vix_percentile_min_periods", 60)),
    )
    features["vix_percentile_bucket"] = features["vix_percentile"].map(percentile_bucket)
    features["iv30_percentile"] = rolling_percentile(
        features["iv30"],
        lookback=int(config.get("iv_percentile_lookback", 252)),
        min_periods=int(config.get("iv_percentile_min_periods", 60)),
    )
    features["vxn_1d_change"] = features["vxn"].diff()
    features["vxn_5d_pct_change"] = features["vxn"].pct_change(5, fill_method=None)
    features["vxn_5d_slope"] = features["vxn"].diff(5) / 5.0
    rising = float(config.get("vxn_rising_pct_threshold", 0.05))
    falling = float(config.get("vxn_falling_pct_threshold", -0.05))
    features["vxn_direction"] = "stable"
    features.loc[features["vxn_5d_pct_change"] >= rising, "vxn_direction"] = "rising"
    features.loc[features["vxn_5d_pct_change"] <= falling, "vxn_direction"] = "falling"
    features["term_spread"] = features["iv30"] - features["iv90"]
    features["term_ratio"] = features["iv30"] / features["iv90"]
    inv = float(config.get("term_inverted_spread", 0.01))
    contango = float(config.get("term_contango_spread", -0.01))
    features["term_state"] = "flat"
    features.loc[features["term_spread"] <= contango, "term_state"] = "contango"
    features.loc[features["term_spread"] >= inv, "term_state"] = "inverted"
    log_return = (features["qqq_close"].map(math.log)).diff()
    features["rv21"] = log_return.rolling(21, min_periods=int(config.get("rv_min_periods", 10))).std() * math.sqrt(252)
    features["rv63"] = log_return.rolling(63, min_periods=int(config.get("rv63_min_periods", 20))).std() * math.sqrt(252)
    features["iv_rv_spread"] = features["iv30"] - features["rv21"]
    features["rv21_5d_change"] = features["rv21"].diff(5)
    features["ma20"] = features["qqq_close"].rolling(20, min_periods=int(config.get("ma20_min_periods", 10))).mean()
    features["ma50"] = features["qqq_close"].rolling(50, min_periods=int(config.get("ma50_min_periods", 20))).mean()
    features["ma200"] = features["qqq_close"].rolling(200, min_periods=int(config.get("ma200_min_periods", 60))).mean()
    features["trend_state"] = features.apply(lambda row: trend_state(row, config), axis=1)
    return assign_regimes(features, config)


def trend_state(row: pd.Series, config: dict[str, Any]) -> str:
    close = row.get("qqq_close")
    ma20 = row.get("ma20")
    ma50 = row.get("ma50")
    ma200 = row.get("ma200")
    tolerance = float(config.get("trend_neutral_band", 0.0))
    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma50):
        return "unknown"
    if close >= ma20 * (1.0 + tolerance) and ma20 >= ma50:
        if pd.isna(ma200) or ma50 >= ma200:
            return "up"
    if close <= ma20 * (1.0 - tolerance) and ma20 <= ma50:
        return "down"
    return "neutral"


def assign_regimes(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = features.copy()
    high_vix = float(config.get("high_vix_threshold", 25.0))
    low_vix = float(config.get("low_vix_threshold", 15.0))
    low_iv_pct = float(config.get("low_iv_percentile_threshold", 20.0))
    elevated_iv_pct = float(config.get("elevated_iv_percentile_threshold", 80.0))
    max_iv_rv = float(config.get("normalized_max_iv_rv_spread", 0.25))
    min_iv_rv = float(config.get("normalized_min_iv_rv_spread", 0.0))
    out["is_high_vix"] = out["vix"] >= high_vix
    out["is_low_vix"] = out["vix"] < low_vix
    out["is_vxn_rising"] = out["vxn_direction"] == "rising"
    out["is_vxn_falling"] = out["vxn_direction"] == "falling"
    out["is_term_inverted"] = out["term_state"] == "inverted"
    out["is_term_normal"] = out["term_state"].isin(["contango", "flat"])
    out["is_iv_low"] = out["iv30_percentile"] <= low_iv_pct
    out["is_iv_elevated"] = out["iv30_percentile"] >= elevated_iv_pct
    out["is_downtrend"] = out["trend_state"] == "down"
    out["is_trend_recovered"] = out["trend_state"].isin(["up", "neutral"])
    out["is_rv_normalizing"] = (out["rv21_5d_change"] < 0) | (out["rv21"] < out["rv63"])
    out["is_iv_rv_positive_normal"] = (
        (out["iv_rv_spread"] > min_iv_rv) & (out["iv_rv_spread"] <= max_iv_rv)
    )
    regimes = []
    reasons = []
    for _, row in out.iterrows():
        regime, reason = classify_regime(row)
        regimes.append(regime)
        reasons.append(reason)
    out["regime"] = regimes
    out["regime_reason"] = reasons
    out["allow_new_short_gamma"] = out["regime"] == "normalized"
    return out


def classify_regime(row: pd.Series) -> tuple[str, str]:
    if bool(row.get("is_high_vix")) and bool(row.get("is_vxn_rising")) and (
        bool(row.get("is_term_inverted")) or bool(row.get("is_iv_elevated")) or bool(row.get("is_downtrend"))
    ):
        return "high_rising", "VIX high, VXN rising, and stress confirmation present"
    if bool(row.get("is_high_vix")) and bool(row.get("is_vxn_falling")):
        return "high_falling", "VIX high and VXN falling"
    if bool(row.get("is_vxn_rising")) and (bool(row.get("is_iv_elevated")) or bool(row.get("is_term_inverted"))):
        return "vol_rising", "VXN rising with elevated QQQ IV or inverted term structure"
    if bool(row.get("is_low_vix")) and bool(row.get("is_iv_low")) and not bool(row.get("is_vxn_rising")) and not bool(row.get("is_term_inverted")):
        return "low_calm", "Low VIX, low QQQ IV percentile, VXN not rising, term not inverted"
    if (
        not bool(row.get("is_vxn_rising"))
        and bool(row.get("is_term_normal"))
        and bool(row.get("is_rv_normalizing"))
        and bool(row.get("is_iv_rv_positive_normal"))
        and bool(row.get("is_trend_recovered"))
    ):
        return "normalized", "VXN not rising, term normal, RV normalizing, IV-RV positive, trend recovered"
    return "transition_no_trade", "No higher-priority regime rule matched"
