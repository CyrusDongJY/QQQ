from __future__ import annotations

import pandas as pd


def rolling_percentile(
    values: pd.Series,
    lookback: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    percentiles: list[float | None] = []
    clean_values = values.reset_index(drop=True)
    for index, value in enumerate(clean_values):
        if pd.isna(value):
            percentiles.append(None)
            continue
        start = max(0, index - int(lookback) + 1)
        window = clean_values.iloc[start : index + 1].dropna().astype(float)
        if len(window) < int(min_periods):
            percentiles.append(None)
            continue
        current = float(value)
        count = int((window <= current).sum())
        percentiles.append(100.0 * count / len(window))
    return pd.Series(percentiles, index=values.index)


def build_daily_features(
    chains: dict[str, pd.DataFrame],
    iv_percentile_lookback: int = 252,
    iv_percentile_min_periods: int = 60,
) -> pd.DataFrame:
    rows: list[dict] = []
    for date_text, chain in sorted(chains.items()):
        if chain.empty:
            continue
        spot = float(pd.to_numeric(chain["underlying_price"], errors="coerce").dropna().iloc[0])
        near = chain.loc[
            chain["dte"].notna()
            & chain["mid_iv"].notna()
            & (chain["mid_iv"].astype(float) > 0)
        ].copy()
        atm_30d_iv = None
        if not near.empty:
            near["atm_score"] = (
                (near["dte"].astype(float) - 30.0).abs()
                + 100.0 * ((near["strike"].astype(float) / spot) - 1.0).abs()
            )
            best_score = near["atm_score"].min()
            best = near.loc[near["atm_score"] == best_score]
            atm_30d_iv = float(best["mid_iv"].astype(float).mean())
        rows.append(
            {
                "date": date_text,
                "underlying_price": spot,
                "atm_30d_iv": atm_30d_iv,
            }
        )
    features = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    features["underlying_return"] = features["underlying_price"].pct_change()
    features["iv_change"] = features["atm_30d_iv"].diff()
    features["iv_percentile"] = rolling_percentile(
        features["atm_30d_iv"],
        lookback=iv_percentile_lookback,
        min_periods=iv_percentile_min_periods,
    )
    return features


def signal_dates_for_trigger(features: pd.DataFrame, trigger_kind: str, threshold: float) -> list[str]:
    if trigger_kind == "return":
        mask = features["underlying_return"] <= threshold
    elif trigger_kind == "iv_percentile":
        mask = features["iv_percentile"] >= threshold
    elif trigger_kind == "iv_change":
        mask = features["iv_change"] >= threshold
    else:
        raise ValueError(f"Unsupported trigger kind: {trigger_kind}")
    return features.loc[mask, "date"].astype(str).tolist()
