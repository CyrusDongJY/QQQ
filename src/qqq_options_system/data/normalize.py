from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from qqq_options_system.market.contracts import (
    calendar_dte,
    canonical_contract_id,
    parse_orats_date,
)


BASE_OUTPUT_COLUMNS = [
    "trade_date",
    "underlying",
    "expiry",
    "dte",
    "right",
    "strike",
    "contract_id",
    "opra_alias",
    "bid",
    "mid",
    "ask",
    "volume",
    "open_interest",
    "bid_iv",
    "mid_iv",
    "ask_iv",
    "smv_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "theo",
    "underlying_price",
    "spot_px",
    "source_file",
    "schema_version",
]


NUMERIC_COLUMNS = [
    "stkPx",
    "yte",
    "strike",
    "cVolu",
    "cOi",
    "pVolu",
    "pOi",
    "cBidPx",
    "cValue",
    "cAskPx",
    "pBidPx",
    "pValue",
    "pAskPx",
    "cBidIv",
    "cMidIv",
    "cAskIv",
    "smoothSmvVol",
    "pBidIv",
    "pMidIv",
    "pAskIv",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "extCTheo",
    "extPTheo",
    "spot_px",
]


@dataclass(frozen=True)
class QuoteFilterConfig:
    """Configurable tradeability filter for normalized option quotes."""

    max_relative_spread: float | None = 0.50
    min_open_interest: int = 1
    min_volume: int = 0
    require_iv_positive: bool = False


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & (numeric != float("inf")) & (numeric != float("-inf"))


def filter_quote_quality(
    frame: pd.DataFrame,
    config: QuoteFilterConfig | None = None,
) -> pd.DataFrame:
    """Return rows with valid, tradeable quotes while preserving the raw lake upstream."""
    cfg = config or QuoteFilterConfig()
    if frame.empty:
        return frame.copy()

    required = {"bid", "mid", "ask"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot filter quotes; missing columns: {sorted(missing)}")

    bid = pd.to_numeric(frame["bid"], errors="coerce")
    mid = pd.to_numeric(frame["mid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    mask = (
        _finite_numeric(frame["bid"])
        & _finite_numeric(frame["mid"])
        & _finite_numeric(frame["ask"])
        & (bid >= 0)
        & (ask >= bid)
        & (mid > 0)
    )

    if cfg.max_relative_spread is not None:
        relative_spread = (ask - bid) / mid
        mask &= relative_spread <= float(cfg.max_relative_spread)

    if "open_interest" in frame.columns:
        oi = pd.to_numeric(frame["open_interest"], errors="coerce").fillna(-1)
        mask &= oi >= int(cfg.min_open_interest)
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(-1)
        mask &= volume >= int(cfg.min_volume)

    if cfg.require_iv_positive:
        iv_cols = [col for col in ("bid_iv", "mid_iv", "ask_iv") if col in frame.columns]
        for col in iv_cols:
            iv = pd.to_numeric(frame[col], errors="coerce")
            mask &= _finite_numeric(frame[col]) & (iv > 0)

    return frame.loc[mask].copy()


def _series_or_default(frame: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _to_iso_dates(values: pd.Series) -> pd.Series:
    return values.map(lambda value: parsed.isoformat() if (parsed := parse_orats_date(value)) else pd.NA)


def _compute_dte(trade_dates: pd.Series, expiries: pd.Series, yte: pd.Series) -> pd.Series:
    values: list[int | None] = []
    for trade_value, expiry_value, yte_value in zip(trade_dates, expiries, yte):
        dte = calendar_dte(trade_value, expiry_value)
        if dte is None and pd.notna(yte_value):
            dte = int(round(float(yte_value) * 365.25))
        values.append(dte)
    return pd.Series(values, index=trade_dates.index, dtype="Int64")


def _contract_ids(
    underlying: pd.Series,
    expiry: pd.Series,
    right: str,
    strike: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            canonical_contract_id(u, e, right, k)
            if pd.notna(u) and pd.notna(e) and pd.notna(k)
            else pd.NA
            for u, e, k in zip(underlying, expiry, strike)
        ],
        index=underlying.index,
    )


def normalize_qqq_wide_to_long(
    frame: pd.DataFrame,
    source_file: str,
    schema_version: str,
    underlying: str = "QQQ",
) -> pd.DataFrame:
    """Convert ORATS call/put wide rows to one row per option side."""
    if "ticker" not in frame.columns:
        raise ValueError("Input frame must include ticker column")

    qqq = frame.loc[frame["ticker"].astype(str).str.upper() == underlying.upper()].copy()
    if qqq.empty:
        return pd.DataFrame(columns=BASE_OUTPUT_COLUMNS)

    for col in NUMERIC_COLUMNS:
        if col in qqq.columns:
            qqq[col] = pd.to_numeric(qqq[col], errors="coerce")

    trade_dates = _series_or_default(qqq, "trade_date")
    expiries = _series_or_default(qqq, "expirDate")
    yte = _series_or_default(qqq, "yte")
    common = pd.DataFrame(
        {
            "trade_date": _to_iso_dates(trade_dates),
            "underlying": qqq["ticker"].astype(str).str.upper(),
            "expiry": _to_iso_dates(expiries),
            "dte": _compute_dte(trade_dates, expiries, yte),
            "strike": _series_or_default(qqq, "strike"),
            "smv_vol": _series_or_default(qqq, "smoothSmvVol"),
            "gamma": _series_or_default(qqq, "gamma"),
            "theta": _series_or_default(qqq, "theta"),
            "vega": _series_or_default(qqq, "vega"),
            "underlying_price": _series_or_default(qqq, "stkPx"),
            "spot_px": _series_or_default(qqq, "spot_px"),
            "source_file": source_file,
            "schema_version": schema_version,
        },
        index=qqq.index,
    )

    call = common.copy()
    call["right"] = "C"
    call["contract_id"] = _contract_ids(call["underlying"], call["expiry"], "C", call["strike"])
    call["opra_alias"] = _series_or_default(qqq, "cOpra")
    call["bid"] = _series_or_default(qqq, "cBidPx")
    call["mid"] = _series_or_default(qqq, "cValue")
    call["ask"] = _series_or_default(qqq, "cAskPx")
    call["volume"] = _series_or_default(qqq, "cVolu")
    call["open_interest"] = _series_or_default(qqq, "cOi")
    call["bid_iv"] = _series_or_default(qqq, "cBidIv")
    call["mid_iv"] = _series_or_default(qqq, "cMidIv")
    call["ask_iv"] = _series_or_default(qqq, "cAskIv")
    call["delta"] = _series_or_default(qqq, "delta")
    call["rho"] = _series_or_default(qqq, "rho")
    call["theo"] = _series_or_default(qqq, "extCTheo")

    put = common.copy()
    put["right"] = "P"
    put["contract_id"] = _contract_ids(put["underlying"], put["expiry"], "P", put["strike"])
    put["opra_alias"] = _series_or_default(qqq, "pOpra")
    put["bid"] = _series_or_default(qqq, "pBidPx")
    put["mid"] = _series_or_default(qqq, "pValue")
    put["ask"] = _series_or_default(qqq, "pAskPx")
    put["volume"] = _series_or_default(qqq, "pVolu")
    put["open_interest"] = _series_or_default(qqq, "pOi")
    put["bid_iv"] = _series_or_default(qqq, "pBidIv")
    put["mid_iv"] = _series_or_default(qqq, "pMidIv")
    put["ask_iv"] = _series_or_default(qqq, "pAskIv")
    put["delta"] = _series_or_default(qqq, "delta").map(
        lambda value: value - 1.0 if pd.notna(value) else pd.NA
    )
    put["rho"] = _series_or_default(qqq, "rho")
    put["theo"] = _series_or_default(qqq, "extPTheo")

    out = pd.concat([call, put], ignore_index=True)
    return out[BASE_OUTPUT_COLUMNS]


def read_qqq_chunks(path: Path, underlying: str, chunk_size: int) -> Any:
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        yield chunk.loc[chunk["ticker"].astype(str).str.upper() == underlying.upper()].copy()


def write_partitioned_parquet(
    frame: pd.DataFrame,
    output_root: Path,
    trade_date: date,
    underlying: str = "QQQ",
) -> Path:
    if frame.empty:
        raise ValueError("Cannot write empty normalized frame")
    partition = (
        output_root
        / "data"
        / "silver_options"
        / f"symbol={underlying.upper()}"
        / f"year={trade_date.year}"
        / f"trade_date={trade_date.isoformat()}"
    )
    partition.mkdir(parents=True, exist_ok=True)
    output_path = partition / "option_chain_eod.parquet"
    frame.to_parquet(output_path, index=False, engine="pyarrow")
    return output_path
