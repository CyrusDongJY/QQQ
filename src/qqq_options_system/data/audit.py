from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qqq_options_system.data.orats_files import OratsFile
from qqq_options_system.market.contracts import parse_orats_date


@dataclass(frozen=True)
class DailyAudit:
    file_date: str
    source_file: str
    schema_version: str
    schema_hash: str
    qqq_rows: int
    min_expiry: str
    max_expiry: str
    min_strike: float | None
    max_strike: float | None
    trade_date_mismatches: int


def audit_qqq_file(
    item: OratsFile,
    underlying: str = "QQQ",
    chunk_size: int = 200_000,
) -> DailyAudit:
    total_rows = 0
    expiry_values: list[str] = []
    min_strike: float | None = None
    max_strike: float | None = None
    trade_date_mismatches = 0
    expected_trade_date = item.file_date.isoformat()

    usecols = [
        col
        for col in ("ticker", "expirDate", "strike", "trade_date")
        if col in item.header
    ]
    for chunk in pd.read_csv(item.path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        qqq = chunk.loc[chunk["ticker"].astype(str).str.upper() == underlying.upper()]
        if qqq.empty:
            continue
        total_rows += len(qqq)
        if "expirDate" in qqq.columns:
            expiry_values.extend(
                parsed.isoformat()
                for value in qqq["expirDate"]
                if (parsed := parse_orats_date(value)) is not None
            )
        if "strike" in qqq.columns:
            strikes = pd.to_numeric(qqq["strike"], errors="coerce").dropna()
            if not strikes.empty:
                chunk_min = float(strikes.min())
                chunk_max = float(strikes.max())
                min_strike = chunk_min if min_strike is None else min(min_strike, chunk_min)
                max_strike = chunk_max if max_strike is None else max(max_strike, chunk_max)
        if "trade_date" in qqq.columns:
            parsed_trade_dates = qqq["trade_date"].map(
                lambda value: parsed.isoformat() if (parsed := parse_orats_date(value)) else ""
            )
            trade_date_mismatches += int((parsed_trade_dates != expected_trade_date).sum())

    return DailyAudit(
        file_date=expected_trade_date,
        source_file=str(item.path),
        schema_version=item.schema_version,
        schema_hash=item.schema_hash,
        qqq_rows=total_rows,
        min_expiry=min(expiry_values) if expiry_values else "",
        max_expiry=max(expiry_values) if expiry_values else "",
        min_strike=min_strike,
        max_strike=max_strike,
        trade_date_mismatches=trade_date_mismatches,
    )


def write_daily_audit_csv(rows: list[DailyAudit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_date",
        "source_file",
        "schema_version",
        "schema_hash",
        "qqq_rows",
        "min_expiry",
        "max_expiry",
        "min_strike",
        "max_strike",
        "trade_date_mismatches",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
