from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def parse_orats_date(value: Any) -> date | None:
    """Parse ORATS date strings such as 6/2/2026 or 2026-06-02."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def format_expiry(value: Any) -> str:
    parsed = parse_orats_date(value)
    if parsed is None:
        raise ValueError(f"Could not parse expiry date: {value!r}")
    return parsed.isoformat()


def strike_millis(strike: Any) -> int:
    number = Decimal(str(strike)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return int((number * 1000).to_integral_value(rounding=ROUND_HALF_UP))


def canonical_contract_id(
    underlying: str,
    expiry: Any,
    right: str,
    strike: Any,
) -> str:
    """Build a stable key without relying on vendor OPRA columns."""
    expiry_date = parse_orats_date(expiry)
    if expiry_date is None:
        raise ValueError(f"Could not parse expiry date: {expiry!r}")
    right_text = str(right).strip().upper()
    if right_text not in {"C", "P"}:
        raise ValueError(f"right must be C or P, got {right!r}")
    return (
        f"{str(underlying).strip().upper()}_"
        f"{expiry_date.strftime('%Y%m%d')}_"
        f"{right_text}_"
        f"{strike_millis(strike):08d}"
    )


def calendar_dte(trade_date: Any, expiry: Any) -> int | None:
    trade = parse_orats_date(trade_date)
    exp = parse_orats_date(expiry)
    if trade is None or exp is None:
        return None
    return (exp - trade).days
