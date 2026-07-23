from __future__ import annotations

import math
from dataclasses import dataclass


VALID_PRICE_MODES = {"bid_ask", "midpoint", "half_spread"}


@dataclass(frozen=True)
class ExecutionCostConfig:
    commission_per_contract: float = 0.65
    exchange_fee_per_contract: float = 0.02
    slippage_per_contract: float = 0.0

    @property
    def round_trip_cost_per_contract(self) -> float:
        return 2.0 * (
            self.commission_per_contract
            + self.exchange_fee_per_contract
            + self.slippage_per_contract
        )


def _validate_quote_number(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Invalid quote: {name} must be finite, got {value!r}")
    if number < 0:
        raise ValueError(f"Invalid quote: {name} must be non-negative, got {number}")
    return number


def fill_price(bid: float, mid: float, ask: float, side: str, price_mode: str = "bid_ask") -> float:
    """Return executable fill for buying or selling one option leg."""
    if price_mode not in VALID_PRICE_MODES:
        raise ValueError(f"Unsupported price mode: {price_mode}")
    bid = _validate_quote_number("bid", bid)
    mid = _validate_quote_number("mid", mid)
    ask = _validate_quote_number("ask", ask)
    if ask < bid:
        raise ValueError(f"Invalid quote with ask < bid: bid={bid}, ask={ask}")
    if price_mode == "midpoint" and mid <= 0:
        raise ValueError(f"Invalid quote: mid must be positive, got {mid}")
    side_text = side.lower()
    if side_text not in {"buy", "sell"}:
        raise ValueError(f"side must be buy or sell, got {side!r}")
    spread = ask - bid
    if price_mode == "midpoint":
        return float(mid)
    if price_mode == "half_spread":
        if side_text == "buy":
            return float(ask - 0.25 * spread)
        return float(bid + 0.25 * spread)
    return float(ask if side_text == "buy" else bid)
