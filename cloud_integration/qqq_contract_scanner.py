from __future__ import annotations

"""Pure selection and formatting for live QQQ option candidates.

The module deliberately has no broker dependency. The cloud integration owns
market-data collection; this layer enforces same-expiry, quote-quality and
leg-level bid/ask execution rules before anything is shown in an email.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_MAX_RELATIVE_SPREAD = 0.35
DEFAULT_MIN_OPEN_INTEREST = 10
DEFAULT_ENTRY_COST_PER_CONTRACT = 0.67


@dataclass(frozen=True)
class LiveOptionQuote:
    underlying: str
    expiry: str
    right: str
    strike: float
    bid: float
    ask: float
    delta: float
    iv: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    market_data_type: Optional[int] = None
    con_id: Optional[int] = None
    local_symbol: Optional[str] = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def relative_spread(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else math.inf

    def is_tradeable(
        self,
        max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
        min_open_interest: int = DEFAULT_MIN_OPEN_INTEREST,
    ) -> bool:
        numeric = (self.strike, self.bid, self.ask, self.delta)
        if not all(math.isfinite(float(value)) for value in numeric):
            return False
        if self.strike <= 0 or self.bid < 0 or self.ask <= 0 or self.ask < self.bid:
            return False
        right = self.right.upper()
        if right not in {"P", "C"}:
            return False
        if right == "P" and not -1.0 <= self.delta < 0:
            return False
        if right == "C" and not 0 < self.delta <= 1.0:
            return False
        if self.relative_spread > max_relative_spread:
            return False
        if self.open_interest is None or int(self.open_interest) < min_open_interest:
            return False
        return True


@dataclass(frozen=True)
class CandidateLeg:
    action: str
    role: str
    quote: LiveOptionQuote
    fill_price: float
    quantity: int = 1


@dataclass(frozen=True)
class ContractScanResult:
    strategy_code: str
    strategy_title: str
    signal_status: str
    quote_time: datetime
    spot: Optional[float]
    expiry: Optional[str] = None
    legs: Tuple[CandidateLeg, ...] = ()
    net_entry_credit: Optional[float] = None
    entry_costs: Optional[float] = None
    capital_required: Optional[float] = None
    max_loss: Optional[float] = None
    failure_reason: Optional[str] = None
    notes: Tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.failure_reason is None and bool(self.legs)


def _eligible(
    quotes: Iterable[LiveOptionQuote],
    right: str,
    expiry: Optional[str] = None,
) -> list[LiveOptionQuote]:
    expected_right = right.upper()
    return [
        quote
        for quote in quotes
        if quote.right.upper() == expected_right
        and (expiry is None or quote.expiry == expiry)
        and quote.is_tradeable()
    ]


def _select_delta(
    quotes: Iterable[LiveOptionQuote],
    right: str,
    target_delta: float,
) -> Optional[LiveOptionQuote]:
    candidates = _eligible(quotes, right)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda quote: (
            abs(quote.delta - target_delta),
            quote.relative_spread,
            -int(quote.open_interest or 0),
        ),
    )


def _select_strike(
    quotes: Iterable[LiveOptionQuote],
    right: str,
    target_strike: float,
    expiry: str,
    below: Optional[float] = None,
    above: Optional[float] = None,
) -> Optional[LiveOptionQuote]:
    candidates = _eligible(quotes, right, expiry=expiry)
    if below is not None:
        candidates = [quote for quote in candidates if quote.strike < below]
    if above is not None:
        candidates = [quote for quote in candidates if quote.strike > above]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda quote: (
            abs(quote.strike - target_strike),
            quote.relative_spread,
            -int(quote.open_interest or 0),
        ),
    )


def _select_affordable_call(
    calls: Iterable[LiveOptionQuote],
    expiry: str,
    budget: float,
) -> Optional[LiveOptionQuote]:
    candidates = [
        quote
        for quote in _eligible(calls, "C", expiry=expiry)
        if quote.ask <= budget + 1e-12
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda quote: (
            quote.ask,
            -quote.relative_spread,
            int(quote.open_interest or 0),
        ),
    )


def _single_expiry(quotes: Sequence[LiveOptionQuote]) -> Optional[str]:
    expiries = sorted({quote.expiry for quote in quotes if quote.expiry})
    return expiries[0] if len(expiries) == 1 else None


def _basic_option_market_valid(quote: LiveOptionQuote, right: str) -> bool:
    numeric = (quote.strike, quote.bid, quote.ask)
    return (
        quote.right.upper() == right.upper()
        and all(math.isfinite(float(value)) for value in numeric)
        and quote.strike > 0
        and quote.bid >= 0
        and quote.ask > 0
        and quote.ask >= quote.bid
    )


def _valid_option_delta(quote: LiveOptionQuote, right: str) -> bool:
    if not math.isfinite(float(quote.delta)):
        return False
    if right.upper() == "P":
        return -1.0 <= quote.delta < 0
    return 0 < quote.delta <= 1.0


def _selection_failure_notes(
    quotes: Sequence[LiveOptionQuote],
    right: str,
    target_delta: float,
    collection_notes: Sequence[str] = (),
) -> Tuple[str, ...]:
    expected_right = right.upper()
    relevant = [quote for quote in quotes if quote.right.upper() == expected_right]
    market_valid = [
        quote for quote in relevant
        if _basic_option_market_valid(quote, expected_right)
    ]
    delta_valid = [
        quote for quote in relevant
        if _valid_option_delta(quote, expected_right)
    ]
    spread_valid = [
        quote for quote in market_valid
        if quote.relative_spread <= DEFAULT_MAX_RELATIVE_SPREAD
    ]
    oi_valid = [
        quote for quote in relevant
        if quote.open_interest is not None
        and int(quote.open_interest) >= DEFAULT_MIN_OPEN_INTEREST
    ]
    tradeable = [quote for quote in relevant if quote.is_tradeable()]
    expiries = sorted({quote.expiry for quote in relevant if quote.expiry})
    data_types = sorted(
        {
            int(quote.market_data_type)
            for quote in relevant
            if quote.market_data_type is not None
        }
    )
    label = "Put" if expected_right == "P" else "Call"
    notes = [
        (
            f"{label}筛选诊断: 收到{len(relevant)}个报价对象，"
            f"bid/ask有效{len(market_valid)}个，Delta有效{len(delta_valid)}个，"
            f"点差不超过{DEFAULT_MAX_RELATIVE_SPREAD:.0%}的{len(spread_valid)}个，"
            f"OI>={DEFAULT_MIN_OPEN_INTEREST}的{len(oi_valid)}个，"
            f"最终可交易{len(tradeable)}个；"
            f"到期日{','.join(expiries) if expiries else '缺失'}，"
            f"marketDataType{','.join(map(str, data_types)) if data_types else '缺失'}。"
        )
    ]
    if delta_valid:
        closest = min(
            delta_valid,
            key=lambda quote: abs(quote.delta - target_delta),
        )
        notes.append(
            f"最接近目标Delta {target_delta:+.2f}的{label}: "
            f"${closest.strike:g}{expected_right} Delta {closest.delta:+.3f}，"
            f"bid/ask ${closest.bid:.2f}/${closest.ask:.2f}，"
            f"OI {_format_optional_int(closest.open_interest)}，"
            f"spread {closest.relative_spread:.0%}。"
        )
    notes.extend(str(note) for note in collection_notes if str(note).strip())
    return tuple(notes)


def _format_quote_brief(quote: LiveOptionQuote) -> str:
    return (
        f"${quote.strike:g}{quote.right.upper()} ask ${quote.ask:.2f} "
        f"Delta {quote.delta:+.3f} OI {_format_optional_int(quote.open_interest)} "
        f"spread {quote.relative_spread:.0%}"
    )


def _basic_call_market_valid(quote: LiveOptionQuote) -> bool:
    numeric = (quote.strike, quote.bid, quote.ask, quote.delta)
    if not all(math.isfinite(float(value)) for value in numeric):
        return False
    return (
        quote.right.upper() == "C"
        and quote.strike > 0
        and quote.bid >= 0
        and quote.ask > 0
        and quote.ask >= quote.bid
        and 0 < quote.delta <= 1.0
    )


def _affordable_call_failure_notes(
    calls: Sequence[LiveOptionQuote],
    expiry: str,
    budget: float,
) -> Tuple[str, ...]:
    same_expiry = [
        quote
        for quote in calls
        if quote.right.upper() == "C" and quote.expiry == expiry
    ]
    if not same_expiry:
        return ("Call诊断: 同到期Call报价为0个；优先检查IB链请求/执行价采样范围。",)

    strikes = [quote.strike for quote in same_expiry if math.isfinite(quote.strike)]
    market_valid = [quote for quote in same_expiry if _basic_call_market_valid(quote)]
    tradeable = [quote for quote in same_expiry if quote.is_tradeable()]
    affordable_tradeable = [quote for quote in tradeable if quote.ask <= budget + 1e-12]
    affordable_market = [
        quote for quote in market_valid if quote.ask <= budget + 1e-12
    ]
    notes = [
        (
            "Call诊断: "
            f"同到期扫描{len(same_expiry)}个"
            + (
                f"，执行价范围 ${min(strikes):g}-${max(strikes):g}"
                if strikes
                else ""
            )
            + f"；报价/Delta有效{len(market_valid)}个，可交易{len(tradeable)}个，"
            f"预算内可交易{len(affordable_tradeable)}个。"
        )
    ]
    if tradeable:
        cheapest = min(tradeable, key=lambda quote: quote.ask)
        richest_affordable = max(
            affordable_tradeable, key=lambda quote: quote.ask
        ) if affordable_tradeable else None
        note = f"最低可交易Call: {_format_quote_brief(cheapest)}。"
        if richest_affordable is not None:
            note += f" 预算内最高ask: {_format_quote_brief(richest_affordable)}。"
        elif cheapest.ask > budget:
            note += f" 最低可交易ask仍高于预算 ${budget:.2f}。"
        notes.append(note)
    if affordable_market and not affordable_tradeable:
        rejected = [
            quote for quote in affordable_market if not quote.is_tradeable()
        ]
        low_oi = sum(
            1
            for quote in rejected
            if quote.open_interest is None
            or int(quote.open_interest) < DEFAULT_MIN_OPEN_INTEREST
        )
        wide_spread = sum(
            1
            for quote in rejected
            if quote.relative_spread > DEFAULT_MAX_RELATIVE_SPREAD
        )
        notes.append(
            "预算内但未通过流动性过滤的Call: "
            f"{len(rejected)}个；OI不足/缺失{low_oi}个，"
            f"价差过宽{wide_spread}个。"
        )
    if not market_valid:
        notes.append("同到期Call均缺少有效bid/ask或Delta；需要检查IB行情权限/延迟行情。")
    return tuple(notes)


def _failed(
    strategy_code: str,
    strategy_title: str,
    signal_status: str,
    quote_time: datetime,
    spot: Optional[float],
    reason: str,
    expiry: Optional[str] = None,
    notes: Sequence[str] = (),
) -> ContractScanResult:
    return ContractScanResult(
        strategy_code=strategy_code,
        strategy_title=strategy_title,
        signal_status=signal_status,
        quote_time=quote_time,
        spot=spot,
        expiry=expiry,
        failure_reason=reason,
        notes=tuple(notes),
    )


def failed_contract_scan_result(
    strategy_code: str,
    strategy_title: str,
    signal_status: str,
    quote_time: datetime,
    reason: str,
    spot: Optional[float] = None,
    expiry: Optional[str] = None,
    notes: Sequence[str] = (),
) -> ContractScanResult:
    """Return an auditable result when broker-side enrichment cannot run."""
    return _failed(
        strategy_code,
        strategy_title,
        signal_status,
        quote_time,
        spot,
        reason,
        expiry=expiry,
        notes=notes,
    )


def build_uptrend_funded_call_candidate(
    puts: Sequence[LiveOptionQuote],
    calls: Sequence[LiveOptionQuote],
    spot: float,
    quote_time: datetime,
    signal_status: str = "TRIGGERED",
    target_put_delta: float = -0.40,
    entry_cost_per_contract: float = DEFAULT_ENTRY_COST_PER_CONTRACT,
    expected_expiry: Optional[str] = None,
    put_collection_notes: Sequence[str] = (),
    call_collection_notes: Sequence[str] = (),
) -> ContractScanResult:
    code = "uptrend_funded_call"
    title = "上涨走势：2%盘中突破 Funded Call"
    short_put = _select_delta(puts, "P", target_put_delta)
    if short_put is None or short_put.bid <= 0:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            "没有符合流动性与Delta要求的可卖Put",
            expiry=expected_expiry or _single_expiry(puts),
            notes=_selection_failure_notes(
                puts,
                "P",
                target_put_delta,
                put_collection_notes,
            ),
        )
    call = _select_affordable_call(calls, short_put.expiry, short_put.bid)
    if call is None:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            f"同到期没有ask不超过Short Put bid ${short_put.bid:.2f}的可交易Call",
            expiry=short_put.expiry,
            notes=_affordable_call_failure_notes(
                calls, short_put.expiry, short_put.bid
            ) + tuple(call_collection_notes),
        )
    net_credit = short_put.bid - call.ask
    costs = 2 * entry_cost_per_contract
    downside_loss = max(0.0, short_put.strike * 100.0 - net_credit * 100.0 + costs)
    return ContractScanResult(
        strategy_code=code,
        strategy_title=title,
        signal_status=signal_status,
        quote_time=quote_time,
        spot=spot,
        expiry=short_put.expiry,
        legs=(
            CandidateLeg("SELL", "Short Put", short_put, short_put.bid),
            CandidateLeg("BUY", "Long Call", call, call.ask),
        ),
        net_entry_credit=net_credit,
        entry_costs=costs,
        capital_required=short_put.strike * 100.0,
        max_loss=downside_loss,
        notes=(
            "筛选优先级：Put最接近-0.40 Delta，再选同到期且ask不超过Put bid、最接近预算上限的Call",
            "现金担保下行口径；Long Call不抵消QQQ跌至零时的Short Put义务",
        ),
    )


def build_downtrend_combo_candidate(
    puts: Sequence[LiveOptionQuote],
    calls: Sequence[LiveOptionQuote],
    spot: float,
    quote_time: datetime,
    signal_status: str = "TRIGGERED",
    target_put_delta: float = -0.50,
    wing_width_pct: float = 0.10,
    entry_cost_per_contract: float = DEFAULT_ENTRY_COST_PER_CONTRACT,
    expected_expiry: Optional[str] = None,
    put_collection_notes: Sequence[str] = (),
    call_collection_notes: Sequence[str] = (),
) -> ContractScanResult:
    code = "downtrend_defined_risk_combo"
    title = "下跌走势：1%回撤后三腿定义风险反弹组合"
    short_put = _select_delta(puts, "P", target_put_delta)
    if short_put is None or short_put.bid <= 0:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            "没有符合流动性与Delta要求的可卖Put",
            expiry=expected_expiry or _single_expiry(puts),
            notes=_selection_failure_notes(
                puts,
                "P",
                target_put_delta,
                put_collection_notes,
            ),
        )
    target_wing = short_put.strike - spot * wing_width_pct
    long_put = _select_strike(
        puts,
        "P",
        target_wing,
        short_put.expiry,
        below=short_put.strike,
    )
    if long_put is None:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            "同到期没有符合报价质量的低执行价保护Put",
            expiry=short_put.expiry,
            notes=tuple(put_collection_notes),
        )
    put_spread_credit = short_put.bid - long_put.ask
    if put_spread_credit <= 0:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            f"Put Spread按bid/ask计算无可用净信用（${put_spread_credit:.2f}）",
            expiry=short_put.expiry,
            notes=tuple(put_collection_notes),
        )
    call = _select_affordable_call(calls, short_put.expiry, put_spread_credit)
    if call is None:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            f"同到期没有ask不超过Put Spread净信用 ${put_spread_credit:.2f}的可交易Call",
            expiry=short_put.expiry,
            notes=_affordable_call_failure_notes(
                calls, short_put.expiry, put_spread_credit
            ) + tuple(call_collection_notes),
        )
    net_credit = put_spread_credit - call.ask
    costs = 3 * entry_cost_per_contract
    width = short_put.strike - long_put.strike
    max_loss = max(0.0, width * 100.0 - net_credit * 100.0 + costs)
    return ContractScanResult(
        strategy_code=code,
        strategy_title=title,
        signal_status=signal_status,
        quote_time=quote_time,
        spot=spot,
        expiry=short_put.expiry,
        legs=(
            CandidateLeg("SELL", "Short Put", short_put, short_put.bid),
            CandidateLeg("BUY", "Protective Put", long_put, long_put.ask),
            CandidateLeg("BUY", "Long Call", call, call.ask),
        ),
        net_entry_credit=net_credit,
        entry_costs=costs,
        capital_required=max_loss,
        max_loss=max_loss,
        notes=(
            "筛选优先级：Short Put最接近-0.50 Delta、保护Put最接近10% spot翼宽，再用可执行净信用购买最接近预算上限的Call",
            f"Put翼宽 ${width:.2f}；最大损失含预计开仓费用、不含退出费用",
        ),
    )


def build_wall_debit_spread_candidate(
    quotes: Sequence[LiveOptionQuote],
    spot: float,
    quote_time: datetime,
    expiry: str,
    right: str,
    wall_strike: float,
    outer_width_pct: float = 0.05,
    signal_status: str = "TRIGGERED",
    entry_cost_per_contract: float = DEFAULT_ENTRY_COST_PER_CONTRACT,
    collection_notes: Sequence[str] = (),
) -> ContractScanResult:
    code = "monthly_wall_break_debit_spread"
    title = "期权墙：标准月权突破顺势 Debit Spread"
    right = right.upper()
    long_leg = _select_strike(quotes, right, wall_strike, expiry)
    if long_leg is None:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            "墙位附近没有可交易Long期权",
            expiry=expiry,
            notes=tuple(collection_notes),
        )
    if right == "C":
        outer_target = wall_strike + spot * outer_width_pct
        short_leg = _select_strike(
            quotes, right, outer_target, expiry, above=long_leg.strike
        )
    elif right == "P":
        outer_target = wall_strike - spot * outer_width_pct
        short_leg = _select_strike(
            quotes, right, outer_target, expiry, below=long_leg.strike
        )
    else:
        raise ValueError("right must be P or C")
    if short_leg is None or short_leg.bid <= 0:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            "外侧卖出腿没有合格bid报价",
            expiry=expiry,
            notes=tuple(collection_notes),
        )
    net_debit = long_leg.ask - short_leg.bid
    if net_debit <= 0:
        return _failed(
            code,
            title,
            signal_status,
            quote_time,
            spot,
            f"Debit Spread逐腿bid/ask净支出异常（${net_debit:.2f}）",
            expiry=expiry,
            notes=tuple(collection_notes),
        )
    costs = 2 * entry_cost_per_contract
    max_loss = net_debit * 100.0 + costs
    return ContractScanResult(
        strategy_code=code,
        strategy_title=title,
        signal_status=signal_status,
        quote_time=quote_time,
        spot=spot,
        expiry=expiry,
        legs=(
            CandidateLeg("BUY", f"Long {right}", long_leg, long_leg.ask),
            CandidateLeg("SELL", f"Short {right}", short_leg, short_leg.bid),
        ),
        net_entry_credit=-net_debit,
        entry_costs=costs,
        capital_required=max_loss,
        max_loss=max_loss,
        notes=("最大损失为逐腿bid/ask净权利金支出加预计开仓费用",),
    )


def _format_optional_int(value: Optional[int]) -> str:
    return str(int(value)) if value is not None else "缺失"


def _format_iv(value: Optional[float]) -> str:
    return f"{value:.1%}" if value is not None and math.isfinite(value) else "缺失"


def _format_market_data_type(value: Optional[int]) -> str:
    labels = {1: "实时", 2: "冻结", 3: "延迟", 4: "延迟冻结"}
    return labels.get(value, f"未知({value})" if value is not None else "缺失")


def format_contract_scan_section(results: Sequence[ContractScanResult]) -> str:
    lines = [
        "",
        "[QQQ IBKR 可执行合约候选]",
        "成交口径: 买入按ask、卖出按bid；仅列候选，不创建或发送订单。",
    ]
    if not results:
        lines.append("没有需要扫描的TRIGGERED/WATCH/BLOCKED且未过期信号。")
        return "\n".join(lines)
    for index, result in enumerate(results, start=1):
        quote_time = result.quote_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        spot_text = f"${result.spot:.2f}" if result.spot is not None else "缺失"
        lines.extend(
            [
                "",
                f"{index}. [{result.signal_status}] {result.strategy_title}",
                f"报价快照: {quote_time} | QQQ {spot_text} | 到期 {result.expiry or '未确定'}",
            ]
        )
        if result.failure_reason:
            lines.append(f"扫描结果: FAILED - {result.failure_reason}")
            for note in result.notes:
                lines.append(f"说明: {note}")
            continue
        lines.append("具体合约:")
        for leg in result.legs:
            quote = leg.quote
            action = "买入" if leg.action == "BUY" else "卖出"
            contract_ref = quote.local_symbol or (
                f"conId={quote.con_id}" if quote.con_id is not None else "合约ID缺失"
            )
            lines.append(
                f"  - {action} {leg.quantity}x {quote.underlying} {quote.expiry} "
                f"${quote.strike:g}{quote.right.upper()} [{leg.role}] | "
                f"IBKR {contract_ref} | "
                f"Delta {quote.delta:+.3f} | bid ${quote.bid:.2f} / ask ${quote.ask:.2f} "
                f"/ mid ${quote.mid:.2f} | IV {_format_iv(quote.iv)} | "
                f"OI {_format_optional_int(quote.open_interest)} | Vol {_format_optional_int(quote.volume)} | "
                f"行情 {_format_market_data_type(quote.market_data_type)} | "
                f"成交口径 ${leg.fill_price:.2f}"
            )
        net_credit = float(result.net_entry_credit or 0.0)
        entry_label = "净信用" if net_credit >= 0 else "净支出"
        lines.append(
            f"组合{entry_label}: ${abs(net_credit):.2f}/股 = ${abs(net_credit) * 100:.2f}/组"
        )
        if result.entry_costs is not None:
            lines.append(f"预计开仓费用: ${result.entry_costs:.2f}/组")
        if result.capital_required is not None:
            lines.append(f"估算资本占用: ${result.capital_required:.2f}/组")
        if result.max_loss is not None:
            lines.append(f"估算最大损失: ${result.max_loss:.2f}/组")
        if result.signal_status == "WATCH":
            lines.append("执行限制: WATCH仅供观察，未通过状态gate，不应据此下单。")
        elif result.signal_status == "BLOCKED":
            lines.append("执行限制: BLOCKED为风险审计候选，状态gate明确禁止下单。")
        for note in result.notes:
            lines.append(f"说明: {note}")
    lines.append("提示: 报价为邮件生成时快照；临近收盘必须重新确认Delta、bid/ask与流动性。")
    return "\n".join(lines)
