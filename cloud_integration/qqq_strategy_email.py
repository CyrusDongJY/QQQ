from __future__ import annotations

"""Authoritative live QQQ email-signal rules.

Use scripts/sync_live_qqq_signal.py to update vendored cloud copies.
"""

import math
from dataclasses import dataclass, replace
from datetime import date
from typing import Optional, Tuple


UPTREND_INTRADAY_THRESHOLD = 0.02
DOWNTREND_CLOSE_THRESHOLD = -0.01
WALL_MAX_BREAK_DISTANCE = 0.03
WALL_OUTER_WIDTH_PCT = 0.05
WALL_SIGNAL_SESSIONS_TO_EXPIRY = 6
BLOCKED_DOWNTREND_REGIMES = {"vol_rising", "high_rising"}
KNOWN_FIVE_FACTOR_REGIMES = {
    "high_rising",
    "high_falling",
    "vol_rising",
    "low_calm",
    "normalized",
    "transition_no_trade",
}
ALLOWED_DOWNTREND_REGIMES = KNOWN_FIVE_FACTOR_REGIMES - BLOCKED_DOWNTREND_REGIMES


@dataclass(frozen=True)
class QQQSignalContext:
    signal_date: date
    entry_date: date
    prior_close: float
    high: float
    close: float
    open: Optional[float] = None
    low: Optional[float] = None
    put_wall: Optional[float] = None
    call_wall: Optional[float] = None
    wall_expiry: Optional[date] = None
    wall_data_expiry: Optional[date] = None
    wall_data_source_date: Optional[date] = None
    wall_sessions_to_expiry: Optional[int] = None
    five_factor_regime: Optional[str] = None
    iv_rank: Optional[float] = None
    skew_25d: Optional[float] = None
    entry_stale_reason: Optional[str] = None

    @property
    def close_return(self) -> float:
        return self.close / self.prior_close - 1.0

    @property
    def intraday_high_return(self) -> float:
        return self.high / self.prior_close - 1.0

    @property
    def gap_open_return(self) -> Optional[float]:
        if self.open is None:
            return None
        return self.open / self.prior_close - 1.0

    @property
    def intraday_low_return(self) -> Optional[float]:
        if self.low is None:
            return None
        return self.low / self.prior_close - 1.0


@dataclass(frozen=True)
class StrategySignal:
    code: str
    title: str
    status: str
    trigger: str
    entry_rule: str
    legs: Tuple[str, ...]
    exit_rule: str
    risk_rule: str
    evidence: Tuple[str, ...] = ()
    caveats: Tuple[str, ...] = ()

    @property
    def should_display(self) -> bool:
        return self.status in {
            "TRIGGERED",
            "WATCH",
            "BLOCKED",
            "STALE",
            "MISSED_WINDOW",
        }


def _valid_positive(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _evaluate_uptrend(context: QQQSignalContext) -> StrategySignal:
    triggered = context.intraday_high_return >= UPTREND_INTRADAY_THRESHOLD
    regime = (context.five_factor_regime or "").strip().lower()
    status = "TRIGGERED" if triggered else "NO_SIGNAL"
    caveats = [
        "冻结回测主规则采用 all-signals，不以五因子状态过滤；该结构含 Short Put 左尾与 Short Gamma 风险"
    ]
    if triggered and regime == "vol_rising":
        status = "WATCH"
        caveats.append("历史状态归因建议在 vol_rising 降低规模或增加确认，不直接视为可执行指令")
    elif triggered and regime == "high_rising":
        caveats.append("冻结状态归因未自动排除 high_rising，但该结构的 Short Put 左尾风险仍需人工控制")
    elif triggered and regime and regime not in KNOWN_FIVE_FACTOR_REGIMES:
        caveats.append(f"五因子值 {regime!r} 不在冻结词表，仅保留原始价格信号")
    elif triggered and not regime:
        caveats.append("五因子状态缺失；邮件只能展示未经状态过滤的冻结价格信号")
    return StrategySignal(
        code="uptrend_funded_call",
        title="上涨走势：2%盘中突破 Funded Call",
        status=status,
        trigger="QQQ 当日最高价 / 前收盘价 - 1 >= 2%，收盘后确认",
        entry_rule=f"{context.entry_date.isoformat()} EOD（T+1）按逐腿 bid/ask 构建",
        legs=(
            "卖出约45DTE、-0.40 Delta 的 QQQ Put",
            "用 Short Put 可执行 bid 作为全部预算",
            "买入同到期、ask 不超过预算的最贵 QQQ Call；组合初始净支出不得为正",
        ),
        exit_rule="持有21个交易日；到期约束需要时提前退出",
        risk_rule="核心风险来自 Short Put 左尾；按现金担保或账户压力测试口径计资本，不能视为免费 Call",
        evidence=(
            f"盘中最高涨幅 {context.intraday_high_return:+.2%}",
            f"收盘涨幅 {context.close_return:+.2%}",
            f"五因子状态 {context.five_factor_regime or '缺失'}",
        ),
        caveats=(
            "45D为默认，60D仅作流动性备选，不按近期收益择优",
            *caveats,
        ),
    )


def _evaluate_downtrend(context: QQQSignalContext) -> StrategySignal:
    triggered = context.close_return <= DOWNTREND_CLOSE_THRESHOLD
    regime = (context.five_factor_regime or "").strip().lower()
    status = "NO_SIGNAL"
    caveats = []
    if triggered:
        if regime in BLOCKED_DOWNTREND_REGIMES:
            status = "BLOCKED"
            caveats.append(f"五因子状态 {regime} 禁止新增 Short Gamma；仅记录，不执行")
        elif regime in ALLOWED_DOWNTREND_REGIMES:
            status = "TRIGGERED"
        elif regime:
            status = "WATCH"
            caveats.append(f"五因子值 {regime!r} 不在冻结词表；未知状态不得放行，需人工复核")
        else:
            status = "WATCH"
            caveats.append("云端数据未提供五因子 regime；未知状态不得放行，需人工复核")
    return StrategySignal(
        code="downtrend_defined_risk_combo",
        title="下跌走势：1%回撤后三腿定义风险反弹组合",
        status=status,
        trigger="QQQ 当日收盘收益 <= -1%，且五因子 gate 不处于 vol_rising/high_rising",
        entry_rule=f"{context.entry_date.isoformat()} EOD（T+1）按逐腿 bid/ask 构建",
        legs=(
            "卖出约45DTE、-0.50 Delta 的 QQQ Put",
            "买入同到期、低约10% spot 的保护 Put",
            "以 Short Put bid - Long Put ask 的可执行净信用为预算，买同到期最贵且 ask 不超预算的 Call",
        ),
        exit_rule="持有21个交易日，exit_dte_min=1",
        risk_rule="最大下行风险 = Put Spread 宽度 - 三腿剩余净信用 + 逐腿双边费用",
        evidence=(
            f"跳空开盘 {context.gap_open_return:+.2%}"
            if context.gap_open_return is not None
            else "跳空开盘 缺失",
            f"盘中最低跌幅 {context.intraday_low_return:+.2%}"
            if context.intraday_low_return is not None
            else "盘中最低跌幅 缺失",
            f"收盘涨跌幅 {context.close_return:+.2%}",
            f"五因子状态 {context.five_factor_regime or '缺失'}",
        ),
        caveats=tuple(caveats),
    )


def _evaluate_wall_break(context: QQQSignalContext) -> StrategySignal:
    wall_values_present = _valid_positive(context.put_wall) and _valid_positive(
        context.call_wall
    )
    walls_inverted = wall_values_present and float(context.put_wall) > float(
        context.call_wall
    )
    walls_valid = wall_values_present and not walls_inverted
    timing_valid = (
        context.wall_expiry is not None
        and context.wall_sessions_to_expiry == WALL_SIGNAL_SESSIONS_TO_EXPIRY
    )
    direction = None
    wall = None
    distance = None
    if walls_valid and timing_valid and context.close > float(context.call_wall):
        direction = "call"
        wall = float(context.call_wall)
        distance = (context.close - wall) / context.close
    elif walls_valid and timing_valid and context.close < float(context.put_wall):
        direction = "put"
        wall = float(context.put_wall)
        distance = wall / context.close - 1.0

    within_break_distance = (
        direction is not None
        and distance is not None
        and distance <= WALL_MAX_BREAK_DISTANCE + 1e-12
    )
    if direction == "call":
        outer_strike = wall + context.close * WALL_OUTER_WIDTH_PCT
        legs = (
            f"Call Debit Spread：买入月权 Call Wall 附近 Call（目标执行价 ${wall:.2f}）",
            f"卖出同到期、约高5% spot 的 Call（目标执行价约 ${outer_strike:.2f}）",
        )
        direction_text = "突破 Call Wall"
    elif direction == "put":
        outer_strike = wall - context.close * WALL_OUTER_WIDTH_PCT
        legs = (
            f"Put Debit Spread：买入月权 Put Wall 附近 Put（目标执行价 ${wall:.2f}）",
            f"卖出同到期、约低5% spot 的 Put（目标执行价约 ${outer_strike:.2f}）",
        )
        direction_text = "跌破 Put Wall"
    else:
        legs = (
            "突破 Call Wall 时构建同到期 Call Debit Spread",
            "跌破 Put Wall 时构建同到期 Put Debit Spread",
        )
        direction_text = "未越墙"

    expiry_text = context.wall_expiry.isoformat() if context.wall_expiry else "未确认"
    data_expiry_text = (
        context.wall_data_expiry.isoformat() if context.wall_data_expiry else "缺失"
    )
    data_source_text = (
        context.wall_data_source_date.isoformat()
        if context.wall_data_source_date
        else "缺失"
    )
    distance_text = f"{distance:.2%}" if distance is not None else "不适用"
    status = "NO_SIGNAL"
    caveats = ["OI墙不含持仓方向，不等同经 dealer side 校准的 Gamma Wall"]
    if walls_inverted:
        status = "WATCH"
        caveats.append("Put Wall 高于 Call Wall，墙数据倒挂；按数据质量异常处理，不生成交易指令")
    elif within_break_distance:
        if context.wall_data_expiry is None:
            status = "WATCH"
            caveats.append("数据库未提供 wall_expiry，无法确认墙值属于目标标准月权；不得执行")
        elif context.wall_data_expiry != context.wall_expiry:
            status = "WATCH"
            caveats.append(
                f"墙数据到期日 {data_expiry_text} 与目标月权 {expiry_text} 不一致；不得执行"
            )
        elif context.wall_data_source_date is None:
            status = "WATCH"
            caveats.append("数据库未提供月权墙 OI 来源日，无法证明其为信号日截面；不得执行")
        elif context.wall_data_source_date != context.signal_date:
            status = "WATCH"
            caveats.append(
                f"月权墙 OI 来源日 {data_source_text} 与信号日 {context.signal_date.isoformat()} 不一致；不得执行"
            )
        else:
            status = "TRIGGERED"
    elif (
        walls_valid
        and context.wall_sessions_to_expiry == WALL_SIGNAL_SESSIONS_TO_EXPIRY - 1
    ):
        status = "MISSED_WINDOW"
        caveats.append("冻结信号日已错过；T+1后只剩4个结算交易日，不得追单")
    return StrategySignal(
        code="monthly_wall_break_debit_spread",
        title="期权墙：标准月权突破顺势 Debit Spread",
        status=status,
        trigger="标准月权结算前第6个交易日，QQQ越过 Put/Call Wall 且越墙距离不超过3%",
        entry_rule=f"{context.entry_date.isoformat()} EOD（T+1）买 ask / 卖 bid；目标月权 {expiry_text}",
        legs=legs,
        exit_rule="持有到该月权到期，以实际成交/到期内在价值核算",
        risk_rule="最大损失为 Debit Spread 净支付权利金，可发生100%权利金损失",
        evidence=(
            f"方向 {direction_text}",
            f"Put Wall {context.put_wall if context.put_wall is not None else '缺失'} / Call Wall {context.call_wall if context.call_wall is not None else '缺失'}",
            f"目标月权 {expiry_text} / 墙数据到期 {data_expiry_text} / OI来源日 {data_source_text}",
            f"信号收盘后至结算共 {context.wall_sessions_to_expiry if context.wall_sessions_to_expiry is not None else '不适用'} 个交易日；越墙距离 {distance_text}",
        ),
        caveats=tuple(caveats),
    )


def evaluate_qqq_signals(context: QQQSignalContext) -> Tuple[StrategySignal, ...]:
    if not all(
        math.isfinite(float(value)) and float(value) > 0
        for value in (context.prior_close, context.high, context.close)
    ):
        raise ValueError("QQQ signal context requires finite positive OHLC values")
    optional_prices = {
        "open": context.open,
        "low": context.low,
    }
    if any(
        value is not None
        and (not math.isfinite(float(value)) or float(value) <= 0)
        for value in optional_prices.values()
    ):
        raise ValueError("QQQ signal context requires finite positive optional OHLC values")
    if context.low is not None:
        observed_prices = [context.high, context.close]
        if context.open is not None:
            observed_prices.append(context.open)
        if context.low > min(observed_prices) or context.high < max(observed_prices):
            raise ValueError("QQQ signal context has inconsistent daily OHLC bounds")
    signals = (
        _evaluate_uptrend(context),
        _evaluate_downtrend(context),
        _evaluate_wall_break(context),
    )
    if not context.entry_stale_reason:
        return signals
    stale = []
    for signal in signals:
        if signal.status in {"TRIGGERED", "WATCH"}:
            stale.append(
                replace(
                    signal,
                    status="STALE",
                    entry_rule=f"不入场（已过期）：{context.entry_stale_reason}；原计划 {signal.entry_rule}",
                    caveats=signal.caveats
                    + ("历史信号仅保留审计，不得按过期价格追单",),
                )
            )
        else:
            stale.append(signal)
    return tuple(stale)


def format_signal_email_section(
    context: QQQSignalContext,
    signals: Tuple[StrategySignal, ...],
) -> str:
    displayed = [signal for signal in signals if signal.should_display]
    lines = [
        "",
        "[QQQ 冻结策略信号监控]",
        (
            f"信号日: {context.signal_date.isoformat()} | 前收 ${context.prior_close:.2f}"
            f" | 开盘 ${context.open:.2f}" if context.open is not None else
            f"信号日: {context.signal_date.isoformat()} | 前收 ${context.prior_close:.2f} | 开盘 缺失"
        )
        + f" | 最高 ${context.high:.2f}"
        + (f" | 最低 ${context.low:.2f}" if context.low is not None else " | 最低 缺失")
        + f" | 收盘 ${context.close:.2f}",
        (
            f"跳空开盘: {context.gap_open_return:+.2%}"
            if context.gap_open_return is not None
            else "跳空开盘: 缺失"
        )
        + f" | 盘中最高涨幅: {context.intraday_high_return:+.2%}"
        + (
            f" | 盘中最低跌幅: {context.intraday_low_return:+.2%}"
            if context.intraday_low_return is not None
            else " | 盘中最低跌幅: 缺失"
        )
        + f" | 收盘涨跌幅: {context.close_return:+.2%}",
        "数据口径: 仅使用已完成日线；未收盘的当前交易日不会进入本次判断。",
    ]
    if context.iv_rank is not None or context.skew_25d is not None:
        lines.append(
            f"IVR: {context.iv_rank if context.iv_rank is not None else '缺失'} | 25D Skew: {context.skew_25d if context.skew_25d is not None else '缺失'}"
        )
    if not displayed:
        lines.append("结论: 今日没有触发上涨、下跌或月权墙策略信号。")
        return "\n".join(lines)

    lines.append(f"结论: 检测到 {len(displayed)} 个触发/观察项；仅作研究提示，不自动下单。")
    for index, signal in enumerate(displayed, start=1):
        lines.extend(
            [
                "",
                f"{index}. [{signal.status}] {signal.title}",
                f"触发条件: {signal.trigger}",
                f"入场: {signal.entry_rule}",
                "结构:",
                *[f"  - {leg}" for leg in signal.legs],
                f"退出: {signal.exit_rule}",
                f"风险: {signal.risk_rule}",
                "证据: " + "；".join(signal.evidence),
            ]
        )
        if signal.caveats:
            lines.append("注意: " + "；".join(signal.caveats))
    return "\n".join(lines)
