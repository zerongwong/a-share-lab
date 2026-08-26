"""Fail-closed intraday range alerts for up to four existing A-share positions.

This module produces research alerts only.  It does not fetch quotes, connect
to a broker, schedule work, send notifications or place orders.  In
particular, an intraday sell alert is capped by the broker-reported quantity
that was already settled before today: shares bought today never increase the
same-day sellable quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_FLOOR, Decimal
from enum import StrEnum
from math import isfinite
from zoneinfo import ZoneInfo

from ashare_lab.domain.market_rules import Board, board_rule, is_continuous_auction

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
MAX_MONITORED_POSITIONS = 4


class PositionMonitorStatus(StrEnum):
    DISABLED = "disabled"
    RESEARCH_ONLY = "research_only"
    OUTSIDE_TRADING_SESSION = "outside_trading_session"
    INCOMPLETE_DATA = "incomplete_data"
    STALE_DATA = "stale_data"
    MODEL_NOT_READY = "model_not_ready"
    NO_ALERTS = "no_alerts"
    READY = "ready"


class RangeAlertKind(StrEnum):
    LOW_BUY_RESEARCH = "low_buy_research"
    HIGH_SELL_RESEARCH = "high_sell_research"


@dataclass(frozen=True, slots=True)
class RangeProbabilityBundle:
    """Calibrated conditional probabilities for one fixed intraday plan.

    The names make the denominators explicit enough for a UI to label them:
    low-to-high estimates are conditional on a simulated fill in the low zone;
    high-to-low estimates are conditional on a simulated fill in the high
    zone.  They must not be presented as unconditional stock-rise odds.
    """

    low_fill_within_60s: float | None = None
    high_sell_fill_within_60s: float | None = None
    low_to_high_before_invalidation: float | None = None
    high_to_low_before_close: float | None = None
    invalidation_before_high: float | None = None
    calibrated: bool = False
    confidence_intervals_available: bool = False
    base_rates_available: bool = False
    out_of_sample_validated: bool = False
    shadow_run_validated: bool = False
    cohort_id: str = ""
    model_version: str = ""
    calibration_version: str = ""
    sample_size: int = 0

    def is_usable(self, minimum_sample_size: int) -> bool:
        probabilities = (
            self.low_fill_within_60s,
            self.high_sell_fill_within_60s,
            self.low_to_high_before_invalidation,
            self.high_to_low_before_close,
            self.invalidation_before_high,
        )
        return (
            self.calibrated
            and self.confidence_intervals_available
            and self.base_rates_available
            and self.out_of_sample_validated
            and self.shadow_run_validated
            and bool(self.cohort_id.strip())
            and bool(self.model_version.strip())
            and bool(self.calibration_version.strip())
            and self.sample_size >= minimum_sample_size
            and all(
                value is not None and isfinite(value) and 0.0 <= value <= 1.0
                for value in probabilities
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IntradayRangePlan:
    symbol: str
    valid_trade_date: date
    generated_at: datetime
    evidence_cutoff: datetime
    low_zone_low: Decimal
    low_zone_high: Decimal
    high_zone_low: Decimal
    high_zone_high: Decimal
    invalidation_below: Decimal
    planned_max_quantity: int
    evidence_ids: tuple[str, ...] = ()
    data_complete: bool = False
    no_lookahead_verified: bool = False
    probabilities: RangeProbabilityBundle = field(default_factory=RangeProbabilityBundle)

    def is_structurally_usable(self, as_of: datetime) -> bool:
        if self.generated_at.tzinfo is None or self.evidence_cutoff.tzinfo is None:
            return False
        generated = self.generated_at.astimezone(SHANGHAI_TZ)
        cutoff = self.evidence_cutoff.astimezone(SHANGHAI_TZ)
        current = as_of.astimezone(SHANGHAI_TZ)
        return (
            self.data_complete
            and self.no_lookahead_verified
            and self.valid_trade_date == current.date()
            and cutoff <= generated <= current
            and Decimal("0") < self.invalidation_below < self.low_zone_low
            and self.low_zone_low <= self.low_zone_high < self.high_zone_low
            and self.high_zone_low <= self.high_zone_high
            and self.planned_max_quantity > 0
            and bool(self.evidence_ids)
            and all(evidence.strip() for evidence in self.evidence_ids)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeQuote:
    symbol: str
    board: Board
    timestamp: datetime
    last_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    official_limit_down: Decimal
    official_limit_up: Decimal
    data_complete: bool = False
    trading_state_normal: bool = False
    is_suspended: bool = True

    def is_usable(self) -> bool:
        return (
            self.data_complete
            and self.trading_state_normal
            and not self.is_suspended
            and self.timestamp.tzinfo is not None
            and Decimal("0") < self.official_limit_down <= self.last_price
            and self.last_price <= self.official_limit_up
            and Decimal("0") < self.best_bid <= self.best_ask
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class BrokerPositionInventory:
    """Broker-authoritative inventory split needed to enforce A-share T+1."""

    symbol: str
    total_quantity: int
    settled_before_today_quantity: int
    sellable_quantity: int
    today_bought_quantity: int
    today_sold_quantity: int
    frozen_sell_quantity: int

    def is_consistent(self) -> bool:
        quantities = (
            self.total_quantity,
            self.settled_before_today_quantity,
            self.sellable_quantity,
            self.today_bought_quantity,
            self.today_sold_quantity,
            self.frozen_sell_quantity,
        )
        expected_total = (
            self.settled_before_today_quantity
            + self.today_bought_quantity
            - self.today_sold_quantity
        )
        return (
            all(value >= 0 for value in quantities)
            and self.today_sold_quantity <= self.settled_before_today_quantity
            and self.total_quantity == expected_total
            and self.sellable_quantity + self.frozen_sell_quantity
            <= self.settled_before_today_quantity - self.today_sold_quantity
            and self.sellable_quantity + self.frozen_sell_quantity
            <= self.total_quantity - self.today_bought_quantity
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionMonitorSnapshot:
    as_of: datetime
    account_timestamp: datetime
    selected_symbols: tuple[str, ...]
    quotes: tuple[RealtimeQuote, ...]
    positions: tuple[BrokerPositionInventory, ...]
    plans: tuple[IntradayRangePlan, ...]
    available_cash: Decimal
    licensed_realtime_data: bool = False
    broker_account_authoritative: bool = False
    broker_account_read_only: bool = False
    account_data_complete: bool = False
    clock_synchronized: bool = False
    trading_calendar_verified: bool = False
    market_halted: bool = True


@dataclass(frozen=True, slots=True)
class PositionMonitorConfig:
    enabled: bool = False
    paper_research_only: bool = True
    max_quote_latency_seconds: float = 3.0
    max_account_latency_seconds: float = 3.0
    minimum_probability_sample_size: int = 1_000
    require_calibrated_probabilities: bool = True
    estimated_buy_fee_rate: Decimal = Decimal("0.001")


@dataclass(frozen=True, slots=True)
class PositionRangeAlert:
    symbol: str
    kind: RangeAlertKind
    last_price: Decimal
    zone_low: Decimal
    zone_high: Decimal
    invalidation_below: Decimal
    maximum_research_quantity: int
    settled_before_today_quantity: int
    sellable_quantity_at_snapshot: int
    today_bought_locked_quantity: int
    probabilities: RangeProbabilityBundle
    inventory_rule: str
    research_only: bool = True
    auto_order_allowed: bool = False


@dataclass(frozen=True, slots=True)
class PositionMonitorResult:
    status: PositionMonitorStatus
    alerts: tuple[PositionRangeAlert, ...] = ()
    reasons: tuple[str, ...] = ()
    auto_order_allowed: bool = False
    notification_sent: bool = False
    system_task_created: bool = False


def _result(status: PositionMonitorStatus, *reasons: str) -> PositionMonitorResult:
    return PositionMonitorResult(status=status, reasons=tuple(reasons))


def _age_seconds(as_of: datetime, timestamp: datetime) -> float | None:
    if as_of.tzinfo is None or timestamp.tzinfo is None:
        return None
    return (as_of.astimezone(SHANGHAI_TZ) - timestamp.astimezone(SHANGHAI_TZ)).total_seconds()


def _is_fresh(as_of: datetime, timestamp: datetime, maximum: float) -> bool:
    age = _age_seconds(as_of, timestamp)
    return age is not None and 0.0 <= age <= maximum


def _whole_lot(quantity: int, lot_size: int) -> int:
    return max(0, quantity // lot_size * lot_size)


def _cash_capacity(cash: Decimal, ask: Decimal, fee_rate: Decimal, lot_size: int) -> int:
    if cash <= 0 or ask <= 0 or fee_rate < 0:
        return 0
    raw = (cash / (ask * (Decimal("1") + fee_rate))).to_integral_value(rounding=ROUND_FLOOR)
    return _whole_lot(int(raw), lot_size)


def _validate_snapshot_shape(snapshot: PositionMonitorSnapshot) -> bool:
    symbols = tuple(symbol.strip().upper() for symbol in snapshot.selected_symbols)
    if not 1 <= len(symbols) <= MAX_MONITORED_POSITIONS or len(set(symbols)) != len(symbols):
        return False
    expected = set(symbols)
    collections = (snapshot.quotes, snapshot.positions, snapshot.plans)
    return all(
        len(items) == len(symbols) and {item.symbol.strip().upper() for item in items} == expected
        for items in collections
    )


def _buy_quantity(
    position: BrokerPositionInventory,
    plan: IntradayRangePlan,
    quote: RealtimeQuote,
    available_cash: Decimal,
    config: PositionMonitorConfig,
) -> int:
    """Cap a low-zone leg by T-1 inventory; never treat today's buy as sellable."""

    lot_size = board_rule(quote.board).buy_lot
    if position.today_sold_quantity > 0:
        inventory_cap = position.today_sold_quantity
    else:
        inventory_cap = position.sellable_quantity
    cash_cap = _cash_capacity(
        available_cash,
        quote.best_ask,
        config.estimated_buy_fee_rate,
        lot_size,
    )
    return _whole_lot(
        min(plan.planned_max_quantity, inventory_cap, cash_cap),
        lot_size,
    )


def _sell_quantity(
    position: BrokerPositionInventory,
    plan: IntradayRangePlan,
    quote: RealtimeQuote,
) -> int:
    """Only the broker's remaining T-1 sellable inventory can be referenced."""

    return _whole_lot(
        min(plan.planned_max_quantity, position.sellable_quantity),
        board_rule(quote.board).buy_lot,
    )


def monitor_position_ranges(
    snapshot: PositionMonitorSnapshot,
    config: PositionMonitorConfig | None = None,
) -> PositionMonitorResult:
    """Evaluate fixed intraday zones without executing or sending anything."""

    config = config or PositionMonitorConfig()
    if not config.enabled:
        return _result(PositionMonitorStatus.DISABLED, "feature_flag_disabled")
    if not config.paper_research_only:
        return _result(PositionMonitorStatus.RESEARCH_ONLY, "live_execution_is_forbidden")
    if snapshot.as_of.tzinfo is None or not is_continuous_auction(snapshot.as_of):
        return _result(
            PositionMonitorStatus.OUTSIDE_TRADING_SESSION,
            "continuous_auction_session_required",
        )
    if not _validate_snapshot_shape(snapshot):
        return _result(
            PositionMonitorStatus.INCOMPLETE_DATA,
            "one_to_four_unique_symbols_with_matching_quote_account_and_plan_required",
        )
    if (
        not all(
            (
                snapshot.licensed_realtime_data,
                snapshot.broker_account_authoritative,
                snapshot.broker_account_read_only,
                snapshot.account_data_complete,
                snapshot.clock_synchronized,
                snapshot.trading_calendar_verified,
            )
        )
        or snapshot.market_halted
    ):
        return _result(
            PositionMonitorStatus.INCOMPLETE_DATA,
            "licensed_quotes_and_authoritative_read_only_broker_inventory_required",
        )
    if snapshot.available_cash < 0 or not _is_fresh(
        snapshot.as_of,
        snapshot.account_timestamp,
        config.max_account_latency_seconds,
    ):
        return _result(
            PositionMonitorStatus.STALE_DATA,
            "broker_inventory_snapshot_exceeds_latency_limit",
        )

    quotes = {item.symbol.strip().upper(): item for item in snapshot.quotes}
    positions = {item.symbol.strip().upper(): item for item in snapshot.positions}
    plans = {item.symbol.strip().upper(): item for item in snapshot.plans}
    for symbol in snapshot.selected_symbols:
        normalized = symbol.strip().upper()
        quote = quotes[normalized]
        position = positions[normalized]
        plan = plans[normalized]
        if (
            not quote.is_usable()
            or not position.is_consistent()
            or position.settled_before_today_quantity <= 0
            or not plan.is_structurally_usable(snapshot.as_of)
        ):
            return _result(
                PositionMonitorStatus.INCOMPLETE_DATA,
                "complete_quote_valid_plan_and_t_minus_one_inventory_required",
            )
        if not _is_fresh(snapshot.as_of, quote.timestamp, config.max_quote_latency_seconds):
            return _result(
                PositionMonitorStatus.STALE_DATA,
                "quote_snapshot_exceeds_latency_limit",
            )
        if config.require_calibrated_probabilities and not plan.probabilities.is_usable(
            config.minimum_probability_sample_size
        ):
            return _result(
                PositionMonitorStatus.MODEL_NOT_READY,
                "oos_shadow_calibrated_conditional_probabilities_required",
            )

    alerts: list[PositionRangeAlert] = []
    remaining_cash = snapshot.available_cash
    for symbol in snapshot.selected_symbols:
        normalized = symbol.strip().upper()
        quote = quotes[normalized]
        position = positions[normalized]
        plan = plans[normalized]
        kind: RangeAlertKind | None = None
        maximum_quantity = 0
        zone_low = Decimal("0")
        zone_high = Decimal("0")

        if plan.high_zone_low <= quote.last_price <= plan.high_zone_high:
            kind = RangeAlertKind.HIGH_SELL_RESEARCH
            zone_low, zone_high = plan.high_zone_low, plan.high_zone_high
            maximum_quantity = _sell_quantity(position, plan, quote)
        elif plan.low_zone_low <= quote.last_price <= plan.low_zone_high:
            kind = RangeAlertKind.LOW_BUY_RESEARCH
            zone_low, zone_high = plan.low_zone_low, plan.low_zone_high
            maximum_quantity = _buy_quantity(position, plan, quote, remaining_cash, config)

        if kind is None or maximum_quantity <= 0:
            continue
        if kind is RangeAlertKind.LOW_BUY_RESEARCH:
            remaining_cash -= (
                Decimal(maximum_quantity)
                * quote.best_ask
                * (Decimal("1") + config.estimated_buy_fee_rate)
            )
        alerts.append(
            PositionRangeAlert(
                symbol=normalized,
                kind=kind,
                last_price=quote.last_price,
                zone_low=zone_low,
                zone_high=zone_high,
                invalidation_below=plan.invalidation_below,
                maximum_research_quantity=maximum_quantity,
                settled_before_today_quantity=position.settled_before_today_quantity,
                sellable_quantity_at_snapshot=position.sellable_quantity,
                today_bought_locked_quantity=position.today_bought_quantity,
                probabilities=plan.probabilities,
                inventory_rule=(
                    "卖出上限仅取券商报告的T-1可卖库存；今日买入不可当日卖出"
                    if kind is RangeAlertKind.HIGH_SELL_RESEARCH
                    else "低位买入当日锁定；后续当日卖出只能使用剩余T-1可卖库存"
                ),
            )
        )

    if not alerts:
        return _result(PositionMonitorStatus.NO_ALERTS, "no_price_entered_a_validated_zone")
    return PositionMonitorResult(
        status=PositionMonitorStatus.READY,
        alerts=tuple(alerts),
        reasons=(
            "research_alert_only",
            "today_bought_shares_are_not_sellable_today",
            "broker_sellable_quantity_is_authoritative",
        ),
    )
