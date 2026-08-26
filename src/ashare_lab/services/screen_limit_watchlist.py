"""Fail-closed 14:45–14:50 A-share limit-watchlist research service.

The service never places or schedules an order. It ranks externally computed,
calibrated estimates only after deterministic timeline, data, liquidity and
risk gates. The feature flag is disabled by default and zero candidates is a
normal successful result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, time
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from zoneinfo import ZoneInfo

from ashare_lab.domain.market_rules import (
    CLOSING_AUCTION_START,
    FINANCING_FRACTION,
    MAX_QUOTE_LATENCY_SECONDS,
    MAX_SPECULATIVE_CAPITAL_FRACTION,
    MAX_WATCHLIST_CANDIDATES,
    MAX_WATCHLIST_SINGLE_NAME_FRACTION,
    T1_EXIT_POLICY_ID,
    T1_EXIT_RESEARCH_END,
    T1_EXIT_RESEARCH_START,
    WATCHLIST_CANCEL_CUTOFF,
    Board,
    daily_price_limit_rate,
    is_watchlist_feature_freeze_window,
    is_watchlist_preflight_window,
    is_watchlist_signal_window,
)

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class WatchlistStatus(StrEnum):
    DISABLED = "disabled"
    RESEARCH_ONLY = "research_only"
    INCOMPLETE_DATA = "incomplete_data"
    STALE_DATA = "stale_data"
    OUTSIDE_SIGNAL_WINDOW = "outside_signal_window"
    MARKET_UNAVAILABLE = "market_unavailable"
    MODEL_NOT_READY = "model_not_ready"
    NO_CANDIDATES = "no_candidates"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class ProbabilityBundle:
    fill_today: float | None = None
    close_limit_today: float | None = None
    hit_limit_t1: float | None = None
    close_limit_t1: float | None = None
    positive_return_t1: float | None = None
    exit_by_t1_1000: float | None = None
    loss_5pct_t1: float | None = None
    expected_net_return_t1: float | None = None
    cvar95_t1: float | None = None
    calibrated: bool = False
    confidence_intervals_available: bool = False
    base_rates_available: bool = False
    cohort_id: str = ""
    sample_size: int = 0

    def is_usable(self, minimum_sample_size: int) -> bool:
        probabilities = (
            self.fill_today,
            self.close_limit_today,
            self.hit_limit_t1,
            self.close_limit_t1,
            self.positive_return_t1,
            self.exit_by_t1_1000,
            self.loss_5pct_t1,
        )
        return (
            self.calibrated
            and self.confidence_intervals_available
            and self.base_rates_available
            and bool(self.cohort_id.strip())
            and self.sample_size >= minimum_sample_size
            and all(
                value is not None and isfinite(value) and 0.0 <= value <= 1.0
                for value in probabilities
            )
            and self.expected_net_return_t1 is not None
            and isfinite(self.expected_net_return_t1)
            and self.cvar95_t1 is not None
            and isfinite(self.cvar95_t1)
            and self.cvar95_t1 <= 0.0
        )


@dataclass(frozen=True, slots=True)
class TraceableScoreInputs:
    """Normalized component values and one source id per fixed component.

    Values use ``0..1`` where larger is better, except ``tail_risk`` where
    larger means worse. ``evidence_ids`` follows the declaration order below.
    """

    market_environment: float | None = None
    sector_confirmation: float | None = None
    price_volume_structure: float | None = None
    limit_behavior: float | None = None
    catalyst_quality: float | None = None
    execution_quality: float | None = None
    next_day_distribution: float | None = None
    tail_risk: float | None = None
    evidence_ids: tuple[str, ...] = ()

    def is_usable(self) -> bool:
        values = (
            self.market_environment,
            self.sector_confirmation,
            self.price_volume_structure,
            self.limit_behavior,
            self.catalyst_quality,
            self.execution_quality,
            self.next_day_distribution,
            self.tail_risk,
        )
        return (
            all(value is not None and isfinite(value) and 0.0 <= value <= 1.0 for value in values)
            and len(self.evidence_ids) == len(values)
            and all(source.strip() for source in self.evidence_ids)
        )


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    market_environment: float
    sector_confirmation: float
    price_volume_structure: float
    limit_behavior: float
    catalyst_quality: float
    execution_quality: float
    next_day_distribution: float
    tail_risk_penalty: float
    total: float
    evidence_ids: tuple[str, ...]


def build_score_breakdown(inputs: TraceableScoreInputs) -> ScoreBreakdown | None:
    """Apply the fixed, auditable 100-point score and a 30-point risk penalty."""

    if not inputs.is_usable():
        return None
    assert inputs.market_environment is not None
    assert inputs.sector_confirmation is not None
    assert inputs.price_volume_structure is not None
    assert inputs.limit_behavior is not None
    assert inputs.catalyst_quality is not None
    assert inputs.execution_quality is not None
    assert inputs.next_day_distribution is not None
    assert inputs.tail_risk is not None

    market = 10.0 * inputs.market_environment
    sector = 15.0 * inputs.sector_confirmation
    structure = 15.0 * inputs.price_volume_structure
    limit_behavior = 20.0 * inputs.limit_behavior
    catalyst = 10.0 * inputs.catalyst_quality
    execution = 15.0 * inputs.execution_quality
    next_day = 15.0 * inputs.next_day_distribution
    tail_penalty = 30.0 * inputs.tail_risk
    total = max(
        0.0,
        market
        + sector
        + structure
        + limit_behavior
        + catalyst
        + execution
        + next_day
        - tail_penalty,
    )
    return ScoreBreakdown(
        market_environment=market,
        sector_confirmation=sector,
        price_volume_structure=structure,
        limit_behavior=limit_behavior,
        catalyst_quality=catalyst,
        execution_quality=execution,
        next_day_distribution=next_day,
        tail_risk_penalty=tail_penalty,
        total=total,
        evidence_ids=inputs.evidence_ids,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class LimitStockSignal:
    symbol: str
    name: str
    board: Board
    theme: str
    quote_timestamp: datetime
    last_price: Decimal
    official_limit_up_price: Decimal
    price_tick: Decimal = Decimal("0.01")

    # Positive evidence must be explicit. Safety-sensitive fields default to
    # false/unknown so a partially populated signal is never eligible.
    data_complete: bool = False
    level2_complete: bool = False
    limit_price_verified: bool = False
    user_has_board_permission: bool = False
    board_model_ready: bool = False
    listed_trading_days: int = 0
    is_st: bool = True
    is_delisting: bool = True
    is_suspended: bool = True
    known_next_session_suspension: bool = True
    has_major_negative_event: bool = True
    severe_abnormal_trading_risk: bool = True
    first_five_day_no_limit: bool = True

    touched_limit: bool = False
    at_limit: bool = False
    estimated_fillable: bool = False
    one_word_limit: bool = True
    board_open_count: int = 99
    cancellation_rate: float = 1.0
    queue_decay_rate: float = 1.0
    sector_limit_count: int = 0
    sector_breadth: float = 0.0
    market_broken_board_rate: float = 1.0

    median_turnover_20d: Decimal = Decimal("0")
    turnover_last_5m: Decimal = Decimal("0")
    minimum_lot_affordable: bool = False
    probabilities: ProbabilityBundle = field(default_factory=ProbabilityBundle)
    score_inputs: TraceableScoreInputs = field(default_factory=TraceableScoreInputs)


@dataclass(frozen=True, slots=True, kw_only=True)
class LimitWatchlistSnapshot:
    as_of: datetime
    preflight_completed_at: datetime
    feature_frozen_at: datetime
    market_quote_timestamp: datetime
    market_data_latency_seconds: float | None
    account_equity: Decimal
    signals: tuple[LimitStockSignal, ...] = ()
    data_complete: bool = False
    licensed_realtime_data: bool = False
    clock_synchronized: bool = False
    trading_calendar_verified: bool = False
    market_halted: bool = True
    model_ready: bool = False
    historical_l2_backtest_ready: bool = False
    out_of_sample_validation_ready: bool = False
    shadow_run_ready: bool = False
    probability_calibration_ready: bool = False
    model_version: str = ""
    data_version: str = ""
    rule_version: str = ""
    exit_policy_id: str = ""


@dataclass(frozen=True, slots=True)
class LimitWatchlistConfig:
    enabled: bool = False
    paper_research_only: bool = True
    allowed_boards: frozenset[Board] = frozenset(Board)
    max_quote_latency_seconds: float = MAX_QUOTE_LATENCY_SECONDS
    min_median_turnover_20d: Decimal = Decimal("100000000")
    max_order_share_of_last_5m_turnover: Decimal = Decimal("0.01")
    min_model_sample_size: int = 1_000
    min_fill_probability: float = 0.05
    min_exit_by_t1_1000_probability: float = 0.50
    min_expected_net_return: float = 0.0
    max_loss_5pct_probability: float = 0.50
    min_total_score: float = 60.0
    max_board_open_count: int = 3
    max_cancellation_rate: float = 0.80
    max_queue_decay_rate: float = 0.80
    min_sector_limit_count: int = 2
    min_sector_breadth: float = 0.50
    max_market_broken_board_rate: float = 0.50
    max_same_theme: int = 2


@dataclass(frozen=True, slots=True)
class WatchlistCandidate:
    symbol: str
    name: str
    board: Board
    theme: str
    rank: int
    score: float
    score_breakdown: ScoreBreakdown
    capital_fraction: Decimal
    probabilities: ProbabilityBundle
    research_only: bool = True
    auto_order_allowed: bool = False


@dataclass(frozen=True, slots=True)
class LimitWatchlistResult:
    status: WatchlistStatus
    candidates: tuple[WatchlistCandidate, ...] = ()
    reasons: tuple[str, ...] = ()
    total_capital_fraction: Decimal = Decimal("0")
    financing_fraction: Decimal = FINANCING_FRACTION
    single_name_capital_fraction: Decimal = MAX_WATCHLIST_SINGLE_NAME_FRACTION
    cancel_unfilled_by: time = WATCHLIST_CANCEL_CUTOFF
    orders_irrevocable_from: time = CLOSING_AUCTION_START
    t1_exit_research_window: tuple[time, time] = (
        T1_EXIT_RESEARCH_START,
        T1_EXIT_RESEARCH_END,
    )
    exit_policy_id: str = T1_EXIT_POLICY_ID
    auto_order_allowed: bool = False
    system_task_created: bool = False


ALLOCATION_PER_CANDIDATE = MAX_WATCHLIST_SINGLE_NAME_FRACTION


def _result(status: WatchlistStatus, *reasons: str) -> LimitWatchlistResult:
    return LimitWatchlistResult(status=status, reasons=tuple(reasons))


def _age_seconds(as_of: datetime, quote_timestamp: datetime) -> float | None:
    if as_of.tzinfo is None or quote_timestamp.tzinfo is None:
        return None
    return (as_of.astimezone(SHANGHAI_TZ) - quote_timestamp.astimezone(SHANGHAI_TZ)).total_seconds()


def _is_fresh(as_of: datetime, quote_timestamp: datetime, maximum: float) -> bool:
    age = _age_seconds(as_of, quote_timestamp)
    return age is not None and 0.0 <= age <= maximum


def _same_trade_date(*timestamps: datetime) -> bool:
    return len({item.astimezone(SHANGHAI_TZ).date() for item in timestamps}) == 1


def _valid_timeline(snapshot: LimitWatchlistSnapshot) -> bool:
    timestamps = (
        snapshot.preflight_completed_at,
        snapshot.feature_frozen_at,
        snapshot.as_of,
    )
    if any(item.tzinfo is None for item in timestamps):
        return False
    if not _same_trade_date(*timestamps):
        return False
    preflight, frozen, output = (item.astimezone(SHANGHAI_TZ) for item in timestamps)
    return (
        is_watchlist_preflight_window(preflight)
        and is_watchlist_feature_freeze_window(frozen)
        and is_watchlist_signal_window(output)
        and preflight < frozen < output
    )


def _near_daily_limit(signal: LimitStockSignal) -> bool:
    if signal.last_price <= 0 or signal.official_limit_up_price <= 0:
        return False
    if signal.price_tick <= 0 or signal.last_price > signal.official_limit_up_price:
        return False
    ticks_away = (signal.official_limit_up_price - signal.last_price) / signal.price_tick
    return signal.at_limit or signal.touched_limit or ticks_away <= Decimal("3")


def _eligible(
    signal: LimitStockSignal,
    snapshot: LimitWatchlistSnapshot,
    config: LimitWatchlistConfig,
) -> tuple[bool, ScoreBreakdown | None]:
    if signal.board not in config.allowed_boards:
        return False, None
    if not _is_fresh(
        snapshot.feature_frozen_at,
        signal.quote_timestamp,
        config.max_quote_latency_seconds,
    ):
        return False, None
    if not all(
        (
            signal.data_complete,
            signal.level2_complete,
            signal.limit_price_verified,
            signal.user_has_board_permission,
            signal.board_model_ready,
            signal.minimum_lot_affordable,
        )
    ):
        return False, None
    if any(
        (
            signal.is_st,
            signal.is_delisting,
            signal.is_suspended,
            signal.known_next_session_suspension,
            signal.has_major_negative_event,
            signal.severe_abnormal_trading_risk,
            signal.first_five_day_no_limit,
        )
    ):
        return False, None
    if (
        daily_price_limit_rate(
            signal.board,
            snapshot.feature_frozen_at.astimezone(SHANGHAI_TZ).date(),
            listed_trading_days=signal.listed_trading_days,
            is_risk_warning=signal.is_st,
        )
        is None
    ):
        return False, None
    if not _near_daily_limit(signal):
        return False, None
    if not signal.estimated_fillable or signal.one_word_limit:
        return False, None
    if signal.median_turnover_20d < config.min_median_turnover_20d:
        return False, None
    if signal.turnover_last_5m <= 0:
        return False, None
    if signal.board_open_count > config.max_board_open_count:
        return False, None
    if not (0.0 <= signal.cancellation_rate <= config.max_cancellation_rate):
        return False, None
    if not (0.0 <= signal.queue_decay_rate <= config.max_queue_decay_rate):
        return False, None
    if signal.sector_limit_count < config.min_sector_limit_count:
        return False, None
    if not (config.min_sector_breadth <= signal.sector_breadth <= 1.0):
        return False, None
    if not (0.0 <= signal.market_broken_board_rate <= config.max_market_broken_board_rate):
        return False, None

    probabilities = signal.probabilities
    if not probabilities.is_usable(config.min_model_sample_size):
        return False, None
    assert probabilities.fill_today is not None
    assert probabilities.exit_by_t1_1000 is not None
    assert probabilities.loss_5pct_t1 is not None
    assert probabilities.expected_net_return_t1 is not None
    if probabilities.fill_today < config.min_fill_probability:
        return False, None
    if probabilities.exit_by_t1_1000 < config.min_exit_by_t1_1000_probability:
        return False, None
    if probabilities.loss_5pct_t1 > config.max_loss_5pct_probability:
        return False, None
    if probabilities.expected_net_return_t1 <= config.min_expected_net_return:
        return False, None

    breakdown = build_score_breakdown(signal.score_inputs)
    if breakdown is None or breakdown.total < config.min_total_score:
        return False, None
    return True, breakdown


def _ranked(
    signals: Iterable[tuple[LimitStockSignal, ScoreBreakdown]],
) -> list[tuple[LimitStockSignal, ScoreBreakdown]]:
    return sorted(signals, key=lambda item: item[1].total, reverse=True)


def screen_limit_watchlist(
    snapshot: LimitWatchlistSnapshot,
    config: LimitWatchlistConfig | None = None,
) -> LimitWatchlistResult:
    config = config or LimitWatchlistConfig()
    if not config.enabled:
        return _result(WatchlistStatus.DISABLED, "feature_flag_disabled")
    if not config.paper_research_only:
        return _result(WatchlistStatus.RESEARCH_ONLY, "live_execution_is_forbidden")
    if not _valid_timeline(snapshot):
        return _result(
            WatchlistStatus.OUTSIDE_SIGNAL_WINDOW,
            "required_timeline_is_14_44_14_49_55_14_50_05",
        )
    if snapshot.market_quote_timestamp.tzinfo is None:
        return _result(
            WatchlistStatus.INCOMPLETE_DATA,
            "timezone_aware_market_timestamp_required",
        )
    if not all(
        (
            snapshot.data_complete,
            snapshot.licensed_realtime_data,
            snapshot.clock_synchronized,
            snapshot.trading_calendar_verified,
        )
    ):
        return _result(
            WatchlistStatus.INCOMPLETE_DATA,
            "required_market_data_gate_failed",
        )
    if snapshot.market_halted or snapshot.account_equity <= 0:
        return _result(
            WatchlistStatus.MARKET_UNAVAILABLE,
            "market_or_account_unavailable",
        )
    if (
        snapshot.market_data_latency_seconds is None
        or not isfinite(snapshot.market_data_latency_seconds)
        or not 0.0 <= snapshot.market_data_latency_seconds <= config.max_quote_latency_seconds
        or not _is_fresh(
            snapshot.feature_frozen_at,
            snapshot.market_quote_timestamp,
            config.max_quote_latency_seconds,
        )
    ):
        return _result(
            WatchlistStatus.STALE_DATA,
            "feature_snapshot_market_data_exceeds_3_seconds",
        )
    if not all(
        (
            snapshot.model_ready,
            snapshot.historical_l2_backtest_ready,
            snapshot.out_of_sample_validation_ready,
            snapshot.shadow_run_ready,
            snapshot.probability_calibration_ready,
            bool(snapshot.model_version),
            bool(snapshot.data_version),
            bool(snapshot.rule_version),
            snapshot.exit_policy_id == T1_EXIT_POLICY_ID,
        )
    ):
        return _result(
            WatchlistStatus.MODEL_NOT_READY,
            "l2_oos_shadow_calibration_and_fixed_exit_policy_required",
        )

    eligible: list[tuple[LimitStockSignal, ScoreBreakdown]] = []
    for signal in snapshot.signals:
        accepted, breakdown = _eligible(signal, snapshot, config)
        if accepted and breakdown is not None:
            eligible.append((signal, breakdown))

    theme_counts: Counter[str] = Counter()
    selected: list[WatchlistCandidate] = []
    for signal, breakdown in _ranked(eligible):
        if theme_counts[signal.theme] >= config.max_same_theme:
            continue
        if len(selected) >= MAX_WATCHLIST_CANDIDATES:
            break
        proposed_order_value = snapshot.account_equity * ALLOCATION_PER_CANDIDATE
        capacity = signal.turnover_last_5m * config.max_order_share_of_last_5m_turnover
        if proposed_order_value > capacity:
            continue
        selected.append(
            WatchlistCandidate(
                symbol=signal.symbol,
                name=signal.name,
                board=signal.board,
                theme=signal.theme,
                rank=len(selected) + 1,
                score=breakdown.total,
                score_breakdown=breakdown,
                capital_fraction=ALLOCATION_PER_CANDIDATE,
                probabilities=signal.probabilities,
            )
        )
        theme_counts[signal.theme] += 1

    if not selected:
        return _result(
            WatchlistStatus.NO_CANDIDATES,
            "no_signal_passed_all_safety_gates",
        )

    total_capital_fraction = sum(
        (candidate.capital_fraction for candidate in selected),
        start=Decimal("0"),
    )
    if total_capital_fraction > MAX_SPECULATIVE_CAPITAL_FRACTION or any(
        candidate.capital_fraction > MAX_WATCHLIST_SINGLE_NAME_FRACTION for candidate in selected
    ):
        return _result(
            WatchlistStatus.NO_CANDIDATES,
            "capital_limit_invariant_failed",
        )
    return LimitWatchlistResult(
        status=WatchlistStatus.READY,
        candidates=tuple(selected),
        reasons=(
            "research_watchlist_only",
            "unfilled_orders_withdraw_by_14_56_45",
            "orders_cannot_be_cancelled_after_14_57",
            "t_plus_one_exit_is_preset_09_35_to_09_45",
            "limit_down_exit_may_be_impossible",
        ),
        total_capital_fraction=total_capital_fraction,
    )
