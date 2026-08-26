"""Deterministic 3:2:2:1 allocation for the three A-share risk profiles."""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose

from ashare_lab.domain.models import (
    LIVERMORE_RATIO,
    FinancingPolicy,
    PortfolioAllocation,
    PositionRole,
    RiskControls,
    RiskProfile,
    RiskProfileName,
    ScoringWeights,
    StockAllocation,
)

RATIO_TOTAL = sum(LIVERMORE_RATIO)
POSITION_ROLES = (
    PositionRole.CORE,
    PositionRole.SECONDARY_CORE,
    PositionRole.DIVERSIFIER,
    PositionRole.SATELLITE,
)


def _weights(stock_exposure: float) -> tuple[float, float, float, float]:
    if not 0.0 < stock_exposure <= 1.10:
        raise ValueError("stock exposure must be in the interval (0, 1.10]")
    return tuple(  # type: ignore[return-value]
        round(stock_exposure * units / RATIO_TOTAL, 12) for units in LIVERMORE_RATIO
    )


DEFAULT_PROFILES: dict[RiskProfileName, RiskProfile] = {
    RiskProfileName.CONSERVATIVE: RiskProfile(
        name=RiskProfileName.CONSERVATIVE,
        display_name="稳健型",
        stock_exposure=0.60,
        base_cash_ratio=0.40,
        position_weights=_weights(0.60),
        risk=RiskControls(
            target_annual_volatility_max=0.12,
            drawdown_alert=0.08,
            drawdown_de_risk=0.12,
            per_position_loss_cap=0.004,
            sector_exposure_cap=0.30,
        ),
        scoring=ScoringWeights(
            drawdown_cvar=0.35,
            risk_adjusted_return=0.25,
            expected_return=0.08,
            diversification=0.05,
            fundamentals=0.14,
            liquidity=0.06,
            trend=0.03,
            news=0.01,
            market_sector=0.03,
        ),
        financing=FinancingPolicy(
            enabled=False,
            max_margin_debt_ratio=0.0,
            max_gross_exposure=0.60,
        ),
    ),
    RiskProfileName.BALANCED: RiskProfile(
        name=RiskProfileName.BALANCED,
        display_name="平衡型",
        stock_exposure=0.80,
        base_cash_ratio=0.20,
        position_weights=_weights(0.80),
        risk=RiskControls(
            target_annual_volatility_max=0.18,
            drawdown_alert=0.12,
            drawdown_de_risk=0.18,
            per_position_loss_cap=0.006,
            sector_exposure_cap=0.40,
        ),
        scoring=ScoringWeights(
            drawdown_cvar=0.25,
            risk_adjusted_return=0.20,
            expected_return=0.20,
            diversification=0.10,
            fundamentals=0.12,
            liquidity=0.05,
            trend=0.04,
            news=0.02,
            market_sector=0.02,
        ),
        financing=FinancingPolicy(
            enabled=False,
            max_margin_debt_ratio=0.0,
            max_gross_exposure=0.80,
        ),
    ),
    RiskProfileName.AGGRESSIVE: RiskProfile(
        name=RiskProfileName.AGGRESSIVE,
        display_name="激进型",
        stock_exposure=0.90,
        base_cash_ratio=0.10,
        position_weights=_weights(0.90),
        risk=RiskControls(
            target_annual_volatility_max=0.25,
            drawdown_alert=0.18,
            drawdown_de_risk=0.25,
            per_position_loss_cap=0.008,
            sector_exposure_cap=0.50,
        ),
        scoring=ScoringWeights(
            drawdown_cvar=0.15,
            risk_adjusted_return=0.15,
            expected_return=0.30,
            diversification=0.08,
            fundamentals=0.08,
            liquidity=0.03,
            trend=0.10,
            news=0.05,
            market_sector=0.06,
        ),
        financing=FinancingPolicy(
            enabled=True,
            max_margin_debt_ratio=0.10,
            max_gross_exposure=1.10,
        ),
    ),
}


def get_default_profile(profile: RiskProfileName | str) -> RiskProfile:
    """Return one immutable validated default profile."""

    try:
        profile_name = RiskProfileName(profile)
    except ValueError as exc:
        choices = ", ".join(item.value for item in RiskProfileName)
        raise ValueError(f"unknown risk profile {profile!r}; choose one of: {choices}") from exc
    return DEFAULT_PROFILES[profile_name]


def allocate_four_stocks(
    tickers: Sequence[str],
    profile: RiskProfileName | str | RiskProfile,
    *,
    margin_debt_ratio: float = 0.0,
    financing_interest_rate: float = 0.0,
) -> PortfolioAllocation:
    """Allocate four unique stocks and keep cash and margin debt separate.

    Margin is opt-in through this explicit function argument and is added to
    stock exposure pro-rata.  Base cash remains untouched.  Consequently the
    accounting identity is always ``stocks + cash - debt == NAV``.
    """

    selected_profile = profile if isinstance(profile, RiskProfile) else get_default_profile(profile)
    normalized = tuple(str(ticker).strip().upper() for ticker in tickers)

    if len(normalized) != 4:
        raise ValueError("exactly four stock tickers are required")
    if any(not ticker for ticker in normalized):
        raise ValueError("stock tickers cannot be blank")
    if len(set(normalized)) != 4:
        raise ValueError("stock tickers must be unique")
    if margin_debt_ratio < 0.0:
        raise ValueError("margin debt cannot be negative")

    policy = selected_profile.financing
    if margin_debt_ratio > 0.0 and not policy.enabled:
        raise ValueError(f"margin financing is disabled for {selected_profile.name.value}")
    if margin_debt_ratio > policy.max_margin_debt_ratio and not isclose(
        margin_debt_ratio, policy.max_margin_debt_ratio, abs_tol=1e-9
    ):
        raise ValueError("requested margin debt exceeds the profile limit")
    if margin_debt_ratio > 0.0 and financing_interest_rate <= 0.0:
        raise ValueError("a positive annual financing rate is required for margin debt")

    securities_exposure = selected_profile.stock_exposure + margin_debt_ratio
    if securities_exposure > policy.max_gross_exposure and not isclose(
        securities_exposure, policy.max_gross_exposure, abs_tol=1e-9
    ):
        raise ValueError("requested financing would exceed maximum gross exposure")

    weights = _weights(securities_exposure)
    positions = tuple(
        StockAllocation(
            ticker=ticker,
            weight=weight,
            ratio_units=units,
            role=role,
        )
        for ticker, weight, units, role in zip(
            normalized, weights, LIVERMORE_RATIO, POSITION_ROLES, strict=True
        )
    )

    return PortfolioAllocation(
        profile=selected_profile.name,
        positions=positions,  # type: ignore[arg-type]
        cash_ratio=selected_profile.base_cash_ratio,
        securities_exposure=securities_exposure,
        margin_debt_ratio=margin_debt_ratio,
        unused_credit_ratio=policy.max_margin_debt_ratio - margin_debt_ratio,
        gross_exposure=securities_exposure,
        financing_interest_rate=financing_interest_rate,
        automatic_financing=False,
    )


def risk_capped_weight(
    template_weight: float, loss_cap: float, invalidation_distance: float
) -> float:
    """Cap a template weight so its planned loss does not exceed the risk budget."""

    if not 0.0 < template_weight <= 1.10:
        raise ValueError("template weight must be in the interval (0, 1.10]")
    if not 0.0 < loss_cap <= 0.05:
        raise ValueError("loss cap must be in the interval (0, 0.05]")
    if not 0.0 < invalidation_distance <= 1.0:
        raise ValueError("invalidation distance must be in the interval (0, 1]")
    return min(template_weight, loss_cap / invalidation_distance)
