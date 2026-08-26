"""Domain models for safe, long-only A-share portfolio allocation.

All ratios use decimal fractions: ``0.20`` means 20% of net asset value.
The models deliberately separate cash, securities exposure, margin debt and
unused credit.  Unused broker credit is not cash and is never invested
automatically.
"""

from __future__ import annotations

from enum import StrEnum
from math import isclose

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RATIO_TOLERANCE = 1e-9
LIVERMORE_RATIO = (3, 2, 2, 1)


class RiskProfileName(StrEnum):
    """Supported portfolio risk profiles."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class PositionRole(StrEnum):
    """The intended role of each slot in the 3:2:2:1 portfolio."""

    CORE = "core"
    SECONDARY_CORE = "secondary_core"
    DIVERSIFIER = "diversifier"
    SATELLITE = "satellite"


class ScoringWeights(BaseModel):
    """Weights for ranking candidates; the fields must sum to exactly one.

    Fundamental, news and market/sector inputs are optional at run time.  The
    portfolio service only activates one of those weights when the complete
    eligible universe has a valid point-in-time value; otherwise it reports the
    missing coverage and redistributes that weight across genuinely available
    factors.  ``quality_liquidity`` and ``trend_catalyst`` remain as deprecated
    compatibility fields for archived configurations.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    drawdown_cvar: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_adjusted_return: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_liquidity: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_return: float = Field(default=0.0, ge=0.0, le=1.0)
    diversification: float = Field(default=0.0, ge=0.0, le=1.0)
    trend_catalyst: float = Field(default=0.0, ge=0.0, le=1.0)
    fundamentals: float = Field(default=0.0, ge=0.0, le=1.0)
    liquidity: float = Field(default=0.0, ge=0.0, le=1.0)
    trend: float = Field(default=0.0, ge=0.0, le=1.0)
    news: float = Field(default=0.0, ge=0.0, le=1.0)
    market_sector: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScoringWeights:
        total = sum(
            (
                self.drawdown_cvar,
                self.risk_adjusted_return,
                self.quality_liquidity,
                self.expected_return,
                self.diversification,
                self.trend_catalyst,
                self.fundamentals,
                self.liquidity,
                self.trend,
                self.news,
                self.market_sector,
            )
        )
        if not isclose(total, 1.0, abs_tol=RATIO_TOLERANCE):
            raise ValueError(f"scoring weights must sum to 1.0, got {total:.12f}")
        return self


class RiskControls(BaseModel):
    """Risk limits are control thresholds, not promises about future losses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_annual_volatility_max: float = Field(gt=0.0, le=1.0)
    drawdown_alert: float = Field(gt=0.0, le=1.0)
    drawdown_de_risk: float = Field(gt=0.0, le=1.0)
    per_position_loss_cap: float = Field(gt=0.0, le=0.05)
    sector_exposure_cap: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def de_risk_follows_alert(self) -> RiskControls:
        if self.drawdown_de_risk <= self.drawdown_alert:
            raise ValueError("drawdown_de_risk must be greater than drawdown_alert")
        return self


class FinancingPolicy(BaseModel):
    """Safety policy for legal broker margin financing.

    Financing starts at zero and can never be activated automatically.  This
    module does not support unlicensed/off-exchange leverage or averaging down.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    default_margin_debt_ratio: float = Field(default=0.0, ge=0.0, le=0.10)
    max_margin_debt_ratio: float = Field(default=0.0, ge=0.0, le=0.10)
    max_gross_exposure: float = Field(default=1.0, gt=0.0, le=1.10)
    automatic_activation: bool = False
    allow_averaging_down: bool = False
    licensed_broker_only: bool = True

    @model_validator(mode="after")
    def financing_is_opt_in_and_bounded(self) -> FinancingPolicy:
        if not isclose(self.default_margin_debt_ratio, 0.0, abs_tol=RATIO_TOLERANCE):
            raise ValueError("margin financing must default to zero")
        if not self.enabled and not isclose(
            self.max_margin_debt_ratio, 0.0, abs_tol=RATIO_TOLERANCE
        ):
            raise ValueError("disabled financing must have a zero debt limit")
        if self.default_margin_debt_ratio > self.max_margin_debt_ratio:
            raise ValueError("default margin debt cannot exceed the configured limit")
        if self.automatic_activation:
            raise ValueError("automatic margin financing is not permitted")
        if self.allow_averaging_down:
            raise ValueError("using margin to average down is not permitted")
        if self.enabled and not self.licensed_broker_only:
            raise ValueError("only licensed broker margin financing is supported")
        return self


class RiskProfile(BaseModel):
    """A validated configuration for one of the three portfolio profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: RiskProfileName
    display_name: str = Field(min_length=1)
    stock_exposure: float = Field(gt=0.0, le=1.0)
    base_cash_ratio: float = Field(ge=0.0, lt=1.0)
    ratio_units: tuple[int, int, int, int] = LIVERMORE_RATIO
    position_weights: tuple[float, float, float, float]
    risk: RiskControls
    scoring: ScoringWeights
    financing: FinancingPolicy

    @field_validator("ratio_units")
    @classmethod
    def ratio_must_be_three_two_two_one(
        cls, value: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if value != LIVERMORE_RATIO:
            raise ValueError("the four stock slots must use the 3:2:2:1 ratio")
        return value

    @model_validator(mode="after")
    def allocation_is_consistent(self) -> RiskProfile:
        if not isclose(
            self.stock_exposure + self.base_cash_ratio,
            1.0,
            abs_tol=RATIO_TOLERANCE,
        ):
            raise ValueError("base stock exposure plus base cash must equal 1.0")

        expected = tuple(
            self.stock_exposure * units / sum(self.ratio_units) for units in self.ratio_units
        )
        if any(
            not isclose(actual, wanted, abs_tol=RATIO_TOLERANCE)
            for actual, wanted in zip(self.position_weights, expected, strict=True)
        ):
            raise ValueError("position_weights must allocate stock exposure at 3:2:2:1")
        if not isclose(sum(self.position_weights), self.stock_exposure, abs_tol=RATIO_TOLERANCE):
            raise ValueError("position_weights must sum to stock_exposure")
        if max(self.position_weights) > self.risk.sector_exposure_cap:
            raise ValueError("sector cap cannot be below the largest individual position")
        if self.financing.max_gross_exposure < self.stock_exposure:
            raise ValueError("max gross exposure cannot be below base stock exposure")
        return self


class StockAllocation(BaseModel):
    """One stock slot in an allocated portfolio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1, max_length=32)
    weight: float = Field(gt=0.0, le=1.10)
    ratio_units: int = Field(gt=0, le=3)
    role: PositionRole

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker cannot be blank")
        return normalized


class PortfolioAllocation(BaseModel):
    """A long-only allocation expressed as fractions of net asset value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: RiskProfileName
    positions: tuple[StockAllocation, StockAllocation, StockAllocation, StockAllocation]
    cash_ratio: float = Field(ge=0.0, le=1.0)
    securities_exposure: float = Field(gt=0.0, le=1.10)
    margin_debt_ratio: float = Field(default=0.0, ge=0.0, le=0.10)
    unused_credit_ratio: float = Field(default=0.0, ge=0.0, le=0.10)
    gross_exposure: float = Field(gt=0.0, le=1.10)
    financing_interest_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    automatic_financing: bool = False

    @model_validator(mode="after")
    def validate_balance_sheet_and_ratio(self) -> PortfolioAllocation:
        tickers = [position.ticker for position in self.positions]
        if len(set(tickers)) != 4:
            raise ValueError("exactly four unique stock tickers are required")

        units = tuple(position.ratio_units for position in self.positions)
        if units != LIVERMORE_RATIO:
            raise ValueError("allocated positions must preserve the 3:2:2:1 order")

        position_total = sum(position.weight for position in self.positions)
        if not isclose(position_total, self.securities_exposure, abs_tol=RATIO_TOLERANCE):
            raise ValueError("position weights must sum to securities_exposure")

        expected_weights = tuple(
            self.securities_exposure * ratio / sum(LIVERMORE_RATIO) for ratio in LIVERMORE_RATIO
        )
        if any(
            not isclose(position.weight, expected, abs_tol=RATIO_TOLERANCE)
            for position, expected in zip(self.positions, expected_weights, strict=True)
        ):
            raise ValueError("position weights must preserve the 3:2:2:1 ratio")

        if not isclose(self.gross_exposure, self.securities_exposure, abs_tol=RATIO_TOLERANCE):
            raise ValueError("gross exposure equals securities exposure for long-only portfolios")

        net_assets = self.securities_exposure + self.cash_ratio - self.margin_debt_ratio
        if not isclose(net_assets, 1.0, abs_tol=RATIO_TOLERANCE):
            raise ValueError("securities + cash - margin debt must equal 1.0 of net asset value")
        if self.automatic_financing:
            raise ValueError("automatic margin financing is not permitted")
        if self.margin_debt_ratio > 0.0 and self.financing_interest_rate <= 0.0:
            raise ValueError("a positive financing rate is required when margin debt is used")
        return self
