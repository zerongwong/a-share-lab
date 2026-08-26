from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.return_ambition import (
    ANNUAL_RETURN_AMBITION_PCTS,
    ReturnAmbitionStatus,
    assess_return_ambition,
    horizon_return_hurdle,
    validate_annual_return_ambition,
    validate_holding_weeks,
)


def test_allowed_ambitions_are_20_to_200_in_steps_of_20() -> None:
    assert tuple(range(20, 201, 20)) == ANNUAL_RETURN_AMBITION_PCTS
    assert validate_annual_return_ambition(20) == 20
    assert validate_annual_return_ambition(200) == 200

    with pytest.raises(ValueError):
        validate_annual_return_ambition(30)
    with pytest.raises(ValueError):
        validate_annual_return_ambition(220)
    with pytest.raises(TypeError):
        validate_annual_return_ambition(20.0)  # type: ignore[arg-type]


def test_horizon_hurdle_uses_compounding_and_validates_one_to_fifty_two_weeks() -> None:
    hurdle = horizon_return_hurdle(200, 12)
    assert hurdle == pytest.approx(3.0 ** (12.0 / 52.0) - 1.0)
    assert hurdle == pytest.approx(0.2886, abs=0.0001)
    assert validate_holding_weeks(1) == 1
    assert validate_holding_weeks(52) == 52

    with pytest.raises(ValueError):
        validate_holding_weeks(0)
    with pytest.raises(ValueError):
        validate_holding_weeks(53)


def test_supported_and_unsupported_are_historical_labels_not_promises() -> None:
    steady_positive = pd.Series(np.full(320, 0.001))
    supported = assess_return_ambition(
        steady_positive,
        annual_return_ambition_pct=20,
        holding_weeks=4,
    )
    unsupported = assess_return_ambition(
        steady_positive,
        annual_return_ambition_pct=200,
        holding_weeks=4,
    )

    assert supported.status == ReturnAmbitionStatus.HISTORICALLY_SUPPORTED
    assert unsupported.status == ReturnAmbitionStatus.UNSUPPORTED
    assert supported.horizon_return_hurdle < supported.return_p50
    assert unsupported.horizon_return_hurdle > unsupported.return_p90
    assert not supported.is_out_of_sample
    assert not supported.is_forecast_probability
    assert not supported.is_promise
    assert "非walk-forward" in supported.method


def test_insufficient_evidence_hides_hit_rate_and_return_quantiles() -> None:
    assessment = assess_return_ambition(
        pd.Series(np.full(40, 0.001)),
        annual_return_ambition_pct=40,
        holding_weeks=12,
        minimum_samples=30,
    )

    assert assessment.status == ReturnAmbitionStatus.INSUFFICIENT_EVIDENCE
    assert assessment.sample_n == 0
    assert assessment.historical_hit_rate is None
    assert assessment.hit_rate_interval is None
    assert assessment.return_p10 is None
    assert assessment.return_p50 is None
    assert assessment.return_p90 is None


def test_stretch_status_for_a_target_between_historical_median_and_p90() -> None:
    returns = pd.Series(np.concatenate((np.full(100, 0.002), np.zeros(220))))
    assessment = assess_return_ambition(
        returns,
        annual_return_ambition_pct=20,
        holding_weeks=4,
    )

    assert assessment.return_p50 < assessment.horizon_return_hurdle
    assert assessment.horizon_return_hurdle <= assessment.return_p90
    assert assessment.status == ReturnAmbitionStatus.STRETCH
    assert assessment.hit_rate_interval is not None
