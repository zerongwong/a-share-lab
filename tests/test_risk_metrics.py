import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.risk_metrics import max_drawdown, risk_metrics


def test_first_return_loss_is_measured_from_initial_investment() -> None:
    returns = pd.Series([-0.10] + [0.0] * 19)
    assert max_drawdown(returns) == pytest.approx(-0.10)
    assert risk_metrics(returns)["max_drawdown"] == pytest.approx(-0.10)


def test_drawdown_tracks_later_high_water_mark_after_recovery() -> None:
    assert max_drawdown(pd.Series([-0.10, 0.50, -0.20])) == pytest.approx(-0.20)


def test_sortino_uses_downside_magnitude_not_variance_between_losses() -> None:
    returns = pd.Series([0.02, -0.01] * 10)
    expected = 0.005 / np.sqrt(0.5 * 0.01**2) * np.sqrt(252)
    assert risk_metrics(returns)["sortino"] == pytest.approx(expected)


def test_sortino_without_any_downside_remains_unavailable() -> None:
    assert np.isnan(risk_metrics(pd.Series([0.01] * 20))["sortino"])
