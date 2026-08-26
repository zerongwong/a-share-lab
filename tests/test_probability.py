import numpy as np
import pandas as pd

from ashare_lab.analytics.probability import empirical_scenario


def test_probability_reports_unavailable_instead_of_inventing_number():
    result = empirical_scenario(pd.Series([0.01] * 10), 5, minimum_samples=30)
    assert result["available"] is False
    assert "historical_positive_rate" not in result


def test_probability_is_empirical_and_has_sample_count():
    rng = np.random.default_rng(7)
    result = empirical_scenario(pd.Series(rng.normal(0.001, 0.02, 200)), 20)
    assert result["available"] is True
    assert result["sample_n"] == 181
    assert 0 <= result["historical_positive_rate"] <= 1
