from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.adaptive_portfolio import AdaptiveCandidate, AdaptiveRiskBudget
from ashare_lab.analytics.continuous_portfolio import (
    ContinuousPortfolioStatus,
    select_continuous_replacement,
)


def _candidate(symbol, seed, *, mean=0.0005, industry=None, returns=None):
    if returns is None:
        returns = pd.Series(
            np.random.default_rng(seed).normal(mean, 0.01, 200),
            index=pd.bdate_range("2025-01-01", periods=200),
        )
    return AdaptiveCandidate(symbol, industry or symbol, 0.8, returns)


def _retained():
    return tuple(_candidate(f"OLD{i}", i) for i in range(3))


def _select(replacements=(), **kwargs):
    defaults = {
        "retained": _retained(),
        "retained_account_weights": {"OLD0": 0.137, "OLD1": 0.163, "OLD2": 0.15},
        "replacements": replacements,
        "cash_weight": 0.55,
        "budget": AdaptiveRiskBudget(),
    }
    return select_continuous_replacement(**(defaults | kwargs))


def test_locks_exact_drifted_account_weights_and_evaluates_every_new_account_grid():
    weights = {"OLD0": 0.137, "OLD1": 0.163, "OLD2": 0.15}
    decision = _select([_candidate("NEW", 9, mean=0.004)], retained_account_weights=weights)
    assert decision.status is ContinuousPortfolioStatus.SELECTED
    assert decision.selected_symbol == "NEW"
    assert decision.new_account_weight in (0.1, 0.2)
    assert {key: dict(decision.account_weights)[key] for key in weights} == weights
    assert weights == {"OLD0": 0.137, "OLD1": 0.163, "OLD2": 0.15}
    assert sum(dict(decision.account_weights).values()) + decision.cash_weight == pytest.approx(1)
    assert decision.evaluated_count == 2  # Cash + the 10% grid within the four-name cap.
    assert any(
        row.new_account_weight == 0.2 and "holding_count_stock_exposure" in row.reasons
        for row in decision.candidate_rejections
    )
    assert not decision.holding_membership_changed
    assert not decision.auto_order_allowed
    assert not decision.metrics.is_out_of_sample


def test_return_proxy_uses_fixed_shares_and_cash_not_daily_rebalanced_returns():
    newcomer = _candidate("NEW", 9, mean=0.004)
    decision = _select([newcomer])
    weights = dict(decision.account_weights)
    rows = (*_retained(), newcomer)
    returns = np.column_stack([row.returns.iloc[-160:] for row in rows])
    blocks = returns.reshape(8, 20, 4)
    expected = (
        (np.prod(1 + blocks, axis=1) - 1) @ np.array([weights[row.symbol] for row in rows])
    ).mean()
    assert decision.metrics.holding_period_return_mean == pytest.approx(expected)
    assert decision.metrics.observation_count == 160


@pytest.mark.parametrize("cash", [0.54, 0.56, -0.1, 1.1, float("nan"), True])
def test_rejects_invalid_or_nonconserving_cash(cash):
    with pytest.raises(ValueError):
        _select(cash_weight=cash)


def test_missing_retained_history_blocks_all_new_purchases():
    retained = list(_retained())
    retained[0] = replace(retained[0], returns=retained[0].returns.iloc[-20:])
    decision = _select([_candidate("NEW", 9, mean=0.004)], retained=retained)
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert decision.metrics is None
    assert decision.selected_symbol is None
    assert decision.evaluated_count == 0
    assert decision.reasons == ("retained_risk_evidence_unavailable",)


def test_retained_risk_breach_cannot_be_rescued_by_a_new_stock():
    decision = _select(
        [_candidate("NEW", 9, mean=0.004)],
        budget=replace(AdaptiveRiskBudget(), max_annual_downside_volatility=0.0001),
    )
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert "annual_downside_volatility" in decision.reasons
    assert decision.selected_symbol is None
    assert decision.evaluated_count == 1
    assert decision.cash_weight == 0.55


def test_overweight_old_stock_requires_review_without_quantizing_it():
    decision = _select(
        [_candidate("NEW", 9, mean=0.004)],
        retained_account_weights={"OLD0": 0.201, "OLD1": 0.10, "OLD2": 0.15},
        cash_weight=0.549,
    )
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert "maximum_20pct_account_weight" in decision.reasons
    assert dict(decision.account_weights)["OLD0"] == 0.201


def test_no_same_industry_replacement_and_no_pyramiding_retained_position():
    decision = _select(
        [
            _candidate("NEW", 9, mean=0.004, industry="OLD0"),
            _retained()[0],
        ]
    )
    assert decision.status is ContinuousPortfolioStatus.HOLD_CASH
    assert decision.evaluated_count == 1
    assert {reason for row in decision.candidate_rejections for reason in row.reasons} == {
        "one_stock_per_industry",
        "retained_position_not_an_addition_candidate",
    }


def test_existing_same_industry_requires_review():
    retained = list(_retained())
    retained[1] = replace(retained[1], industry="OLD0")
    decision = _select(retained=retained)
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert "one_stock_per_industry" in decision.reasons


def test_bad_candidate_does_not_suppress_valid_candidate():
    bad = _candidate("BAD", 4)
    values = bad.returns.copy()
    values.iloc[-10] = np.nan
    bad = replace(bad, returns=values)
    decision = _select([bad, _candidate("GOOD", 9, mean=0.004)])
    assert decision.selected_symbol == "GOOD"
    assert any(
        row.symbol == "BAD" and "candidate_history_unavailable" in row.reasons
        for row in decision.candidate_rejections
    )
    assert decision.evaluated_count == 2


def test_stale_or_different_candidate_calendar_never_shrinks_baseline_window():
    bad = _candidate("STALE", 4)
    bad = replace(bad, returns=bad.returns.iloc[:-1])
    decision = _select([bad, _candidate("GOOD", 9, mean=0.004)])
    assert decision.selected_symbol == "GOOD"
    assert decision.baseline_metrics.observation_count == 160
    assert any(
        "candidate_history_not_aligned" in row.reasons for row in decision.candidate_rejections
    )


def test_highly_correlated_high_return_candidate_is_rejected_jointly():
    old = _retained()[0]
    correlated = _candidate("CORRELATED", 99, returns=old.returns + 0.005)
    independent = _candidate("INDEPENDENT", 9, mean=0.003)
    decision = _select([correlated, independent])
    assert decision.selected_symbol == "INDEPENDENT"
    assert any(
        row.symbol == "CORRELATED" and "down_period_correlation" in row.reasons
        for row in decision.candidate_rejections
    )


def test_all_valid_candidates_are_enumerated_not_top36_or_beam_pruned():
    candidates = [_candidate(f"NEW{i:02}", 9, mean=0.0001) for i in range(40)]
    candidates.append(_candidate("ZZZ_BEST", 9, mean=0.004))
    decision = _select(candidates)
    assert decision.candidate_count == 41
    assert decision.evaluated_count == 1 + 41
    assert decision.selected_symbol == "ZZZ_BEST"
    reversed_decision = _select(tuple(reversed(candidates)))
    assert decision == reversed_decision


def test_cash_option_wins_when_every_new_stock_worsens_return_proxy():
    decision = _select([_candidate("LOSER", 4, mean=-0.004)])
    assert decision.status is ContinuousPortfolioStatus.HOLD_CASH
    assert decision.selected_symbol is None
    assert decision.new_account_weight == 0
    assert decision.metrics == decision.baseline_metrics
    assert decision.cash_weight == 0.55


def test_only_confirmed_cash_and_cycle_exposure_are_available_for_purchase():
    decision = _select(
        [_candidate("NEW", 9, mean=0.004)],
        budget=replace(AdaptiveRiskBudget(), maximum_stock_exposure=0.55),
    )
    assert decision.new_account_weight == 0.1
    assert decision.evaluated_count == 2
    assert any("maximum_stock_exposure" in row.reasons for row in decision.candidate_rejections)


def test_zero_stock_cash_metrics_are_not_fabricated_equity_correlations():
    decision = select_continuous_replacement([], {}, [], cash_weight=1, budget=AdaptiveRiskBudget())
    assert decision.status is ContinuousPortfolioStatus.HOLD_CASH
    assert decision.metrics.cash_only
    assert decision.metrics.max_down_period_correlation is None
    assert decision.metrics.max_position_downside_risk_contribution is None
    assert not decision.metrics.correlation_applicable
    assert decision.metrics.observation_count == 0


@pytest.mark.parametrize("count", [1, 2])
def test_one_to_two_holdings_are_valid_transition_states_without_normalized_risk_rejection(
    count,
):
    retained = _retained()[:count]
    weights = {row.symbol: 0.15 for row in retained}
    decision = select_continuous_replacement(
        retained,
        weights,
        [_candidate("NEW", 9, mean=0.004)],
        cash_weight=1 - sum(weights.values()),
        budget=AdaptiveRiskBudget(),
    )
    assert decision.status is ContinuousPortfolioStatus.SELECTED
    assert "position_downside_risk_contribution" not in decision.reasons
    assert decision.baseline_metrics.max_position_downside_risk_contribution >= 1 / count
    assert decision.baseline_metrics.position_risk_contribution_applicable is False
    assert decision.new_account_weight == 0.1
    assert any(
        row.new_account_weight == 0.2 and "holding_count_stock_exposure" in row.reasons
        for row in decision.candidate_rejections
    )
    if count == 1:
        assert decision.baseline_metrics.max_down_period_correlation is None
        assert not decision.baseline_metrics.correlation_applicable


def test_initial_empty_portfolio_can_add_one_stock_without_normalized_risk_false_rejection():
    decision = select_continuous_replacement(
        [], {}, [_candidate("NEW", 9, mean=0.004)], cash_weight=1, budget=AdaptiveRiskBudget()
    )
    assert decision.status is ContinuousPortfolioStatus.SELECTED
    assert decision.selected_symbol == "NEW"
    assert decision.new_account_weight == 0.1
    assert decision.metrics.position_risk_contribution_applicable is False
    assert all(
        "position_downside_risk_contribution" not in row.reasons
        for row in decision.candidate_rejections
    )


@pytest.mark.parametrize("count", (3, 4, 5))
def test_three_to_five_holdings_keep_normalized_position_risk_constraint(count):
    retained = tuple(_candidate(f"OLD{i}", i) for i in range(count))
    decision = select_continuous_replacement(
        retained,
        {row.symbol: 0.1 for row in retained},
        [],
        cash_weight=1 - 0.1 * count,
        budget=replace(
            AdaptiveRiskBudget(),
            max_position_downside_risk_contribution=0.01,
        ),
    )
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert decision.metrics.position_risk_contribution_applicable is True
    assert "position_downside_risk_contribution" in decision.reasons


@pytest.mark.parametrize(
    ("count", "weight"),
    [
        (1, 0.151),
        (2, 0.151),
        (3, 0.151),
        (4, 0.151),
        (5, 0.151),
    ],
)
def test_count_specific_stock_exposure_ceiling_blocks_overdeployed_retained_state(
    count,
    weight,
):
    retained = tuple(_candidate(f"OLD{i}", i) for i in range(count))
    weights = {row.symbol: weight for row in retained}
    decision = select_continuous_replacement(
        retained,
        weights,
        [],
        cash_weight=1 - sum(weights.values()),
        budget=AdaptiveRiskBudget(),
    )
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert "holding_count_stock_exposure" in decision.reasons
    assert decision.evaluated_count == 0


def test_five_positions_have_no_slot_but_still_get_baseline_risk_review():
    retained = tuple(_candidate(f"OLD{i}", i) for i in range(5))
    decision = select_continuous_replacement(
        retained,
        {row.symbol: 0.1 for row in retained},
        [_candidate("NEW", 9, mean=0.004)],
        cash_weight=0.5,
        budget=AdaptiveRiskBudget(),
    )
    assert decision.status is ContinuousPortfolioStatus.HOLD_CASH
    assert decision.reasons == ("maximum_five_holdings_no_free_slot",)
    assert decision.evaluated_count == 1


@pytest.mark.parametrize("count", (6, 7, 8))
def test_more_than_five_positions_requires_review_before_market_comparison(count):
    retained = tuple(_candidate(f"OLD{i}", i) for i in range(count))
    decision = select_continuous_replacement(
        retained,
        {row.symbol: 0.09 for row in retained},
        [],
        cash_weight=1 - 0.09 * count,
        budget=AdaptiveRiskBudget(),
    )
    assert decision.status is ContinuousPortfolioStatus.REVIEW_REQUIRED
    assert "maximum_five_holdings" in decision.reasons
    assert decision.evaluated_count == 0


def test_other_proxy_horizon_is_not_silently_relabeled_twenty_sessions():
    with pytest.raises(ValueError, match="20-session"):
        _select(budget=replace(AdaptiveRiskBudget(), holding_period_sessions=60))


def test_undefined_correlation_and_all_positive_history_are_not_treated_as_zero_risk():
    candidate = _candidate("NO_DOWNS", 9)
    candidate = replace(candidate, returns=pd.Series(0.01, index=candidate.returns.index))
    decision = _select([candidate])
    assert decision.status is ContinuousPortfolioStatus.HOLD_CASH
    assert any(
        "candidate_joint_risk_unavailable" in row.reasons for row in decision.candidate_rejections
    )
    assert any("holding_count_stock_exposure" in row.reasons for row in decision.candidate_rejections)
