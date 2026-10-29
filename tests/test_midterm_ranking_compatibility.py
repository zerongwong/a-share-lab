"""Regression: continuous location preferences cannot re-rank legacy research."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ashare_lab.analytics.multi_timeframe import ExecutionState
from ashare_lab.services.build_midterm_portfolio import _rank_candidates


def _same_market_evidence(symbol, stage_score):
    return SimpleNamespace(
        symbol=symbol,
        name=symbol,
        industry=symbol,
        returns=pd.Series(np.tile([0.003, -0.001], 100)),
        timeframe=SimpleNamespace(score=0.8),
        continuous_stage_rank_score=stage_score,
        entry=None,
        evidence_unknown=(),
        risk_history_available=True,
        risk_history_reasons=(),
        risk_history_available_returns=200,
        risk_history_required_returns=160,
    )


def test_legacy_keeps_frozen_score_and_ignores_continuous_location_bonus():
    rows = [_same_market_evidence("000001", 0.0), _same_market_evidence("000002", 1.0)]
    ranked = _rank_candidates(rows, 4, {"000001": 0.7, "000002": 0.7})
    assert [row.symbol for row in ranked] == ["000001", "000002"]
    assert [row.signal_score for row in ranked] == pytest.approx([0.6725, 0.6725])


def test_continuous_location_bonus_changes_only_continuous_ranking():
    rows = [_same_market_evidence("000001", 0.0), _same_market_evidence("000002", 1.0)]
    ranked = _rank_candidates(
        rows, 4, {"000001": 0.7, "000002": 0.7}, continuous_entry_policy=True
    )
    assert [row.symbol for row in ranked] == ["000002", "000001"]
    assert ranked[0].signal_score > ranked[1].signal_score


def test_stabilized_retest_is_bounded_preference_not_breakout_exclusion():
    breakout = _same_market_evidence("000001", 1.0)
    retest = _same_market_evidence("000002", 1.0)
    for row, state in ((breakout, ExecutionState.READY_BREAKOUT),
                       (retest, ExecutionState.READY_PULLBACK)):
        row.timeframe.execution = SimpleNamespace(state=state)
        row.timeframe.execution_ready = True
    rows = [breakout, retest]
    strengths = {row.symbol: 0.7 for row in rows}
    ranked = _rank_candidates(rows, 4, strengths, continuous_entry_policy=True)
    assert [row.symbol for row in ranked] == ["000002", "000001"]
    assert ranked[0].signal_score - ranked[1].signal_score == pytest.approx(0.05)
    legacy = _rank_candidates(rows, 4, strengths)
    assert legacy[0].signal_score == legacy[1].signal_score


def test_unready_retest_receives_no_preference():
    rows = [_same_market_evidence("000001", 1.0), _same_market_evidence("000002", 1.0)]
    rows[1].timeframe.execution = SimpleNamespace(state=ExecutionState.READY_PULLBACK)
    rows[1].timeframe.execution_ready = False
    ranked = _rank_candidates(
        rows, 4, {row.symbol: 0.7 for row in rows}, continuous_entry_policy=True,
    )
    assert ranked[0].signal_score == ranked[1].signal_score
