"""Regression: continuous location preferences cannot re-rank legacy research."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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
