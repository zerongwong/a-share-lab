from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_lab.analytics.medium_term_stage import MediumTermStage, assess_medium_term_stage


def _series(parts: list[np.ndarray]) -> pd.Series:
    return pd.Series(np.concatenate(parts), dtype=float)


def test_orderly_multihorizon_uptrend_is_preferred() -> None:
    close = pd.Series(np.linspace(20.0, 29.0, 260), dtype=float)
    result = assess_medium_term_stage(close)
    assert result.stage == MediumTermStage.ORDERLY_UPTREND
    assert result.quality_score == 1.0
    assert not result.hard_freeze_new_entry


def test_five_session_vertical_move_is_frozen() -> None:
    close = _series([np.linspace(10.0, 12.0, 255), np.linspace(12.2, 16.5, 5)])
    result = assess_medium_term_stage(close)
    assert result.stage == MediumTermStage.PARABOLIC
    assert result.hard_freeze_new_entry
    assert "five_session_vertical_acceleration" in result.reasons


def test_long_rise_with_current_extension_is_not_mistaken_for_low_risk() -> None:
    close = _series(
        [np.linspace(8.0, 15.0, 140), np.linspace(15.0, 24.0, 100), np.linspace(24.0, 33.0, 20)]
    )
    result = assess_medium_term_stage(close)
    assert result.stage in {MediumTermStage.EXTENDED, MediumTermStage.PARABOLIC}
    assert result.quality_score <= 0.20


def test_a_long_rise_can_requalify_after_a_stable_base() -> None:
    close = _series([np.linspace(8.0, 24.0, 160), np.linspace(23.0, 25.0, 100)])
    result = assess_medium_term_stage(close)
    assert result.stage != MediumTermStage.PARABOLIC
    assert not result.hard_freeze_new_entry


def test_insufficient_history_never_invents_a_stage_score() -> None:
    result = assess_medium_term_stage(pd.Series(np.linspace(10.0, 12.0, 60)))
    assert result.stage == MediumTermStage.INSUFFICIENT
    assert result.quality_score == 0.0
    assert not result.hard_freeze_new_entry
