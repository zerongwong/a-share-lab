"""A conditional recommendation must archive sufficient replay evidence."""

import json
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from ashare_lab.services.build_continuous_digest import _entry
from ashare_lab.services.build_midterm_portfolio import (
    ConditionalEntryPlan,
    ConditionalEntryPlanKind,
)


def _plan(kind):
    return ConditionalEntryPlan(
        kind=kind,
        data_cutoff=pd.Timestamp("2026-10-09"),
        horizon="continuous_weekly_breakout_daily_execution_v4",
        sessions=20,
        price_low=9.95 if kind is ConditionalEntryPlanKind.HEALTHY_PULLBACK else None,
        price_high=10.05 if kind is ConditionalEntryPlanKind.HEALTHY_PULLBACK else None,
        trigger_price=10.0 if kind is not ConditionalEntryPlanKind.HEALTHY_PULLBACK else None,
        confirmation_rule="按已归档条件确认",
        structure_timeframe="weekly",
        structure_cutoff=pd.Timestamp("2026-10-09"),
        primary_structure_timeframe="weekly",
        primary_structure_cutoff=pd.Timestamp("2026-10-09"),
        price_source_timeframe="daily",
        invalidation_price=9.5,
        maximum_entry_price=10.05,
        initial_risk_qualified=True,
        confirmation_activity_metric="amount_cny",
        confirmation_activity_min=120_000_000.0,
        initial_protection_support=9.6,
        initial_protection_atr=0.2,
        initial_protection_evidence_date="2026-09-25",
        initial_protection_atr_cutoff="2026-10-09",
        initial_protection_method_version="magee-inspired-pivot-trailing-v0.2.0",
    )


@pytest.mark.parametrize("kind", tuple(ConditionalEntryPlanKind))
def test_entry_keeps_structured_condition_and_all_risk_evidence(kind):
    plan = _plan(kind)
    entry = _entry("600001", "示例", 0.1, plan, expected_cutoff=date(2026, 10, 9))
    rule = json.loads(json.dumps(entry["entry_rule"], ensure_ascii=False, allow_nan=False))

    assert rule["schema_version"] == "recommendation-entry-rule-v1"
    assert rule["kind"] == kind.value
    assert rule["data_cutoff"] == rule["primary_structure_cutoff"] == "2026-10-09"
    assert rule["structure_timeframe"] == "weekly"
    assert rule["price_source_timeframe"] == "daily"
    assert rule["price_low"] == plan.price_low
    assert rule["price_high"] == plan.price_high
    assert rule["trigger_price"] == plan.trigger_price
    assert rule["confirmation_activity_metric"] == "amount_cny"
    assert rule["confirmation_activity_min"] == 120_000_000.0
    assert rule["structural_invalidation_price"] == 9.5
    assert rule["maximum_entry_price"] == 10.05
    assert rule["initial_protection_support"] == 9.6
    assert rule["initial_protection_atr"] == 0.2
    assert rule["initial_protection_evidence_date"] == "2026-09-25"
    assert rule["method_version"] == plan.method_version
    assert rule["execution_policy"]["holding_deadline"] is None
    assert rule["execution_policy"]["cost_stop_loss_fraction"] == 0.08
    assert rule["execution_policy"]["simulated_fill_requires_verified_execution_evidence"]
    assert entry["entry_qualified"] is True


def test_optional_unknown_evidence_remains_unknown_and_rule_is_detached():
    plan = replace(_plan(ConditionalEntryPlanKind.RECLAIM), initial_protection_atr=float("nan"))
    entry = _entry("600001", "示例", 0.1, plan, expected_cutoff=date(2026, 10, 9))
    assert entry["entry_rule"]["initial_protection_atr"] is None
    json.dumps(entry, allow_nan=False)
    entry["entry_rule"]["maximum_entry_price"] = 99.0
    assert plan.maximum_entry_price == 10.05
