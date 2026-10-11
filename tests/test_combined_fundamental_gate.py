"""Versioned combined risk receipts must not disable legacy/known risk checks."""

import pytest

from ashare_lab.services.build_midterm_portfolio import _metadata_hard_gate, _unknown_evidence


def _item(**overrides):
    return {
        "name": "合成研究样本", "industry": "制造业", "is_st": False,
        "is_delisting": False, "is_buyable_at_cutoff": True,
        "fundamental_gate": "pass", "announcement_gate": "unknown",
        "candidate_evidence_method_version": "candidate-evidence-v2.0.0",
        "fundamental_risk_gate": "pass", **overrides,
    }


def test_combined_current_review_replaces_separate_legacy_approval_requirement():
    assert _unknown_evidence(_item()) == ()
    assert _unknown_evidence(_item(fundamental_risk_gate="unknown")) == (
        "fundamental_risk_evidence_unknown",
    )


@pytest.mark.parametrize("version", [None, "candidate-evidence-v1.2.0", "unexpected"])
def test_unversioned_or_legacy_combined_pass_cannot_bypass_missing_announcement_evidence(version):
    assert "official_announcement_evidence_unknown" in _unknown_evidence(
        _item(candidate_evidence_method_version=version)
    )


@pytest.mark.parametrize("change,reason", [
    ({"fundamental_gate": "veto"}, "fundamental_veto"),
    ({"announcement_gate": "veto"}, "official_announcement_veto"),
    ({"announcement_veto": True}, "official_announcement_veto"),
    ({"fundamental_risk_gate": "veto"}, "fundamental_risk_veto"),
    ({"is_buyable_at_cutoff": False}, "formation_unbuyable_at_cutoff"),
])
def test_combined_pass_never_overrides_known_veto_or_execution(change, reason):
    assert reason in _metadata_hard_gate("600001.SH", _item(**change), minimum_balance_sheet_strength=0)


def test_combined_pass_still_requires_execution():
    assert _unknown_evidence(_item(is_buyable_at_cutoff=None)) == (
        "formation_execution_evidence_unknown",
    )


def test_inconsistent_combined_pass_cannot_override_unknown_financial_quality():
    assert _unknown_evidence(_item(fundamental_gate="unknown")) == (
        "fundamental_risk_evidence_unknown",
    )
