from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ashare_lab.analytics.allocation import (  # noqa: E402
    DEFAULT_PROFILES,
    allocate_four_stocks,
    get_default_profile,
    risk_capped_weight,
)
from ashare_lab.domain.models import (  # noqa: E402
    FinancingPolicy,
    RiskControls,
    RiskProfile,
    RiskProfileName,
    ScoringWeights,
)

TICKERS = ("600000.SH", "000001.SZ", "600519.SH", "300750.SZ")


class AllocationTests(unittest.TestCase):
    def test_three_default_profiles_preserve_ratio_cash_and_zero_debt(self) -> None:
        expected = {
            RiskProfileName.CONSERVATIVE: ((0.225, 0.15, 0.15, 0.075), 0.40),
            RiskProfileName.BALANCED: ((0.30, 0.20, 0.20, 0.10), 0.20),
            RiskProfileName.AGGRESSIVE: ((0.3375, 0.225, 0.225, 0.1125), 0.10),
        }

        for profile_name, (weights, cash) in expected.items():
            with self.subTest(profile=profile_name):
                allocation = allocate_four_stocks(TICKERS, profile_name)
                self.assertEqual(
                    tuple(position.ratio_units for position in allocation.positions), (3, 2, 2, 1)
                )
                for actual, wanted in zip(
                    (position.weight for position in allocation.positions), weights, strict=True
                ):
                    self.assertAlmostEqual(actual, wanted)
                self.assertAlmostEqual(allocation.cash_ratio, cash)
                self.assertAlmostEqual(allocation.margin_debt_ratio, 0.0)
                self.assertAlmostEqual(
                    allocation.securities_exposure
                    + allocation.cash_ratio
                    - allocation.margin_debt_ratio,
                    1.0,
                )

    def test_tickers_are_normalized_and_must_be_unique(self) -> None:
        allocation = allocate_four_stocks(
            (" 600000.sh ", "000001.sz", "600519.sh", "300750.sz"), "balanced"
        )
        self.assertEqual(allocation.positions[0].ticker, "600000.SH")

        with self.assertRaisesRegex(ValueError, "unique"):
            allocate_four_stocks(("600000.SH", "600000.sh", "1", "2"), "balanced")
        with self.assertRaisesRegex(ValueError, "exactly four"):
            allocate_four_stocks(TICKERS[:3], "balanced")

    def test_financing_is_separate_opt_in_and_disabled_for_safer_profiles(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            allocate_four_stocks(
                TICKERS,
                "balanced",
                margin_debt_ratio=0.01,
                financing_interest_rate=0.05,
            )

        allocation = allocate_four_stocks(
            TICKERS,
            "aggressive",
            margin_debt_ratio=0.10,
            financing_interest_rate=0.06,
        )
        self.assertAlmostEqual(allocation.securities_exposure, 1.0)
        self.assertAlmostEqual(allocation.cash_ratio, 0.10)
        self.assertAlmostEqual(allocation.margin_debt_ratio, 0.10)
        self.assertAlmostEqual(allocation.unused_credit_ratio, 0.0)
        self.assertAlmostEqual(allocation.gross_exposure, 1.0)
        self.assertEqual(
            tuple(round(position.weight, 6) for position in allocation.positions),
            (0.375, 0.25, 0.25, 0.125),
        )

    def test_financing_rejects_excess_debt_and_missing_interest(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            allocate_four_stocks(
                TICKERS,
                "aggressive",
                margin_debt_ratio=0.11,
                financing_interest_rate=0.06,
            )
        with self.assertRaisesRegex(ValueError, "positive annual financing rate"):
            allocate_four_stocks(TICKERS, "aggressive", margin_debt_ratio=0.01)

    def test_policy_forbids_automatic_financing_and_averaging_down(self) -> None:
        with self.assertRaises(ValidationError):
            FinancingPolicy(
                enabled=True,
                max_margin_debt_ratio=0.05,
                automatic_activation=True,
            )
        with self.assertRaises(ValidationError):
            FinancingPolicy(
                enabled=True,
                max_margin_debt_ratio=0.05,
                allow_averaging_down=True,
            )

    def test_profile_rejects_any_ratio_other_than_three_two_two_one(self) -> None:
        base = get_default_profile("balanced")
        payload = base.model_dump()
        payload["ratio_units"] = (4, 2, 1, 1)
        with self.assertRaises(ValidationError):
            RiskProfile.model_validate(payload)

    def test_risk_cap_never_increases_template_weight(self) -> None:
        self.assertAlmostEqual(risk_capped_weight(0.30, 0.006, 0.05), 0.12)
        self.assertAlmostEqual(risk_capped_weight(0.10, 0.006, 0.05), 0.10)

    def test_toml_config_matches_validated_defaults(self) -> None:
        with (PROJECT_ROOT / "config" / "risk_profiles.toml").open("rb") as stream:
            config = tomllib.load(stream)

        self.assertEqual(config["ratio_units"], [3, 2, 2, 1])
        for name, raw in config["profiles"].items():
            profile = RiskProfile(
                name=name,
                display_name=raw["display_name"],
                stock_exposure=raw["stock_exposure"],
                base_cash_ratio=raw["base_cash_ratio"],
                ratio_units=tuple(config["ratio_units"]),
                position_weights=tuple(raw["position_weights"]),
                risk=RiskControls.model_validate(raw["risk"]),
                scoring=ScoringWeights.model_validate(raw["scoring"]),
                financing=FinancingPolicy.model_validate(raw["financing"]),
            )
            default = DEFAULT_PROFILES[RiskProfileName(name)]
            self.assertEqual(profile, default)


if __name__ == "__main__":
    unittest.main()
