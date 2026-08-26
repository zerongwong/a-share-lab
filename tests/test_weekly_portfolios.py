from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ashare_lab.analytics.return_ambition import ReturnAmbitionStatus  # noqa: E402
from ashare_lab.domain.models import RiskProfileName  # noqa: E402
from ashare_lab.services.build_weekly_portfolios import (  # noqa: E402
    NON_PROMISE_NOTICE,
    WeeklyPortfolioStatus,
    build_weekly_portfolios,
)

DATES = pd.bdate_range("2025-01-01", periods=300)


def _history(
    seed: int,
    *,
    drift: float = 0.0003,
    volatility: float = 0.006,
    returns: np.ndarray | None = None,
    periods: int = 300,
) -> pd.DataFrame:
    if returns is None:
        rng = np.random.default_rng(seed)
        generated = drift + rng.normal(0.0, volatility, periods)
    else:
        generated = np.asarray(returns, dtype=float)[:periods]
    close = 100.0 * np.cumprod(1.0 + generated)
    return pd.DataFrame({"close": close}, index=DATES[:periods])


def _meta(
    industry: str,
    *,
    quality: float = 0.5,
    liquidity: float = 0.5,
    catalyst: float = 0.0,
    is_st: bool = False,
    is_delisting: bool = False,
    name: str | None = None,
) -> dict[str, object]:
    return {
        "name": name or industry,
        "industry": industry,
        "is_st": is_st,
        "is_delisting": is_delisting,
        "quality_score": quality,
        "liquidity_score": liquidity,
        "catalyst_score": catalyst,
    }


def _valid_universe() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    histories = {
        "SAFE.SH": _history(1, drift=0.00025, volatility=0.002),
        "MOM.SZ": _history(2, drift=0.0010, volatility=0.008),
        "UTIL.SH": _history(3, drift=0.00035, volatility=0.003),
        "BANK.SH": _history(4, drift=0.00030, volatility=0.003),
        "TECH.SZ": _history(5, drift=0.00075, volatility=0.009),
        "MED.SZ": _history(6, drift=0.00050, volatility=0.006),
        "CONS.SH": _history(7, drift=0.00040, volatility=0.004),
        "MAT.SZ": _history(8, drift=0.00055, volatility=0.007),
    }
    metadata = {
        "SAFE.SH": _meta("公用事业", quality=1.0, liquidity=1.0),
        "MOM.SZ": _meta("高端制造", quality=0.25, liquidity=0.8, catalyst=1.0),
        "UTIL.SH": _meta("电力", quality=0.9, liquidity=0.8),
        "BANK.SH": _meta("银行", quality=0.85, liquidity=1.0),
        "TECH.SZ": _meta("电子", quality=0.5, liquidity=0.9, catalyst=0.9),
        "MED.SZ": _meta("医药", quality=0.7, liquidity=0.7, catalyst=0.4),
        "CONS.SH": _meta("消费", quality=0.8, liquidity=0.8, catalyst=0.2),
        "MAT.SZ": _meta("材料", quality=0.55, liquidity=0.6, catalyst=0.6),
    }
    return histories, metadata


class WeeklyPortfolioTests(unittest.TestCase):
    def test_builds_three_profiles_with_exact_allocation_and_no_financing(self) -> None:
        histories, metadata = _valid_universe()
        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])

        expected = {
            RiskProfileName.CONSERVATIVE: ((0.225, 0.15, 0.15, 0.075), 0.40),
            RiskProfileName.BALANCED: ((0.30, 0.20, 0.20, 0.10), 0.20),
            RiskProfileName.AGGRESSIVE: ((0.3375, 0.225, 0.225, 0.1125), 0.10),
        }
        for profile, (weights, cash) in expected.items():
            result = batch.for_profile(profile)
            self.assertEqual(result.status, WeeklyPortfolioStatus.READY)
            self.assertIsNotNone(result.allocation)
            allocation = result.allocation
            assert allocation is not None
            self.assertEqual(len(result.selected), 4)
            for position, wanted in zip(allocation.positions, weights, strict=True):
                self.assertAlmostEqual(position.weight, wanted)
            self.assertAlmostEqual(allocation.cash_ratio, cash)
            self.assertAlmostEqual(allocation.margin_debt_ratio, 0.0)

    def test_excludes_st_delisting_and_insufficient_history(self) -> None:
        histories, metadata = _valid_universe()
        histories["STBAD.SH"] = _history(20)
        metadata["STBAD.SH"] = _meta("垃圾", is_st=True, name="ST坏样本")
        histories["DELIST.SZ"] = _history(21)
        metadata["DELIST.SZ"] = _meta("垃圾", is_delisting=True, name="退市样本")
        histories["SHORT.SH"] = _history(22, periods=100)
        histories["SHORT.SH"].index = DATES[-100:]
        metadata["SHORT.SH"] = _meta("其他")

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        exclusions = {item.symbol: item.reasons for item in batch.exclusions}
        self.assertIn("st_stock_excluded", exclusions["STBAD.SH"])
        self.assertIn("delisting_stock_excluded", exclusions["DELIST.SZ"])
        self.assertTrue(
            any(reason.startswith("insufficient_history") for reason in exclusions["SHORT.SH"])
        )
        selected = {stock.symbol for result in batch.portfolios for stock in result.selected}
        self.assertFalse({"STBAD.SH", "DELIST.SZ", "SHORT.SH"} & selected)

    def test_profiles_apply_different_scoring_priorities(self) -> None:
        histories, metadata = _valid_universe()
        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])

        conservative = batch.for_profile("conservative")
        aggressive = batch.for_profile("aggressive")
        conservative_scores = {stock.symbol: stock.score for stock in conservative.selected}
        aggressive_scores = {stock.symbol: stock.score for stock in aggressive.selected}
        shared = set(conservative_scores) & set(aggressive_scores)
        self.assertTrue(shared)
        self.assertTrue(
            any(conservative_scores[symbol] != aggressive_scores[symbol] for symbol in shared)
        )
        self.assertNotEqual(
            tuple(stock.symbol for stock in conservative.selected),
            tuple(stock.symbol for stock in aggressive.selected),
        )

    def test_missing_optional_factors_are_disabled_not_neutral_filled(self) -> None:
        histories, metadata = _valid_universe()
        for item in metadata.values():
            item.pop("quality_score")
            item.pop("catalyst_score")
            item["sector_score"] = None

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        coverage = {item.factor: item for item in batch.factor_coverage}
        for factor in ("fundamentals", "news", "sector_context"):
            self.assertFalse(coverage[factor].enabled)
            self.assertEqual(coverage[factor].provided, 0)
        self.assertTrue(coverage["liquidity"].enabled)

        balanced = batch.for_profile("balanced")
        self.assertEqual(balanced.status, WeeklyPortfolioStatus.READY)
        for selected in balanced.selected:
            component_names = {name for name, _ in selected.component_scores}
            weight_names = {name for name, _ in selected.effective_weights}
            self.assertNotIn("fundamentals", component_names)
            self.assertNotIn("news", component_names)
            self.assertNotIn("fundamentals", weight_names)
            self.assertNotIn("news", weight_names)
            self.assertAlmostEqual(sum(value for _, value in selected.effective_weights), 1.0)
        self.assertTrue(any("财务基本面因子未启用" in item for item in balanced.risk_warnings))
        self.assertTrue(any("公司新闻/公告因子未启用" in item for item in balanced.risk_warnings))

    def test_risk_off_market_pauses_new_portfolio(self) -> None:
        histories, metadata = _valid_universe()
        falling = np.linspace(100.0, 45.0, len(DATES))
        for symbol in histories:
            histories[symbol] = pd.DataFrame({"close": falling}, index=DATES)

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        assert batch.market_regime is not None
        self.assertEqual(batch.market_regime.state.value, "risk_off")
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.UNAVAILABLE)
            self.assertIn("market_regime_risk_off_new_entries_paused", result.reasons)

    def test_risk_off_core_indices_pause_even_when_stock_breadth_is_not_risk_off(self) -> None:
        histories, metadata = _valid_universe()
        falling = np.linspace(150.0, 90.0, len(DATES))
        index_histories = {
            code: pd.DataFrame(
                {
                    "trade_date": DATES,
                    "close": falling,
                    "historical_backtest_eligible": True,
                    "common_cutoff_date": DATES[-1],
                }
            )
            for code in ("000001", "000300", "000905")
        }

        batch = build_weekly_portfolios(
            histories,
            metadata,
            as_of=DATES[-1],
            market_index_histories=index_histories,
        )

        assert batch.market_regime is not None
        assert batch.index_regime is not None
        self.assertNotEqual(batch.market_regime.state.value, "risk_off")
        self.assertEqual(batch.index_regime.state.value, "risk_off")
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.UNAVAILABLE)
            self.assertIn("core_index_regime_risk_off_new_entries_paused", result.reasons)

    def test_current_balance_sheet_strength_is_separate_from_full_fundamentals(self) -> None:
        histories, metadata = _valid_universe()
        for index, item in enumerate(metadata.values()):
            item.pop("quality_score")
            item["balance_sheet_strength_score"] = index / (len(metadata) - 1)

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        coverage = {item.factor: item for item in batch.factor_coverage}

        self.assertFalse(coverage["fundamentals"].enabled)
        self.assertTrue(coverage["balance_sheet_strength"].enabled)
        selected = batch.for_profile("balanced").selected
        self.assertTrue(selected)
        for stock in selected:
            components = {name for name, _ in stock.component_scores}
            self.assertIn("balance_sheet_strength", components)
            self.assertNotIn("fundamentals", components)

    def test_partial_factor_coverage_is_disabled_for_the_whole_universe(self) -> None:
        histories, metadata = _valid_universe()
        metadata["SAFE.SH"].pop("quality_score")

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        fundamentals = next(item for item in batch.factor_coverage if item.factor == "fundamentals")
        self.assertFalse(fundamentals.enabled)
        self.assertEqual(fundamentals.provided, len(histories) - 1)
        self.assertIn("partial_coverage", fundamentals.reason)

    def test_formation_limit_up_is_substituted_with_next_constrained_candidate(self) -> None:
        histories, metadata = _valid_universe()
        baseline = build_weekly_portfolios(
            histories,
            metadata,
            as_of=DATES[-1],
            exclude_formation_limit_up=False,
        ).for_profile("balanced")
        blocked = baseline.selected[0].symbol
        metadata[blocked]["is_limit_up_at_cutoff"] = True
        metadata[blocked]["is_buyable_at_cutoff"] = False

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        replacement = batch.for_profile("balanced")
        self.assertEqual(replacement.status, WeeklyPortfolioStatus.READY)
        self.assertEqual(len(replacement.selected), 4)
        self.assertNotIn(blocked, {item.symbol for item in replacement.selected})
        exclusions = {item.symbol: item.reasons for item in batch.exclusions}
        self.assertIn("formation_limit_up_unbuyable", exclusions[blocked])

    def test_overheated_acceleration_is_frozen_without_inventing_execution_data(self) -> None:
        histories, metadata = _valid_universe()
        overheated = histories["MOM.SZ"].copy()
        overheated.iloc[-20:, overheated.columns.get_loc("close")] *= np.linspace(1.0, 1.60, 20)
        histories["MOM.SZ"] = overheated

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        exclusions = {item.symbol: item.reasons for item in batch.exclusions}
        self.assertIn("overheated_acceleration_excluded", exclusions["MOM.SZ"])
        selected = {stock.symbol for result in batch.portfolios for stock in result.selected}
        self.assertNotIn("MOM.SZ", selected)

    def test_sector_and_correlation_constraints_prevent_false_diversification(self) -> None:
        histories, metadata = _valid_universe()
        common = np.random.default_rng(99).normal(0.0008, 0.005, len(DATES))
        histories["CLONE1.SH"] = _history(30, returns=common)
        histories["CLONE2.SH"] = _history(31, returns=common)
        metadata["CLONE1.SH"] = _meta("热门主题", quality=1.0, liquidity=1.0, catalyst=1.0)
        metadata["CLONE2.SH"] = _meta("热门主题", quality=1.0, liquidity=1.0, catalyst=1.0)

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.READY)
            selected = {stock.symbol for stock in result.selected}
            self.assertFalse({"CLONE1.SH", "CLONE2.SH"} <= selected)

            allocation = result.allocation
            assert allocation is not None
            sectors: dict[str, float] = {}
            industry_by_symbol = {stock.symbol: stock.industry for stock in result.selected}
            for position in allocation.positions:
                industry = industry_by_symbol[position.ticker]
                sectors[industry] = sectors.get(industry, 0.0) + position.weight
            profile_cap = {
                "conservative": 0.30,
                "balanced": 0.40,
                "aggressive": 0.50,
            }[result.profile.value]
            self.assertLessEqual(max(sectors.values()), profile_cap + 1e-9)

    def test_fewer_than_four_eligible_returns_unavailable_instead_of_filling(self) -> None:
        histories, metadata = _valid_universe()
        keep = {"SAFE.SH", "UTIL.SH", "BANK.SH"}
        histories = {key: value for key, value in histories.items() if key in keep}
        metadata = {key: value for key, value in metadata.items() if key in keep}

        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.UNAVAILABLE)
            self.assertIsNone(result.allocation)
            self.assertEqual(result.selected, ())
            self.assertIn("four_required", result.reasons[0])

    def test_historical_metrics_and_scenarios_are_explicitly_not_promises(self) -> None:
        histories, metadata = _valid_universe()
        batch = build_weekly_portfolios(histories, metadata, as_of=DATES[-1])
        result = batch.for_profile("balanced")

        self.assertEqual(batch.disclaimer, NON_PROMISE_NOTICE)
        self.assertIsNotNone(result.historical_risk)
        self.assertIsNotNone(result.historical_scenario)
        risk = result.historical_risk
        scenario = result.historical_scenario
        assert risk is not None and scenario is not None
        self.assertFalse(risk.is_promise)
        self.assertFalse(risk.is_out_of_sample)
        self.assertFalse(risk.net_of_costs)
        self.assertIsNotNone(risk.historical_sortino)
        self.assertIsNotNone(risk.historical_calmar)
        self.assertEqual(
            {item.window_sessions for item in risk.drawdown_windows},
            {20, 40, 60},
        )
        for distribution in risk.drawdown_windows:
            self.assertTrue(distribution.available)
            self.assertIsNotNone(distribution.breach_probability)
            self.assertIsNotNone(distribution.breach_interval)
            self.assertFalse(distribution.is_forecast_probability)
            self.assertFalse(distribution.is_promise)
        self.assertFalse(scenario.is_promise)
        self.assertFalse(scenario.is_forecast_probability)
        self.assertTrue(scenario.available)
        self.assertLessEqual(scenario.return_p10, scenario.return_p50)
        self.assertLessEqual(scenario.return_p50, scenario.return_p90)

    def test_holding_weeks_and_return_ambition_drive_horizon_evidence(self) -> None:
        histories, metadata = _valid_universe()
        batch = build_weekly_portfolios(
            histories,
            metadata,
            as_of=DATES[-1],
            holding_weeks=8,
            annual_return_ambition_pct=200,
        )

        self.assertEqual(batch.holding_weeks, 8)
        self.assertEqual(batch.annual_return_ambition_pct, 200)
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.READY)
            assessment = result.return_ambition_assessment
            scenario = result.historical_scenario
            assert assessment is not None and scenario is not None
            self.assertEqual(assessment.holding_weeks, 8)
            self.assertEqual(assessment.horizon_sessions, 40)
            self.assertEqual(scenario.horizon_sessions, 40)
            self.assertIn(
                assessment.status,
                {
                    ReturnAmbitionStatus.UNSUPPORTED,
                    ReturnAmbitionStatus.STRETCH,
                    ReturnAmbitionStatus.HISTORICALLY_SUPPORTED,
                },
            )
            self.assertFalse(assessment.is_out_of_sample)
            self.assertFalse(assessment.is_forecast_probability)
            self.assertFalse(assessment.is_promise)

    def test_default_service_sets_a_horizon_without_an_arbitrary_return_target(self) -> None:
        histories, metadata = _valid_universe()
        batch = build_weekly_portfolios(
            histories,
            metadata,
            as_of=DATES[-1],
            holding_weeks=13,
        )

        self.assertEqual(batch.holding_weeks, 13)
        self.assertIsNone(batch.annual_return_ambition_pct)
        for result in batch.portfolios:
            self.assertEqual(result.status, WeeklyPortfolioStatus.READY)
            self.assertIsNone(result.return_ambition_assessment)
            scenario = result.historical_scenario
            assert scenario is not None
            self.assertEqual(scenario.horizon_sessions, 65)

    def test_invalid_holding_period_or_ambition_fails_before_portfolio_selection(self) -> None:
        histories, metadata = _valid_universe()
        with self.assertRaisesRegex(ValueError, "between 1 and 52"):
            build_weekly_portfolios(histories, metadata, holding_weeks=0)
        with self.assertRaisesRegex(ValueError, "between 1 and 52"):
            build_weekly_portfolios(histories, metadata, holding_weeks=53)
        with self.assertRaisesRegex(ValueError, "must be one of"):
            build_weekly_portfolios(
                histories,
                metadata,
                annual_return_ambition_pct=30,
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            build_weekly_portfolios(
                histories,
                metadata,
                holding_weeks=8,
                scenario_horizon_sessions=20,
            )


if __name__ == "__main__":
    unittest.main()
