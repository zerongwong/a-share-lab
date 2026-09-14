"""Howard Marks-inspired risk experiment, NOT a reproduction or timing oracle.

Frozen v1 thresholds are research hypotheses. Four non-price dimensions must
be independently evidenced. Price trend is kept separate. Output cannot enter
production allocations, protective stops or order/notification decisions.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

METHOD = "marks-cycle-shadow-v1"
DIMENSIONS = {
    "valuation_percentile": 14,
    "sentiment_percentile": 7,
    "credit_stress_percentile": 62,
    "earnings_improving_share": 125,
}


def assess_marks_cycle(
    observations, *, price_cutoff: date, known_at: datetime, incumbent_cap: float, previous=None
):
    """Read point-in-time evidence; missing/stale dimensions are not neutral."""
    if known_at.tzinfo is None or not math.isfinite(incumbent_cap) or not 0 <= incumbent_cap <= 1:
        raise ValueError("aware knowledge time and valid incumbent cap required")
    known_at = known_at.astimezone(ZoneInfo("Asia/Shanghai"))
    if price_cutoff > known_at.date():
        raise ValueError("future price cutoff")
    accepted, gaps = {}, []
    for dimension, max_age in DIMENSIONS.items():
        candidates = []
        for obs in observations:
            if obs["dimension"] != dimension:
                continue
            observed = date.fromisoformat(obs["observed_through"])
            times = [
                datetime.fromisoformat(obs[key])
                for key in ("published_at", "retrieved_at", "recorded_at")
            ]
            if (
                all(t.tzinfo is not None and t <= known_at for t in times)
                and observed <= price_cutoff
                and 0 <= (known_at.date() - observed).days <= max_age
            ):
                candidates.append(obs)
        if candidates:
            accepted[dimension] = max(
                candidates,
                key=lambda x: (
                    date.fromisoformat(x["observed_through"]),
                    datetime.fromisoformat(x["published_at"]),
                    datetime.fromisoformat(x["recorded_at"]),
                    x["id"],
                ),
            )
        else:
            gaps.append(dimension)
    base = {
        "method_version": METHOD,
        "mode": "shadow_only",
        "price_cutoff": price_cutoff.isoformat(),
        "known_at": known_at.isoformat(),
        "production_decision_input": False,
        "external_delivery_allowed": False,
        "auto_order_allowed": False,
        "validation_status": "pending_point_in_time_walk_forward",
        "incumbent_cap": incumbent_cap,
        "evidence_coverage": len(accepted) / len(DIMENSIONS),
        "evidence_ids": [accepted[key]["id"] for key in sorted(accepted)],
        "missing_dimensions": gaps,
        "confidence_probability": None,
        "state": "data_not_ready",
        "shadow_cap": None,
        "minimum_shadow_cash": None,
        "weekly_state_date": None,
        "reason": "四类非价格证据未齐，不以价格趋势冒充马克斯周期。",
    }
    if gaps:
        return base
    values = {key: float(obs["value"]) for key, obs in accepted.items()}
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
        raise ValueError("cycle evidence must be in [0, 100]")
    v, s, c, e = (values[key] for key in DIMENSIONS)
    # Separate investor excess from credit stress: high fear alone is NOT a buy.
    if v >= 80 and s >= 80:
        state, ceiling, reason = "overheated_defense", 0.30, "估值与情绪同时偏热，防范乐观定价。"
    elif c >= 80 and e <= 40:
        state, ceiling, reason = (
            "credit_stress_defense",
            0.30,
            "信用压力与盈利恶化并存，不能仅因下跌抄底。",
        )
    elif v <= 20 and s <= 20:
        state, ceiling, reason = (
            "opportunity_watch",
            0.50,
            "低估值与恐惧并存；仅扩大观察，仍等待个股确认。",
        )
    elif v < 80 and c < 60 and e >= 60:
        state, ceiling, reason = (
            "recovery_offense",
            0.80,
            "盈利改善、信用压力可控；进攻资格不等于立即买入。",
        )
    else:
        state, ceiling, reason = "mixed_baseline", 0.50, "证据混合，维持基准风险观察。"
    state_date = known_at.date()
    # Freeze ordinary changes within an ISO week. Fresh extreme-risk evidence
    # may tighten immediately. Missing evidence never inherits an old green light.
    if previous and previous.get("method_version") == METHOD and previous.get("weekly_state_date"):
        prior_date = date.fromisoformat(previous["weekly_state_date"])
        if (
            prior_date <= state_date
            and prior_date.isocalendar()[:2] == state_date.isocalendar()[:2]
            and previous.get("shadow_ceiling") is not None
            and ceiling >= previous["shadow_ceiling"]
        ):
            state, ceiling, reason = (
                previous["state"],
                previous["shadow_ceiling"],
                "周内维持已冻结判断；新证据留档，不频繁反转。",
            )
            state_date = prior_date
    cap = min(incumbent_cap, ceiling)
    return {
        **base,
        "state": state,
        "shadow_ceiling": ceiling,
        "shadow_cap": cap,
        "minimum_shadow_cash": 1 - cap,
        "weekly_state_date": state_date.isoformat(),
        "reason": reason,
        "observations": accepted,
        "cap_difference": cap - incumbent_cap,
        "note": "影子仓位上限，不是目标仓位；不放宽原有买点、止损或组合风控。",
    }
