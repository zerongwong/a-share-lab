from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd

from ashare_lab.adapters.sqlite_repository import SQLiteRepository
from ashare_lab.analytics.indicators import enrich_indicators
from ashare_lab.analytics.levels import build_horizon_levels
from ashare_lab.analytics.livermore import build_livermore_plan
from ashare_lab.analytics.probability import empirical_three_way_scenarios
from ashare_lab.analytics.risk_metrics import risk_metrics
from ashare_lab.analytics.trend import breakout_evidence, classify_trend
from ashare_lab.domain.enums import TrendState
from ashare_lab.domain.errors import DataQualityError, InsufficientHistoryError
from ashare_lab.ports.market_data import MarketDataPort, normalize_symbol

STRATEGY_VERSION = "ashare-evidence-v0.2.0"
HORIZON_LABELS = {5: "一周", 20: "一月", 60: "三个月", 120: "六个月", 252: "一年"}


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _hash_frame(frame: pd.DataFrame) -> str:
    stable = frame.copy()
    if "retrieved_at" in stable:
        stable = stable.drop(columns=["retrieved_at"])
    payload = stable.to_json(orient="split", date_format="iso", double_precision=10)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _actions(trend: TrendState, close: float, levels: dict[str, Any]) -> tuple[str, str]:
    in_entry = levels["pullback_entry_low"] <= close <= levels["pullback_entry_high"]
    if trend == TrendState.UP:
        empty = "回踩计划区可小仓分批试错" if in_entry else "等待回踩计划区或有效突破，不追涨"
        holder = "趋势未破可继续观察；只有盈利仓越过加仓线才考虑加仓"
    elif trend == TrendState.DOWN:
        empty = "观望：下降趋势中不逢低摊平"
        holder = "反弹至减仓区评估降仓；跌破结构失效位执行风控"
    else:
        empty = "等待区间下沿企稳或放量突破，当前不抢跑"
        holder = "区间内控制仓位；临近减仓区分批收缩，失效位下方不硬扛"
    return empty, holder


def analyze_stock(
    provider: MarketDataPort,
    symbol: str,
    as_of: date,
    *,
    cost_price: float | None = None,
    lookback_days: int = 1_200,
) -> dict[str, Any]:
    code = normalize_symbol(symbol)
    start = as_of - timedelta(days=lookback_days)
    frame = provider.fetch_daily(code, start, as_of, adjust="qfq")
    if frame.empty:
        raise InsufficientHistoryError(f"{code} 没有可用日线数据")
    frame = frame.sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    cutoff = pd.Timestamp(as_of)
    if (frame["trade_date"] > cutoff).any():
        raise DataQualityError("数据包含分析截止日之后的K线，已拒绝分析")
    if len(frame) < 120:
        raise InsufficientHistoryError(f"至少需要120根日线，目前只有{len(frame)}根")

    enriched = enrich_indicators(frame)
    latest = enriched.iloc[-1]
    trend = classify_trend(enriched)
    breakout = breakout_evidence(enriched)
    current_profitable = bool(
        cost_price is not None and cost_price > 0 and latest["close"] > cost_price
    )
    livermore = build_livermore_plan(
        enriched,
        trend,
        current_position_profitable=current_profitable,
    )
    levels = [item.as_dict() for item in build_horizon_levels(enriched)]
    returns = enriched["close"].pct_change()
    metrics = risk_metrics(returns.tail(252))
    scenarios: dict[int, list[dict[str, object]]] = {
        sessions: empirical_three_way_scenarios(returns, sessions) for sessions in HORIZON_LABELS
    }
    for row in levels:
        empty, holder = _actions(trend, float(latest["close"]), row)
        row["action_for_empty"] = empty
        row["action_for_holder"] = holder
        row["scenarios"] = scenarios[row["sessions"]]

    source = str(frame.attrs.get("source", latest.get("source", "unknown")))
    retrieved_at = str(frame.attrs.get("retrieved_at", latest.get("retrieved_at", "")))
    is_stale = bool(
        frame.attrs.get("is_stale", False)
        or frame.attrs.get("is_cache_fallback", False)
        or frame.attrs.get("data_quality") == "cached"
    )
    warning = [
        "价格区间是规则化研究计划，不是保证成交或保证收益。",
        "艾德华兹–麦吉和利弗摩尔理论已转译为可测试规则，不代表原著给出这些固定ATR参数。",
        "历史情景频率尚未做样本外概率校准，不能当作未来真实概率。",
        "未纳入最新公告与持牌实时行情时，应降低结论置信度。",
    ]
    if is_stale:
        warning.insert(0, "当前使用缓存数据，可能不是最新交易日。")

    return {
        "symbol": code,
        "as_of": as_of.isoformat(),
        "data_cutoff": pd.Timestamp(latest["trade_date"]).date().isoformat(),
        "source": source,
        "retrieved_at": retrieved_at,
        "is_stale": is_stale,
        "row_count": len(frame),
        "data_hash": _hash_frame(frame),
        "latest_price": float(latest["close"]),
        "trend": trend.value,
        "breakout": breakout,
        "livermore": asdict(livermore),
        "metrics": metrics,
        "levels": levels,
        "warnings": warning,
        "frame": enriched,
    }


def archive_stock_analysis(
    result: dict[str, Any],
    repository: SQLiteRepository,
) -> str:
    run_id = str(uuid4())
    created_at = datetime.now(UTC)
    config_hash = hashlib.sha256(STRATEGY_VERSION.encode()).hexdigest()
    analyses: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for level in result["levels"]:
        analysis_id = str(uuid4())
        confidence = "medium" if level["scenarios"] else "unavailable"
        analyses.append(
            {
                "id": analysis_id,
                "run_id": run_id,
                "symbol": result["symbol"],
                "horizon_sessions": level["sessions"],
                "trend_state": result["trend"],
                "action_for_empty": level["action_for_empty"],
                "action_for_holder": level["action_for_holder"],
                "entry_low": level["pullback_entry_low"],
                "entry_high": level["pullback_entry_high"],
                "add_above": result["livermore"]["add_only_above"],
                "reduce_low": level["reduce_low"],
                "reduce_high": level["reduce_high"],
                "invalidation": level["invalidation"],
                "confidence": confidence,
                "rationale_json": {
                    "breakout": result["breakout"],
                    "livermore": result["livermore"]["note"],
                    "level_method": level["level_method"],
                    "level_evidence_dates": level["level_evidence_dates"],
                    "breakout_confirmation_rule": level["breakout_confirmation_rule"],
                    "stop_execution_rule": level["stop_execution_rule"],
                },
            }
        )
        for scenario in level["scenarios"]:
            scenarios.append({"id": str(uuid4()), "analysis_id": analysis_id, **scenario})

    repository.archive_run(
        {
            "id": run_id,
            "run_type": "stock_analysis",
            "as_of": result["as_of"],
            "data_cutoff": result["data_cutoff"],
            "created_at": created_at,
            "strategy_version": STRATEGY_VERSION,
            "model_id": None,
            "config_hash": config_hash,
            "data_hash": result["data_hash"],
            "status": "completed",
            "warning_json": result["warnings"],
        },
        data_snapshots=[
            {
                "id": str(uuid4()),
                "run_id": run_id,
                "source": result["source"],
                "dataset": "daily_ohlcv",
                "symbol": result["symbol"],
                "first_at": pd.Timestamp(result["frame"].iloc[0]["trade_date"]).date(),
                "last_at": pd.Timestamp(result["frame"].iloc[-1]["trade_date"]).date(),
                "row_count": result["row_count"],
                "adjustment": "qfq",
                "unit_json": {"volume_shares": "share", "amount_cny": "CNY"},
                "checksum": result["data_hash"],
                "retrieved_at": result["retrieved_at"] or created_at,
                "is_stale": result["is_stale"],
            }
        ],
        analyses=analyses,
        scenarios=scenarios,
    )
    return run_id
