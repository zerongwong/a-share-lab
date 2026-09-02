"""Load an immutable CSMAR baseline plus a verified daily market overlay.

The historical CSMAR catalogue remains the source of identity, eligibility and
long history.  This service never updates that catalogue and never lets a
newer provider row replace an overlapping CSMAR observation.  It only appends
post-baseline, checksum-verified, unadjusted rows whose stock and core-index
cutoffs agree.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from ashare_lab.domain.data_sources import DEFAULT_MARKET_OVERLAY_SOURCE_ID
from ashare_lab.domain.errors import DataQualityError, DataUnavailableError
from ashare_lab.services.load_csmar_universe import (
    DEFAULT_CORE_INDEX_CODES,
    CSMARUniverseSnapshot,
    _closed_at_formation_limit,
    load_csmar_universe,
)


class _VerifiedOverlayReader(Protocol):
    def read_verified_manifest(self, *, source_id: str | None = None) -> pd.DataFrame: ...

    def read_verified_daily(
        self,
        trade_date: date,
        *,
        source_id: str,
        asset_kind: Literal["stocks", "indices"],
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class HybridUniverseLoad:
    """One auditable hybrid load and its immutable CSMAR-compatible snapshot."""

    snapshot: CSMARUniverseSnapshot
    historical_baseline_cutoff: date
    automatic_increment_cutoff: date | None
    common_cutoff: date
    sources: tuple[str, ...]
    overlay_trading_days: tuple[date, ...] = ()
    isolated_overlay_symbols: tuple[str, ...] = ()


_STOCK_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume_shares",
    "amount_cny",
    "turnover_pct",
    "source",
    "retrieved_at",
)
_INDEX_PRICE_COLUMNS = ("trade_date", "open", "high", "low", "close")
_PROVIDER_INDEX_ALIASES: dict[str, dict[str, str]] = {
    # Infoway exposes CSI 500 only under the Shenzhen-style vendor identifier.
    # CSMAR and the research domain use 000905 as the canonical identity.
    "infoway": {"399905": "000905"},
}
_MANIFEST_COLUMNS = (
    "source_id",
    "adjustment",
    "trade_date",
    "previous_trade_date",
    "stock_count",
    "expected_stock_count",
    "stock_coverage_ratio",
    "index_count",
    "core_index_symbols",
    "latest_retrieved_at",
    "verified_at",
)

# Full-market membership needs enough complete daily history for the common
# market-regime and basic liquidity gates.  Holding-horizon risk history is a
# candidate-level requirement and must not remove nearly half the market before
# the horizon screen has even run.
HYBRID_QUALIFICATION_MINIMUM_SESSIONS = 252


def load_hybrid_universe(
    dataset_root: str | Path,
    *,
    overlay_root: str | Path,
    as_of: date,
    decision_date: date | None = None,
    mode: Literal["live", "historical"] = "live",
    overlay_source_id: str = DEFAULT_MARKET_OVERLAY_SOURCE_ID,
    core_index_codes: tuple[str, ...] = DEFAULT_CORE_INDEX_CODES,
    overlay_store: _VerifiedOverlayReader | None = None,
    minimum_qualification_sessions: int = HYBRID_QUALIFICATION_MINIMUM_SESSIONS,
    **csmar_options: object,
) -> HybridUniverseLoad:
    """Return a CSMAR snapshot extended only by verified, later EOD rows.

    A stale live CSMAR baseline is intentionally reopened in historical mode;
    freshness is enforced again after the verified overlay is merged.  Other
    CSMAR failures are never swallowed.  Historical replay accepts an overlay
    only when both its retrieval and verification times were already known by
    that decision date.
    """

    if mode not in {"live", "historical"}:
        raise ValueError("mode must be live or historical")
    resolved_decision_date = as_of if decision_date is None else decision_date
    if resolved_decision_date < as_of:
        raise ValueError("decision_date cannot be earlier than as_of")
    if not overlay_source_id.strip():
        raise ValueError("overlay_source_id cannot be blank")
    if len(core_index_codes) < 3 or len(set(core_index_codes)) != len(core_index_codes):
        raise ValueError("core_index_codes must contain at least three unique codes")
    if "core_index_codes" in csmar_options:
        raise ValueError("pass core_index_codes only once")
    if (
        isinstance(minimum_qualification_sessions, bool)
        or not isinstance(minimum_qualification_sessions, int)
        or minimum_qualification_sessions < HYBRID_QUALIFICATION_MINIMUM_SESSIONS
    ):
        raise ValueError("minimum_qualification_sessions must be an integer of at least 252")

    # ``minimum_sessions`` was historically supplied by the six-horizon
    # orchestrator as the downstream portfolio-risk requirement (for one year,
    # 2017 prices).  Preserve it as a requested read-depth validation, but do
    # not apply it as a full-market coverage gate.  Each horizon service later
    # excludes a structurally qualified stock whose own risk history is short.
    requested_risk_sessions_raw = csmar_options.pop(
        "minimum_sessions",
        minimum_qualification_sessions,
    )
    history_sessions_raw = csmar_options.get("history_sessions", 320)
    if (
        isinstance(requested_risk_sessions_raw, bool)
        or not isinstance(requested_risk_sessions_raw, int)
        or requested_risk_sessions_raw < HYBRID_QUALIFICATION_MINIMUM_SESSIONS
    ):
        raise ValueError("minimum_sessions must be an integer of at least 252")
    if (
        isinstance(history_sessions_raw, bool)
        or not isinstance(history_sessions_raw, int)
        or history_sessions_raw < max(requested_risk_sessions_raw, minimum_qualification_sessions)
    ):
        raise ValueError(
            "history_sessions cannot be shorter than the requested risk or qualification history"
        )
    csmar_options["minimum_sessions"] = minimum_qualification_sessions

    baseline = _load_baseline_allowing_verified_live_extension(
        dataset_root,
        as_of=as_of,
        decision_date=resolved_decision_date,
        mode=mode,
        core_index_codes=core_index_codes,
        csmar_options=csmar_options,
    )
    if requested_risk_sessions_raw > minimum_qualification_sessions:
        baseline = replace(
            baseline,
            reference_warnings=(
                *baseline.reference_warnings,
                "全市场资格门仅要求"
                f"{minimum_qualification_sessions}个完整价格点；"
                f"{requested_risk_sessions_raw}个价格点的持有期风险历史"
                "仅对各期限结构合格候选逐股核验。",
            ),
        )
    baseline_cutoff = baseline.data_cutoff
    master_symbols = _read_csmar_master_symbols(dataset_root)
    if not set(baseline.histories).issubset(master_symbols):
        raise DataQualityError("CSMAR合格历史包含证券主表之外的代码")

    reader = overlay_store or _build_overlay_store(overlay_root)
    manifest = _normalise_manifest(reader.read_verified_manifest(source_id=overlay_source_id))
    selected_manifest = _select_manifest_rows(
        manifest,
        source_id=overlay_source_id,
        baseline_cutoff=baseline_cutoff,
        as_of=as_of,
        decision_date=resolved_decision_date,
        mode=mode,
    )
    if selected_manifest.empty:
        _enforce_live_freshness(
            baseline_cutoff,
            decision_date=resolved_decision_date,
            mode=mode,
        )
        return HybridUniverseLoad(
            snapshot=baseline,
            historical_baseline_cutoff=baseline_cutoff,
            automatic_increment_cutoff=None,
            common_cutoff=baseline_cutoff,
            sources=("CSMAR只读历史基线",),
        )

    _validate_manifest_chain(
        selected_manifest,
        baseline_cutoff=baseline_cutoff,
        core_index_codes=core_index_codes,
        source_id=overlay_source_id,
    )
    overlay_dates = tuple(selected_manifest["trade_date"].dt.date)
    stock_days: list[pd.DataFrame] = []
    index_days: list[pd.DataFrame] = []
    isolated: set[str] = set()
    for manifest_row, trade_day in zip(
        selected_manifest.to_dict("records"),
        overlay_dates,
        strict=True,
    ):
        stocks = _normalise_stock_day(
            reader.read_verified_daily(
                trade_day,
                source_id=overlay_source_id,
                asset_kind="stocks",
            ),
            trade_day=trade_day,
            source_id=overlay_source_id,
            decision_date=resolved_decision_date,
            mode=mode,
        )
        indices = _normalise_index_day(
            reader.read_verified_daily(
                trade_day,
                source_id=overlay_source_id,
                asset_kind="indices",
            ),
            trade_day=trade_day,
            source_id=overlay_source_id,
            core_index_codes=core_index_codes,
            decision_date=resolved_decision_date,
            mode=mode,
        )
        if len(stocks) != int(manifest_row["stock_count"]):
            raise DataQualityError(f"{trade_day.isoformat()}股票文件行数与已验证清单不一致")
        if len(indices) != int(manifest_row["index_count"]):
            raise DataQualityError(f"{trade_day.isoformat()}指数文件行数与已验证清单不一致")
        stock_symbols = set(stocks["symbol"])
        isolated.update(stock_symbols.difference(master_symbols))
        stock_days.append(stocks.loc[stocks["symbol"].isin(master_symbols)].copy())
        index_days.append(indices)

    required_symbols = set(baseline.histories)
    complete_symbols = required_symbols.intersection(
        *(set(frame["symbol"]) for frame in stock_days)
    )
    incomplete_symbols = required_symbols.difference(complete_symbols)
    minimum_eligible_symbols = int(csmar_options.get("minimum_eligible_symbols", 1_000))
    minimum_eligible_active_coverage = float(
        csmar_options.get("minimum_eligible_active_coverage", 0.70)
    )
    eligible_active_coverage = len(complete_symbols) / baseline.active_symbols
    if (
        len(complete_symbols) < minimum_eligible_symbols
        or eligible_active_coverage < minimum_eligible_active_coverage
    ):
        raise DataUnavailableError(
            "自动增量缺失标的排除后，不再满足全市场研究资格门："
            f"完整{len(complete_symbols)}只/基线活跃{baseline.active_symbols}只，"
            f"最低要求{minimum_eligible_symbols}只且覆盖率"
            f"{minimum_eligible_active_coverage:.0%}。"
        )

    filtered_histories = {
        symbol: history
        for symbol, history in baseline.histories.items()
        if symbol in complete_symbols
    }
    filtered_metadata = {
        symbol: item for symbol, item in baseline.metadata.items() if symbol in complete_symbols
    }

    merged_histories = _append_stock_histories(
        filtered_histories,
        stock_days,
        overlay_dates=overlay_dates,
        baseline_cutoff=baseline_cutoff,
    )
    common_cutoff = overlay_dates[-1]
    merged_indices = _append_index_histories(
        baseline.market_index_histories,
        index_days,
        overlay_dates=overlay_dates,
        baseline_cutoff=baseline_cutoff,
        common_cutoff=common_cutoff,
        core_index_codes=core_index_codes,
    )
    merged_metadata = _refresh_cutoff_metadata(merged_histories, filtered_metadata)
    _enforce_live_freshness(
        common_cutoff,
        decision_date=resolved_decision_date,
        mode=mode,
    )

    overlay_sources = sorted(
        {
            str(value)
            for frame in (*stock_days, *index_days)
            for value in frame["source"].dropna().unique()
        }
    )
    warnings = list(baseline.reference_warnings)
    warnings.append(
        "CSMAR历史库保持只读；共同截止日之后仅追加通过manifest校验的"
        f"{overlay_source_id}未复权收盘增量。"
    )
    if isolated:
        warnings.append(f"增量中{len(isolated)}只股票不在CSMAR证券主表，已隔离且未进入研究。")
    if incomplete_symbols:
        warnings.append(
            f"{len(incomplete_symbols)}只CSMAR合格股票未覆盖全部自动增量交易日，"
            "已从本轮共同样本排除。"
        )
    snapshot = replace(
        baseline,
        histories=merged_histories,
        metadata=merged_metadata,
        data_cutoff=common_cutoff,
        eligible_symbols=len(merged_histories),
        excluded_symbols=baseline.active_symbols - len(merged_histories),
        market_index_histories=merged_indices,
        reference_common_cutoff=common_cutoff,
        reference_warnings=tuple(warnings),
    )
    return HybridUniverseLoad(
        snapshot=snapshot,
        historical_baseline_cutoff=baseline_cutoff,
        automatic_increment_cutoff=common_cutoff,
        common_cutoff=common_cutoff,
        sources=("CSMAR只读历史基线", *(f"{item}已验证收盘增量" for item in overlay_sources)),
        overlay_trading_days=overlay_dates,
        isolated_overlay_symbols=tuple(sorted(isolated)),
    )


def _load_baseline_allowing_verified_live_extension(
    dataset_root: str | Path,
    *,
    as_of: date,
    decision_date: date,
    mode: Literal["live", "historical"],
    core_index_codes: tuple[str, ...],
    csmar_options: dict[str, object],
) -> CSMARUniverseSnapshot:
    try:
        return load_csmar_universe(
            dataset_root,
            as_of=as_of,
            decision_date=decision_date,
            mode=mode,
            core_index_codes=core_index_codes,
            **csmar_options,
        )
    except DataUnavailableError as exc:
        stale_live = "当前研究的数据不是最新完整截面" in str(exc)
        if mode != "live" or not stale_live:
            raise
    return load_csmar_universe(
        dataset_root,
        as_of=as_of,
        decision_date=decision_date,
        mode="historical",
        core_index_codes=core_index_codes,
        **csmar_options,
    )


def _build_overlay_store(root: str | Path) -> _VerifiedOverlayReader:
    from ashare_lab.adapters.market_overlay_store import MarketOverlayStore

    return MarketOverlayStore(root)


def _read_csmar_master_symbols(dataset_root: str | Path) -> set[str]:
    database_path = Path(dataset_root).expanduser().resolve() / "csmar.duckdb"
    if not database_path.is_file():
        raise DataUnavailableError(f"CSMAR本地数据库不存在：{database_path}")
    try:
        import duckdb
    except ImportError as exc:
        raise DataUnavailableError("缺少duckdb依赖，无法读取CSMAR证券主表") from exc
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        values = connection.execute("SELECT DISTINCT symbol FROM security_master").fetchall()
    finally:
        connection.close()
    symbols = {str(value[0]).zfill(6) for value in values if value and value[0] is not None}
    if not symbols:
        raise DataQualityError("CSMAR证券主表为空")
    return symbols


def _normalise_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise DataQualityError("已验证增量清单不是DataFrame")
    if raw.empty:
        return pd.DataFrame(columns=list(_MANIFEST_COLUMNS))
    missing = set(_MANIFEST_COLUMNS).difference(raw.columns)
    if missing:
        raise DataQualityError("已验证增量清单缺列：" + "、".join(sorted(missing)))
    frame = raw.copy()
    frame["source_id"] = frame["source_id"].astype(str).str.strip().str.lower()
    frame["adjustment"] = frame["adjustment"].astype(str).str.strip().str.lower()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["previous_trade_date"] = pd.to_datetime(
        frame["previous_trade_date"], errors="coerce"
    ).dt.normalize()
    frame["latest_retrieved_at"] = pd.to_datetime(
        frame["latest_retrieved_at"], errors="coerce", utc=True
    )
    frame["verified_at"] = pd.to_datetime(frame["verified_at"], errors="coerce", utc=True)
    if bool(
        frame[
            [
                "trade_date",
                "previous_trade_date",
                "latest_retrieved_at",
                "verified_at",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise DataQualityError("已验证增量清单含无效日期或审计时间")
    for column in ("stock_count", "expected_stock_count", "index_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if bool(frame[column].isna().any()) or bool((frame[column] <= 0).any()):
            raise DataQualityError(f"已验证增量清单的{column}无效")
    frame["stock_coverage_ratio"] = pd.to_numeric(frame["stock_coverage_ratio"], errors="coerce")
    if (
        bool(frame["stock_coverage_ratio"].isna().any())
        or bool((frame["stock_coverage_ratio"] <= 0).any())
        or bool((frame["stock_coverage_ratio"] > 1).any())
    ):
        raise DataQualityError("已验证增量清单的stock_coverage_ratio无效")
    calculated_coverage = frame["stock_count"] / frame["expected_stock_count"]
    if not bool(np.isclose(calculated_coverage, frame["stock_coverage_ratio"], atol=1e-12).all()):
        raise DataQualityError("已验证增量清单的股票覆盖率与计数不一致")
    if "verified" in frame and not bool(frame["verified"].fillna(False).astype(bool).all()):
        raise DataQualityError("读取结果混入未验证manifest")
    return frame.sort_values(["source_id", "trade_date"]).reset_index(drop=True)


def _validate_manifest_chain(
    manifest: pd.DataFrame,
    *,
    baseline_cutoff: date,
    core_index_codes: tuple[str, ...],
    source_id: str,
) -> None:
    previous = baseline_cutoff
    expected_indices = set(core_index_codes)
    for row in manifest.to_dict("records"):
        trade_day = pd.Timestamp(row["trade_date"]).date()
        declared_previous = pd.Timestamp(row["previous_trade_date"]).date()
        if declared_previous != previous:
            raise DataQualityError(
                "已验证自动增量存在交易日断链："
                f"{trade_day.isoformat()}声明前序为{declared_previous.isoformat()}，"
                f"共同链要求{previous.isoformat()}。"
            )
        raw_indices = [
            item.strip() for item in str(row["core_index_symbols"]).split("|") if item.strip()
        ]
        declared_indices = _canonical_provider_index_codes(
            raw_indices,
            source_id=source_id,
            label=f"{trade_day.isoformat()}manifest",
        )
        if set(declared_indices) != expected_indices:
            raise DataQualityError(
                f"{trade_day.isoformat()}manifest的六核心指数集合与研究配置不一致"
            )
        previous = trade_day


def _select_manifest_rows(
    manifest: pd.DataFrame,
    *,
    source_id: str,
    baseline_cutoff: date,
    as_of: date,
    decision_date: date,
    mode: Literal["live", "historical"],
) -> pd.DataFrame:
    normalized_source = source_id.strip().lower()
    source_rows = manifest.loc[manifest["source_id"] == normalized_source].copy()
    if source_rows.empty:
        return source_rows
    if bool((source_rows["adjustment"] != "none").any()):
        raise DataQualityError("自动增量必须为none未复权，不能与CSMAR原始价格混接")
    if bool(source_rows["trade_date"].duplicated().any()):
        raise DataQualityError("同一来源同一交易日存在多个已验证manifest")
    relevant = source_rows.loc[
        (source_rows["trade_date"].dt.date > baseline_cutoff)
        & (source_rows["trade_date"].dt.date <= as_of)
    ].copy()
    if relevant.empty:
        return relevant
    if mode == "historical":
        decision_end = datetime.combine(decision_date + timedelta(days=1), datetime.min.time())
        decision_end = decision_end.replace(tzinfo=UTC)
        late = relevant.loc[
            (relevant["latest_retrieved_at"] >= decision_end)
            | (relevant["verified_at"] >= decision_end)
        ]
        if not late.empty:
            first = late.iloc[0]
            raise DataQualityError(
                "历史回放禁止使用决策时点之后才取得或验证的增量："
                f"{pd.Timestamp(first['trade_date']).date().isoformat()}"
            )
    return relevant.sort_values("trade_date").reset_index(drop=True)


def _normalise_stock_day(
    raw: pd.DataFrame,
    *,
    trade_day: date,
    source_id: str,
    decision_date: date,
    mode: Literal["live", "historical"],
) -> pd.DataFrame:
    frame = _require_frame(raw, _STOCK_COLUMNS, label=f"{trade_day.isoformat()}股票增量")
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
    if bool(frame["symbol"].isna().any()) or bool(frame["symbol"].duplicated().any()):
        raise DataQualityError(f"{trade_day.isoformat()}股票增量含无效或重复代码")
    _validate_daily_values(frame, trade_day=trade_day, label="股票")
    _validate_source_and_retrieval(
        frame,
        source_id=source_id,
        trade_day=trade_day,
        decision_date=decision_date,
        mode=mode,
    )
    return frame.loc[:, list(_STOCK_COLUMNS)].sort_values("symbol").reset_index(drop=True)


def _normalise_index_day(
    raw: pd.DataFrame,
    *,
    trade_day: date,
    source_id: str,
    core_index_codes: tuple[str, ...],
    decision_date: date,
    mode: Literal["live", "historical"],
) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise DataQualityError(f"{trade_day.isoformat()}指数增量不是DataFrame")
    frame = raw.copy()
    code_column = "index_code" if "index_code" in frame else "symbol"
    required = {*_INDEX_PRICE_COLUMNS, code_column, "source", "retrieved_at"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataQualityError(
            f"{trade_day.isoformat()}指数增量缺列：" + "、".join(sorted(missing))
        )
    frame["index_code"] = _canonical_provider_index_codes(
        frame[code_column].astype(str).str.strip().tolist(),
        source_id=source_id,
        label=f"{trade_day.isoformat()}指数增量",
    )
    if bool(frame["index_code"].duplicated().any()):
        raise DataQualityError(f"{trade_day.isoformat()}指数增量含无效或重复代码")
    expected = set(core_index_codes)
    actual = set(frame["index_code"])
    if actual != expected:
        raise DataQualityError(
            f"{trade_day.isoformat()}六核心指数不完整：实际{len(actual)}个，要求{len(expected)}个"
        )
    _validate_daily_values(frame, trade_day=trade_day, label="指数")
    _validate_source_and_retrieval(
        frame,
        source_id=source_id,
        trade_day=trade_day,
        decision_date=decision_date,
        mode=mode,
    )
    return frame.sort_values("index_code").reset_index(drop=True)


def _canonical_provider_index_codes(
    raw_codes: list[str],
    *,
    source_id: str,
    label: str,
) -> list[str]:
    aliases = _PROVIDER_INDEX_ALIASES.get(source_id.strip().lower(), {})
    canonical: list[str] = []
    for raw_code in raw_codes:
        code = str(raw_code).strip()
        if len(code) != 6 or not code.isdigit():
            raise DataQualityError(f"{label}含无效指数代码：{code or '空值'}")
        canonical.append(aliases.get(code, code))
    if len(canonical) != len(set(canonical)):
        raise DataQualityError(f"{label}在供应商别名映射后发生指数代码碰撞")
    return canonical


def _require_frame(raw: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise DataQualityError(f"{label}不是DataFrame")
    missing = set(columns).difference(raw.columns)
    if missing:
        raise DataQualityError(f"{label}缺列：" + "、".join(sorted(missing)))
    if raw.empty:
        raise DataQualityError(f"{label}为空")
    return raw.copy()


def _validate_daily_values(frame: pd.DataFrame, *, trade_day: date, label: str) -> None:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if bool(dates.isna().any()) or set(dates.dt.date) != {trade_day}:
        raise DataQualityError(f"{trade_day.isoformat()}{label}文件混入其他交易日")
    frame["trade_date"] = dates
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if (
            bool(values.isna().any())
            or not bool(np.isfinite(values).all())
            or bool((values <= 0).any())
        ):
            raise DataQualityError(f"{trade_day.isoformat()}{label}{column}含无效值")
        frame[column] = values.astype(float)
    if bool((frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any()):
        raise DataQualityError(f"{trade_day.isoformat()}{label}最高价关系无效")
    if bool((frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any()):
        raise DataQualityError(f"{trade_day.isoformat()}{label}最低价关系无效")


def _validate_source_and_retrieval(
    frame: pd.DataFrame,
    *,
    source_id: str,
    trade_day: date,
    decision_date: date,
    mode: Literal["live", "historical"],
) -> None:
    source = frame["source"].astype(str).str.strip().str.lower()
    normalized_source = source_id.strip().lower()
    source_matches = source.eq(normalized_source) | source.str.startswith(f"{normalized_source}:")
    if bool((source == "").any()) or not bool(source_matches.all()):
        raise DataQualityError(f"{trade_day.isoformat()}增量来源与manifest不一致")
    retrieved = pd.to_datetime(frame["retrieved_at"], errors="coerce", utc=True)
    if bool(retrieved.isna().any()):
        raise DataQualityError(f"{trade_day.isoformat()}增量缺少有效retrieved_at")
    if mode == "historical":
        decision_end = datetime.combine(decision_date + timedelta(days=1), datetime.min.time())
        decision_end = decision_end.replace(tzinfo=UTC)
        if bool((retrieved >= decision_end).any()):
            raise DataQualityError("历史回放禁止使用决策时点之后取得的overlay行")


def _append_stock_histories(
    baseline: dict[str, pd.DataFrame],
    stock_days: list[pd.DataFrame],
    *,
    overlay_dates: tuple[date, ...],
    baseline_cutoff: date,
) -> dict[str, pd.DataFrame]:
    by_day = [frame.set_index("symbol", drop=False) for frame in stock_days]
    merged: dict[str, pd.DataFrame] = {}
    for symbol, raw_history in baseline.items():
        history = raw_history.copy()
        history_dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
        if bool(history_dates.isna().any()) or bool(history_dates.duplicated().any()):
            raise DataQualityError(f"CSMAR历史{symbol}含无效或重复交易日")
        if history_dates.dt.date.max() != baseline_cutoff:
            raise DataQualityError(f"CSMAR历史{symbol}未到共同基线截止日")
        rows = [frame.loc[[symbol], list(_STOCK_COLUMNS)] for frame in by_day]
        appended = pd.concat([history, *rows], ignore_index=True, sort=False)
        final_dates = pd.to_datetime(appended["trade_date"], errors="coerce").dt.normalize()
        if bool(final_dates.duplicated().any()):
            raise DataQualityError(f"overlay试图覆盖{symbol}已有交易日")
        actual_overlay_dates = tuple(final_dates.dt.date.iloc[-len(overlay_dates) :])
        if actual_overlay_dates != overlay_dates:
            raise DataQualityError(f"{symbol}增量交易日链不一致")
        appended["trade_date"] = final_dates
        merged[symbol] = appended.sort_values("trade_date").reset_index(drop=True)
    return merged


def _append_index_histories(
    baseline: dict[str, pd.DataFrame],
    index_days: list[pd.DataFrame],
    *,
    overlay_dates: tuple[date, ...],
    baseline_cutoff: date,
    common_cutoff: date,
    core_index_codes: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    if set(baseline) != set(core_index_codes):
        raise DataQualityError("CSMAR参考库与配置的六核心指数集合不一致")
    merged: dict[str, pd.DataFrame] = {}
    for code in core_index_codes:
        history = baseline[code].copy()
        history_dates = pd.to_datetime(history["trade_date"], errors="coerce").dt.normalize()
        if bool(history_dates.isna().any()) or bool(history_dates.duplicated().any()):
            raise DataQualityError(f"CSMAR指数{code}含无效或重复交易日")
        if history_dates.dt.date.max() != baseline_cutoff:
            raise DataQualityError(f"CSMAR指数{code}未到共同基线截止日")
        additions = []
        for trade_day, frame in zip(overlay_dates, index_days, strict=True):
            raw = frame.loc[frame["index_code"] == code].iloc[0]
            prev_close = pd.to_numeric(pd.Series([raw.get("prev_close")]), errors="coerce").iloc[0]
            index_return = (
                float(raw["close"]) / float(prev_close) - 1.0
                if np.isfinite(prev_close) and float(prev_close) > 0.0
                else np.nan
            )
            additions.append(
                {
                    "index_code": code,
                    "trade_date": pd.Timestamp(trade_day),
                    "open": float(raw["open"]),
                    "high": float(raw["high"]),
                    "low": float(raw["low"]),
                    "close": float(raw["close"]),
                    "component_volume": raw.get("component_volume", raw.get("volume_shares")),
                    "component_amount_cny": raw.get("component_amount_cny", raw.get("amount_cny")),
                    "index_return": index_return,
                    "knowledge_date": pd.Timestamp(raw["retrieved_at"]).date(),
                    "data_role": "verified_daily_overlay",
                    "historical_backtest_eligible": True,
                    "common_cutoff_date": common_cutoff,
                    "source": raw["source"],
                    "retrieved_at": raw["retrieved_at"],
                }
            )
        appended = pd.concat([history, pd.DataFrame(additions)], ignore_index=True, sort=False)
        final_dates = pd.to_datetime(appended["trade_date"], errors="coerce").dt.normalize()
        if bool(final_dates.duplicated().any()):
            raise DataQualityError(f"overlay试图覆盖指数{code}已有交易日")
        actual_overlay_dates = tuple(final_dates.dt.date.iloc[-len(overlay_dates) :])
        if actual_overlay_dates != overlay_dates:
            raise DataQualityError(f"指数{code}增量交易日链不一致")
        appended["trade_date"] = final_dates
        appended["common_cutoff_date"] = common_cutoff
        merged[code] = appended.sort_values("trade_date").reset_index(drop=True)
    return merged


def _refresh_cutoff_metadata(
    histories: dict[str, pd.DataFrame],
    baseline: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    amounts: dict[str, float] = {}
    output: dict[str, dict[str, object]] = {}
    for symbol, history in histories.items():
        item = dict(baseline[symbol])
        amount = pd.to_numeric(history["amount_cny"], errors="coerce")
        median_amount = float(amount.tail(20).median())
        if not np.isfinite(median_amount):
            raise DataQualityError(f"{symbol}最新20日成交额中位数不可用")
        amounts[symbol] = median_amount
        item["median_amount_20d_cny"] = median_amount
        item["is_limit_up_at_cutoff"] = _closed_at_formation_limit(
            history,
            item.get("board", ""),
        )
        output[symbol] = item
    scores = pd.Series(amounts, dtype=float).rank(method="average", pct=True)
    for symbol in output:
        output[symbol]["liquidity_score"] = float(scores.loc[symbol])
    return output


def _required_live_cutoff(decision_date: date) -> date:
    weekday = decision_date.weekday()
    if weekday == 0:
        return decision_date - timedelta(days=3)
    if weekday == 6:
        return decision_date - timedelta(days=2)
    return decision_date - timedelta(days=1)


def _enforce_live_freshness(
    cutoff: date,
    *,
    decision_date: date,
    mode: Literal["live", "historical"],
) -> None:
    if mode != "live":
        return
    required = _required_live_cutoff(decision_date)
    if cutoff < required:
        raise DataUnavailableError(
            "当前研究的股票与六核心指数共同截止日仍然过旧："
            f"决策日{decision_date.isoformat()}至少需要{required.isoformat()}，"
            f"混合数据实际只有{cutoff.isoformat()}。"
        )
