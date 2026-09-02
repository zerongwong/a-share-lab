from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from ashare_lab.services.chart_publisher_settings import (
    CHART_PUBLISHER_SETTINGS_VERSION,
    CLOUDFLARE_R2_PUBLISHER_ID,
    DEFAULT_R2_OBJECT_PREFIX,
    DEFAULT_SIGNED_URL_TTL_SECONDS,
    ChartPublisherSettings,
    build_configured_chart_publisher,
    delete_chart_publisher_settings,
    load_chart_publisher_settings,
    save_chart_publisher_settings,
)


def _settings() -> ChartPublisherSettings:
    return ChartPublisherSettings(
        account_id="0123456789abcdef0123456789abcdef",
        bucket_name="private-holding-charts",
        private_bucket_verified=True,
        lifecycle_delete_after_days=1,
        lifecycle_rule_verified=True,
    )


def test_round_trip_is_private_and_contains_no_credentials(tmp_path: Path) -> None:
    path = tmp_path / "private" / "chart-publisher.json"

    saved = save_chart_publisher_settings(_settings(), path)
    loaded = load_chart_publisher_settings(path)

    assert saved == path.absolute()
    assert loaded == _settings()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "account_id": "0123456789abcdef0123456789abcdef",
        "bucket_name": "private-holding-charts",
        "object_prefix": DEFAULT_R2_OBJECT_PREFIX,
        "publisher_id": CLOUDFLARE_R2_PUBLISHER_ID,
        "private_bucket_verified": True,
        "lifecycle_delete_after_days": 1,
        "lifecycle_rule_verified": True,
        "settings_version": CHART_PUBLISHER_SETTINGS_VERSION,
        "signed_url_ttl_seconds": DEFAULT_SIGNED_URL_TTL_SECONDS,
    }
    serialized = path.read_text(encoding="utf-8").lower()
    assert "access_key" not in serialized
    assert "secret" not in serialized
    assert "image_url" not in serialized
    assert "signature" not in serialized
    assert "https://" not in serialized


def test_missing_and_delete_are_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "chart-publisher.json"

    assert load_chart_publisher_settings(path) is None
    delete_chart_publisher_settings(path)
    save_chart_publisher_settings(_settings(), path)
    delete_chart_publisher_settings(path)
    delete_chart_publisher_settings(path)

    assert load_chart_publisher_settings(path) is None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("account_id", "not-an-account-id"),
        ("bucket_name", "contains/a/slash"),
        ("object_prefix", "../holding-charts"),
        ("signed_url_ttl_seconds", 60),
        ("signed_url_ttl_seconds", 3_601),
        ("lifecycle_delete_after_days", 2),
        ("publisher_id", "public_image_host"),
    ),
)
def test_invalid_or_unsafe_coordinates_fail_closed(field: str, value: object) -> None:
    values = {
        "account_id": "0123456789abcdef0123456789abcdef",
        "bucket_name": "private-holding-charts",
        "object_prefix": DEFAULT_R2_OBJECT_PREFIX,
        "signed_url_ttl_seconds": DEFAULT_SIGNED_URL_TTL_SECONDS,
        "publisher_id": CLOUDFLARE_R2_PUBLISHER_ID,
        "private_bucket_verified": True,
        "lifecycle_delete_after_days": 1,
        "lifecycle_rule_verified": True,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError)):
        ChartPublisherSettings(**values)


def test_unknown_file_fields_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "chart-publisher.json"
    document = {
        "account_id": "0123456789abcdef0123456789abcdef",
        "bucket_name": "private-holding-charts",
        "object_prefix": DEFAULT_R2_OBJECT_PREFIX,
        "publisher_id": CLOUDFLARE_R2_PUBLISHER_ID,
        "settings_version": CHART_PUBLISHER_SETTINGS_VERSION,
        "signed_url_ttl_seconds": DEFAULT_SIGNED_URL_TTL_SECONDS,
        "secret_access_key": "must-not-be-accepted",
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="未知字段"):
        load_chart_publisher_settings(path)


def test_symlink_paths_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "chart-publisher.json"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="普通文件"):
        load_chart_publisher_settings(symlink)
    with pytest.raises(ValueError, match="符号链接"):
        save_chart_publisher_settings(_settings(), symlink)


def test_factory_constructs_only_after_settings_and_both_keys_exist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chart-publisher.json"
    save_chart_publisher_settings(_settings(), path)
    constructed: list[tuple[object, object]] = []
    sentinel = object()

    def factory(config, credentials):
        constructed.append((config, credentials))
        return sentinel

    result = build_configured_chart_publisher(
        path,
        access_key_loader=lambda: "r2-access-key-id",
        secret_key_loader=lambda: "r2-secret-access-key",
        publisher_factory=factory,
    )

    assert result is sentinel
    assert len(constructed) == 1
    config, credentials = constructed[0]
    assert config.enabled is True
    assert config.account_id == _settings().account_id
    assert config.bucket == _settings().bucket_name
    assert config.presign_ttl_seconds == DEFAULT_SIGNED_URL_TTL_SECONDS
    assert credentials.access_key_id == "r2-access-key-id"
    assert credentials.secret_access_key == "r2-secret-access-key"


def test_factory_remains_disabled_for_missing_settings_or_one_key(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"
    key_calls: list[str] = []

    assert (
        build_configured_chart_publisher(
            missing_path,
            access_key_loader=lambda: key_calls.append("access") or "access-key",
            secret_key_loader=lambda: key_calls.append("secret") or "secret-key",
        )
        is None
    )
    assert key_calls == []

    path = tmp_path / "chart-publisher.json"
    save_chart_publisher_settings(_settings(), path)
    publisher_calls: list[object] = []
    assert (
        build_configured_chart_publisher(
            path,
            access_key_loader=lambda: "r2-access-key-id",
            secret_key_loader=lambda: None,
            publisher_factory=lambda *_args: publisher_calls.append(object()),
        )
        is None
    )
    assert publisher_calls == []


def test_factory_requires_explicit_private_bucket_and_one_day_lifecycle_confirmation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chart-publisher.json"
    for settings in (
        ChartPublisherSettings(
            account_id="0123456789abcdef0123456789abcdef",
            bucket_name="private-holding-charts",
            private_bucket_verified=False,
            lifecycle_rule_verified=True,
        ),
        ChartPublisherSettings(
            account_id="0123456789abcdef0123456789abcdef",
            bucket_name="private-holding-charts",
            private_bucket_verified=True,
            lifecycle_rule_verified=False,
        ),
    ):
        save_chart_publisher_settings(settings, path)
        assert (
            build_configured_chart_publisher(
                path,
                access_key_loader=lambda: "r2-access-key-id",
                secret_key_loader=lambda: "r2-secret-access-key",
                publisher_factory=lambda *_args: object(),
            )
            is None
        )
