"""Private local configuration for an optional holding-chart publisher.

This module stores only non-secret Cloudflare R2 coordinates.  Access keys
belong in macOS Keychain, while signed URLs and uploaded object identifiers are
runtime evidence and must never enter this settings file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from ashare_lab.bootstrap import application_data_dir

CHART_PUBLISHER_SETTINGS_VERSION: Final = "chart-publisher-settings-v0.2.0"
CLOUDFLARE_R2_PUBLISHER_ID: Final = "cloudflare_r2"
DEFAULT_R2_OBJECT_PREFIX: Final = "holding-charts"
DEFAULT_SIGNED_URL_TTL_SECONDS: Final = 3_600

_ACCOUNT_ID = re.compile(r"^[0-9a-fA-F]{32}$")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_MIN_SIGNED_URL_TTL_SECONDS = 300
_MAX_SIGNED_URL_TTL_SECONDS = 86_400


@dataclass(frozen=True, slots=True)
class ChartPublisherSettings:
    """Non-secret coordinates for one explicitly selected chart publisher."""

    account_id: str
    bucket_name: str
    publisher_id: str = CLOUDFLARE_R2_PUBLISHER_ID
    object_prefix: str = DEFAULT_R2_OBJECT_PREFIX
    signed_url_ttl_seconds: int = DEFAULT_SIGNED_URL_TTL_SECONDS
    private_bucket_verified: bool = False
    lifecycle_delete_after_days: int = 1
    lifecycle_rule_verified: bool = False
    settings_version: str = CHART_PUBLISHER_SETTINGS_VERSION

    def __post_init__(self) -> None:
        publisher_id = _nonblank(self.publisher_id, field="publisher_id")
        if publisher_id != CLOUDFLARE_R2_PUBLISHER_ID:
            raise ValueError("当前只支持Cloudflare R2图片发布配置")
        account_id = _nonblank(self.account_id, field="account_id").lower()
        if _ACCOUNT_ID.fullmatch(account_id) is None:
            raise ValueError("Cloudflare Account ID应为32位十六进制字符")
        bucket_name = _bucket_name(self.bucket_name)
        object_prefix = _object_prefix(self.object_prefix)
        if object_prefix != DEFAULT_R2_OBJECT_PREFIX:
            raise ValueError("R2对象目录固定为holding-charts")
        ttl = self.signed_url_ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int):
            raise TypeError("签名地址有效期必须是整数秒")
        if not _MIN_SIGNED_URL_TTL_SECONDS <= ttl <= DEFAULT_SIGNED_URL_TTL_SECONDS:
            raise ValueError("持仓图签名地址有效期必须在5分钟至1小时之间")
        if not isinstance(self.private_bucket_verified, bool):
            raise TypeError("private_bucket_verified must be a boolean")
        if (
            isinstance(self.lifecycle_delete_after_days, bool)
            or not isinstance(self.lifecycle_delete_after_days, int)
            or self.lifecycle_delete_after_days != 1
        ):
            raise ValueError("持仓图R2对象生命周期必须固定为1日删除")
        if not isinstance(self.lifecycle_rule_verified, bool):
            raise TypeError("lifecycle_rule_verified must be a boolean")
        if self.settings_version != CHART_PUBLISHER_SETTINGS_VERSION:
            raise ValueError("图片发布配置版本不受支持")
        object.__setattr__(self, "publisher_id", publisher_id)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "bucket_name", bucket_name)
        object.__setattr__(self, "object_prefix", object_prefix)


def chart_publisher_settings_path() -> Path:
    """Return the private application-data path without exposing a credential."""

    return application_data_dir() / "settings" / "chart-publisher.json"


def load_chart_publisher_settings(
    path: str | Path | None = None,
) -> ChartPublisherSettings | None:
    """Load one non-secret configuration; absence means publishing is disabled."""

    target = _resolved_path(path)
    if not target.exists():
        return None
    if target.is_symlink() or not target.is_file():
        raise ValueError("图片发布配置路径必须是本机普通文件")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("图片发布配置无法读取") from exc
    if not isinstance(document, dict):
        raise ValueError("图片发布配置格式无效")
    allowed = {
        "account_id",
        "bucket_name",
        "publisher_id",
        "object_prefix",
        "signed_url_ttl_seconds",
        "private_bucket_verified",
        "lifecycle_delete_after_days",
        "lifecycle_rule_verified",
        "settings_version",
    }
    if set(document) != allowed:
        raise ValueError("图片发布配置字段不完整或包含未知字段")
    return ChartPublisherSettings(**document)


def save_chart_publisher_settings(
    settings: ChartPublisherSettings,
    path: str | Path | None = None,
) -> Path:
    """Atomically save only non-secret coordinates with private file modes."""

    if not isinstance(settings, ChartPublisherSettings):
        raise TypeError("settings must be ChartPublisherSettings")
    target = _resolved_path(path)
    parent = target.parent
    if parent.is_symlink():
        raise ValueError("图片发布配置目录不能是符号链接")
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError("图片发布配置目录无效")
    parent.chmod(0o700)
    if target.is_symlink():
        raise ValueError("图片发布配置文件不能是符号链接")
    payload = json.dumps(
        asdict(settings),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        target.chmod(0o600)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()
    return target


def delete_chart_publisher_settings(path: str | Path | None = None) -> None:
    """Idempotently disable publisher coordinates without touching Keychain."""

    target = _resolved_path(path)
    if not target.exists():
        return
    if target.is_symlink() or not target.is_file():
        raise ValueError("图片发布配置路径必须是本机普通文件")
    target.unlink()


def build_configured_chart_publisher(
    path: str | Path | None = None,
    *,
    access_key_loader: Callable[[], str | None] | None = None,
    secret_key_loader: Callable[[], str | None] | None = None,
    publisher_factory: Callable[[object, object], object] | None = None,
) -> object | None:
    """Construct a configured R2 publisher without making a network request.

    This factory is deliberately all-or-nothing.  Missing local coordinates or
    either Keychain item keeps image publishing disabled.  Construction alone
    does not upload, test a bucket, mint a signed URL, or authorize a channel.
    """

    from ashare_lab.adapters.cloudflare_r2_image_publisher import (
        CloudflareR2PrivateImagePublisher,
        R2Credentials,
        R2PublisherConfig,
    )
    from ashare_lab.adapters.macos_keychain import (
        load_cloudflare_r2_access_key_id,
        load_cloudflare_r2_secret_access_key,
    )

    settings = load_chart_publisher_settings(path)
    if settings is None:
        return None
    if not settings.private_bucket_verified or not settings.lifecycle_rule_verified:
        return None
    load_access_key = access_key_loader or load_cloudflare_r2_access_key_id
    load_secret_key = secret_key_loader or load_cloudflare_r2_secret_access_key
    access_key_id = load_access_key()
    secret_access_key = load_secret_key()
    if not access_key_id or not secret_access_key:
        return None
    config = R2PublisherConfig(
        enabled=True,
        account_id=settings.account_id,
        bucket=settings.bucket_name,
        presign_ttl_seconds=settings.signed_url_ttl_seconds,
    )
    credentials = R2Credentials(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
    factory = publisher_factory or CloudflareR2PrivateImagePublisher
    return factory(config, credentials)


def _resolved_path(path: str | Path | None) -> Path:
    target = chart_publisher_settings_path() if path is None else Path(path).expanduser()
    return target.absolute()


def _nonblank(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def _bucket_name(value: object) -> str:
    normalized = _nonblank(value, field="bucket_name").lower()
    if (
        _BUCKET_NAME.fullmatch(normalized) is None
        or ".." in normalized
        or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", normalized)
    ):
        raise ValueError("R2存储桶名称格式无效")
    return normalized


def _object_prefix(value: object) -> str:
    normalized = _nonblank(value, field="object_prefix").strip("/")
    if (
        not normalized
        or len(normalized) > 255
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or any(character in normalized for character in ("\r", "\n", "\x00"))
    ):
        raise ValueError("R2对象目录格式无效")
    return normalized
