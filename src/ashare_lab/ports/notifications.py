"""Provider-neutral notification contracts for local research alerts."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

MAX_COMPACT_NOTIFICATION_BODY_BYTES = 2_400
_IMAGE_DELIVERY_CHANNELS = frozenset({"serverchan", "bark"})


class NotificationUrgency(StrEnum):
    NORMAL = "normal"
    TIME_SENSITIVE = "time_sensitive"


@dataclass(frozen=True)
class NotificationMessage:
    title: str
    body: str
    urgency: NotificationUrgency = NotificationUrgency.NORMAL
    group: str = "A股研究室"
    compact_body: str | None = None
    image_url: str | None = field(default=None, repr=False)
    image_authorized_channels: frozenset[str] = field(
        default_factory=frozenset,
        repr=False,
    )
    holding_authorization_guard: Callable[[str], bool] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    image_authorization_guard: Callable[[str], bool] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    unauthorized_body: str | None = field(default=None, repr=False, compare=False)
    unauthorized_compact_body: str | None = field(default=None, repr=False, compare=False)
    image_revoke_callback: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        title = self.title.strip()
        body = self.body.strip()
        group = self.group.strip()
        compact_body = None if self.compact_body is None else self.compact_body.strip()
        image_url = None if self.image_url is None else _normalize_image_url(self.image_url)
        image_authorized_channels = _normalize_image_authorized_channels(
            self.image_authorized_channels
        )
        if not title:
            raise ValueError("通知标题不能为空")
        if not body:
            raise ValueError("通知正文不能为空")
        if not group:
            raise ValueError("通知分组不能为空")
        if len(title) > 100:
            raise ValueError("通知标题不能超过100个字符")
        if len(body) > 8_000:
            raise ValueError("通知正文不能超过8000个字符")
        if len(group) > 64:
            raise ValueError("通知分组不能超过64个字符")
        if self.compact_body is not None and not compact_body:
            raise ValueError("紧凑通知正文不能为空")
        if (
            compact_body is not None
            and len(compact_body.encode("utf-8")) > MAX_COMPACT_NOTIFICATION_BODY_BYTES
        ):
            raise ValueError(
                f"紧凑通知正文不能超过{MAX_COMPACT_NOTIFICATION_BODY_BYTES}个UTF-8字节"
            )
        if image_url is not None and not image_authorized_channels:
            raise ValueError("图表图片没有明确授权的通知通道")
        if image_url is None and image_authorized_channels:
            raise ValueError("没有图表图片时不能设置图片通知通道")
        for guard in (self.holding_authorization_guard, self.image_authorization_guard):
            if guard is not None and not callable(guard):
                raise TypeError("通知授权复核器必须可调用")
        if self.image_revoke_callback is not None and not callable(self.image_revoke_callback):
            raise TypeError("图片撤销回调必须可调用")
        unauthorized_body = (
            None if self.unauthorized_body is None else self.unauthorized_body.strip()
        )
        unauthorized_compact_body = (
            None
            if self.unauthorized_compact_body is None
            else self.unauthorized_compact_body.strip()
        )
        if self.holding_authorization_guard is not None and not unauthorized_body:
            raise ValueError("持仓披露复核必须提供无持仓回退正文")
        if self.unauthorized_body is not None and not unauthorized_body:
            raise ValueError("无持仓回退正文不能为空")
        if self.unauthorized_compact_body is not None and not unauthorized_compact_body:
            raise ValueError("无持仓紧凑回退正文不能为空")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "compact_body", compact_body)
        object.__setattr__(self, "image_url", image_url)
        object.__setattr__(self, "image_authorized_channels", image_authorized_channels)
        object.__setattr__(self, "unauthorized_body", unauthorized_body)
        object.__setattr__(self, "unauthorized_compact_body", unauthorized_compact_body)


def _normalize_image_authorized_channels(value: object) -> frozenset[str]:
    if isinstance(value, (str, bytes, bool)) or not isinstance(
        value,
        (set, frozenset, list, tuple),
    ):
        raise ValueError("图片通知通道必须是serverchan/bark明确允许列表")
    normalized: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("图片通知通道必须只包含提供商名称")
        channel = item.strip().lower()
        if channel not in _IMAGE_DELIVERY_CHANNELS:
            raise ValueError("图片通知通道只支持serverchan/bark")
        normalized.add(channel)
    return frozenset(normalized)


def prepare_notification_for_channel(
    message: NotificationMessage,
    *,
    channel_name: str,
) -> NotificationMessage | None:
    """Recheck runtime disclosure grants immediately before one provider call.

    A failed or unreadable grant never blocks the public research-plan fallback,
    but it strips both the holding details and private image URL.  The callbacks
    stay attached so the provider adapter can repeat the same check at the last
    possible point before its HTTPS submission.
    """

    if not isinstance(message, NotificationMessage):
        raise TypeError("message must be NotificationMessage")
    if channel_name not in _IMAGE_DELIVERY_CHANNELS:
        raise ValueError("通知通道只支持serverchan/bark")

    holding_allowed = True
    if message.holding_authorization_guard is not None:
        try:
            holding_allowed = message.holding_authorization_guard(channel_name) is True
        except Exception:  # noqa: BLE001 - an authorization read must fail closed
            holding_allowed = False
    if not holding_allowed:
        _best_effort_revoke_image(message, channel_name=channel_name)
        if message.unauthorized_body is None:
            return None
        fallback_body = (
            message.unauthorized_compact_body
            if channel_name == "bark" and message.unauthorized_compact_body is not None
            else message.unauthorized_body
        )
        return replace(
            message,
            body=fallback_body,
            compact_body=(None if channel_name == "bark" else message.unauthorized_compact_body),
            image_url=None,
            image_authorized_channels=frozenset(),
        )

    if message.image_url is None:
        return message
    if channel_name not in message.image_authorized_channels:
        return replace(
            message,
            image_url=None,
            image_authorized_channels=frozenset(),
        )
    image_allowed = True
    if message.image_authorization_guard is not None:
        try:
            image_allowed = message.image_authorization_guard(channel_name) is True
        except Exception:  # noqa: BLE001 - an authorization read must fail closed
            image_allowed = False
    if image_allowed:
        return message
    _best_effort_revoke_image(message, channel_name=channel_name)
    return replace(
        message,
        image_url=None,
        image_authorized_channels=frozenset(),
    )


def _best_effort_revoke_image(message: NotificationMessage, *, channel_name: str) -> None:
    if (
        channel_name not in message.image_authorized_channels
        or message.image_revoke_callback is None
    ):
        return
    try:
        message.image_revoke_callback()
    except Exception:  # noqa: BLE001 - lifecycle cleanup cannot disclose its error
        return


def _normalize_image_url(value: str) -> str:
    """Return one public HTTPS image URL without exposing it in validation errors."""

    if not isinstance(value, str):
        raise ValueError("图表图片地址必须是HTTPS URL")
    if "\r" in value or "\n" in value:
        raise ValueError("图表图片地址必须是HTTPS URL")
    normalized = value.strip()
    if not normalized or "<" in normalized or ">" in normalized:
        raise ValueError("图表图片地址必须是HTTPS URL")
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except ValueError:
        raise ValueError("图表图片地址必须是HTTPS URL") from None
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("图表图片地址必须是HTTPS URL")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        raise ValueError("图表图片地址必须是HTTPS URL")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("图表图片地址必须是HTTPS URL")
    return normalized


@dataclass(frozen=True)
class NotificationReceipt:
    channel: str
    accepted: bool
    provider_status: str | None = None
    provider_receipt_id: str | None = None
    image_accepted: bool = False


class NotificationChannel(Protocol):
    channel_name: str

    def send(self, message: NotificationMessage) -> NotificationReceipt: ...


class PrivateImagePublishReceipt(Protocol):
    """Opaque private-publication evidence; secret fields must stay runtime-only."""

    provider_id: str
    expires_at: datetime
    image_url: str
    revoke_key: str


class PrivateImagePublisher(Protocol):
    """Provider-neutral private PNG publisher used by the delivery orchestrator."""

    provider_id: str

    def publish_png(self, payload: bytes) -> PrivateImagePublishReceipt: ...

    def revoke(self, revoke_key: str) -> object: ...

    def close(self) -> None: ...
