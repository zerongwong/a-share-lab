"""Provider-neutral notification contracts for local research alerts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

MAX_COMPACT_NOTIFICATION_BODY_BYTES = 2_400


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

    def __post_init__(self) -> None:
        title = self.title.strip()
        body = self.body.strip()
        group = self.group.strip()
        compact_body = None if self.compact_body is None else self.compact_body.strip()
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
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "compact_body", compact_body)


@dataclass(frozen=True)
class NotificationReceipt:
    channel: str
    accepted: bool
    provider_status: str | None = None
    provider_receipt_id: str | None = None


class NotificationChannel(Protocol):
    channel_name: str

    def send(self, message: NotificationMessage) -> NotificationReceipt: ...
