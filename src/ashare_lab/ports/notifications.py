"""Provider-neutral notification contracts for local research alerts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class NotificationUrgency(StrEnum):
    NORMAL = "normal"
    TIME_SENSITIVE = "time_sensitive"


@dataclass(frozen=True)
class NotificationMessage:
    title: str
    body: str
    urgency: NotificationUrgency = NotificationUrgency.NORMAL
    group: str = "A股研究室"

    def __post_init__(self) -> None:
        title = self.title.strip()
        body = self.body.strip()
        group = self.group.strip()
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
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "group", group)


@dataclass(frozen=True)
class NotificationReceipt:
    channel: str
    accepted: bool


class NotificationChannel(Protocol):
    channel_name: str

    def send(self, message: NotificationMessage) -> NotificationReceipt: ...
