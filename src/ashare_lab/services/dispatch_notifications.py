"""Best-effort fan-out across independently configured notification channels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import NotificationChannel, NotificationMessage


@dataclass(frozen=True)
class NotificationDispatchReport:
    successful_channels: tuple[str, ...]
    failed_channels: tuple[str, ...]

    @property
    def any_succeeded(self) -> bool:
        return bool(self.successful_channels)


def dispatch_notification(
    channels: Iterable[NotificationChannel],
    message: NotificationMessage,
) -> NotificationDispatchReport:
    """Try every channel so one provider outage never suppresses the other."""

    successful: list[str] = []
    failed: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        name = channel.channel_name
        if name in seen:
            raise ValueError(f"重复通知通道：{name}")
        seen.add(name)
        try:
            receipt = channel.send(message)
        except NotificationDeliveryError:
            failed.append(name)
            continue
        if receipt.accepted:
            successful.append(name)
        else:
            failed.append(name)
    return NotificationDispatchReport(tuple(successful), tuple(failed))
