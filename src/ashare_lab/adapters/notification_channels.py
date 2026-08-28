"""HTTPS notification adapters for ServerChan Turbo and Bark.

Credentials are supplied by the caller from a local secret store. Provider
responses and transport exceptions are deliberately not copied into user-facing
errors because either may contain a credential-bearing request URL.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import (
    NotificationMessage,
    NotificationReceipt,
    NotificationUrgency,
)

_SERVERCHAN_KEY = re.compile(r"^SCT[A-Za-z0-9_-]{8,192}$")
_BARK_DEVICE_KEY = re.compile(r"^[A-Za-z0-9_-]{8,256}$")


class ServerChanNotificationChannel:
    """Send one-way personal WeChat notifications through ServerChan Turbo."""

    channel_name = "serverchan"

    def __init__(
        self,
        sendkey: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://sctapi.ftqq.com",
    ) -> None:
        normalized = normalize_serverchan_sendkey(sendkey)
        if not base_url.startswith("https://"):
            raise ValueError("Server酱地址必须使用HTTPS")
        self._sendkey = normalized
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0))
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def send(self, message: NotificationMessage) -> NotificationReceipt:
        if "\n" in message.title or "\r" in message.title:
            raise ValueError("Server酱标题不能包含换行符")
        if len(message.title) > 32:
            raise ValueError("Server酱标题不能超过32个字符")
        # ServerChan's documented endpoint embeds the SendKey in the request
        # path. Never log the request URL or include provider text in errors.
        try:
            response = self._client.post(
                f"{self._base_url}/{self._sendkey}.send",
                data={"title": message.title, "desp": message.body},
            )
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError):
            # The original httpx exception may include the credential-bearing
            # request URL.  Suppress chaining so background tracebacks cannot
            # leak the SendKey into logs.
            raise NotificationDeliveryError("Server酱通知发送失败，请检查网络和SendKey。") from None

        if (
            not isinstance(document, dict)
            or type(document.get("code")) is not int
            or document.get("code") != 0
            or not isinstance(document.get("message"), str)
        ):
            raise NotificationDeliveryError("Server酱未接受通知，请检查通道状态和额度。")
        # ``code == 0`` confirms provider ingestion only.  ServerChan's
        # downstream WeChat/Bark channel may still fail after this response.
        return NotificationReceipt(
            channel=self.channel_name,
            accepted=True,
            provider_status="provider_accepted",
            provider_receipt_id=_serverchan_receipt_id(document.get("data")),
        )


class BarkNotificationChannel:
    """Send native iPhone notifications through Bark's HTTPS API v2."""

    channel_name = "bark"

    def __init__(
        self,
        device_key_or_url: str,
        *,
        client: httpx.Client | None = None,
        base_url: str = "https://api.day.app",
    ) -> None:
        self._device_key = normalize_bark_device_key(device_key_or_url)
        if not base_url.startswith("https://"):
            raise ValueError("Bark地址必须使用HTTPS")
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(15.0))
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def send(self, message: NotificationMessage) -> NotificationReceipt:
        payload: dict[str, Any] = {
            "device_key": self._device_key,
            "title": message.title,
            "body": message.body,
            "group": message.group,
            "level": (
                "timeSensitive"
                if message.urgency is NotificationUrgency.TIME_SENSITIVE
                else "active"
            ),
        }
        try:
            response = self._client.post(f"{self._base_url}/push", json=payload)
            response.raise_for_status()
            document = response.json()
        except (httpx.HTTPError, ValueError):
            raise NotificationDeliveryError("Bark通知发送失败，请检查网络和设备Key。") from None

        if not isinstance(document, dict) or document.get("code") != 200:
            raise NotificationDeliveryError("Bark未接受通知，请检查App注册状态。")
        return NotificationReceipt(
            channel=self.channel_name,
            accepted=True,
            provider_status="provider_accepted",
        )


def _serverchan_receipt_id(data: object) -> str | None:
    """Return an irreversible identifier without retaining provider response data."""

    if not isinstance(data, dict):
        return None
    pushid = data.get("pushid")
    if not isinstance(pushid, (str, int)) or isinstance(pushid, bool):
        return None
    normalized = str(pushid).strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def normalize_bark_device_key(value: str) -> str:
    """Accept the Bark device key or its standard api.day.app push URL."""

    normalized = value.strip()
    if normalized.startswith("https://"):
        parsed = urlparse(normalized)
        if parsed.hostname != "api.day.app" or parsed.username or parsed.password:
            raise ValueError("只接受Bark官方api.day.app推送地址")
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            raise ValueError("Bark推送地址缺少设备Key")
        normalized = segments[0]
    if not _BARK_DEVICE_KEY.fullmatch(normalized):
        raise ValueError("Bark设备Key格式无效")
    return normalized


def normalize_serverchan_sendkey(value: str) -> str:
    """Accept a Turbo SendKey or an official sctapi.ftqq.com endpoint URL."""

    normalized = value.strip()
    if normalized.startswith("https://"):
        parsed = urlparse(normalized)
        if (
            parsed.hostname != "sctapi.ftqq.com"
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise ValueError("只接受Server酱官方sctapi.ftqq.com地址")
        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) != 1:
            raise ValueError("Server酱推送地址缺少SendKey")
        normalized = segments[0]
        if normalized.endswith(".send"):
            normalized = normalized.removesuffix(".send")
    if not _SERVERCHAN_KEY.fullmatch(normalized):
        raise ValueError("Server酱 SendKey格式无效")
    return normalized
