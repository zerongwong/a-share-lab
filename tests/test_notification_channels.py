from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from ashare_lab.adapters.notification_channels import (
    BarkNotificationChannel,
    ServerChanNotificationChannel,
    normalize_bark_device_key,
    normalize_serverchan_sendkey,
)
from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import (
    MAX_COMPACT_NOTIFICATION_BODY_BYTES,
    NotificationMessage,
    NotificationReceipt,
    NotificationUrgency,
)
from ashare_lab.services.dispatch_notifications import dispatch_notification


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _message(urgency: NotificationUrgency = NotificationUrgency.NORMAL):
    return NotificationMessage("行动单", "今日全部持有。", urgency=urgency)


def test_serverchan_posts_expected_fields_and_accepts_success() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={"code": 0, "message": "SUCCESS", "data": {"pushid": "one"}},
        )

    channel = ServerChanNotificationChannel(
        "SCTabcdefgh12345678",
        client=_client(handler),
    )
    receipt = channel.send(_message())

    assert receipt == NotificationReceipt(
        channel="serverchan",
        accepted=True,
        provider_status="provider_accepted",
        provider_receipt_id=hashlib.sha256(b"one").hexdigest()[:16],
    )
    assert "one" not in repr(receipt)
    assert str(seen["url"]).endswith("/SCTabcdefgh12345678.send")
    assert "title=" in str(seen["body"])
    assert "desp=" in str(seen["body"])


def test_serverchan_never_reveals_sendkey_in_errors() -> None:
    secret = "SCTsupersecret123456"
    channel = ServerChanNotificationChannel(
        secret,
        client=_client(lambda _: httpx.Response(403, json={"key": secret})),
    )

    with pytest.raises(NotificationDeliveryError) as exc_info:
        channel.send(_message())
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "document",
    [
        {"code": 1, "message": "quota"},
        {"code": "0", "message": "SUCCESS"},
        {"code": True, "message": "SUCCESS"},
        {"code": 0},
        {"code": 0, "message": 0},
        ["not", "an", "object"],
    ],
)
def test_serverchan_requires_the_official_success_contract(document: object) -> None:
    channel = ServerChanNotificationChannel(
        "SCTabcdefgh12345678",
        client=_client(lambda _: httpx.Response(200, json=document)),
    )

    with pytest.raises(NotificationDeliveryError, match="未接受"):
        channel.send(_message())


@pytest.mark.parametrize("title", ["第一行\n第二行", "第一行\r第二行", "长" * 33])
def test_serverchan_rejects_unsafe_titles_before_network(title: str) -> None:
    channel = ServerChanNotificationChannel(
        "SCTabcdefgh12345678",
        client=_client(lambda _: (_ for _ in ()).throw(AssertionError("must not send"))),
    )

    with pytest.raises(ValueError, match="标题"):
        channel.send(NotificationMessage(title, "正文"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("SCTabcdefgh12345678", "SCTabcdefgh12345678"),
        (
            "https://sctapi.ftqq.com/SCTabcdefgh12345678",
            "SCTabcdefgh12345678",
        ),
        (
            "https://sctapi.ftqq.com/SCTabcdefgh12345678.send",
            "SCTabcdefgh12345678",
        ),
    ],
)
def test_serverchan_accepts_sendkey_or_official_url(value: str, expected: str) -> None:
    assert normalize_serverchan_sendkey(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://sctapi.ftqq.com/SCTabcdefgh12345678",
        "https://example.com/SCTabcdefgh12345678",
        "https://user:pass@sctapi.ftqq.com/SCTabcdefgh12345678",
        "https://sctapi.ftqq.com/",
        "https://sctapi.ftqq.com/one/two",
    ],
)
def test_serverchan_rejects_unsafe_or_invalid_urls(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_serverchan_sendkey(value)


def test_bark_accepts_official_url_and_sends_time_sensitive_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.read())
        return httpx.Response(200, json={"code": 200, "message": "success"})

    channel = BarkNotificationChannel(
        "https://api.day.app/barkDevice123/测试/消息",
        client=_client(handler),
    )
    receipt = channel.send(_message(NotificationUrgency.TIME_SENSITIVE))

    assert receipt == NotificationReceipt(
        channel="bark",
        accepted=True,
        provider_status="provider_accepted",
    )
    assert seen["url"] == "https://api.day.app/push"
    assert seen["payload"] == {
        "device_key": "barkDevice123",
        "title": "行动单",
        "body": "今日全部持有。",
        "group": "A股研究室",
        "level": "timeSensitive",
    }


@pytest.mark.parametrize(
    "value",
    [
        "http://api.day.app/not-secure",
        "https://example.com/device-key",
        "https://user:pass@api.day.app/device-key",
        "https://api.day.app/",
        "spaces are invalid",
    ],
)
def test_bark_rejects_unsafe_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_bark_device_key(value)


def test_bark_never_reveals_device_key_in_errors() -> None:
    secret = "barkSecretDevice123"
    channel = BarkNotificationChannel(
        secret,
        client=_client(lambda _: httpx.Response(500, json={"device_key": secret})),
    )

    with pytest.raises(NotificationDeliveryError) as exc_info:
        channel.send(_message())
    assert secret not in str(exc_info.value)


def test_dispatcher_keeps_second_channel_when_first_fails() -> None:
    class Channel:
        def __init__(self, name: str, fails: bool) -> None:
            self.channel_name = name
            self.fails = fails

        def send(self, _message: NotificationMessage) -> NotificationReceipt:
            if self.fails:
                raise NotificationDeliveryError("sanitized")
            return NotificationReceipt(channel=self.channel_name, accepted=True)

    report = dispatch_notification(
        [Channel("wechat", True), Channel("iphone", False)],
        _message(),
    )

    assert report.successful_channels == ("iphone",)
    assert report.failed_channels == ("wechat",)
    assert report.any_succeeded


def test_message_rejects_empty_and_overlong_content() -> None:
    with pytest.raises(ValueError, match="标题"):
        NotificationMessage(" ", "正文")
    with pytest.raises(ValueError, match="8000"):
        NotificationMessage("标题", "x" * 8_001)
    with pytest.raises(ValueError, match="紧凑通知正文不能为空"):
        NotificationMessage("标题", "正文", compact_body=" ")
    with pytest.raises(ValueError, match="UTF-8字节"):
        NotificationMessage(
            "标题",
            "正文",
            compact_body="中" * (MAX_COMPACT_NOTIFICATION_BODY_BYTES // 3 + 1),
        )
