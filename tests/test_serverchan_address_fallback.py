from __future__ import annotations

import socket

import httpx
import pytest

from ashare_lab.adapters import notification_channels as channels
from ashare_lab.domain.errors import NotificationDeliveryError
from ashare_lab.ports.notifications import NotificationMessage


def _record(address):
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))


def _client(handler, resolver):
    return httpx.Client(
        transport=channels._ServerChanAddressFallbackTransport(
            transport=httpx.MockTransport(handler), resolver=resolver
        ),
        follow_redirects=False,
    )


def _message():
    return NotificationMessage("合成测试", "此消息仅在模拟通道内测试。")


def test_connect_failure_tries_dynamic_addresses_with_original_host_and_tls_name():
    seen, resolutions = [], []

    def handler(request):
        seen.append(request)
        if request.url.host in {"sctapi.ftqq.com", "198.51.100.10"}:
            raise httpx.ConnectError("synthetic TLS connection failed")
        return httpx.Response(200, json={"code": 0, "message": "SUCCESS"})

    def resolver(host, port, **kwargs):
        resolutions.append((host, port, kwargs))
        return [_record("198.51.100.10"), _record("198.51.100.10"), _record("2001:db8::20")]

    with _client(handler, resolver) as client:
        result = channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client).send(
            _message()
        )
    assert result.accepted is True
    assert [request.url.host for request in seen] == [
        "sctapi.ftqq.com",
        "198.51.100.10",
        "2001:db8::20",
    ]
    assert resolutions == [
        ("sctapi.ftqq.com", 443, {"type": socket.SOCK_STREAM, "proto": socket.IPPROTO_TCP})
    ]
    for request in seen[1:]:
        assert request.headers["host"] == "sctapi.ftqq.com"
        assert request.extensions["sni_hostname"] == "sctapi.ftqq.com"
        assert request.extensions["timeout"] == seen[0].extensions["timeout"]
        assert request.method == "POST"
        assert request.url.scheme == "https"
        assert request.url.raw_path == seen[0].url.raw_path
        assert request.read() == seen[0].read()


def test_successful_initial_connection_does_not_resolve_or_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"code": 0, "message": "SUCCESS"})

    with _client(handler, lambda *_args, **_kwargs: pytest.fail("no fallback needed")) as client:
        assert (
            channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client)
            .send(_message())
            .accepted
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ReadTimeout,
        httpx.ReadError,
        httpx.WriteTimeout,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    ],
)
def test_ambiguous_post_submission_failure_is_never_retried(exception):
    calls = []

    def handler(request):
        calls.append(request)
        raise exception("SCTsynthetic-secret-must-stay-private")

    with _client(handler, lambda *_args, **_kwargs: pytest.fail("must not retry")) as client:
        channel = channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client)
        with pytest.raises(NotificationDeliveryError) as caught:
            channel.send(_message())
    assert len(calls) == 1
    assert "SCTsynthetic-secret" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("status", [302, 403, 429, 500])
def test_http_response_is_not_retried_or_redirected(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://other.example/"})

    with (
        _client(handler, lambda *_args, **_kwargs: pytest.fail("HTTP response forbids retry")) as client,
        pytest.raises(NotificationDeliveryError),
    ):
        channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client).send(_message())
    assert len(calls) == 1


def test_alternate_read_timeout_stops_before_trying_a_second_address():
    hosts = []

    def handler(request):
        hosts.append(request.url.host)
        if request.url.host == "sctapi.ftqq.com":
            raise httpx.ConnectTimeout("synthetic connect timeout")
        raise httpx.ReadTimeout("provider may already have accepted POST")

    with (
        _client(
            handler,
            lambda *_args, **_kwargs: [_record("198.51.100.10"), _record("198.51.100.20")],
        ) as client,
        pytest.raises(NotificationDeliveryError),
    ):
        channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client).send(_message())
    assert hosts == ["sctapi.ftqq.com", "198.51.100.10"]


def test_dns_failure_is_sanitized_by_existing_notification_boundary():
    def resolver(*_args, **_kwargs):
        raise socket.gaierror("unsafe provider diagnostics")

    def handler(_request):
        raise httpx.ConnectError("https://sctapi.ftqq.com/SCTprivate-diagnostic.send")

    with _client(handler, resolver) as client, pytest.raises(NotificationDeliveryError) as caught:
        channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client).send(_message())
    assert "SCTprivate" not in str(caught.value)
    assert "diagnostics" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_fallback_is_limited_to_original_official_https_host():
    def handler(_request):
        raise httpx.ConnectError("synthetic failure")

    with (
        _client(handler, lambda *_args, **_kwargs: pytest.fail("wrong host must not resolve")) as client,
        pytest.raises(NotificationDeliveryError),
    ):
        channels.ServerChanNotificationChannel(
            "SCTsynthetic123456", client=client, base_url="https://other.example"
        ).send(_message())


def test_default_client_keeps_tls_verification_no_redirects_and_zero_post_retries(monkeypatch):
    transport_options = []

    def transport(**kwargs):
        transport_options.append(kwargs)
        return httpx.MockTransport(lambda _request: pytest.fail("construction must not send"))

    monkeypatch.setattr(channels.httpx, "HTTPTransport", transport)
    with channels.ServerChanNotificationChannel("SCTsynthetic123456") as channel:
        assert channel._client.follow_redirects is False
        assert isinstance(channel._client._transport, channels._ServerChanAddressFallbackTransport)
    assert transport_options == [{"verify": True, "retries": 0, "trust_env": False}]


def test_injected_client_is_not_replaced_or_closed():
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    ) as client:
        channel = channels.ServerChanNotificationChannel("SCTsynthetic123456", client=client)
        assert channel._client is client
        channel.close()
        assert client.is_closed is False
