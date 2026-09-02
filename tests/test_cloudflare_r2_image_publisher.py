from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from ashare_lab.adapters.cloudflare_r2_image_publisher import (
    R2_DEFAULT_PRESIGN_TTL_SECONDS,
    R2_MAX_PRESIGN_TTL_SECONDS,
    R2_MIN_PRESIGN_TTL_SECONDS,
    R2_OBJECT_PREFIX,
    R2_PROVIDER_ID,
    CloudflareR2PrivateImagePublisher,
    R2Credentials,
    R2ImagePublisherError,
    R2PublisherConfig,
)

ACCOUNT_ID = "a" * 32
ENDPOINT = f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET = "private-holding-charts"
ACCESS_KEY_ID = "R2ACCESSKEY123456"
SECRET_ACCESS_KEY = "r2-secret-value-that-must-never-appear"
NOW = datetime(2026, 9, 1, 13, 0, 0, tzinfo=UTC)
PNG = b"\x89PNG\r\n\x1a\n" + b"private-chart-content"


def _config(**changes: object) -> R2PublisherConfig:
    values: dict[str, object] = {
        "enabled": True,
        "account_id": ACCOUNT_ID,
        "bucket": BUCKET,
    }
    values.update(changes)
    return R2PublisherConfig(**values)  # type: ignore[arg-type]


def _credentials() -> R2Credentials:
    return R2Credentials(ACCESS_KEY_ID, SECRET_ACCESS_KEY)


def test_config_is_disabled_by_default_and_uses_one_hour_signed_urls() -> None:
    config = R2PublisherConfig()

    assert config.enabled is False
    assert config.account_id is None
    assert config.bucket is None
    assert config.endpoint is None
    assert config.presign_ttl_seconds == R2_DEFAULT_PRESIGN_TTL_SECONDS == 3_600


def test_enabled_config_derives_and_hides_the_strict_r2_endpoint() -> None:
    config = _config()

    assert config.endpoint == ENDPOINT
    assert ENDPOINT not in repr(config)
    assert config.presign_ttl_seconds == 3_600


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://" + ACCOUNT_ID + ".r2.cloudflarestorage.com",
        "https://evil.example",
        "https://" + ACCOUNT_ID + ".r2.cloudflarestorage.com.evil.example",
        "https://user:password@" + ACCOUNT_ID + ".r2.cloudflarestorage.com",
        ENDPOINT + ":443",
        ENDPOINT + "/bucket",
        ENDPOINT + "?token=secret",
        ENDPOINT + "#fragment",
        "https://127.0.0.1",
    ],
)
def test_config_rejects_every_endpoint_not_bound_to_the_exact_r2_host(endpoint: str) -> None:
    with pytest.raises(R2ImagePublisherError) as exc_info:
        _config(endpoint=endpoint)

    assert str(exc_info.value) == "R2图片发布配置无效。"
    assert endpoint not in str(exc_info.value)


@pytest.mark.parametrize(
    "ttl",
    [
        0,
        R2_MIN_PRESIGN_TTL_SECONDS - 1,
        R2_MAX_PRESIGN_TTL_SECONDS + 1,
        True,
        1.5,
    ],
)
def test_config_rejects_ttl_outside_five_minutes_to_twenty_four_hours(
    ttl: object,
) -> None:
    with pytest.raises(R2ImagePublisherError, match="^R2图片发布配置无效。$"):
        _config(presign_ttl_seconds=ttl)


def test_credentials_and_publish_receipt_repr_hide_every_secret_and_url() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: NOW,
            nonce_factory=lambda: "0" * 32,
        )
        receipt = publisher.publish_png(PNG)

    assert len(requests) == 1
    assert receipt.provider_id == R2_PROVIDER_ID == "cloudflare_r2"
    assert receipt.expires_at == NOW + timedelta(seconds=3_600)
    assert receipt.expires_at.tzinfo is UTC
    assert receipt.revoke_key == f"{R2_OBJECT_PREFIX}/{'0' * 32}.png"

    credentials_repr = repr(_credentials())
    receipt_repr = repr(receipt)
    assert ACCESS_KEY_ID not in credentials_repr
    assert SECRET_ACCESS_KEY not in credentials_repr
    assert receipt.image_url not in receipt_repr
    assert receipt.revoke_key not in receipt_repr
    assert "X-Amz-Signature" not in receipt_repr


def test_publish_uploads_one_private_png_under_a_random_nonsemantic_object_name() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    nonces = iter(("1" * 32, "2" * 32))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(presign_ttl_seconds=R2_MAX_PRESIGN_TTL_SECONDS),
            _credentials(),
            client=client,
            clock=lambda: NOW,
            nonce_factory=lambda: next(nonces),
        )
        first = publisher.publish_png(PNG)
        second = publisher.publish_png(PNG)

    assert first.revoke_key != second.revoke_key
    assert len(requests) == 2
    for request, receipt in zip(requests, (first, second), strict=True):
        assert request.method == "PUT"
        assert request.url.host == f"{ACCOUNT_ID}.r2.cloudflarestorage.com"
        assert request.url.path == f"/{BUCKET}/{receipt.revoke_key}"
        assert request.content == PNG
        assert request.headers["content-type"] == "image/png"
        assert request.headers["x-amz-content-sha256"]
        assert request.headers["x-amz-date"] == "20260901T130000Z"
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=")
        assert "public-read" not in request.headers.get("x-amz-acl", "")
        assert re.fullmatch(rf"{R2_OBJECT_PREFIX}/[0-9a-f]{{32}}\.png", receipt.revoke_key)
        assert not any(
            value in request.url.path
            for value in ("600919", "江苏银行", "holding_version", "2026-09-01")
        )

        signed = urlsplit(receipt.image_url)
        query = parse_qs(signed.query)
        assert signed.scheme == "https"
        assert signed.hostname == f"{ACCOUNT_ID}.r2.cloudflarestorage.com"
        assert signed.path == request.url.path
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert query["X-Amz-Expires"] == [str(R2_MAX_PRESIGN_TTL_SECONDS)]
        assert query["X-Amz-SignedHeaders"] == ["host"]
        assert re.fullmatch(r"[0-9a-f]{64}", query["X-Amz-Signature"][0])


def test_revoke_deletes_only_the_opaque_object_and_is_idempotent_on_missing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = 200 if request.method == "PUT" else 404
        return httpx.Response(status, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: NOW,
            nonce_factory=lambda: "3" * 32,
        )
        published = publisher.publish_png(PNG)
        revoked = publisher.revoke(published.revoke_key)

    assert [request.method for request in requests] == ["PUT", "DELETE"]
    assert requests[1].url.path == f"/{BUCKET}/{published.revoke_key}"
    assert requests[1].headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=")
    assert revoked.provider_id == R2_PROVIDER_ID
    assert revoked.revoked is True


@pytest.mark.parametrize(
    "revoke_key",
    [
        "",
        "holding-charts/../../other.png",
        "holding-charts/600919.png",
        "other/" + "4" * 32 + ".png",
        "holding-charts/" + "4" * 31 + ".png",
    ],
)
def test_revoke_rejects_nonopaque_or_out_of_scope_keys_without_request(revoke_key: str) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: NOW,
        )
        with pytest.raises(R2ImagePublisherError, match="^R2图片撤销凭据无效。$"):
            publisher.revoke(revoke_key)

    assert request_count == 0


def test_transport_failure_does_not_copy_secret_url_or_provider_exception() -> None:
    private_detail = ENDPOINT + "/private?X-Amz-Signature=do-not-log"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"{private_detail} {ACCESS_KEY_ID} {SECRET_ACCESS_KEY}",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: NOW,
            nonce_factory=lambda: "5" * 32,
        )
        with pytest.raises(R2ImagePublisherError) as exc_info:
            publisher.publish_png(PNG)

    message = str(exc_info.value)
    assert message == "R2私有图片请求失败。"
    assert private_detail not in message
    assert ACCESS_KEY_ID not in message
    assert SECRET_ACCESS_KEY not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not-a-png",
        bytearray(PNG),
    ],
)
def test_publish_rejects_non_png_without_network(payload: object) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: NOW,
        )
        with pytest.raises(R2ImagePublisherError):
            publisher.publish_png(payload)  # type: ignore[arg-type]

    assert request_count == 0


def test_publisher_refuses_disabled_configuration() -> None:
    with pytest.raises(R2ImagePublisherError, match="^R2私有图片发布器未启用。$"):
        CloudflareR2PrivateImagePublisher(R2PublisherConfig(), _credentials())


def test_clock_must_return_an_aware_datetime() -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
        publisher = CloudflareR2PrivateImagePublisher(
            _config(),
            _credentials(),
            client=client,
            clock=lambda: datetime(2026, 9, 1, 13, 0, 0),
        )
        with pytest.raises(R2ImagePublisherError, match="^R2图片发布时钟不可用。$"):
            publisher.publish_png(PNG)
