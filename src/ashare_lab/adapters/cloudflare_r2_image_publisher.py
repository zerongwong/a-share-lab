"""Private Cloudflare R2 image publishing with bounded signed URLs.

This adapter is deliberately independent from notification and holding code.
It accepts only PNG bytes, writes one private object through Cloudflare's R2
S3 endpoint, and returns a short-lived SigV4 GET URL.  Credentials are supplied
by the caller (normally from macOS Keychain); this module never reads files,
environment variables, or command-line arguments for secrets.

URL expiry is not object expiry.  This adapter deletes only through explicit
``revoke``; the local settings/acceptance layer must require an independent R2
bucket lifecycle rule for routine deletion of objects that are never revoked.

The adapter does not log.  Transport failures are converted to fixed messages
without exception chaining so credential-bearing authorization material and
signed URLs cannot be copied into application logs by this boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final
from urllib.parse import quote, urlsplit

import httpx

R2_PROVIDER_ID: Final = "cloudflare_r2"
R2_OBJECT_PREFIX: Final = "holding-charts"
R2_MIN_PRESIGN_TTL_SECONDS: Final = 300
R2_DEFAULT_PRESIGN_TTL_SECONDS: Final = 3_600
R2_MAX_PRESIGN_TTL_SECONDS: Final = 86_400
R2_MAX_PNG_BYTES: Final = 20 * 1024 * 1024

_R2_REGION: Final = "auto"
_R2_SERVICE: Final = "s3"
_R2_REQUEST_KIND: Final = "aws4_request"
_R2_ALGORITHM: Final = "AWS4-HMAC-SHA256"
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_ACCOUNT_ID = re.compile(r"^[0-9a-f]{32}$")
_BUCKET_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_ACCESS_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{8,256}$")
_REVOKE_KEY = re.compile(rf"^{R2_OBJECT_PREFIX}/[0-9a-f]{{32}}\.png$")


class R2ImagePublisherError(RuntimeError):
    """A sanitized configuration, signing, or transport failure."""


@dataclass(frozen=True, slots=True)
class R2PublisherConfig:
    """Non-secret R2 settings supplied by a separate local settings service.

    ``enabled`` is false by default.  When enabled, the endpoint is either
    derived from the account id or must bind to that exact Cloudflare R2 host.
    Custom domains, IP literals, ports, paths, and redirects are not accepted.
    """

    enabled: bool = False
    account_id: str | None = None
    bucket: str | None = None
    endpoint: str | None = field(default=None, repr=False)
    presign_ttl_seconds: int = R2_DEFAULT_PRESIGN_TTL_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise R2ImagePublisherError("R2图片发布配置无效。")
        if isinstance(self.presign_ttl_seconds, bool) or not isinstance(
            self.presign_ttl_seconds, int
        ):
            raise R2ImagePublisherError("R2图片发布配置无效。")
        if not (
            R2_MIN_PRESIGN_TTL_SECONDS <= self.presign_ttl_seconds <= R2_MAX_PRESIGN_TTL_SECONDS
        ):
            raise R2ImagePublisherError("R2图片发布配置无效。")

        # The default disabled value intentionally needs no account details.
        if not self.enabled and self.account_id is None and self.bucket is None:
            if self.endpoint is not None:
                raise R2ImagePublisherError("R2图片发布配置无效。")
            return

        account_id = _normalize_account_id(self.account_id)
        bucket = _normalize_bucket(self.bucket)
        endpoint = _normalize_endpoint(self.endpoint, account_id=account_id)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "endpoint", endpoint)


@dataclass(frozen=True, slots=True)
class R2Credentials:
    """R2 S3 API credentials; both fields are deliberately absent from repr."""

    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)

    def __post_init__(self) -> None:
        access_key_id = _secret_text(self.access_key_id)
        secret_access_key = _secret_text(self.secret_access_key)
        if not _ACCESS_KEY_ID.fullmatch(access_key_id) or not 8 <= len(secret_access_key) <= 512:
            raise R2ImagePublisherError("R2访问凭据无效。")
        object.__setattr__(self, "access_key_id", access_key_id)
        object.__setattr__(self, "secret_access_key", secret_access_key)


@dataclass(frozen=True, slots=True)
class R2ImagePublishReceipt:
    """One private publication; ``expires_at`` covers the URL, not object retention."""

    provider_id: str
    expires_at: datetime
    image_url: str = field(repr=False)
    revoke_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.provider_id != R2_PROVIDER_ID:
            raise R2ImagePublisherError("R2图片发布回执无效。")
        _require_aware_utc(self.expires_at, error="R2图片发布回执无效。")
        if not isinstance(self.image_url, str) or not self.image_url.startswith("https://"):
            raise R2ImagePublisherError("R2图片发布回执无效。")
        if not isinstance(self.revoke_key, str) or not _REVOKE_KEY.fullmatch(self.revoke_key):
            raise R2ImagePublisherError("R2图片发布回执无效。")


@dataclass(frozen=True, slots=True)
class R2ImageRevokeReceipt:
    provider_id: str
    revoked: bool

    def __post_init__(self) -> None:
        if self.provider_id != R2_PROVIDER_ID or not isinstance(self.revoked, bool):
            raise R2ImagePublisherError("R2图片撤销回执无效。")


class CloudflareR2PrivateImagePublisher:
    """Upload private PNGs and mint short-lived Cloudflare R2 GET URLs."""

    provider_id: Final = R2_PROVIDER_ID

    def __init__(
        self,
        config: R2PublisherConfig,
        credentials: R2Credentials,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
        nonce_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(config, R2PublisherConfig) or not config.enabled:
            raise R2ImagePublisherError("R2私有图片发布器未启用。")
        if not isinstance(credentials, R2Credentials):
            raise R2ImagePublisherError("R2访问凭据无效。")
        if config.account_id is None or config.bucket is None or config.endpoint is None:
            raise R2ImagePublisherError("R2图片发布配置无效。")

        self._config = config
        self._credentials = credentials
        self._endpoint = config.endpoint
        self._host = urlsplit(config.endpoint).hostname or ""
        self._clock = clock or (lambda: datetime.now(UTC))
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(16))
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(15.0),
            follow_redirects=False,
        )
        self._owns_client = client is None

    def __enter__(self) -> CloudflareR2PrivateImagePublisher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def publish_png(self, payload: bytes) -> R2ImagePublishReceipt:
        """Upload one PNG under a random, non-semantic private object key."""

        image = _validated_png(payload)
        now = self._now()
        object_key = self._new_object_key()
        path = self._object_path(object_key)
        payload_hash = hashlib.sha256(image).hexdigest()
        headers = self._signed_headers(
            method="PUT",
            canonical_path=path,
            payload_hash=payload_hash,
            now=now,
            extra_headers={"content-type": "image/png"},
        )
        self._request("PUT", path, content=image, headers=headers)

        image_url = self._presigned_get_url(path, now=now)
        return R2ImagePublishReceipt(
            provider_id=R2_PROVIDER_ID,
            expires_at=now + timedelta(seconds=self._config.presign_ttl_seconds),
            image_url=image_url,
            revoke_key=object_key,
        )

    def revoke(self, revoke_key: str) -> R2ImageRevokeReceipt:
        """Delete exactly one object previously identified by an opaque handle."""

        if not isinstance(revoke_key, str) or not _REVOKE_KEY.fullmatch(revoke_key):
            raise R2ImagePublisherError("R2图片撤销凭据无效。")
        now = self._now()
        path = self._object_path(revoke_key)
        empty_hash = hashlib.sha256(b"").hexdigest()
        headers = self._signed_headers(
            method="DELETE",
            canonical_path=path,
            payload_hash=empty_hash,
            now=now,
        )
        self._request("DELETE", path, content=None, headers=headers, missing_is_success=True)
        return R2ImageRevokeReceipt(provider_id=R2_PROVIDER_ID, revoked=True)

    def _now(self) -> datetime:
        try:
            value = self._clock()
        except Exception:
            raise R2ImagePublisherError("R2图片发布时钟不可用。") from None
        return _require_aware_utc(value, error="R2图片发布时钟不可用。")

    def _new_object_key(self) -> str:
        try:
            nonce = self._nonce_factory()
        except Exception:
            raise R2ImagePublisherError("R2图片随机对象名生成失败。") from None
        if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
            raise R2ImagePublisherError("R2图片随机对象名生成失败。")
        return f"{R2_OBJECT_PREFIX}/{nonce}.png"

    def _object_path(self, object_key: str) -> str:
        assert self._config.bucket is not None
        return quote(f"/{self._config.bucket}/{object_key}", safe="/-_.~")

    def _request(
        self,
        method: str,
        path: str,
        *,
        content: bytes | None,
        headers: dict[str, str],
        missing_is_success: bool = False,
    ) -> None:
        try:
            response = self._client.request(
                method,
                f"{self._endpoint}{path}",
                content=content,
                headers=headers,
            )
        except Exception:
            raise R2ImagePublisherError("R2私有图片请求失败。") from None
        if 200 <= response.status_code < 300:
            return
        if missing_is_success and response.status_code == 404:
            return
        raise R2ImagePublisherError("R2私有图片请求未被服务接受。")

    def _signed_headers(
        self,
        *,
        method: str,
        canonical_path: str,
        payload_hash: str,
        now: datetime,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        headers = {
            "host": self._host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            **(extra_headers or {}),
        }
        canonical_headers, signed_headers = _canonical_headers(headers)
        canonical_request = (
            f"{method}\n{canonical_path}\n\n{canonical_headers}\n\n{signed_headers}\n{payload_hash}"
        )
        scope = _credential_scope(now)
        string_to_sign = _string_to_sign(amz_date, scope, canonical_request)
        signature = self._signature(now, string_to_sign)
        authorization = (
            f"{_R2_ALGORITHM} "
            f"Credential={self._credentials.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return {
            **headers,
            "authorization": authorization,
        }

    def _presigned_get_url(self, canonical_path: str, *, now: datetime) -> str:
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        scope = _credential_scope(now)
        parameters = {
            "X-Amz-Algorithm": _R2_ALGORITHM,
            "X-Amz-Credential": f"{self._credentials.access_key_id}/{scope}",
            "X-Amz-Date": amz_date,
            "X-Amz-Expires": str(self._config.presign_ttl_seconds),
            "X-Amz-SignedHeaders": "host",
        }
        canonical_query = _canonical_query(parameters)
        canonical_request = (
            f"GET\n{canonical_path}\n{canonical_query}\nhost:{self._host}\n\nhost\nUNSIGNED-PAYLOAD"
        )
        string_to_sign = _string_to_sign(amz_date, scope, canonical_request)
        parameters["X-Amz-Signature"] = self._signature(now, string_to_sign)
        return f"{self._endpoint}{canonical_path}?{_canonical_query(parameters)}"

    def _signature(self, now: datetime, string_to_sign: str) -> str:
        date_key = _hmac_sha256(
            f"AWS4{self._credentials.secret_access_key}".encode(),
            now.strftime("%Y%m%d"),
        )
        region_key = _hmac_sha256(date_key, _R2_REGION)
        service_key = _hmac_sha256(region_key, _R2_SERVICE)
        signing_key = _hmac_sha256(service_key, _R2_REQUEST_KIND)
        return hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def _normalize_account_id(value: str | None) -> str:
    if not isinstance(value, str):
        raise R2ImagePublisherError("R2图片发布配置无效。")
    normalized = value.strip().lower()
    if not _ACCOUNT_ID.fullmatch(normalized):
        raise R2ImagePublisherError("R2图片发布配置无效。")
    return normalized


def _normalize_bucket(value: str | None) -> str:
    if not isinstance(value, str):
        raise R2ImagePublisherError("R2图片发布配置无效。")
    normalized = value.strip()
    if (
        not _BUCKET_NAME.fullmatch(normalized)
        or ".." in normalized
        or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", normalized)
    ):
        raise R2ImagePublisherError("R2图片发布配置无效。")
    return normalized


def _normalize_endpoint(value: str | None, *, account_id: str) -> str:
    expected_host = f"{account_id}.r2.cloudflarestorage.com"
    if value is None:
        return f"https://{expected_host}"
    if not isinstance(value, str) or "\r" in value or "\n" in value:
        raise R2ImagePublisherError("R2图片发布配置无效。")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError:
        raise R2ImagePublisherError("R2图片发布配置无效。") from None
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.rstrip(".").lower() != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise R2ImagePublisherError("R2图片发布配置无效。")
    return f"https://{expected_host}"


def _secret_text(value: str) -> str:
    if not isinstance(value, str):
        raise R2ImagePublisherError("R2访问凭据无效。")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 512
        or any(ord(character) < 33 or ord(character) > 126 for character in normalized)
    ):
        raise R2ImagePublisherError("R2访问凭据无效。")
    return normalized


def _validated_png(payload: bytes) -> bytes:
    if not isinstance(payload, bytes):
        raise R2ImagePublisherError("R2图片内容必须是PNG字节。")
    if not payload.startswith(_PNG_SIGNATURE) or not len(payload) <= R2_MAX_PNG_BYTES:
        raise R2ImagePublisherError("R2图片内容必须是有效且大小受限的PNG。")
    return payload


def _require_aware_utc(value: datetime, *, error: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise R2ImagePublisherError(error)
    return value.astimezone(UTC)


def _credential_scope(now: datetime) -> str:
    return f"{now.strftime('%Y%m%d')}/{_R2_REGION}/{_R2_SERVICE}/{_R2_REQUEST_KIND}"


def _canonical_headers(headers: dict[str, str]) -> tuple[str, str]:
    normalized = {
        name.strip().lower(): " ".join(str(value).strip().split())
        for name, value in headers.items()
    }
    names = tuple(sorted(normalized))
    return "\n".join(f"{name}:{normalized[name]}" for name in names), ";".join(names)


def _canonical_query(parameters: dict[str, str]) -> str:
    encoded = [
        (quote(str(name), safe="-_.~"), quote(str(value), safe="-_.~"))
        for name, value in parameters.items()
    ]
    return "&".join(f"{name}={value}" for name, value in sorted(encoded))


def _string_to_sign(amz_date: str, scope: str, canonical_request: str) -> str:
    request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    return f"{_R2_ALGORITHM}\n{amz_date}\n{scope}\n{request_hash}"


def _hmac_sha256(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


__all__ = (
    "CloudflareR2PrivateImagePublisher",
    "R2Credentials",
    "R2ImagePublishReceipt",
    "R2ImagePublisherError",
    "R2ImageRevokeReceipt",
    "R2PublisherConfig",
    "R2_DEFAULT_PRESIGN_TTL_SECONDS",
    "R2_MAX_PRESIGN_TTL_SECONDS",
    "R2_MIN_PRESIGN_TTL_SECONDS",
    "R2_OBJECT_PREFIX",
    "R2_PROVIDER_ID",
)
