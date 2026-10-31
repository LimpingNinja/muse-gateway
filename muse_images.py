"""Image-input normalization shared by every gateway route.

Structured image parts (inline base64, remote URL, or an opt-in ``file://``
reference) are turned into checked image bytes exactly once, before any
provider translation, and rewritten to one canonical Responses shape:
``{"type": "input_image", "image_url": "data:<mime>;base64,<b64>"}``.

Every failure is a ``ProviderError`` raised before the request is forwarded
upstream (the model catalog may be consulted first to learn whether the
model accepts images). Error messages and log events never contain URLs,
paths, or image bytes.

Remote fetches use ``http.client`` with the TCP connection pinned to a
DNS-validated public address, so proxy environment variables are ignored:
operators behind a mandatory proxy cannot fetch image URLs.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import os
import socket
import ssl
import stat
import time
import urllib.parse
import urllib.request
from typing import Any

from muse_providers import ProviderError, _input_items


MAX_IMAGE_BYTES = 20 * 1024 * 1024
ROUTE_IMAGE_BYTES = {"claude": 5 * 1024 * 1024}
MAX_REQUEST_IMAGE_BYTES = 24 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 100
MAX_URL_LENGTH = 2048
MAX_IMAGE_REDIRECTS = 3
IMAGE_CONNECT_TIMEOUT = 5.0
IMAGE_READ_TIMEOUT = 20.0
IMAGE_FETCH_BUDGET = 60.0
READ_CHUNK = 65536
# Same value as MuseShimHandler.server_version (muse_gateway.py); not
# imported because muse_gateway imports this module.
USER_AGENT = "muse-gateway/0.5.0"

_IMAGE_TYPES = frozenset({"input_image", "image_url", "image"})
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_EMBEDDED_IPV4_NETWORKS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("::/96"))
_LOCAL_USE_NAT64 = ipaddress.ip_network("64:ff9b:1::/48")
_ROUTE_LABELS = {"codex": "Codex", "claude": "Claude", "meta": "Meta", "local": "Local",
                 "generic": "generic", "devin": "Devin"}
_MALFORMED = "malformed image part"
_NOT_IMAGE = "not a supported image (PNG, JPEG, GIF, WebP)"
_TOO_LARGE = "image input exceeds the per-image or per-request size limit"
_CANNOT_READ = "cannot read the referenced image file"
# Percent-encode only non-ASCII characters in the request target; control
# characters, spaces, and backslashes were already rejected.
_TARGET_SAFE = "".join(chr(code) for code in range(33, 127))


def _error(code: str, message: str, *, status: int = 400) -> ProviderError:
    return ProviderError(message, status=status, code=code)


def _is_image(part: Any) -> bool:
    return isinstance(part, dict) and str(part.get("type") or "").strip().lower() in _IMAGE_TYPES


def _scan(body: dict[str, Any]) -> list[tuple[int, str, int]]:
    """Locate image parts as (item index, field, part index); no I/O."""
    for key in ("instructions", "system"):
        value = body.get(key)
        if _is_image(value) or (isinstance(value, list) and any(_is_image(part) for part in value)):
            raise _error("image_invalid", "image parts are not accepted in system instructions")
    found: list[tuple[int, str, int]] = []
    for index, item in enumerate(_input_items(body)):
        if not isinstance(item, dict):
            continue
        field = "output" if item.get("type") == "function_call_output" else "content"
        parts = item.get(field)
        if _is_image(parts):
            raise _error("image_invalid", _MALFORMED)
        if not isinstance(parts, list):
            continue
        for position, part in enumerate(parts):
            if not _is_image(part):
                continue
            if field == "content" and item.get("role") not in ("user", "developer"):
                raise _error("image_invalid", "image parts are accepted only in user, developer, or tool-result content")
            found.append((index, field, position))
    return found


def _source_of(part: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    """Classify shapes A-D as (kind, reference, declared mime, detail)."""
    if "url" in part:
        raise _error("image_invalid", _MALFORMED)
    detail = part.get("detail")
    if str(part.get("type") or "").strip().lower() == "image":
        source = part.get("source")
        if not isinstance(source, dict):
            raise _error("image_invalid", _MALFORMED)
        if source.get("type") == "base64":
            media_type, data = source.get("media_type"), source.get("data")
            if not isinstance(media_type, str) or not isinstance(data, str) or not data:
                raise _error("image_invalid", _MALFORMED)
            return "base64", data, media_type.strip().lower(), detail if isinstance(detail, str) else None
        if source.get("type") != "url":
            raise _error("image_invalid", _MALFORMED)
        reference = source.get("url")
    else:
        reference = part.get("image_url")
        if isinstance(reference, dict):
            detail = reference.get("detail", detail)
            reference = reference.get("url")
    if not isinstance(reference, str) or not reference:
        raise _error("image_invalid", _MALFORMED)
    scheme = reference.split(":", 1)[0].lower() if ":" in reference else ""
    kinds = {"data": "data", "https": "url", "http": "url", "file": "file"}
    if scheme not in kinds:
        raise _error("image_invalid", _MALFORMED)
    return kinds[scheme], reference, None, detail if isinstance(detail, str) else None


def _sniff(raw: bytes) -> str:
    if len(raw) >= 12:
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    raise _error("image_invalid", _NOT_IMAGE)


def _budget(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _error("image_url_unreachable", "image loading exceeded its time budget", status=502)
    return remaining


def _read_capped(read: Any, cap: int, deadline: float) -> bytes:
    """Stream chunks, aborting the moment the running length passes ``cap``."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = read()
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > cap:
            raise _error("image_too_large", _TOO_LARGE)
        chunks.append(chunk)
        _budget(deadline)


def _load_data(payload: str, cap: int) -> bytes:
    estimate = len(payload) * 3 // 4 - (len(payload) - len(payload.rstrip("=")))
    if estimate > cap:
        raise _error("image_too_large", _TOO_LARGE)
    try:
        raw = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise _error("image_invalid", "image data is not valid base64") from None
    if len(raw) > cap:
        raise _error("image_too_large", _TOO_LARGE)
    return raw


def _check_address(address: str) -> str:
    """Return a connectable address only when it is globally routable."""
    rejected = _error("image_url_rejected", "image URL host is not a public address")
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        raise rejected from None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    checks: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if ip.version == 6:
        if ip in _LOCAL_USE_NAT64:
            raise rejected
        # NAT64 and IPv4-compatible forms carry an IPv4 target in the low 32 bits.
        if any(ip in network for network in _EMBEDDED_IPV4_NETWORKS):
            checks.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip.sixtofour is not None:
            checks.append(ip.sixtofour)
    if not all(checked.is_global for checked in checks):
        raise rejected
    return str(ip)


def _validate_url(url: str) -> tuple[str, str, int, str]:
    rejected = _error("image_url_rejected", "image URL is not allowed")
    if len(url) > MAX_URL_LENGTH or any(ord(c) <= 32 or c in "\x7f\\" for c in url):
        raise rejected
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
        username, password = parts.username, parts.password
    except ValueError:
        raise rejected from None
    scheme = parts.scheme.lower()
    if scheme not in ("https", "http") or username is not None or password is not None:
        raise rejected
    host = parts.hostname or ""
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise rejected from None
    if not host:
        raise rejected
    if port is None:
        port = 443 if scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise rejected
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return scheme, host, port, urllib.parse.quote(target, safe=_TARGET_SAFE)


def _resolve(host: str, port: int) -> str:
    # Resolution uses the system resolver's own timeout; IMAGE_FETCH_BUDGET
    # bounds connect/read time, not DNS (documented in README).
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise _error("image_url_unreachable", "image URL host could not be resolved", status=502) from None
    if not infos:
        raise _error("image_url_unreachable", "image URL host could not be resolved", status=502)
    # Every address must be public, not just the one we connect to.
    addresses = [_check_address(str(info[4][0])) for info in infos]
    return addresses[0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose TCP peer is a pre-validated address."""

    def __init__(self, host: str, port: int, address: str, *, connect_timeout: float,
                 read_timeout: float) -> None:
        super().__init__(host, port, timeout=read_timeout)
        self._address = address
        self._connect_timeout = connect_timeout

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), timeout=self._connect_timeout)
        self.sock.settimeout(self.timeout)


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    """TLS to the pinned address; SNI and certificate checks use the hostname."""

    def connect(self) -> None:
        super().connect()
        # Always verified, independent of --insecure-upstream.
        self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)


def _load_url(url: str, cap: int, deadline: float) -> bytes:
    current = url
    for _hop in range(MAX_IMAGE_REDIRECTS + 1):
        scheme, host, port, target = _validate_url(current)
        _budget(deadline)
        address = _resolve(host, port)
        remaining = _budget(deadline)
        connection_class = _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
        connection = connection_class(
            host, port, address,
            connect_timeout=min(IMAGE_CONNECT_TIMEOUT, remaining),
            read_timeout=min(IMAGE_READ_TIMEOUT, remaining),
        )
        try:
            connection.request("GET", target, headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
            response = connection.getresponse()
            location = response.getheader("Location")
            if response.status in _REDIRECT_STATUSES and location:
                current = urllib.parse.urljoin(current, location)
                continue
            if response.status != 200:
                raise _error("image_url_unreachable", f"image URL returned HTTP {response.status}", status=502)
            length = (response.getheader("Content-Length") or "").strip()
            if length.isdigit() and int(length) > cap:
                raise _error("image_too_large", _TOO_LARGE)
            return _read_capped(lambda: response.read(READ_CHUNK), cap, deadline)
        except (OSError, http.client.HTTPException, ValueError):
            raise _error("image_url_unreachable", "image URL could not be fetched", status=502) from None
        finally:
            connection.close()
    raise _error("image_url_rejected", "image URL redirected too many times")


def _load_file(reference: str, cap: int, deadline: float) -> bytes:
    rejected = _error("image_file_rejected", "image file reference is not allowed")
    try:
        parts = urllib.parse.urlsplit(reference)
    except ValueError:
        raise rejected from None
    if parts.scheme.lower() != "file" or parts.netloc != "" or parts.query or parts.fragment:
        raise rejected
    path = urllib.request.url2pathname(parts.path)
    if "\x00" in path or not os.path.isabs(path):
        raise rejected
    if os.path.realpath(path) != os.path.normpath(path):
        raise _error("image_file_rejected", "image file paths must not involve symbolic links")
    _budget(deadline)
    try:
        before = os.lstat(path)
    except OSError:
        raise _error("image_file_rejected", _CANNOT_READ) from None
    if not stat.S_ISREG(before.st_mode):
        raise _error("image_file_rejected", "image file reference is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise _error("image_file_rejected", _CANNOT_READ) from None
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise _error("image_file_rejected", "image file changed while it was being opened")
        if opened.st_size > cap:
            raise _error("image_too_large", _TOO_LARGE)
        return _read_capped(lambda: os.read(fd, READ_CHUNK), cap, deadline)
    except OSError:
        raise _error("image_file_rejected", _CANNOT_READ) from None
    finally:
        os.close(fd)


def _capability_error(route: str) -> ProviderError:
    if route == "claude-cli":
        return ProviderError(
            "Claude CLI transport supports text input only; use --claude-transport http for image input.",
            code="claude_cli_images_unsupported",
        )
    if route == "kiro":
        return ProviderError(
            "Kiro currently supports text input only; media and unsupported content blocks cannot be replayed.",
            code="kiro_unsupported_input",
        )
    label = _ROUTE_LABELS.get(route, route)
    return ProviderError(f"The selected {label} model does not accept image input.", code="image_input_unsupported")


def _safe_mime(value: str) -> str:
    """A declared media type is client text; log it only when it looks like one."""
    kind, slash, subtype = value.partition("/")
    if (
        slash
        and kind
        and subtype
        and "/" not in subtype
        and len(value) <= 100
        and kind[0].isascii() and kind[0].isalnum()
        and subtype[0].isascii() and subtype[0].isalnum()
        and all(c.isascii() and (c.isalnum() or c in "/.+-") for c in value)
    ):
        return value
    return "other"


def normalize_request_images(body: dict[str, Any], *, route: str, supports_images: bool | None,
                             allow_files: bool, notify: Any = None) -> tuple[dict[str, Any], dict[str, int]]:
    """Return ``(new_body, stats)`` with every image part loaded and checked.

    Copy-on-write: ``body`` is never mutated. Bodies without image parts are
    returned unchanged without any I/O.
    """
    stats = {"count": 0, "data": 0, "url": 0, "file": 0, "bytes": 0}
    found = _scan(body)
    if not found:
        return body, stats
    if route in ("claude-cli", "kiro"):
        supports_images = False
    if supports_images is False:
        raise _capability_error(route)
    if route == "devin" and any(field == "output" for _index, field, _position in found):
        raise ProviderError(
            "Devin cannot carry images inside tool results; send the image in a user message instead.",
            code="image_input_unsupported",
        )
    if len(found) > MAX_IMAGES_PER_REQUEST:
        raise _error("image_too_large", f"too many images in one request (limit {MAX_IMAGES_PER_REQUEST})")
    items = _input_items(body)
    sources = [_source_of(items[index][field][position]) for index, field, position in found]
    if not allow_files and any(kind == "file" for kind, _ref, _declared, _detail in sources):
        raise _error("image_file_rejected",
                     "local image files are disabled; start the gateway with --allow-image-files")
    per_image = ROUTE_IMAGE_BYTES.get(route, MAX_IMAGE_BYTES)
    deadline = time.monotonic() + IMAGE_FETCH_BUDGET
    rewritten: dict[tuple[int, str], list[Any]] = {}
    for (index, field, position), (kind, reference, declared, detail) in zip(found, sources):
        cap = min(per_image, MAX_REQUEST_IMAGE_BYTES - stats["bytes"])
        if kind == "base64":
            raw = _load_data(reference, cap)
            stats["data"] += 1
        elif kind == "data":
            marker = reference.find(";base64,")
            if marker < 0:
                raise _error("image_invalid", "image data URI must be base64-encoded")
            declared = reference[5:marker].split(";", 1)[0].strip().lower() or None
            raw = _load_data(reference[marker + len(";base64,"):], cap)
            stats["data"] += 1
        elif kind == "url":
            raw = _load_url(reference, cap, deadline)
            stats["url"] += 1
        else:
            raw = _load_file(reference, cap, deadline)
            stats["file"] += 1
        mime = _sniff(raw)
        if declared and declared != mime and callable(notify):
            notify("image_mime_corrected", declared=_safe_mime(declared), sniffed=mime)
        stats["count"] += 1
        stats["bytes"] += len(raw)
        part: dict[str, Any] = {
            "type": "input_image",
            "image_url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        }
        if detail is not None:
            part["detail"] = detail
        parts = rewritten.setdefault((index, field), list(items[index][field]))
        parts[position] = part
    new_items = list(items)
    for (index, field), parts in rewritten.items():
        new_items[index] = dict(new_items[index], **{field: parts})
    new_body = dict(body)
    value = body.get("input")
    if value is None:
        new_body["messages"] = new_items
    elif isinstance(value, dict):
        new_body["input"] = new_items[0]
    else:
        new_body["input"] = new_items
    if callable(notify):
        notify("images_normalized", **stats)
    return new_body, stats
