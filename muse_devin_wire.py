"""Connect-protocol wire layer for the Devin (Codeium Cascade) provider.

Mirrors, action by action, the OMP sources at pinned rev 818dd42:
  packages/catalog/src/wire/devin.ts        (identities, base URL, token prefix)
  packages/catalog/src/wire/devin-proto.ts  (unary gzip-or-plain decode)
  packages/ai/src/providers/connect-frame.ts (streaming frame codec)
  packages/ai/src/providers/devin.ts        (RPC calls, trailers, error detail)

Rule 2 applies throughout: match OMP where mechanically sound; our own
mechanisms are additive only (URL validation, decompression bound). Every
deviation is marked ADDITIVE.

Requires ``protobuf`` (a hard install dependency): this module imports the
generated ``devin_pb2`` bindings.
"""

import gzip
import io
import json
import re
import struct
import sys
import urllib.error
import urllib.request

import devin_pb2
from muse_capture import current_capture


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DevinWireError(Exception):
    """Base class for Devin transport failures."""


class DevinHttpError(DevinWireError):
    def __init__(self, operation, status, detail=None):
        self.operation = operation
        self.status = status
        self.detail = detail
        message = f"Devin {operation} error {status}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class DevinAuthError(DevinWireError):
    """GetUserJwt failed or returned no usable JWT."""


class DevinFrameError(DevinWireError):
    """A Connect frame violated the payload cap."""


class DevinTrailerError(DevinWireError):
    """The server rejected the stream via a Connect end-stream trailer."""

    def __init__(self, code, message, formatted, detail=None, raw=""):
        self.code = code
        self.message = message
        self.formatted = formatted
        self.detail = detail
        self.raw = raw
        super().__init__(formatted)


# ---------------------------------------------------------------------------
# Constants (wire/devin.ts, connect-frame.ts, devin.ts)
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://server.codeium.com"
AUTH_PATH = "/exa.auth_pb.AuthService/GetUserJwt"
ASSIGN_MODEL_PATH = "/exa.api_server_pb.ApiServerService/AssignModel"
CHAT_MESSAGE_PATH = "/exa.api_server_pb.ApiServerService/GetChatMessage"
GET_CLI_MODEL_CONFIGS_PATH = (
    "/exa.api_server_pb.ApiServerService/GetCliModelConfigs"
)

SESSION_TOKEN_PREFIX = "devin-session-token$"

# Released Devin CLI identity. The backend gates on this tuple: ide_type
# "chisel" unlocks AssignModel and the CLI model surface.
CLI_IDE_NAME = "devin-cli"
CLI_IDE_TYPE = "chisel"
CLI_VERSION = "3000.11.3"
CLI_EXTENSION = "chisel"

# Discovery identity: dev-channel chisel unlocks the full native config set.
DISCOVERY_IDE_NAME = "chisel"
DISCOVERY_VERSION = "0.0.0-dev"

# Legacy Windsurf identity for enterprise seats whose roster only the editor
# identity can see.
LEGACY_IDE_NAME = "windsurf"
LEGACY_IDE_VERSION = "3.2.23"
LEGACY_EXTENSION_VERSION = "1.48.2"

LOCALE = "en"

# Display slots requested at discovery (raw ints: OMP defines no names).
SUPPORTED_MODEL_DISPLAYS = (3, 4, 6, 7, 8)
# Slots requested but never surfaced (quick-review + internal defaults).
INTERNAL_MODEL_DISPLAYS = frozenset({4, 6})

# Connect streaming envelope: 1 flag byte + 4-byte big-endian length.
CONNECT_COMPRESSED_FLAG = 0x01
CONNECT_END_STREAM_FLAG = 0x02
CONNECT_HEADER_BYTES = 5

# Hard upper bound on one frame payload, checked before buffering (a corrupt
# length prefix fails fast instead of eating memory).
MAX_CONNECT_FRAME_PAYLOAD = 16 * 1024 * 1024
# History-size heuristic for invalid_argument trailer recovery.
LARGE_HISTORY_RECOVERY_BYTES = 512 * 1024
# Bounds on retained error evidence.
MAX_TRAILER_EVIDENCE_CHARS = 2000
MAX_ERROR_DETAIL_CHARS = 4096
# ADDITIVE: cap gunzip expansion (OMP gunzipSync is unbounded). Far above any
# legitimate unary body; trips only on corrupt/zip-bomb input.
MAX_UNARY_DECOMPRESSED_BYTES = 64 * 1024 * 1024

CONNECT_UA = "connect-go/1.18.1 (go1.26.3)"

_HTML_BODY_PATTERN = re.compile(r"^\s*(?:<!doctype\s+html\b|<html\b)", re.IGNORECASE)


def _platform_os():
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    return "linux"


# ---------------------------------------------------------------------------
# Metadata (wire/devin.ts)
# ---------------------------------------------------------------------------


def normalize_session_token(api_key):
    """Wire form of a session credential: the scheme prefix is required."""
    if not api_key:
        return ""
    if api_key.startswith(SESSION_TOKEN_PREFIX):
        return api_key
    return SESSION_TOKEN_PREFIX + api_key


def _base_metadata(**fields):
    metadata = devin_pb2.Metadata()
    for name, value in fields.items():
        if value:
            setattr(metadata, name, value)
    return metadata


def discovery_metadata(api_key):
    """Metadata for dev-channel GetCliModelConfigs (no ide_type, no JWT)."""
    metadata = _base_metadata(
        api_key=normalize_session_token(api_key),
        ide_name=DISCOVERY_IDE_NAME,
        ide_version=DISCOVERY_VERSION,
        extension_name=DISCOVERY_IDE_NAME,
        extension_version=DISCOVERY_VERSION,
        locale=LOCALE,
        os=_platform_os(),
    )
    metadata.supported_model_displays.extend(SUPPORTED_MODEL_DISPLAYS)
    return metadata


def legacy_metadata(api_key):
    """Metadata for the legacy Windsurf-identity discovery fallback."""
    return _base_metadata(
        api_key=api_key or "",
        ide_name=LEGACY_IDE_NAME,
        ide_version=LEGACY_IDE_VERSION,
        extension_name=LEGACY_IDE_NAME,
        extension_version=LEGACY_EXTENSION_VERSION,
        locale=LOCALE,
    )


def cli_metadata(api_key, user_jwt=""):
    """Released-CLI metadata with the credential already prefixed for the wire."""
    return wire_metadata(normalize_session_token(api_key), user_jwt)


def wire_metadata(api_key, user_jwt=""):
    """Released-CLI metadata with credential bytes used verbatim."""
    metadata = _base_metadata(
        api_key=api_key or "",
        user_jwt=user_jwt,
        ide_name=CLI_IDE_NAME,
        ide_type=CLI_IDE_TYPE,
        ide_version=CLI_VERSION,
        extension_name=CLI_EXTENSION,
        extension_version=CLI_VERSION,
        locale=LOCALE,
        os=_platform_os(),
    )
    return metadata


# ---------------------------------------------------------------------------
# Connect framing (connect-frame.ts)
# ---------------------------------------------------------------------------


def frame_message(data, flags=0):
    """Wrap one Connect message: flag byte + BE32 length + payload."""
    return bytes([flags & 0xFF]) + struct.pack(">I", len(data)) + bytes(data)


def decompress_frame(payload, limit=MAX_UNARY_DECOMPRESSED_BYTES):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(bytes(payload))) as handle:
            expanded = handle.read(limit + 1)
    except (OSError, EOFError) as error:
        raise DevinFrameError("Devin stream frame is not valid gzip") from error
    if len(expanded) > limit:
        raise DevinFrameError("Devin stream frame exceeds the decompressed size limit")
    return expanded


class ConnectFrameDecoder:
    """Incremental Connect frame decoder.

    Feed arbitrary HTTP chunks to :meth:`decode`; complete frames are
    yielded in wire order. A length prefix above the cap raises
    :class:`DevinFrameError` before any of that payload is buffered, after
    the frames that precede it — mirroring OMP exactly.
    """

    def __init__(self, max_payload_bytes=MAX_CONNECT_FRAME_PAYLOAD):
        self._buffer = bytearray()
        self._max_payload_bytes = max_payload_bytes

    def decode(self, chunk=None):
        """Buffer ``chunk`` (if any) and yield ``(flags, payload)`` tuples."""
        if chunk:
            self._buffer += chunk
        frames = []
        view = memoryview(self._buffer)
        offset = 0
        size = len(self._buffer)
        while size - offset >= CONNECT_HEADER_BYTES:
            (length,) = struct.unpack_from(">I", view, offset + 1)
            if length > self._max_payload_bytes:
                raise DevinFrameError(
                    f"Devin Connect frame length {length} exceeds "
                    f"{self._max_payload_bytes}-byte cap"
                )
            if size - offset < CONNECT_HEADER_BYTES + length:
                break
            flags = view[offset]
            start = offset + CONNECT_HEADER_BYTES
            frames.append((flags, bytes(view[start : start + length])))
            offset = start + length
        view.release()
        del self._buffer[:offset]
        return frames


# ---------------------------------------------------------------------------
# Unary decode + trailers + error detail (devin-proto.ts, devin.ts)
# ---------------------------------------------------------------------------


def decode_unary(message_cls, payload):
    """Decode a unary body: bare protobuf first, gzipped fallback.

    Edges variously return either representation. Returns None when neither
    decodes. (ADDITIVE: gunzip expansion is bounded; OMP is unbounded.)
    """
    try:
        message = message_cls()
        message.ParseFromString(bytes(payload))
        return message
    except Exception:
        pass
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(bytes(payload))) as handle:
            expanded = handle.read(MAX_UNARY_DECOMPRESSED_BYTES + 1)
        if len(expanded) > MAX_UNARY_DECOMPRESSED_BYTES:
            return None
        message = message_cls()
        message.ParseFromString(expanded)
        return message
    except Exception:
        return None


def truncate_trailer_evidence(text):
    if len(text) > MAX_TRAILER_EVIDENCE_CHARS:
        return text[:MAX_TRAILER_EVIDENCE_CHARS] + "…"
    return text


def summarize_trailer_details(details):
    """Summarize loosely-shaped ``{type, value, debug}`` detail entries."""
    if not isinstance(details, list) or not details:
        return None
    summary = ""
    for entry in details:
        if not isinstance(entry, dict):
            continue
        entry_type = entry.get("type")
        entry_type = (
            entry_type if isinstance(entry_type, str) and entry_type else None
        )
        value = entry.get("value")
        value = value if isinstance(value, str) and value else None
        debug = entry.get("debug")
        if debug is not None:
            try:
                debug = debug if isinstance(debug, str) else json.dumps(debug)
            except (TypeError, ValueError):
                debug = None
        bounded_type = truncate_trailer_evidence(entry_type) if entry_type else None
        evidence = debug if debug is not None else value
        bounded_evidence = (
            truncate_trailer_evidence(evidence) if evidence else None
        )
        if bounded_type and bounded_evidence:
            part = f"{bounded_type}: {bounded_evidence}"
        else:
            part = bounded_type if bounded_type is not None else bounded_evidence
        if not part:
            continue
        candidate = f"{summary}; {part}" if summary else part
        if len(candidate) > MAX_TRAILER_EVIDENCE_CHARS:
            return truncate_trailer_evidence(candidate)
        summary = candidate
    return summary or None


def parse_trailer_error(text):
    """Parse a Connect end-of-stream JSON trailer into a DevinTrailerError.

    Returns None for empty text, malformed JSON, or shapes without
    ``{error: {code, message}}`` — callers treat None as clean EOF.
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or "error" not in parsed:
        return None
    error = parsed["error"]
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    code = code if isinstance(code, str) else ""
    message = error.get("message")
    message = message if isinstance(message, str) else ""
    if not code and not message:
        return None
    formatted = f"Devin stream error{f' {code}' if code else ''}: {message}"
    detail = None
    if "details" in error:
        detail = summarize_trailer_details(error["details"])
        if detail:
            formatted += f" [details: {detail}]"
    return DevinTrailerError(code, message, formatted, detail, truncate_trailer_evidence(text))


def http_error_detail(content_type, payload):
    """Extract a bounded human error from an HTTP error body.

    Returns None for binary bodies, HTML bodies, empty text, or text with
    control characters — mirroring OMP's suppress-to-status-only behavior.
    """
    try:
        text = bytes(payload).decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    if content_type and "text/html" in content_type.lower():
        return None
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            error = decoded.get("error")
            if isinstance(error, dict) and isinstance(
                error.get("message"), str
            ):
                text = error["message"].strip()
            elif isinstance(error, str):
                text = error.strip()
            elif isinstance(decoded.get("message"), str):
                text = decoded["message"].strip()
    except ValueError:
        pass
    # Validate after envelope extraction: escapes materialize bytes the raw
    # scan cannot see. Collapse whitespace first so benign CRLF cannot trip
    # control detection.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized or _HTML_BODY_PATTERN.match(normalized):
        return None
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", normalized):
        return None
    if len(normalized) <= MAX_ERROR_DETAIL_CHARS:
        return normalized
    return normalized[:MAX_ERROR_DETAIL_CHARS]


# ---------------------------------------------------------------------------
# RPC plumbing (devin.ts requestDevinAuth / assign / chat / discovery)
# ---------------------------------------------------------------------------


_UNARY_HEADERS = {
    "Content-Type": "application/proto",
    "connect-protocol-version": "1",
    "Accept": "*/*",
}

_CHAT_HEADERS = {
    "Content-Type": "application/connect+proto",
    "connect-protocol-version": "1",
    "connect-content-encoding": "gzip",
    "Accept-Encoding": "identity",
    "User-Agent": CONNECT_UA,
    "connect-accept-encoding": "gzip",
}


def normalize_base_url(base_url):
    """Strip trailing slashes (ADDITIVE: also require http/https + host)."""
    cleaned = (base_url or "").strip().rstrip("/")
    if not re.match(r"^https?://[^/]+", cleaned):
        raise DevinWireError(
            f"Devin base URL must be http(s) with a host, got {base_url!r}"
        )
    return cleaned


def _post(url, body, headers, timeout):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    capture = current_capture()
    if capture is not None:
        capture.event("devin.rpc.open", url=url, request_bytes=len(body))
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        try:
            payload = error.read()
        finally:
            # The chained exception can outlive this request; do not leave
            # its HTTP response/socket open until garbage collection.
            error.close()
        if capture is not None:
            label = capture.label("devin.http_error")
            capture.json(label + ".headers", {"status": error.code, "headers": dict(error.headers or {})})
            capture.bytes(label + ".body", payload)
            capture.event("devin.rpc.error", status=error.code, bytes=len(payload))
        raise DevinHttpError(
            "request",
            getattr(error, "code", "?"),
            http_error_detail(
                error.headers.get("Content-Type", "") if error.headers else "",
                payload,
            ),
        ) from error
    if capture is not None:
        capture.json(capture.label("devin.rpc.response"), {"status": response.getcode(), "headers": dict(response.headers)})
        capture.event("devin.rpc.headers", status=response.getcode())
    return response


def unary_rpc(base_url, path, request_bytes, timeout=30.0):
    """POST a bare-protobuf unary RPC; return ``(status, content_type, body)``.

    Non-2xx responses raise :class:`DevinHttpError` with bounded detail.
    """
    url = normalize_base_url(base_url) + path
    capture = current_capture()
    classes = {
        AUTH_PATH: (devin_pb2.GetUserJwtRequest, devin_pb2.GetUserJwtResponse),
        ASSIGN_MODEL_PATH: (devin_pb2.AssignModelRequest, devin_pb2.AssignModelResponse),
        GET_CLI_MODEL_CONFIGS_PATH: (devin_pb2.GetCliModelConfigsRequest, devin_pb2.GetCliModelConfigsResponse),
    }
    label = None
    if capture is not None and path in classes:
        label = capture.label("devin." + path.rsplit("/", 1)[-1])
        request = classes[path][0]()
        request.ParseFromString(request_bytes)
        capture.protobuf(label + ".request", request, request_bytes)
    response = _post(url, bytes(request_bytes), dict(_UNARY_HEADERS), timeout)
    try:
        status = response.getcode()
        content_type = response.headers.get("Content-Type", "")
        body = response.read()
        if capture is not None and label is not None:
            decoded = decode_unary(classes[path][1], body)
            if decoded is not None:
                capture.protobuf(label + ".response", decoded, body)
            else:
                capture.event(label + ".decode_failed", bytes=len(body))
    finally:
        response.close()
    if not 200 <= status < 300:
        raise DevinHttpError(
            "request", status, http_error_detail(content_type, body)
        )
    return status, content_type, body


def open_chat_stream(base_url, request_bytes, timeout=60.0):
    """POST a gzipped single-frame chat request; return the open response.

    The caller reads ``(flags, payload)`` frames via :class:`ConnectFrameDecoder`
    and must close the response. Non-2xx responses raise DevinHttpError.
    """
    url = normalize_base_url(base_url) + CHAT_MESSAGE_PATH
    frame = frame_message(gzip.compress(bytes(request_bytes)), CONNECT_COMPRESSED_FLAG)
    capture = current_capture()
    if capture is not None:
        request = devin_pb2.GetChatMessageRequest()
        request.ParseFromString(request_bytes)
        capture.protobuf("devin.chat.request", request, request_bytes)
        capture.event("devin.chat.frame_encoded", bytes=len(frame))
    response = _post(url, frame, dict(_CHAT_HEADERS), timeout)
    status = response.getcode()
    if not 200 <= status < 300:
        try:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
        finally:
            response.close()
        raise DevinHttpError(
            "API", status, http_error_detail(content_type, body)
        )
    return response


def iter_chat_frames(response, chunk_bytes=65536):
    """Yield ``(flags, payload)`` frames from an open chat response."""
    decoder = ConnectFrameDecoder()
    while True:
        capture = current_capture()
        if capture is not None:
            capture.event("devin.chat.read_start", requested_bytes=chunk_bytes)
        reader = getattr(response, "read1", None) or response.read
        chunk = reader(chunk_bytes)
        if capture is not None:
            capture.event("devin.chat.read_end", bytes=len(chunk))
        if not chunk:
            if capture is not None:
                capture.event("devin.chat.eof", pending_frame_bytes=len(decoder._buffer))
            if decoder._buffer:
                raise DevinFrameError("Devin stream ended inside a Connect frame")
            break
        items = decoder.decode(chunk)
        for item in items:
            if capture is not None:
                flags, payload = item
                label = capture.label("devin.chat.frame")
                capture.event(label, flags=flags, payload_bytes=len(payload))
                try:
                    raw = decompress_frame(payload) if flags & CONNECT_COMPRESSED_FLAG else payload
                    if flags & CONNECT_END_STREAM_FLAG:
                        safe = capture.redact(raw)
                        capture.bytes(label + ".trailer", raw)
                        capture.event("devin.chat.trailer")
                    else:
                        message = devin_pb2.GetChatMessageResponse()
                        message.ParseFromString(raw)
                        scrubbed = capture.protobuf(label, message, raw)
                        safe = raw if capture.redact(raw) == raw else scrubbed
                    kept_payload = payload if safe == raw else (gzip.compress(safe, mtime=0) if flags & CONNECT_COMPRESSED_FLAG else safe)
                    capture._append("devin.chat.response.connect", frame_message(kept_payload, flags))
                except Exception as error:
                    capture.event(label + ".capture_decode_failed", kind=type(error).__name__, opaque_bytes_omitted=True)
        yield from items
