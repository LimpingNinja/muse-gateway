"""Opt-in private wire evidence. Never used to decide protocol behavior."""

from __future__ import annotations

import contextvars
from array import array
import csv
import hashlib
import heapq
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time


_ACTIVE = contextvars.ContextVar("gateway_capture", default=None)
_LIMIT = 64 * 1024 * 1024
_MAX_OVERLAP = 65536
_SECRET_FIELDS = {
    "authorization", "proxy_authorization", "api_key", "apikey", "access_token",
    "refresh_token", "user_jwt", "assignment_jwt", "model_assignment_jwt",
    "cookie", "set_cookie", "client_secret", "password",
}
_TOKEN = re.compile(
    rb"(?:devin-session-token\$[^\s\"'<>]+|"
    rb"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    rb"sk-[A-Za-z0-9_-]{16,})"
)


def current_capture():
    return _ACTIVE.get()


def _secret_field(name):
    normalized = re.sub(r"[^a-z0-9]", "", str(name).casefold())
    return (
        str(name).lower().replace("-", "_") in _SECRET_FIELDS
        or any(marker in normalized for marker in (
            "authorization", "authentication", "credential", "password",
            "passwd", "secret", "cookie", "apikey", "accesstoken",
            "refreshtoken", "authtoken", "bearertoken", "sessiontoken",
            "identitytoken", "idtoken", "userjwt", "assignmentjwt",
        ))
        or normalized.endswith(("token", "jwt"))
    )


def encoded_secret_forms(value: str) -> set[bytes]:
    """Literal UTF-8 and the JSON string forms used by our serializers.

    Two JSON layers cover strings inside tool-argument JSON inside SSE/JSON.
    This is deliberately not a claim about arbitrary encodings in user data.
    """
    if not isinstance(value, str) or not value:
        return set()
    texts = {value}
    frontier = {value}
    for _ in range(2):
        frontier = {json.dumps(text, ensure_ascii=ascii_only)[1:-1]
                    for text in frontier for ascii_only in (True, False)}
        texts.update(frontier)
    forms = set()
    for text in texts:
        try:
            forms.add(text.encode("utf-8"))
        except UnicodeError:
            # A malformed semantic string still has a safe ASCII JSON form.
            # Its literal UTF-8 representation does not exist.
            continue
    return forms


def _literal_spans(data, secret):
    start = data.find(secret)
    while start >= 0:
        yield start, start + len(secret)
        start = data.find(secret, start + 1)


def _known_secret_spans(data, secret):
    """Locate full keys and identifying prefixes in linear, bounded space.

    Require at least 16 bytes and half the known form. Only the existing
    literal/two-layer JSON forms are considered, never arbitrary encodings.
    Fragments are masked even inside JSON strings or before a non-key byte.
    """
    minimum = max(16, (len(secret) + 1) // 2)
    if len(secret) <= minimum or len(secret) > _MAX_OVERLAP:
        yield from _literal_spans(data, secret)
        return
    anchor = secret[:minimum]
    start = data.find(anchor)
    if start < 0:
        return
    # KMP avoids enumerating/copying all prefixes, even for repetitive keys.
    fallback = array("I", [0]) * len(secret)
    matched = 0
    for index in range(1, len(secret)):
        while matched and secret[index] != secret[matched]:
            matched = fallback[matched - 1]
        if secret[index] == secret[matched]:
            matched += 1
        fallback[index] = matched
    matched = minimum
    index = start + matched
    while index < len(data):
        byte = data[index]
        if byte != secret[matched] and matched >= minimum:
            yield index - matched, index
        while matched and byte != secret[matched]:
            matched = fallback[matched - 1]
        if byte == secret[matched]:
            matched += 1
        index += 1
        if matched == len(secret):
            yield index - matched, index
            matched = fallback[matched - 1]
        if not matched:
            # Most capture bytes are clean: search for the identifying anchor
            # in C instead of visiting every unrelated byte in Python.
            start = data.find(anchor, index)
            if start < 0:
                return
            matched = minimum
            index = start + matched
    if matched >= minimum:
        yield len(data) - matched, len(data)


def _private_directory(path):
    os.chmod(path, 0o700)
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        output = subprocess.check_output(
            ["whoami", "/user", "/fo", "csv", "/nh"], text=True,
            creationflags=flags,
        )
        sid = next(csv.reader(io.StringIO(output.strip())))[1]
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{sid}:(OI)(CI)F"],
            check=True, capture_output=True, creationflags=flags,
        )


def _source_identity():
    root = Path(__file__).resolve().parent
    files = {}
    for name in ("muse_gateway.py", "muse_providers.py", "muse_devin.py",
                 "muse_devin_wire.py", "devin_pb2.py", "muse_capture.py", "muse_kiro.py"):
        path = root / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    revision = None
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "source_root": str(root), "revision": revision, "files": files,
        "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "python": sys.version, "executable": sys.executable,
        "protobuf": importlib.metadata.version("protobuf"),
        "platform": platform.platform(), "pid": os.getpid(),
    }


class CaptureSession:
    def __init__(self, directory):
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="gateway-", dir=parent))
        _private_directory(self.directory)
        self.identity = _source_identity()
        self.identity["capture_id"] = self.directory.name
        self.identity["started_unix_ns"] = time.time_ns()
        self._save("session.json", self.identity)

    def _save(self, name, value):
        path = self.directory / name
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        os.chmod(path, 0o600)

    def health(self):
        return {key: self.identity[key] for key in ("capture_id", "fingerprint", "source_root", "pid")}

    def request(self, request_id, secrets=()):
        return RequestCapture(self.directory / request_id, request_id, secrets)


class RequestCapture:
    def __init__(self, directory, request_id, secrets=(), limit=_LIMIT):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        self.request_id = request_id
        self.started = time.monotonic_ns()
        self.secrets = set()
        self.streams = {}
        self.artifacts = {}
        self.limit = limit
        self.written = 0
        self.errors = []
        self.events = 0
        self.labels = {}
        self.header_pending = bytearray()
        self.headers_done = False
        self.sse_pending = {}
        self.first_events = {}
        self.terminals = {}
        self._finished = False
        for secret in secrets:
            self.add_secret(secret)
        self.token = _ACTIVE.set(self)
        self.event("request.open", unix_ns=time.time_ns())

    def add_secret(self, value):
        if isinstance(value, str) and value:
            if value.lower().startswith("bearer "):
                self.add_secret(value[7:])
            self.secrets.update(encoded_secret_forms(value))
            prefix = "devin-session-token$"
            if value.startswith(prefix):
                raw = value[len(prefix):]
                if raw:
                    self.secrets.update(encoded_secret_forms(raw))
            elif not value.lower().startswith("bearer "):
                self.secrets.update(encoded_secret_forms(prefix + value))

    def label(self, name):
        self.labels[name] = self.labels.get(name, 0) + 1
        return f"{name}.{self.labels[name]:03d}"

    def _redaction_spans(self, data):
        # Locate full matches before replacing anything. Unioning their spans
        # with identifying prefixes preserves overlapping short/long keys.
        sources = [_known_secret_spans(data, secret) for secret in self.secrets]
        sources.append(((match.start(), match.end()) for match in _TOKEN.finditer(data)))
        pending = None
        for start, end in heapq.merge(*sources):
            if pending is not None and start > pending[1]:
                yield pending
                pending = None
            pending = (start, end) if pending is None else (pending[0], max(pending[1], end))
        if pending is not None:
            yield pending

    def redact(self, data):
        if any(len(secret) > _MAX_OVERLAP for secret in self.secrets):
            if "stream_redaction_secret_limit" not in self.errors:
                self.errors.append("stream_redaction_secret_limit")
            return b"[REDACTED]" if data else data
        parts = []
        end = 0
        for start, stop in self._redaction_spans(data):
            parts.extend((data[end:start], b"[REDACTED]"))
            end = stop
        return b"".join((*parts, data[end:])) if parts else data

    def sanitize(self, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                safe_key = self.sanitize(key) if isinstance(key, str) else key
                if _secret_field(key):
                    if isinstance(item, str):
                        self.add_secret(item)
                    result[safe_key] = "[REDACTED]"
                else:
                    result[safe_key] = self.sanitize(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        if isinstance(value, str):
            return self.redact(value.encode()).decode("utf-8", errors="replace")
        return value

    def _append(self, name, data):
        stats = self.artifacts.setdefault(name, {"captured_bytes": 0, "truncated": False})
        if stats["truncated"]:
            return
        available = max(0, self.limit - self.written)
        if len(data) > available:
            stats["truncated"] = True
        selected = data[:available]
        try:
            path = self.directory / name
            offset = None
            if selected and len(data) >= available:
                # The budget creates an EOF of its own. Include bounded bytes
                # from the prior append so a split prefix cannot escape here.
                previous = b""
                if stats["captured_bytes"] and len(selected) < _MAX_OVERLAP:
                    with open(path, "rb") as handle:
                        handle.seek(max(0, stats["captured_bytes"] - _MAX_OVERLAP))
                        previous = handle.read(_MAX_OVERLAP)
                context = previous + selected[-_MAX_OVERLAP:]
                scrubbed = self.redact(context)
                if scrubbed != context:
                    # A replacement for a very short full key can grow; it
                    # must still fit the original capture byte budget.
                    scrubbed = scrubbed[:len(context)]
                    start = stats["captured_bytes"] + len(selected) - len(context)
                    if start < stats["captured_bytes"]:
                        offset = start
                        selected = scrubbed
                    else:
                        selected = selected[:start - stats["captured_bytes"]] + scrubbed
            with open(path, "r+b" if offset is not None and path.exists() else "ab") as handle:
                if offset is not None:
                    handle.seek(offset)
                    handle.write(selected)
                    handle.truncate()
                else:
                    handle.write(selected)
            os.chmod(path, 0o600)
            added = len(selected) if offset is None else offset + len(selected) - stats["captured_bytes"]
            self.written += added
            stats["captured_bytes"] += added
        except OSError as error:
            if type(error).__name__ not in self.errors:
                self.errors.append(type(error).__name__)

    def event(self, name, **fields):
        if self.events >= 50000:
            return
        self.events += 1
        record = {"event": name, "elapsed_ns": time.monotonic_ns() - self.started, **fields}
        self._append("timeline.jsonl", json.dumps(self.sanitize(record), ensure_ascii=True).encode() + b"\n")

    def json(self, name, value):
        def register(node, protected=False):
            if isinstance(node, dict):
                for key, item in node.items():
                    register(item, protected or _secret_field(key))
            elif isinstance(node, (list, tuple)):
                for item in node:
                    register(item, protected)
            elif protected:
                self.add_secret(node)
        register(value)
        if name + ".json" in self.artifacts:
            name = self.label(name)
        self._append(name + ".json", json.dumps(self.sanitize(value), ensure_ascii=True, indent=2).encode() + b"\n")

    def bytes(self, name, data):
        """Redact before disk, retaining boundary overlap only in memory."""
        state = self.streams.setdefault(name, {"tail": b"", "redacting": None, "suppressed": False, "observed_bytes": 0, "sha256": hashlib.sha256()})
        state["observed_bytes"] += len(data)
        state["sha256"].update(data)
        longest = max(map(len, self.secrets), default=0)
        if state["suppressed"]:
            return
        if longest > _MAX_OVERLAP:
            state["tail"] = b""
            if "stream_redaction_secret_limit" not in self.errors:
                self.errors.append("stream_redaction_secret_limit")
            return
        if state["redacting"] is not None:
            continuation = rb"[^\s\"'<>]*" if state["redacting"] == "session" else rb"[A-Za-z0-9_-]*"
            consumed = re.match(continuation, data).end()
            if consumed == len(data):
                return
            data = data[consumed:]
            state["redacting"] = None
        combined = state["tail"] + data
        cut = max(0, len(combined) - max(8192, longest))
        open_token = next((match for match in _TOKEN.finditer(combined)
                           if match.end() == len(combined)), None)
        for start, end in self._redaction_spans(combined):
            if open_token is not None and end == len(combined) and start < cut:
                self._append(name, self.redact(combined[:start]) + b"[REDACTED]")
                state["tail"] = b""
                state["redacting"] = "session" if open_token.group().startswith(b"devin-session-token$") else "token"
                return
            if start < cut < end:
                cut = start
                break
        self._append(name, self.redact(combined[:cut]))
        # Retain raw bytes: replacing a short key here would destroy evidence
        # needed to recognize a longer overlapping key on the next chunk.
        if len(combined) - cut > _MAX_OVERLAP:
            state["tail"] = b""
            state["suppressed"] = True
            if "stream_redaction_overlap_limit" not in self.errors:
                self.errors.append("stream_redaction_overlap_limit")
        else:
            state["tail"] = combined[cut:]

    def observe_sse(self, channel, data):
        pending = self.sse_pending.setdefault(channel, bytearray())
        pending.extend(data)
        while b"\n\n" in pending or b"\r\n\r\n" in pending:
            match = re.search(rb"\r?\n\r?\n", pending)
            frame = bytes(pending[:match.start()])
            del pending[:match.end()]
            event_name = None
            payload = []
            for line in frame.splitlines():
                if line.startswith(b"event:"):
                    event_name = line[6:].strip().decode(errors="replace")
                elif line.startswith(b"data:"):
                    payload.append(line[5:].lstrip())
            text = b"\n".join(payload)
            try:
                value = json.loads(text)
            except (ValueError, UnicodeError):
                value = {}
            if isinstance(value, dict):
                event_name = event_name or value.get("type")
            if text == b"[DONE]":
                event_name = "[DONE]"
            if event_name:
                self.first_events.setdefault(channel, event_name)
                if event_name in {"response.completed", "response.failed", "response.incomplete", "error", "[DONE]"}:
                    self.terminals.setdefault(channel, []).append(event_name)
                self.event(channel + ".event", type=event_name, frame_bytes=len(frame),
                           sequence_number=value.get("sequence_number") if isinstance(value, dict) else None)
        if len(pending) > 1024 * 1024:
            pending.clear()
            self.event(channel + ".observer_limit", bytes=1024 * 1024)

    def downstream(self, data):
        self.event("downstream.write", bytes=len(data))
        self.bytes("downstream.http", data)
        body = data
        if not self.headers_done:
            self.header_pending.extend(data)
            boundary = self.header_pending.find(b"\r\n\r\n")
            if boundary < 0:
                return
            body = bytes(self.header_pending[boundary + 4:])
            self.header_pending.clear()
            self.headers_done = True
        if body:
            self.bytes("downstream.body", body)
            self.observe_sse("downstream", body)

    def protobuf(self, name, message, original=None):
        """Retain a scrubbed protobuf plus readable fields; never auth bytes."""
        from google.protobuf.json_format import MessageToDict
        clone = type(message)()
        clone.CopyFrom(message)
        clone.DiscardUnknownFields()

        def register(node):
            for field, value in node.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    for child in value if field.is_repeated else (value,):
                        register(child)
                elif field.type == field.TYPE_STRING and _secret_field(field.name):
                    for item in value if field.is_repeated else (value,):
                        self.add_secret(item)

        def scrub(node):
            for field, value in node.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    for child in value if field.is_repeated else (value,):
                        scrub(child)
                elif field.type == field.TYPE_STRING:
                    if _secret_field(field.name):
                        replacement = "[REDACTED]"
                        if not field.is_repeated:
                            setattr(node, field.name, replacement)
                        else:
                            value[:] = [replacement] * len(value)
                    elif field.is_repeated:
                        value[:] = [self.redact(item.encode()).decode(errors="replace") for item in value]
                    else:
                        setattr(node, field.name, self.redact(value.encode()).decode(errors="replace"))
        # Credential discovery must finish before any earlier string field is
        # scrubbed; protobuf field order is not an authentication boundary.
        register(clone)
        scrub(clone)
        self.json(name, MessageToDict(clone, preserving_proto_field_name=True))
        self._append(name + ".pb", clone.SerializeToString())
        if original is not None:
            self.event(name, original_bytes=len(original), original_sha256=hashlib.sha256(original).hexdigest(),
                       representation="credentials_scrubbed_protobuf")
        return clone.SerializeToString()

    def finish(self, **outcome):
        if self._finished:
            return
        self._finished = True
        self.event("request.close", **outcome)
        for name, state in self.streams.items():
            self._append(name, self.redact(state["tail"]))
        manifest = {
            "request_id": self.request_id, "outcome": outcome,
            "first_events": self.first_events, "terminal_events": self.terminals,
            "artifacts": self.artifacts, "capture_errors": self.errors,
            "timeline_event_limit_reached": self.events >= 50000,
            "streams": {name: {"observed_bytes": state["observed_bytes"],
                                "original_sha256": state["sha256"].hexdigest()}
                        for name, state in self.streams.items()},
        }
        try:
            with open(self.directory / "manifest.json", "x", encoding="utf-8") as handle:
                json.dump(self.sanitize(manifest), handle, indent=2)
            os.chmod(self.directory / "manifest.json", 0o600)
        except OSError:
            pass
        finally:
            _ACTIVE.reset(self.token)


class CapturingWriter:
    def __init__(self, writer):
        self.writer = writer

    def write(self, data):
        capture = current_capture()
        try:
            result = self.writer.write(data)
        except OSError as error:
            if capture is not None:
                capture.event("downstream.write_error", kind=type(error).__name__, attempted_bytes=len(data))
            raise
        if capture is not None:
            capture.downstream(bytes(data[:result] if isinstance(result, int) else data))
        return result

    def flush(self):
        result = self.writer.flush()
        capture = current_capture()
        if capture is not None:
            capture.event("downstream.flush")
        return result

    def __getattr__(self, name):
        return getattr(self.writer, name)
