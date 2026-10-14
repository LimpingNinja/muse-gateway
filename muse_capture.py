"""Opt-in private wire evidence. Never used to decide protocol behavior."""

from __future__ import annotations

import contextvars
import csv
import hashlib
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
                 "muse_devin_wire.py", "devin_pb2.py", "muse_capture.py"):
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
            self.secrets.add(value.encode())
            prefix = "devin-session-token$"
            if value.startswith(prefix):
                raw = value[len(prefix):]
                if raw:
                    self.secrets.add(raw.encode())
            elif not value.lower().startswith("bearer "):
                self.secrets.add((prefix + value).encode())

    def label(self, name):
        self.labels[name] = self.labels.get(name, 0) + 1
        return f"{name}.{self.labels[name]:03d}"

    def redact(self, data):
        for secret in sorted(self.secrets, key=len, reverse=True):
            data = data.replace(secret, b"[REDACTED]")
        return _TOKEN.sub(b"[REDACTED]", data)

    def sanitize(self, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if str(key).lower().replace("-", "_") in _SECRET_FIELDS:
                    if isinstance(item, str):
                        self.add_secret(item)
                    result[key] = "[REDACTED]"
                else:
                    result[key] = self.sanitize(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        if isinstance(value, str):
            return self.redact(value.encode()).decode("utf-8", errors="replace")
        return value

    def _append(self, name, data):
        stats = self.artifacts.setdefault(name, {"captured_bytes": 0, "truncated": False})
        available = max(0, self.limit - self.written)
        if len(data) > available:
            stats["truncated"] = True
        selected = data[:available]
        try:
            with open(self.directory / name, "ab") as handle:
                handle.write(selected)
            os.chmod(self.directory / name, 0o600)
            self.written += len(selected)
            stats["captured_bytes"] += len(selected)
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
        def register(node):
            if isinstance(node, dict):
                for key, item in node.items():
                    if str(key).lower().replace("-", "_") in _SECRET_FIELDS:
                        self.add_secret(item)
                    register(item)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    register(item)
        register(value)
        if name + ".json" in self.artifacts:
            name = self.label(name)
        self._append(name + ".json", json.dumps(self.sanitize(value), ensure_ascii=True, indent=2).encode() + b"\n")

    def bytes(self, name, data):
        """Redact before disk, retaining boundary overlap only in memory."""
        state = self.streams.setdefault(name, {"tail": b"", "redacting": None, "observed_bytes": 0, "sha256": hashlib.sha256()})
        state["observed_bytes"] += len(data)
        state["sha256"].update(data)
        longest = max(map(len, self.secrets), default=0)
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
        # An open-ended token is replaced once, then its continuation is
        # discarded until a delimiter. It must not accumulate in the tail.
        for match in _TOKEN.finditer(combined):
            if match.end() == len(combined):
                self._append(name, self.redact(combined[:match.start()]) + b"[REDACTED]")
                state["tail"] = b""
                state["redacting"] = "session" if match.group().startswith(b"devin-session-token$") else "token"
                return
        sanitized = self.redact(combined)
        cut = max(0, len(sanitized) - max(8192, longest))
        self._append(name, sanitized[:cut])
        state["tail"] = sanitized[cut:]

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

        def scrub(node):
            for field, value in node.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    for child in value if field.is_repeated else (value,):
                        scrub(child)
                elif field.type == field.TYPE_STRING:
                    if field.name in _SECRET_FIELDS:
                        for item in value if field.is_repeated else (value,):
                            self.add_secret(item)
                        replacement = "[REDACTED]"
                        if not field.is_repeated:
                            setattr(node, field.name, replacement)
                        else:
                            value[:] = [replacement] * len(value)
                    elif field.is_repeated:
                        value[:] = [self.redact(item.encode()).decode(errors="replace") for item in value]
                    else:
                        setattr(node, field.name, self.redact(value.encode()).decode(errors="replace"))
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
