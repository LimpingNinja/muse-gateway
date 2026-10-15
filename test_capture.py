import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import devin_pb2
from muse_capture import CaptureSession, current_capture
from muse_devin_wire import frame_message
from muse_gateway import Config, MuseShimServer


@contextmanager
def serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def post(base, body, headers=None):
    request = urllib.request.Request(base + "/responses", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read()


class CaptureTests(unittest.TestCase):
    def test_split_json_serialized_secrets_cannot_be_decoded_from_evidence(self):
        key = 'serialized-quote"backslash\\secret-\u03bb'
        cases = []
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("serialized-secret", [key])
            try:
                for ascii_only in (True, False):
                    for layers in (1, 2):
                        encoded = key
                        for _ in range(layers):
                            encoded = json.dumps(encoded, ensure_ascii=ascii_only)[1:-1]
                        payload = {"padding": "x" * 8190, "text": key} if layers == 1 else {
                            "padding": "x" * 8190, "arguments": json.dumps({"text": key}, ensure_ascii=ascii_only)}
                        wire = json.dumps(payload, ensure_ascii=ascii_only).encode()
                        needle = encoded.encode()
                        start = wire.index(needle)
                        for offset in range(1, len(needle)):
                            name = f"split-{int(ascii_only)}-{layers}-{offset}"
                            cut = start + offset
                            capture.bytes(name, wire[:cut])
                            capture.bytes(name, wire[cut:])
                            self.assertLessEqual(len(capture.streams[name]["tail"]), 65536)
                            cases.append((name, wire, layers))
                clean = b'event: ping\ndata: {"type":"ping","text":"credential-free"}\n\n'
                capture.bytes("clean", clean)
            finally:
                capture.finish()
            manifest = json.loads((capture.directory / "manifest.json").read_text())
            for name, original, layers in cases:
                saved = json.loads((capture.directory / name).read_bytes())
                text = saved["text"] if layers == 1 else json.loads(saved["arguments"])["text"]
                self.assertNotIn(key, text, "decoding serialized capture recovered a complete credential")
                self.assertEqual(text, "[REDACTED]")
                self.assertEqual(manifest["streams"][name]["original_sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(manifest["streams"][name]["observed_bytes"], len(original))
            self.assertEqual((capture.directory / "clean").read_bytes(), clean)
            self.assertEqual(manifest["capture_errors"], [])

    def test_scrubs_split_secrets_and_records_explicit_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            session = CaptureSession(directory)
            capture = session.request("scrub", ["a-private-credential-value"])
            try:
                capture.bytes("stream", b"a-private-cred")
                capture.bytes("stream", b"ential-value END")
                capture.bytes("boundary", b"x" * 8190 + b"a-private-cred")
                capture.bytes("boundary", b"ential-value" + b"y" * 8180)
                capture.json("nested", {"message": "a-different-secret", "metadata": {"api_key": "a-different-secret"}})
            finally:
                capture.finish()
            contents = b"".join(path.read_bytes() for path in capture.directory.iterdir())
            self.assertNotIn(b"a-private-credential-value", contents)
            self.assertNotIn(b"a-different-secret", contents)
            self.assertEqual((capture.directory / "stream").read_bytes(), b"[REDACTED] END")
            self.assertEqual((capture.directory / "boundary").read_bytes(), b"x" * 8190 + b"[REDACTED]" + b"y" * 8180)
            self.assertIsNone(current_capture())
            capture = session.request("limited")
            capture.limit = capture.written + 8
            try:
                capture.bytes("large", b"x" * 20)
            finally:
                capture.finish()
            manifest = json.loads((capture.directory / "manifest.json").read_text())
            self.assertTrue(manifest["artifacts"]["large"]["truncated"])
            self.assertEqual(manifest["streams"]["large"]["observed_bytes"], 20)
            key = ("budget-private|" + "0123456789abcdef" * 9)[:150].encode()
            for surface in ("json", "protobuf", "split"):
                with self.subTest(surface=surface):
                    capture = session.request("budget-" + surface, [key.decode()])
                    try:
                        if surface == "json":
                            capture.limit = capture.written + len(b'{\n  "text": "') + 149
                            payload = json.dumps({"text": key[:149].decode() + "!"}, indent=2).encode()
                            capture._append("budget.json", payload)
                            path = capture.directory / "budget.json"
                        else:
                            payload = devin_pb2.GetChatMessageRequest(prompt=key[:149].decode() + "!").SerializeToString()
                            end = payload.index(key[:149]) + 149
                            if surface == "split":
                                cut = payload.index(key[:149]) + 60
                                capture._append("budget.pb", payload[:cut])
                                payload, end = payload[cut:], end - cut
                            capture.limit = capture.written + end
                            capture._append("budget.pb", payload)
                            path = capture.directory / "budget.pb"
                    finally:
                        capture.finish()
                    self.assertNotIn(key[:75], path.read_bytes())
                    self.assertIn(b"[REDACTED]", path.read_bytes())
                    self.assertTrue(capture.artifacts[path.name]["truncated"])
                    self.assertLessEqual(capture.written, capture.limit)

    def test_protobuf_credentials_are_removed_before_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("protobuf")
            message = devin_pb2.GetChatMessageRequest(prompt="public prompt")
            message.metadata.api_key = ("protobuf-echo|" + "0123456789abcdef" * 9)[:150]
            message.metadata.ide_name = message.metadata.api_key + " " + message.metadata.api_key[:-1]
            message.metadata.user_jwt = "private-user-jwt"
            message.model_assignment_jwt = "private-assignment-jwt"
            message.prompt = message.model_assignment_jwt + " " + message.model_assignment_jwt[:-1]
            original = message.SerializeToString()
            try:
                returned = capture.protobuf("chat", message, original)
            finally:
                capture.finish()
            saved = devin_pb2.GetChatMessageRequest.FromString((capture.directory / "chat.pb").read_bytes())
            self.assertEqual(message.SerializeToString(), original)
            self.assertEqual(devin_pb2.GetChatMessageRequest.FromString(returned), saved)
            self.assertEqual(saved.prompt, "[REDACTED] [REDACTED]")
            self.assertEqual(saved.metadata.ide_name, "[REDACTED] [REDACTED]")
            self.assertEqual(json.loads((capture.directory / "chat.json").read_text())["prompt"], saved.prompt)
            self.assertEqual(saved.metadata.api_key, "[REDACTED]")
            self.assertEqual(saved.metadata.user_jwt, "[REDACTED]")
            self.assertEqual(saved.model_assignment_jwt, "[REDACTED]")
            contents = b"".join(path.read_bytes() for path in capture.directory.iterdir())
            for secret in (message.metadata.api_key[:75].encode(), b"private-user-jwt", b"private-assignment-jwt"):
                self.assertNotIn(secret, contents)

    def test_auth_error_scrubs_raw_cli_token_when_wire_uses_prefix(self):
        from muse_devin_wire import AUTH_PATH, DevinHttpError, unary_rpc
        token = "unprefixed-cli-login-secret"
        request = devin_pb2.GetUserJwtRequest()
        request.metadata.api_key = "devin-session-token$" + token
        error = urllib.error.HTTPError("https://x" + AUTH_PATH, 401, "unauthorized", {},
                                       io.BytesIO(json.dumps({"error": {"message": token}}).encode()))
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("raw-cli-token")
            try:
                with patch("muse_devin_wire.urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(DevinHttpError):
                        unary_rpc("https://x", AUTH_PATH, request.SerializeToString())
            finally:
                error.close()
                capture.finish()
            contents = b"".join(path.read_bytes() for path in capture.directory.iterdir())
            self.assertNotIn(token.encode(), contents)
            self.assertNotIn(("devin-session-token$" + token).encode(), contents)

    def test_stream_redaction_overlap_stays_bounded_for_open_token(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("open-token")
            try:
                capture.bytes("body", b"sk-")
                for _ in range(20):
                    capture.bytes("body", b"a" * 10000)
                    self.assertLessEqual(len(capture.streams["body"]["tail"]), 65536)
                capture.bytes("body", b" SAFE")
            finally:
                capture.finish()
            self.assertEqual((capture.directory / "body").read_bytes(), b"[REDACTED] SAFE")
            key = ("arbitrary-proxy|" + "0123456789ABCDEF" * 9)[:150]
            escaped = ('quoted"slash\\unicode-\u03bb|' + "abcdef0123456789" * 9)[:150]
            other = ("other-private|" + "fedcba9876543210" * 9)[:150]
            full = "leading-full-key|" + other[:75]
            capture = CaptureSession(directory).request("known-prefixes", [key, key[:32], escaped, other, full])
            cases = [("half", key[:75].encode(), b"[REDACTED]"),
                     ("overlap-prefix", key[:-1].encode(), b"[REDACTED]"),
                     ("full", key.encode(), b"[REDACTED]"),
                     ("full-overlap-suffix", full.encode(), b"[REDACTED]"),
                     ("clean", b"credential-free tail", b"credential-free tail"),
                     ("short", key[:15].encode(), key[:15].encode()),
                     ("divergent", other[:-1].encode() + b"!", b"[REDACTED]!")]
            for ascii_only in (True, False):
                encoded = escaped
                for layer in (1, 2):
                    encoded = json.dumps(encoded, ensure_ascii=ascii_only)[1:-1]
                    cases.append((f"encoded-{ascii_only}-{layer}", encoded.encode()[:-1], b"[REDACTED]"))
            try:
                for name, suffix, expected in cases:
                    wire = b"x" * 8190 + suffix
                    for chunk in (wire[:8222], wire[8222:8250], wire[8250:]):
                        capture.bytes(name, chunk)
                        self.assertLessEqual(len(capture.streams[name]["tail"]), 65536)
                capture.json("semantic-echo", {"text": key[:-1], "clean": "safe tail"})
                capture.json("secret-keys", {other: "full", key[:-1]: "partial", "safe": "unchanged"})
            finally:
                capture.finish()
            for name, suffix, expected in cases:
                with self.subTest(name=name):
                    self.assertEqual((capture.directory / name).read_bytes(), b"x" * 8190 + expected)
            self.assertEqual(json.loads((capture.directory / "semantic-echo.json").read_text()),
                             {"text": "[REDACTED]", "clean": "safe tail"})
            saved_keys = json.loads((capture.directory / "secret-keys.json").read_text())
            self.assertEqual(saved_keys, {"[REDACTED]": "partial", "safe": "unchanged"})

    def test_capture_preserves_socket_sse_bytes(self):
        wire = b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            outputs = []
            for enabled in (False, True):
                config = Config(argparse.Namespace(provider="generic", upstream=upstream,
                    api_key="fake-capture-credential", capture_dir=directory if enabled else None,
                    log_file=str(Path(directory) / "gateway.log")))
                server = MuseShimServer(("127.0.0.1", 0), config)
                with serving(server) as base:
                    outputs.append(post(base, {"model": "test", "input": "hello", "stream": True}))
                    if enabled:
                        with patch.object(server.capture_session, "request", side_effect=PermissionError("denied")):
                            self.assertEqual(post(base, {"model": "test", "input": "hello", "stream": True}), wire)
                if enabled:
                    request_dir = next(path for path in server.capture_session.directory.iterdir() if path.is_dir())
                    self.assertEqual((request_dir / "downstream.body").read_bytes(), wire)
                    self.assertEqual((request_dir / "upstream.body").read_bytes(), wire)
                    manifest = json.loads((request_dir / "manifest.json").read_text())
                    self.assertEqual(manifest["first_events"]["downstream"], "response.completed")
                    self.assertEqual(manifest["terminal_events"]["downstream"], ["response.completed"])
            self.assertEqual(outputs, [wire, wire])

    def test_real_connect_route_captures_frames_without_changing_events(self):
        from muse_devin_wire import ConnectFrameDecoder, iter_chat_frames
        key = ("connect-proxy|" + "0123456789abcdef" * 9)[:150]
        first = devin_pb2.GetChatMessageResponse(message_id="resp_capture", delta_text="Hello")
        echo = devin_pb2.GetChatMessageResponse(delta_text=key[:-1], stop_reason=2)
        cases = [("json", 2, json.dumps({"error": {"message": key[:-1]}}).encode()),
                 ("malformed", 2, b'{"error":{"message":"' + key[:-1].encode()),
                 ("proto", 0, echo.SerializeToString()),
                 ("compressed", 1, gzip.compress(echo.SerializeToString())),
                 ("shadowed", 0, echo.SerializeToString() +
                  devin_pb2.GetChatMessageResponse(delta_text="public").SerializeToString())]
        for name, flags, payload in cases:
            with self.subTest(trailer=name), tempfile.TemporaryDirectory() as directory:
                capture = CaptureSession(directory).request(name, [key])
                try:
                    with io.BytesIO(frame_message(payload, flags)) as response:
                        self.assertEqual(list(iter_chat_frames(response)), [(flags, payload)])
                finally:
                    capture.finish()
                saved_path = capture.directory / "devin.chat.response.connect"
                if name not in ("json", "malformed"):
                    saved = ConnectFrameDecoder().decode(saved_path.read_bytes())
                    raw = gzip.decompress(saved[0][1]) if flags & 1 else saved[0][1]
                    self.assertEqual(devin_pb2.GetChatMessageResponse.FromString(raw).delta_text,
                                     "public" if name == "shadowed" else "[REDACTED]")
                elif name == "json":
                    saved = ConnectFrameDecoder().decode(saved_path.read_bytes())
                    self.assertEqual(json.loads(saved[0][1]), {"error": {"message": "[REDACTED]"}})
                else:
                    self.assertFalse(saved_path.exists())
                    timeline = [json.loads(line) for line in (capture.directory / "timeline.jsonl").read_text().splitlines()]
                    self.assertTrue(any(row["event"].endswith(".capture_decode_failed") and
                                        row["opaque_bytes_omitted"] for row in timeline))
                self.assertNotIn(key[:75].encode(), b"".join(path.read_bytes() for path in capture.directory.iterdir()))
        clean_frame = frame_message(gzip.compress(first.SerializeToString()), 1)
        wire = clean_frame + frame_message(echo.SerializeToString(), 0) + frame_message(b"{}", 2)
        chat_requests = []
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                encoded = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path.endswith("GetUserJwt"):
                    body = devin_pb2.GetUserJwtResponse(user_jwt="private-session-jwt").SerializeToString()
                    content_type = "application/proto"
                else:
                    chat_requests.append(devin_pb2.GetChatMessageRequest.FromString(gzip.decompress(encoded[5:])))
                    body, content_type = wire, "application/connect+proto"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            outputs = []
            for enabled in (False, True):
                config = Config(argparse.Namespace(provider="auto", devin_upstream=upstream,
                    devin_api_key=key, model_discovery=False,
                    capture_dir=directory if enabled else None, log_file=str(Path(directory) / "gateway.log")))
                server = MuseShimServer(("127.0.0.1", 0), config)
                with serving(server) as base:
                    body = {"model": "devin/swe-1-6", "input": "hello", "stream": True}
                    headers = {"x-tbh-session-id": "stable-conversation"}
                    outputs.append(post(base, body, headers))
                    if enabled:
                        self.assertEqual(post(base, body, headers), outputs[-1])
                if enabled:
                    request_dir = next(path for path in server.capture_session.directory.iterdir() if path.is_dir())
                    stored = (request_dir / "devin.chat.response.connect").read_bytes()
                    self.assertTrue(stored.startswith(clean_frame))
                    frames = ConnectFrameDecoder().decode(stored)
                    self.assertEqual(len(frames), 3)
                    self.assertEqual(frames[2], (2, b"{}"))
                    self.assertEqual(devin_pb2.GetChatMessageResponse.FromString(frames[1][1]).delta_text, "[REDACTED]")
                    self.assertNotIn(key[:75].encode(), (request_dir / "downstream.body").read_bytes())
                    contents = b"".join(path.read_bytes() for path in request_dir.iterdir())
                    self.assertNotIn(key[:75].encode(), contents)
                    self.assertNotIn(b"private-session-jwt", contents)
            self.assertEqual(outputs[0], outputs[1])
            self.assertIn(key[:-1].encode(), outputs[0])
            self.assertEqual(len({request.cascade_id for request in chat_requests}), 1)
            self.assertEqual(len({request.execution_id for request in chat_requests}), 3)

    def test_catalog_is_bound_to_the_credential_used_for_discovery(self):
        import muse_devin
        from muse_gateway import provider_model_catalog
        config = Config(argparse.Namespace(provider="auto", model_discovery=True, devin_api_key="account-a"))
        def discover(token, *args):
            model = muse_devin.DevinModel(uid=token, logical_id=token, label=token,
                                          context_limit=200000, output_limit=64000)
            return [model], 0, []
        with patch("muse_devin.discover_models", side_effect=discover) as discovery:
            first = provider_model_catalog(config, "devin")
            config.devin_api_key = "account-b"
            second = provider_model_catalog(config, "devin")
            config.devin_api_key = "account-a"
            repeated = provider_model_catalog(config, "devin")
        self.assertEqual(first[0]["model_id"], "devin/account-a")
        self.assertEqual(second[0]["model_id"], "devin/account-b")
        self.assertEqual(repeated, first)
        self.assertEqual(discovery.call_count, 2)
        from concurrent.futures import ThreadPoolExecutor
        from muse_gateway import _devin_catalog_entries, _devin_cache_key
        barrier = threading.Barrier(2)
        class SynchronizedModels(list):
            first_iteration = True
            def __iter__(self):
                if self.first_iteration:
                    self.first_iteration = False
                    barrier.wait(timeout=3)
                return super().__iter__()
        def concurrent_discovery(token, *args):
            rows, dropped, warnings = discover(token)
            return SynchronizedModels(rows), dropped, warnings
        with patch("muse_devin.discover_models", side_effect=concurrent_discovery):
            with ThreadPoolExecutor(max_workers=2) as executor:
                jobs = [executor.submit(_devin_catalog_entries, config, token)
                        for token in ("concurrent-a", "concurrent-b")]
                for job in jobs:
                    job.result(timeout=5)
        for token in ("concurrent-a", "concurrent-b"):
            roster, _ = config._devin_rosters[_devin_cache_key(config, token)]
            self.assertEqual(set(roster), {token})

    def test_upstream_disconnect_before_headers_returns_502(self):
        from http.client import RemoteDisconnected
        from muse_devin import DevinModel
        for boundary in ("fetch_auth", "assign_if_router", "open_chat_stream"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as directory:
                model = DevinModel(uid="swe-1-6", logical_id="swe-1-6", label="Test",
                                   context_limit=200000, output_limit=64000,
                                   is_router=boundary == "assign_if_router")
                config = Config(argparse.Namespace(provider="auto", devin_api_key="fake-token",
                    model_discovery=False, log_file=str(Path(directory) / "gateway.log")))
                server = MuseShimServer(("127.0.0.1", 0), config)
                with patch("muse_devin.seed_models", return_value=[model]), patch(
                    "muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")
                ), patch("muse_devin." + boundary, side_effect=RemoteDisconnected("closed")):
                    with serving(server) as base:
                        with self.assertRaises(urllib.error.HTTPError) as raised:
                            post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
                        self.assertEqual(raised.exception.code, 502)
                        self.assertEqual(json.loads(raised.exception.read())["error"]["type"], "devin_chat_error")
                        raised.exception.close()
                self.assertNotIn("client_disconnected", (Path(directory) / "gateway.log").read_text())


if __name__ == "__main__":
    unittest.main()
