import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import gzip
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
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

    def test_protobuf_credentials_are_removed_before_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("protobuf")
            message = devin_pb2.GetChatMessageRequest(prompt="public prompt")
            message.metadata.api_key = "private-login-token"
            message.metadata.user_jwt = "private-user-jwt"
            message.model_assignment_jwt = "private-assignment-jwt"
            try:
                capture.protobuf("chat", message, message.SerializeToString())
            finally:
                capture.finish()
            saved = devin_pb2.GetChatMessageRequest.FromString((capture.directory / "chat.pb").read_bytes())
            self.assertEqual(saved.prompt, message.prompt)
            self.assertEqual(saved.metadata.api_key, "[REDACTED]")
            self.assertEqual(saved.metadata.user_jwt, "[REDACTED]")
            self.assertEqual(saved.model_assignment_jwt, "[REDACTED]")
            contents = b"".join(path.read_bytes() for path in capture.directory.iterdir())
            for secret in (b"private-login-token", b"private-user-jwt", b"private-assignment-jwt"):
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

    def test_error_capture_drain_failure_preserves_first_error_chunk(self):
        from muse_gateway import _capture_http_error
        first = b'{"error":{"code":"invalid_encrypted_content","message":"rejected"}}'
        error = SimpleNamespace(code=400, headers={}, read=unittest.mock.Mock(side_effect=[first, TimeoutError()]))
        with tempfile.TemporaryDirectory() as directory:
            capture = CaptureSession(directory).request("drain-timeout")
            try:
                self.assertEqual(_capture_http_error(error), first)
            finally:
                capture.finish()
            manifest = json.loads((capture.directory / "manifest.json").read_text())
            self.assertIn("upstream_error_capture_incomplete", manifest["capture_errors"])

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

    def test_codex_http_400_retains_full_error_and_responds(self):
        outgoing = []
        error_body = json.dumps({"error": {"type": "invalid_request_error", "code": "invalid_value",
            "param": "input[17].id", "message": "x" * 70000 + "FULL_ERROR_TAIL"}}).encode()
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                outgoing.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error_body)))
                self.end_headers()
                self.wfile.write(error_body)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            config = Config(argparse.Namespace(provider="codex", upstream=upstream,
                allow_oauth_host_override=True, model_discovery=False, capture_dir=directory,
                log_file=str(Path(directory) / "gateway.log")))
            server = MuseShimServer(("127.0.0.1", 0), config)
            with serving(server) as base, patch("muse_gateway.resolve_codex_credentials", return_value=SimpleNamespace(access_token="private-codex-token")):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    post(base, {"model": "gpt-test", "input": "hello", "stream": True})
                self.assertEqual(raised.exception.code, 400)
                with raised.exception as response:
                    self.assertEqual(json.load(response)["error"]["type"], "codex_upstream_error")
            request_dir = next(path for path in server.capture_session.directory.iterdir() if path.is_dir())
            self.assertEqual((request_dir / "upstream.error.body").read_bytes(), error_body)
            captured_request = json.loads((request_dir / "outgoing.body").read_bytes())
            self.assertEqual(captured_request, outgoing[0])
            self.assertFalse(captured_request["store"])
            self.assertIsInstance(captured_request["input"], list)
            self.assertNotIn("FULL_ERROR_TAIL", (Path(directory) / "gateway.log").read_text())
            contents = b"".join(path.read_bytes() for path in request_dir.iterdir())
            self.assertNotIn(b"private-codex-token", contents)

    def test_real_connect_route_captures_frames_without_changing_events(self):
        first = devin_pb2.GetChatMessageResponse(message_id="resp_capture", delta_text="Hello", stop_reason=2)
        wire = frame_message(gzip.compress(first.SerializeToString()), 1) + frame_message(b"{}", 2)
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
                    devin_api_key="private-devin-token", model_discovery=False,
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
                    self.assertEqual((request_dir / "devin.chat.response.connect").read_bytes(), wire)
                    self.assertEqual((request_dir / "downstream.body").read_bytes(), outputs[-1])
                    contents = b"".join(path.read_bytes() for path in request_dir.iterdir())
                    self.assertNotIn(b"private-devin-token", contents)
                    self.assertNotIn(b"private-session-jwt", contents)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(len({request.cascade_id for request in chat_requests}), 1)
            self.assertEqual(len({request.execution_id for request in chat_requests}), 3)

    def test_codex_recovers_once_from_rejected_encrypted_state(self):
        attempts = []
        wire = b'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                attempts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                rejected = len(attempts) == 1
                body = json.dumps({"error": {"code": "invalid_encrypted_content", "message": "cannot decrypt"}}).encode() if rejected else wire
                self.send_response(400 if rejected else 200)
                self.send_header("Content-Type", "application/json" if rejected else "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        history = [
            {"type": "message", "role": "user", "content": "remember me"},
            {"type": "reasoning", "id": "rs_response:rs_item", "encrypted_content": "foreign-state",
             "summary": [{"type": "summary_text", "text": "visible reasoning"}]},
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "visible result"},
            {"type": "item_reference", "id": "rs_response:rs_item"},
        ]
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            config = Config(argparse.Namespace(provider="codex", upstream=upstream,
                allow_oauth_host_override=True, model_discovery=False, capture_dir=directory,
                log_file=str(Path(directory) / "gateway.log")))
            server = MuseShimServer(("127.0.0.1", 0), config)
            with serving(server) as base, patch("muse_gateway.resolve_codex_credentials", return_value=SimpleNamespace(access_token="fake-codex-token")):
                self.assertEqual(post(base, {"model": "gpt-test", "input": history, "stream": True}), wire)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["input"][1]["encrypted_content"], "foreign-state")
            self.assertEqual(attempts[1]["input"][0], history[0])
            self.assertEqual(attempts[1]["input"][1]["content"][0]["text"], "visible reasoning")
            self.assertEqual(attempts[1]["input"][2:], history[2:-1])
            self.assertFalse(any(item.get("type") == "item_reference" for item in attempts[1]["input"]))
            self.assertEqual(history[1]["encrypted_content"], "foreign-state")
            request_dir = next(path for path in server.capture_session.directory.iterdir() if path.is_dir())
            self.assertEqual(json.loads((request_dir / "outgoing.recovered.body").read_bytes()), attempts[1])
            self.assertIn("codex_reasoning_demoted", (Path(directory) / "gateway.log").read_text())

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

    def test_partial_connect_frame_fails_instead_of_completing(self):
        message = devin_pb2.GetChatMessageResponse(message_id="resp_truncated", delta_text="partial")
        wire = frame_message(message.SerializeToString()) + b"\x00\x00\x00\x00\x10short"
        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                auth = self.path.endswith("GetUserJwt")
                body = devin_pb2.GetUserJwtResponse(user_jwt="fake-jwt").SerializeToString() if auth else wire
                self.send_response(200)
                self.send_header("Content-Type", "application/proto" if auth else "application/connect+proto")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), Upstream)) as upstream:
            config = Config(argparse.Namespace(provider="auto", devin_upstream=upstream,
                devin_api_key="fake-token", model_discovery=False, capture_dir=directory,
                log_file=str(Path(directory) / "gateway.log")))
            server = MuseShimServer(("127.0.0.1", 0), config)
            with serving(server) as base:
                body = post(base, {"model": "devin/swe-1-6", "input": "hello", "stream": True})
            self.assertIn(b"response.failed", body)
            self.assertNotIn(b"response.completed", body)
            self.assertNotIn(b"event: error\n", body)
            request_dir = next(path for path in server.capture_session.directory.iterdir() if path.is_dir())
            manifest = json.loads((request_dir / "manifest.json").read_text())
            self.assertTrue(manifest["outcome"]["stream_failed"])
            self.assertEqual(manifest["terminal_events"]["downstream"], ["response.failed"])

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
