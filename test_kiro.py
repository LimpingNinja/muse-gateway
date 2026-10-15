"""Kiro contracts through real HTTP boundaries, without provider accounts."""

import argparse
import copy
from contextlib import ExitStack
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from muse_gateway import Config, MuseShimServer
from muse_capture import RequestCapture
import muse_kiro
from muse_providers import ProviderError
from tests.contracts import consume
from tests.peers import anthropic_text, post, request, serving


def event(kind, **payload):
    return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **payload}) + "\n\n").encode()


def function_tool(name="echo"):
    return {"type": "function", "name": name, "parameters": {"type": "object"}}


def inline_tool_history(name="echo"):
    return [{"role": "user", "content": "Earlier request"},
            {"type": "function_call", "call_id": "call_old", "name": name, "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_old", "output": "literal tool result"},
            {"role": "user", "content": "Continue"}]


def usage_response(usage, phase):
    message = {"id": "msg_usage", "type": "message", "role": "assistant", "content": []}
    if phase == "json":
        return (200, "application/json", json.dumps({**message, "content": [{"type": "text", "text": "usage"}],
                                                     "stop_reason": "end_turn", "usage": usage}).encode())
    start = {"usage": usage} if phase == "message_start" else {
        "usage": {"input_tokens": 2, "output_tokens": 1, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 5}}
    delta = {"usage": usage} if phase == "message_delta" else {}
    return (event("message_start", message={**message, **start})
            + event("content_block_start", index=0, content_block={"type": "text", "text": "usage"})
            + event("content_block_stop", index=0)
            + event("message_delta", delta={"stop_reason": "end_turn"}, **delta) + event("message_stop"))


def tool_response():
    return (event("message_start", message={"id": "msg_tool", "role": "assistant", "content": []})
            + event("content_block_start", index=0, content_block={"type": "tool_use", "id": "call_echo", "name": "echo", "input": {}})
            + event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"text":'})
            + event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '"OK"}'})
            + event("content_block_stop", index=0)
            + event("message_delta", delta={"stop_reason": "tool_use"})
            + event("message_stop"))


def interleaved_response(*, stream=True):
    blocks = [{"type": "text", "text": "Before the tool."},
              {"type": "tool_use", "id": "call_interleaved", "name": "echo", "input": {"text": "OK"}},
              {"type": "text", "text": "After the tool."}]
    if not stream:
        return (200, "application/json", json.dumps({"type": "message", "id": "msg_interleaved", "role": "assistant",
                                                    "content": blocks, "stop_reason": "tool_use"}).encode())
    wire = event("message_start", message={"id": "msg_interleaved", "role": "assistant", "content": []})
    for index, block in enumerate(blocks):
        wire += event("content_block_start", index=index, content_block=block) + event("content_block_stop", index=index)
    return wire + event("message_delta", delta={"stop_reason": "tool_use"}) + event("message_stop")


class MessagesPeer:
    def __init__(self, turns=(), *, hold_after=None, catalog_status=200, catalog=None, catalog_mime="application/json"):
        self.turns = list(turns)
        self.requests = []
        self.catalog_requests = []
        self.errors = []
        self.hold_after = hold_after
        self.catalog_status = catalog_status
        self.catalog = catalog
        self.catalog_mime = catalog_mime
        self.release = threading.Event()

    def handler(self):
        peer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                peer.catalog_requests.append((self.path, dict(self.headers)))
                body = peer.catalog if peer.catalog is not None else json.dumps({"data": [{"id": "claude-haiku-4.5"}, {"id": "glm-5"}]}).encode()
                self.reply(peer.catalog_status, peer.catalog_mime, body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                peer.requests.append((self.path, dict(self.headers), body))
                if not peer.turns:
                    peer.errors.append("unexpected extra inference request")
                    self.reply(500, "application/json", b"{}")
                    return
                turn = peer.turns.pop(0)
                if callable(turn):
                    turn = turn(body)
                status, mime, wire = turn[:3] if isinstance(turn, tuple) else (200, "text/event-stream", turn)
                headers = turn[3] if isinstance(turn, tuple) and len(turn) == 4 else ()
                self.reply(status, mime, wire, headers)

            def reply(self, status, mime, wire, headers=()):
                self.send_response(status)
                self.send_header("Content-Type", mime)
                for name, value in headers:
                    self.send_header(name, value)
                # An open HTTP body deliberately outlives semantic completion.
                if not any(name.lower() == "transfer-encoding" for name, _ in headers):
                    self.send_header("Content-Length", str(len(wire) + (1 if peer.hold_after is not None else 0)))
                self.end_headers()
                try:
                    if peer.hold_after is None:
                        self.wfile.write(wire)
                    else:
                        self.wfile.write(wire[:peer.hold_after])
                        self.wfile.flush()
                        peer.release.wait(6)
                        self.wfile.write(wire[peer.hold_after:] + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
        return Handler


class KiroContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        clean = {name: value for name, value in os.environ.items()
                 if not name.upper().startswith(("MUSE_", "META_", "CODEX_", "CLAUDE_", "ANTHROPIC_", "DEVIN_", "KIRO_"))
                 and name.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
        clean["NO_PROXY"] = "127.0.0.1,localhost"
        self.stack.enter_context(patch.dict(os.environ, clean, clear=True))
        for boundary in ("discover_codex_credentials", "discover_claude_credentials",
                         "resolve_codex_credentials", "resolve_claude_credentials"):
            self.stack.enter_context(patch("muse_gateway." + boundary,
                side_effect=AssertionError("Kiro must never inspect native credentials")))

    def start(self, peer, **overrides):
        upstream = self.stack.enter_context(serving(ThreadingHTTPServer(("127.0.0.1", 0), peer.handler())))
        options = dict(provider="kiro", upstream=upstream + "/v1", kiro_upstream=upstream + "/v1",
                       kiro_api_key="synthetic-proxy-credential", model_discovery=False,
                       log_file=str(Path(self.directory) / "gateway.log"), timeout=2)
        options.update(overrides)
        self.config = Config(argparse.Namespace(**options))
        base = self.stack.enter_context(serving(MuseShimServer(("127.0.0.1", 0), self.config)))
        self.addCleanup(peer.release.set)
        self.addCleanup(lambda: self.assertEqual(peer.errors, [], "fixture failed"))
        return base

    def test_route_uses_only_proxy_key_and_shared_conversion(self):
        peer = MessagesPeer([anthropic_text("KIRO_OK")])
        base = self.start(peer, provider="auto", api_key="generic-secret", access_token="native-secret")
        req = request(base, {"model": "kiro/claude-haiku-4.5", "input": "hello", "stream": True})
        req.add_header("Authorization", "Bearer muse-subscription-secret")
        with urllib.request.urlopen(req, timeout=3) as response:
            final = consume(response.read())
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["output"][0]["content"][0]["text"], "KIRO_OK")
        path, headers, body = peer.requests[0]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(headers["Authorization"], "Bearer synthetic-proxy-credential", "KIRO_AUTH: wrong credential")
        self.assertNotIn("muse-subscription-secret", json.dumps(peer.requests))
        self.assertNotIn("native-secret", json.dumps(peer.requests))
        self.assertEqual(body["model"], "claude-haiku-4.5")
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(self.config.resolve_model("claude-haiku-4.5"), ("claude", "claude-haiku-4.5"))
        self.assertEqual(self.config.resolve_model("kiro/auto"), ("kiro", "auto"))

    def test_inline_tool_replay_keeps_history_and_drops_only_redundant_state(self):
        peer = MessagesPeer([tool_response(), anthropic_text("REPLAY_OK")])
        base = self.start(peer)
        tools = [{"type": "function", "name": "echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}]
        history = [{"role": "developer", "content": "DEVELOPER_SENTINEL"}, {"role": "user", "content": "Echo OK"}]
        _, wire = post(base, {"model": "kiro/claude-haiku-4.5", "input": history, "tools": tools, "stream": True})
        first = consume(wire)
        call = first["output"][0]
        self.assertEqual(call["name"], "echo")
        self.assertEqual(json.loads(call["arguments"]), {"text": "OK"})
        giant = ('BEGIN_SENTINEL\n1|C:\\fixture\\report.txt\n'
                 '{"exit_code":7,"is_error":true,"output":"ordinary file contents"}\n'
                 + "x" * 340000 + "END_SENTINEL")
        history += [{"type": "reasoning", "id": "rs_foreign", "encrypted_content": "foreign-state",
                     "summary": [{"type": "summary_text", "text": "VISIBLE_SENTINEL"}]},
                    {"type": "item_reference", "id": "rs_foreign"}, call,
                    {"type": "function_call_output", "call_id": call["call_id"], "output": giant},
                    {"role": "user", "content": "Continue"}]
        original = json.loads(json.dumps(history))
        _, wire = post(base, {"model": "kiro/claude-haiku-4.5", "input": history, "tools": tools,
                              "previous_response_id": first["id"], "stream": True})
        self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "REPLAY_OK")
        self.assertEqual(len(peer.requests), 2, "KIRO_REPLAY: extra upstream inference request")
        outgoing = peer.requests[-1][2]
        self.assertNotIn("previous_response_id", outgoing)
        self.assertNotIn("foreign-state", json.dumps(outgoing))
        self.assertIn("VISIBLE_SENTINEL", json.dumps(outgoing))
        self.assertIn("DEVELOPER_SENTINEL", json.dumps(outgoing["system"]))
        calls = [part for message in outgoing["messages"] for part in message["content"] if part["type"] == "tool_use"]
        self.assertEqual([part["id"] for part in calls], [call["call_id"]], "KIRO_REPLAY: call ID changed")
        results = [part for message in outgoing["messages"] for part in message["content"] if part["type"] == "tool_result"]
        self.assertEqual(len(results), 1, "KIRO_REPLAY: tool result count changed")
        self.assertEqual(results[0]["tool_use_id"], call["call_id"], "KIRO_REPLAY: tool ID changed")
        self.assertEqual(results[0]["content"], giant, "KIRO_REPLAY: tool result lost")
        self.assertEqual(history, original)

    def test_completion_does_not_wait_for_http_eof(self):
        wire = anthropic_text("READY")
        peer = MessagesPeer([wire], hold_after=len(wire))
        base = self.start(peer)
        try:
            with urllib.request.urlopen(request(base, {"model": "kiro/claude-haiku-4.5", "input": "hi", "stream": True}), timeout=1) as response:
                final = consume(response.read())
            self.assertEqual(final["status"], "completed")
            self.assertFalse(peer.release.is_set())
        except (TimeoutError, OSError) as error:
            self.fail("KIRO_COMPLETION: waiting for upstream EOF: " + type(error).__name__)
        finally:
            peer.release.set()

    def test_text_arrives_while_peer_holds_completion(self):
        wire = anthropic_text("SMALL_FRAME_\u03bb")
        peer = MessagesPeer([wire], hold_after=wire.index(b"event: content_block_stop"))
        base = self.start(peer)
        received = bytearray()
        try:
            with urllib.request.urlopen(request(base, {"model": "kiro/claude-haiku-4.5", "input": "hi", "stream": True}), timeout=1) as response:
                while True:
                    line = response.readline()
                    self.assertTrue(line, "stream ended before its text delta")
                    received.extend(line)
                    if line.startswith(b"event: response.output_text.delta"):
                        payload = response.readline()
                        received.extend(payload)
                        self.assertEqual(json.loads(payload[5:])["delta"], "SMALL_FRAME_\u03bb")
                        break
                self.assertFalse(peer.release.is_set(), "text was buffered until completion")
                peer.release.set()
                received.extend(response.read())
            self.assertEqual(consume(bytes(received))["status"], "completed")
        finally:
            peer.release.set()

    def test_http_error_evidence_survives_the_public_boundary_without_retry(self):
        cases = [
            (429, "rate_limit_error", None, "rate_limit", True, [("Retry-After", "1")]),
            (401, "authentication_error", None, "authentication", False, []),
            (403, "permission_error", None, "permission", False, []),
            (413, "invalid_request_error", None, "request", False, []),
            (429, "rate_limit_error", "insufficient_quota", "quota", False, [("Retry-After", "1")]),
            (529, "overloaded_error", None, "overload", True, []),
            (400, "invalid_request_error", "context_length_exceeded", "context", False, []),
            (503, "authentication_error", None, "authentication", False, [("Retry-After", "1")]),
            (401, "overloaded_error", None, "authentication", False, [("Retry-After", "1")]),
            (400, "future_error", None, "unknown", None, []),
            (404, "api_error", "future_code", "unknown", None, []),
        ]
        turns = []
        for status, kind, code, _, _, headers in cases:
            details = {"type": kind, "message": "private prose authentication quota busy", "param": "messages.0.content"}
            if code:
                details["code"] = code
            payload = {"type": "error", "error": details, "request_id": "req_body"}
            if kind == "authentication_error" and status == 401:
                payload = {"detail": payload}  # Stock FastAPI HTTPException envelope.
            turns.append((status, "application/json", json.dumps(payload).encode(), [*headers, ("request-id", "req_header")]))
        peer = MessagesPeer(turns)
        base = self.start(peer)
        for count, (status, kind, code, category, transient, headers) in enumerate(cases, 1):
            with self.subTest(status=status, kind=kind, code=code), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
            with caught.exception as response:
                self.assertEqual(response.code, status)
                self.assertEqual(response.headers.get("Retry-After"), "1" if headers else None)
                error = json.load(response)["error"]
            self.assertEqual(error["type"], "kiro_upstream_error")
            self.assertEqual(error["code"], code or kind)
            metadata = error["metadata"]
            self.assertEqual((metadata["provider"], metadata["layer"], metadata["source"], metadata["phase"]),
                             ("kiro", "kiro_gateway", "http", "before_response"))
            self.assertEqual((metadata["upstream_status"], metadata["category"], metadata["transient"], metadata["output_exposure"]),
                             (status, category, transient, "none"))
            self.assertEqual(metadata["upstream"]["type"], kind)
            self.assertEqual(metadata["upstream"]["param"], "messages.0.content")
            self.assertEqual(metadata["upstream"]["body_request_id"], "req_body")
            self.assertEqual(metadata["upstream"]["header_request_id"], "req_header")
            self.assertTrue(metadata["upstream"]["request_id_conflict"])
            self.assertNotEqual(error["request_id"], "req_body")
            if status == 503 or status == 401 and kind == "overloaded_error":
                self.assertTrue(metadata["evidence_conflict"])
            elif code in {"insufficient_quota", "context_length_exceeded"} or kind == "invalid_request_error":
                self.assertNotIn("evidence_conflict", metadata, "specific codes refine generic error types")
            self.assertNotIn("private prose", json.dumps(error))
            self.assertEqual(len(peer.requests), count, "classification must not replay inference")

    def test_retry_after_dates_and_rejected_metadata_are_passive(self):
        accepted = ["4", "Wed, 21 Oct 2015 07:28:00 GMT"]
        rejected = ["-1", "NaN", "Wed, 21 Oct 2015 07:28:00 PST", "synthetic-proxy-credential"]
        headers = [[("Retry-After", value)] for value in accepted + rejected]
        headers += [[("Retry-After", "1"), ("Retry-After", "1")], [("Retry-After", "1\r\n\t2")]]
        payload = b'{"type":"error","error":{"type":"rate_limit_error","message":"private"}}'
        peer = MessagesPeer([(429, "application/json", payload, items) for items in headers])
        base = self.start(peer)
        for count, items in enumerate(headers, 1):
            with self.subTest(headers=items), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "kiro/haiku", "input": "hi", "stream": False})
            with caught.exception as response:
                self.assertEqual(response.code, 429)
                error = json.load(response)["error"]
                if count <= len(accepted):
                    value = accepted[count - 1]
                    self.assertEqual(response.headers.get("Retry-After"), value)
                    self.assertEqual(error["metadata"]["retry_after"],
                                     {"value": value, "kind": "delay-seconds" if value.isascii() and value.isdigit() else "http-date"})
                    self.assertTrue(error["metadata"]["transient"])
                else:
                    self.assertIsNone(response.headers.get("Retry-After"))
                    self.assertNotIn("retry_after", error["metadata"])
                    self.assertIsNone(error["metadata"]["transient"])
            self.assertEqual(len(peer.requests), count)

    def test_sse_and_json_typed_errors_preserve_category_and_observed_status(self):
        cases = [("authentication_error", "authentication", False), ("overloaded_error", "overload", True),
                 ("rate_limit_error", "rate_limit", None), ("invented_error", "unknown", None)]
        turns = []
        for stream in (True, False):
            for kind, _, _ in cases:
                payload = {"type": "error", "error": {"type": kind, "message": "private prose"}, "request_id": "req_same"}
                wire = event("error", error=payload["error"], request_id="req_same") if stream else json.dumps(payload).encode()
                turns.append((200, "text/event-stream" if stream else "application/json", wire, [("request-id", "req_same")]))
        peer = MessagesPeer(turns)
        base = self.start(peer)
        count = 0
        for stream in (True, False):
            for kind, category, transient in cases:
                count += 1
                with self.subTest(stream=stream, kind=kind), self.assertRaises(urllib.error.HTTPError) as caught:
                    post(base, {"model": "kiro/haiku", "input": "hi", "stream": stream})
                with caught.exception as response:
                    self.assertEqual(response.code, 502, "first pass must preserve native status/retry behavior")
                    error = json.load(response)["error"]
                self.assertEqual(error["code"], kind)
                metadata = error["metadata"]
                self.assertEqual((metadata["source"], metadata["upstream_status"], metadata["category"], metadata["transient"]),
                                 ("sse" if stream else "json", 200, category, transient))
                self.assertNotIn("evidence_conflict", metadata)
                self.assertNotIn("request_id_conflict", metadata["upstream"])
                self.assertNotIn("private prose", json.dumps(error))
                self.assertEqual(len(peer.requests), count)

    def test_partial_typed_error_keeps_one_lifecycle_and_private_tools_unpublished(self):
        wire = anthropic_text("partial")
        wire = wire[:wire.index(b"event: message_delta")]
        wire += event("content_block_start", index=1, content_block={"type": "tool_use", "id": "call_private", "name": "echo", "input": {}})
        wire += event("content_block_delta", index=1, delta={"type": "input_json_delta", "partial_json": '{"private":'})
        wire += event("error", error={"type": "overloaded_error", "message": "private error prose"})
        peer = MessagesPeer([wire])
        base = self.start(peer)
        with urllib.request.urlopen(request(base, {"model": "kiro/haiku", "input": "hi", "stream": True, "tools": [function_tool()]}), timeout=3) as response:
            self.assertEqual(response.status, 200)
            downstream = response.read()
        final = consume(downstream)  # Independently checks IDs, contiguous sequences and one terminal.
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"]["type"], "kiro_upstream_error")
        self.assertEqual(final["error"]["code"], "overloaded_error")
        self.assertEqual(final["error"]["metadata"]["category"], "overload")
        self.assertEqual(final["error"]["metadata"]["output_exposure"], "content")
        self.assertEqual(final["error"]["metadata"]["phase"], "stream")
        self.assertEqual(final["error"]["metadata"]["upstream_status"], 200)
        self.assertEqual(final["output"][0]["content"][0]["text"], "partial")
        self.assertNotIn(b"call_private", downstream)
        self.assertNotIn(b"private error prose", downstream)
        self.assertEqual(len(peer.requests), 1)

    def test_error_metadata_cannot_echo_secrets_invalid_fields_or_guess_from_prose(self):
        bodies = [
            {"type": "error", "error": {"type": "synthetic-proxy-credential", "code": "generic-secret", "param": "muse-secret", "message": "authentication_error overloaded_error insufficient_quota"}, "request_id": "req_synthetic-proxy-credential"},
            {"type": "error", "error": {"type": ["authentication_error"], "code": 429, "param": {"x": "y"}}, "request_id": "a" * 200},
            {"type": "error", "error": {"type": "bad\nheader", "code": "code with spaces", "param": "x" * 200}, "request_id": "bad\rvalue"},
        ]
        peer = MessagesPeer([(400, "application/json", json.dumps(body).encode(), [("request-id", "muse-secret")]) for body in bodies]
                            + [(429, "application/json", b'{"error":{"type":"authentication_error","type":"overloaded_error"}}')])
        base = self.start(peer, api_key="generic-secret")
        for count in range(1, len(bodies) + 2):
            req = request(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
            req.add_header("Authorization", "Bearer muse-secret")
            with self.subTest(case=count), self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=3)
            with caught.exception as response:
                error = json.load(response)["error"]
            self.assertEqual(error["metadata"]["category"], "unknown")
            self.assertIsNone(error["metadata"]["transient"])
            self.assertEqual(error["metadata"]["upstream"], {})
            for secret in ("synthetic-proxy-credential", "generic-secret", "muse-secret"):
                self.assertNotIn(secret, json.dumps(error))
            self.assertEqual(len(peer.requests), count)

    def test_unreadable_error_bodies_keep_bounded_header_evidence_in_wire_logs_and_capture(self):
        payload = b'{"type":"error","error":{"type":"overloaded_error","message":"private-body-prose"}}'
        chunk = (f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n")
        cases = [
            (payload, (), len(payload)),  # Complete-looking JSON, HTTP entity still incomplete.
            (chunk, (("Transfer-Encoding", "chunked"),), None),  # Deliberately never drained.
            (payload + b" " * 70000, (), None),  # Beyond the passive prefix limit.
            (b"not JSON", (), None),
        ]
        captures = Path(self.directory) / "captures"
        for index, (wire, extra_headers, hold_after) in enumerate(cases):
            peer = MessagesPeer([(429, "application/json", wire,
                                  [*extra_headers, ("Retry-After", "1"), ("request-id", "req_header")])], hold_after=hold_after)
            base = self.start(peer, capture_dir=str(captures))
            with self.subTest(case=index), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
            with caught.exception as response:
                self.assertEqual(response.code, 429)
                self.assertEqual(response.headers.get("Retry-After"), "1")
                error = json.load(response)["error"]
            metadata = error["metadata"]
            self.assertEqual(metadata["upstream_status"], 429)
            self.assertEqual(metadata["category"], "unknown", "an unread complete-looking prefix is not typed evidence")
            self.assertEqual(metadata["upstream"], {"header_request_id": "req_header"})
            self.assertEqual(metadata["retry_after"], {"value": "1", "kind": "delay-seconds"})
            self.assertNotIn("private-body-prose", json.dumps(error))
            self.assertEqual(len(peer.requests), 1)
            peer.release.set()
        timelines = list(captures.rglob("timeline.jsonl"))
        self.assertEqual(len(timelines), len(cases))
        for path in timelines:
            records = [json.loads(line) for line in path.read_text().splitlines()]
            failures = [entry for entry in records if entry["event"] == "kiro.upstream.failed"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["metadata"]["upstream"], {"header_request_id": "req_header"})
            self.assertEqual(failures[0]["metadata"]["upstream_status"], 429)
        logs = (Path(self.directory) / "gateway.log").read_text()
        records = [json.loads(line[line.index("{"):]) for line in logs.splitlines() if "{" in line]
        errors = [entry for entry in records if entry["event"] == "provider_error"]
        self.assertEqual(len(errors), len(cases))
        self.assertTrue(all(entry["metadata"]["upstream_status"] == 429 for entry in errors))
        self.assertNotIn("private-body-prose", logs)
        self.assertNotIn("synthetic-proxy-credential", logs)

    def test_error_classification_is_independent_of_optional_capture_storage(self):
        payload = {"type": "error", "error": {"type": "api_error", "code": "insufficient_quota", "message": "private prose"},
                   "request_id": "req_body"}
        original_bytes = RequestCapture.bytes
        original_finish = RequestCapture.finish
        finished = threading.Event()

        def unavailable_error_body(capture, name, data):
            if name == "upstream.error.body":
                raise OSError("private-recorder-error")
            return original_bytes(capture, name, data)

        def finish_capture(capture, *args, **kwargs):
            result = original_finish(capture, *args, **kwargs)
            finished.set()
            return result

        baseline = None
        for mode in ("disabled", "enabled", "failed"):
            finished.clear()
            captures = Path(self.directory) / mode
            peer = MessagesPeer([(403, "application/json", json.dumps(payload).encode(),
                                  [("Retry-After", "1"), ("request-id", "req_header")])])
            base = self.start(peer, capture_dir=None if mode == "disabled" else str(captures))
            with ExitStack() as stack:
                if mode != "disabled":
                    stack.enter_context(patch.object(RequestCapture, "finish", new=finish_capture))
                if mode == "failed":
                    stack.enter_context(patch.object(RequestCapture, "bytes", new=unavailable_error_body))
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
                with caught.exception as response:
                    error = json.load(response)["error"]
                    error.pop("request_id")  # Local correlation is unique per request.
                    observed = (response.code, response.headers.get("Retry-After"), error)
                if mode != "disabled":
                    self.assertTrue(finished.wait(2), "capture did not finish")
            self.assertEqual(len(peer.requests), 1)
            if baseline is None:
                baseline = observed
                self.assertEqual(error["code"], "insufficient_quota")
                self.assertEqual(error["metadata"]["category"], "quota")
                self.assertFalse(error["metadata"]["transient"])
            else:
                self.assertEqual(observed, baseline, "optional capture storage must not change client error evidence")
            if mode != "disabled":
                manifests = list(captures.rglob("manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text())
                records = [json.loads(line) for line in (manifests[0].parent / "timeline.jsonl").read_text().splitlines()]
                rejected = next(record for record in records if record["event"] == "upstream.error.body.rejected")
                self.assertEqual(rejected["complete"], mode == "enabled")
                if mode == "failed":
                    self.assertIn("kiro_error_capture_failed", manifest["capture_errors"])
                    self.assertIn("kiro_http_error_body_incomplete", manifest["capture_errors"])
                    self.assertNotIn("private-recorder-error", json.dumps(manifest))

    def test_frontend_bearer_spacing_does_not_leak_credentials_in_error_metadata(self):
        cases = [("bEaReR   audit_frontend_secret  ", "audit_frontend_secret", True),
                 ("Bearer  audit frontend secret  ", "auditfrontendsecret", False)]
        turns = []
        for _, marker, _ in cases:
            payload = {"type": "error", "error": {"type": "api_error", "code": marker, "param": marker}, "request_id": marker}
            turns.append((400, "application/json", json.dumps(payload).encode(), [("request-id", marker)]))
        peer = MessagesPeer(turns)
        base = self.start(peer)
        for count, (authorization, marker, excluded) in enumerate(cases, 1):
            req = request(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
            req.add_header("Authorization", authorization)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=3)
            with caught.exception as response:
                self.assertEqual(response.code, 400)
                error = json.load(response)["error"]
            if excluded:
                self.assertEqual(error["code"], "api_error")
                self.assertEqual(error["metadata"]["upstream"], {"type": "api_error"})
                self.assertNotIn(marker, json.dumps(error))
            else:
                self.assertEqual(error["code"], marker, "interior spaces must not be removed from a credential")
                self.assertEqual(error["metadata"]["upstream"]["body_request_id"], marker)
            self.assertEqual(len(peer.requests), count)

    def test_redirect_stays_blocked_and_nested_correlation_ids_remain_distinct(self):
        body = {"request_id": "req_outer", "detail": {"type": "error", "request_id": "req_nested",
                 "error": {"type": "authentication_error", "message": "private"}}}
        peer = MessagesPeer([(302, "application/json", b"{}", [("Location", "/credential-sink")]),
                             (401, "application/json", json.dumps(body).encode(), [("request-id", "req_header")])])
        base = self.start(peer)
        for count, status in enumerate((502, 401), 1):
            with self.subTest(status=status), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "kiro/haiku", "input": "hi", "stream": True})
            with caught.exception as response:
                self.assertEqual(response.code, status)
                self.assertIsNone(response.headers.get("Location"))
                error = json.load(response)["error"]
            if count == 1:
                self.assertEqual(error["metadata"]["upstream_status"], 302)
            else:
                self.assertEqual(error["code"], "authentication_error")
                self.assertEqual(error["metadata"]["upstream"], {
                    "type": "authentication_error", "body_request_id": "req_outer",
                    "detail_request_id": "req_nested", "header_request_id": "req_header", "request_id_conflict": True})
            self.assertEqual(len(peer.requests), count)
        self.assertEqual([path for path, _, _ in peer.requests], ["/v1/messages", "/v1/messages"])

    def test_partial_malformed_and_error_streams_never_complete(self):
        valid = anthropic_text("partial")
        peer = MessagesPeer([valid[:valid.index(b"event: message_stop")],
                             valid[:valid.index(b"event: content_block_stop")] + b'event: content_block_delta\ndata: INVALID\n\n',
                             event("error", error={"type": "overloaded_error", "message": "busy"})])
        base = self.start(peer)
        for label in ("truncated", "malformed", "prestream error"):
            with self.subTest(label=label):
                try:
                    _, wire = post(base, {"model": "kiro/claude-haiku-4.5", "input": "hi", "stream": True})
                except urllib.error.HTTPError as error:
                    with error:
                        self.assertEqual(error.code, 502)
                else:
                    final = consume(wire)
                    self.assertEqual(final["status"], "failed", "KIRO_FAILURE: malformed stream reported success")
                    self.assertEqual(final["error"]["metadata"]["source"], "protocol")
                    self.assertEqual(final["error"]["metadata"]["category"], "protocol")
                    self.assertIsNone(final["error"]["metadata"]["transient"])
                    self.assertNotIn(b"event: error\n", wire)

    def test_stream_and_json_interleaved_output_round_trip_as_one_assistant_turn(self):
        peer = MessagesPeer([interleaved_response(), anthropic_text("CONTINUED"),
                             interleaved_response(stream=False), anthropic_text("CONTINUED")])
        base = self.start(peer)
        for stream in (True, False):
            with self.subTest(stream=stream):
                history = [{"role": "user", "content": "Use echo"}]
                _, raw = post(base, {"model": "kiro/haiku", "input": history, "tools": [function_tool()], "stream": stream})
                first = consume(raw) if stream else json.loads(raw)
                self.assertEqual(first["status"], "completed")
                history += first["output"] + [{"type": "function_call_output", "call_id": "call_interleaved", "output": "OK"}]
                _, raw = post(base, {"model": "kiro/haiku", "input": history,
                                     "previous_response_id": first["id"], "tools": [function_tool()], "stream": True})
                self.assertEqual(consume(raw)["status"], "completed")
                sent = peer.requests[-1][2]["messages"]
                self.assertEqual([part["type"] for part in sent[1]["content"]], ["text", "tool_use"])
                self.assertEqual(sent[1]["content"][0]["text"], "Before the tool.After the tool.")
                self.assertEqual(sent[2]["content"][0]["tool_use_id"], "call_interleaved")

    def test_history_wire_ids_are_injective_and_result_groups_cannot_cross_turns(self):
        def call(cid):
            return {"type": "function_call", "call_id": cid, "name": "echo", "arguments": "{}"}
        def result(cid):
            return {"type": "function_call_output", "call_id": cid, "output": "result:" + cid}
        peer = MessagesPeer([anthropic_text("PAIRED")])
        base = self.start(peer)
        ids = ("x:y", "muse_call_eDp5")
        history = [{"role": "user", "content": "two tools"}] + [call(cid) for cid in ids] + [result(cid) for cid in ids]
        original = copy.deepcopy(history)
        prepared, _ = muse_kiro.prepare_request({"model": "haiku", "input": history, "tools": [function_tool()]})
        self.assertEqual(history, original)
        _, raw = post(base, {"model": "kiro/haiku", "input": history, "tools": [function_tool()], "stream": True})
        self.assertEqual(consume(raw)["status"], "completed")
        sent = peer.requests[0][2]
        self.assertEqual(sent["messages"], prepared["messages"])
        calls = [part["id"] for message in sent["messages"] for part in message["content"] if part["type"] == "tool_use"]
        results = [part for message in sent["messages"] for part in message["content"] if part["type"] == "tool_result"]
        self.assertEqual(len(set(calls)), 2)
        self.assertEqual([part["tool_use_id"] for part in results], calls)
        self.assertEqual([part["content"] for part in results], ["result:" + cid for cid in ids])
        invalid = [call("a"), call("b"), result("a"), call("c"), result("b"), result("c")]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(base, {"model": "kiro/haiku", "input": [{"role": "user", "content": "bad grouping"}] + invalid, "stream": True})
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertEqual(json.load(response)["error"]["type"], "kiro_invalid_history")
        self.assertEqual(len(peer.requests), 1, "invalid result groups reached inference")

    def test_malformed_discriminators_are_400_without_upstream_calls(self):
        peer = MessagesPeer([anthropic_text("UNEXPECTED_DISPATCH")] * 7)
        base = self.start(peer)
        cases = [{"tool_choice": {"type": []}}, {"input": [{"role": [], "content": "hi"}]},
                 {"input": [{"type": [], "role": "user", "content": "hi"}]},
                 {"input": [{"role": "user", "content": [{"type": [], "text": "hi"}]}]}]
        cases += [{"parallel_tool_calls": value} for value in (False, 0, 1, "false", [], {})]
        for body in cases:
            with self.subTest(body=body):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post(base, {"model": "kiro/haiku", "input": "hello", **body, "stream": True})
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    error = json.load(response)["error"]
                    self.assertTrue(error["type"].startswith("kiro_"))
                    if "parallel_tool_calls" in body:
                        self.assertEqual(error["type"], "kiro_unsupported_option" if body["parallel_tool_calls"] is False else "kiro_invalid_request")
                        self.assertIn("parallel_tool_calls", error["message"])
        self.assertEqual(peer.requests, [])

    def test_surrogate_semantic_strings_have_explicit_terminal_failure(self):
        bad_id = event("message_start", message={"id": "msg_\ud800", "role": "assistant", "content": []})
        bad_name = (KiroMessages.START + event("content_block_start", index=0,
                    content_block={"type": "tool_use", "id": "call_bad", "name": "echo\ud800", "input": {}}))
        bad_argument = (KiroMessages.START + event("content_block_start", index=0,
                        content_block={"type": "tool_use", "id": "call_bad", "name": "echo", "input": {}})
                        + event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"text":"\\ud800"}'})
                        + event("content_block_stop", index=0))
        peer = MessagesPeer([b"data: \xff\n\n", bad_id, anthropic_text("\ud800"), bad_name, bad_argument])
        base = self.start(peer)
        for label in ("raw malformed UTF-8", "surrogate ID"):
            with self.subTest(label=label), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "kiro/haiku", "input": "hello", "stream": True})
            with caught.exception as response:
                self.assertEqual(response.code, 502)
        for label in ("text", "tool name", "decoded tool input"):
            with self.subTest(label=label):
                _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "tools": [function_tool()], "stream": True})
                self.assertEqual(consume(raw)["status"], "failed", "must emit a terminal, not merely close")

    def test_partial_tool_is_incomplete_only_for_max_tokens_and_never_executable(self):
        partial = (KiroMessages.START + event("content_block_start", index=0,
                   content_block={"type": "tool_use", "id": "call_partial", "name": "echo", "input": {}})
                   + event("content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"text":"unfinished'})
                   + event("content_block_stop", index=0))
        peer = MessagesPeer([partial + event("message_delta", delta={"stop_reason": reason}) + event("message_stop")
                             for reason in ("max_tokens", "tool_use")])
        base = self.start(peer)
        for status in ("incomplete", "failed"):
            _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "tools": [function_tool()], "stream": True})
            final = consume(raw)
            self.assertEqual(final["status"], status)
            self.assertNotIn(b"event: response.function_call_arguments.done", raw)
            self.assertFalse(any(item["type"] == "function_call" for item in final["output"]))
            if status == "incomplete":
                self.assertEqual(final["incomplete_details"], {"reason": "max_output_tokens"})

    def test_nonfinite_tool_values_cannot_complete_or_enter_replay(self):
        wire = tool_response().replace(b'{\\"text\\":', b'{\\"x\\":').replace(b'\\"OK\\"}', b'1e309}')
        overflowing_json = (b'{"id":"msg_json","type":"message","role":"assistant","stop_reason":"tool_use",'
                            b'"content":[{"type":"tool_use","id":"call_bad","name":"echo","input":{"x":1e309}}]}')
        peer = MessagesPeer([wire, (200, "application/json", overflowing_json)])
        base = self.start(peer)
        _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "tools": [function_tool()], "stream": True})
        self.assertEqual(consume(raw)["status"], "failed")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(base, {"model": "kiro/haiku", "input": "hello", "tools": [function_tool()], "stream": False})
        with caught.exception as response:
            self.assertEqual(response.code, 502)
            self.assertEqual(json.load(response)["error"]["type"], "kiro_invalid_response")
        with self.assertRaises(ProviderError) as caught:
            muse_kiro.prepare_request({"model": "haiku", "input": [{"role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "call_bad", "name": "echo", "arguments": '{"x":1e309}'},
                {"type": "function_call_output", "call_id": "call_bad", "output": "done"}]})
        self.assertEqual(caught.exception.status, 400)
        invalid = [("input_tokens", -1), ("output_tokens", True), ("prompt_tokens", -1),
                   ("completion_tokens", "invalid"), ("cache_read_input_tokens", "invalid"),
                   ("cache_read_input_tokens", -1), ("cache_read_input_tokens", False),
                   ("cache_creation_input_tokens", 1.5), ("cache_creation_input_tokens", None)]
        phases = ("json", "message_start", "message_delta")
        valid = {"input_tokens": 0, "output_tokens": 7, "cache_read_input_tokens": 11, "cache_creation_input_tokens": 13}
        cases = [(phase, {"input_tokens": 2, "output_tokens": 1, name: value}, False)
                 for name, value in invalid for phase in phases]
        cases += [(phase, valid, True) for phase in phases]
        cases += [("json", {"prompt_tokens": 17, "completion_tokens": 19}, True)]
        peer = MessagesPeer([usage_response(usage, phase) for phase, usage, _ in cases])
        base = self.start(peer)
        for count, (phase, usage, accepted) in enumerate(cases, 1):
            with self.subTest(phase=phase, usage=usage):
                body = {"model": "kiro/haiku", "input": "usage", "stream": phase != "json"}
                if not accepted and phase != "message_delta":
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        post(base, body)
                    with caught.exception as response:
                        self.assertEqual(response.code, 502)
                        self.assertEqual(json.load(response)["error"]["type"], "kiro_invalid_response")
                else:
                    _, raw = post(base, body)
                    final = json.loads(raw) if phase == "json" else consume(raw)
                    self.assertEqual(final["status"], "completed" if accepted else "failed")
                    if accepted:
                        expected = {"input_tokens": 17, "output_tokens": 19, "total_tokens": 36} if "prompt_tokens" in usage else {
                            "input_tokens": 0, "output_tokens": 7, "total_tokens": 7,
                            "input_tokens_details": {"cached_tokens": 11, "anthropic_cache_creation_input_tokens": 13}}
                        self.assertEqual(final["usage"], expected)
                    else:
                        self.assertEqual(final["error"]["type"], "kiro_invalid_response")
                        self.assertNotIn(b"event: response.completed", raw)
                        self.assertEqual(final["usage"], {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
                            "input_tokens_details": {"cached_tokens": 3, "anthropic_cache_creation_input_tokens": 5}},
                            "a failed delta must retain only previously validated counters")
                self.assertEqual(len(peer.requests), count, "usage failures must not replay inference")

    def test_none_tool_policy_rejects_synthesized_calls_in_stream_and_json(self):
        peer = MessagesPeer([tool_response(), interleaved_response(stream=False)])
        base = self.start(peer)
        body = {"model": "kiro/haiku", "input": "Quote a tool call, do not execute it", "tool_choice": "none",
                "tools": [{"type": "function", "name": "echo", "parameters": {"type": "object"}}]}
        _, raw = post(base, {**body, "stream": True})
        final = consume(raw)
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"]["type"], "kiro_unexpected_tool")
        self.assertNotIn(b"event: response.function_call_arguments.done", raw)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(base, {**body, "stream": False})
        with caught.exception as response:
            self.assertEqual(response.code, 502)
            self.assertEqual(json.load(response)["error"]["type"], "kiro_unexpected_tool")
        self.assertTrue(all("tools" not in body for _, _, body in peer.requests))
        disabled = [{}, {"tools": None}, {"tools": []},
                    {"tools": [{"type": "namespace", "name": "empty", "tools": []}]},
                    {"tools": [function_tool()], "tool_choice": "none"},
                    {"tools": [function_tool()], "tool_choice": {"type": "none"}}]
        peer.turns.extend([lambda body: anthropic_text("TOOLS_DISABLED") if body["stream"] else usage_response({}, "json")]
                          * (len(disabled) * 2 + 2))
        for options in disabled:
            for stream in (False, True):
                with self.subTest(options=options, stream=stream):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        post(base, {"model": "kiro/haiku", "input": inline_tool_history(), **options, "stream": stream})
                    with caught.exception as response:
                        self.assertEqual(response.code, 400)
                        error = json.load(response)["error"]
                        self.assertEqual(error["type"], "kiro_unsupported_history_tool_policy")
                        self.assertIn("structured", error["message"])
                        self.assertIn("current tools are disabled or empty", error["message"])
        self.assertEqual(len(peer.requests), 2, "disabled structured history reached inference")
        for stream in (False, True):
            _, raw = post(base, {**body, "stream": stream})
            final = consume(raw) if stream else json.loads(raw)
            self.assertEqual(final["status"], "completed", "none remains valid without tool history")
        self.assertEqual(len(peer.requests), 4)

    def test_only_currently_offered_wire_tools_are_accepted_in_http_and_json(self):
        previous = [{"role": "user", "content": "Earlier request"},
                    {"type": "function_call", "call_id": "call_old", "name": "read_file", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_old", "output": "old result"},
                    {"role": "user", "content": "No more file reads"}]
        previous_alias = copy.deepcopy(previous)
        previous_alias[1]["name"] = "web_search"
        cases = [
            ("no tools field", {}, "read_file", False),
            ("unknown function", {"tools": [function_tool()]}, "read_file", False),
            ("historical alias omitted", {"input": previous_alias, "tools": [function_tool()]}, None, False),
            ("declared alias", {"tools": [function_tool("web_search")]}, None, True),
        ]
        for label, options, reply_name, accepted in cases:
            for stream in (True, False):
                with self.subTest(case=label, stream=stream):
                    def upstream_call(body):
                        name = reply_name
                        if name is None:
                            if accepted:
                                name = body["tools"][0]["name"]
                            else:
                                name = next(part["name"] for message in body["messages"] for part in message["content"] if part["type"] == "tool_use")
                        block = {"type": "tool_use", "id": "call_current", "name": name, "input": {}}
                        if not body["stream"]:
                            return (200, "application/json", json.dumps({"type": "message", "id": "msg_offers", "role": "assistant",
                                    "content": [block], "stop_reason": "tool_use"}).encode())
                        return (KiroMessages.START + event("content_block_start", index=0, content_block=block)
                                + event("content_block_stop", index=0) + event("message_delta", delta={"stop_reason": "tool_use"}) + event("message_stop"))
                    peer = MessagesPeer([upstream_call])
                    base = self.start(peer)
                    body = {"model": "kiro/haiku", "input": "Current request", **options, "stream": stream}
                    if not stream and not accepted:
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            post(base, body)
                        with caught.exception as response:
                            self.assertEqual(response.code, 502)
                            self.assertEqual(json.load(response)["error"]["type"], "kiro_unexpected_tool")
                    else:
                        _, raw = post(base, body)
                        final = consume(raw) if stream else json.loads(raw)
                        self.assertEqual(final["status"], "completed" if accepted else "failed")
                        if accepted:
                            self.assertEqual(final["output"][0]["name"], reply_name or "web_search")
                        else:
                            self.assertEqual(final["error"]["code"], "kiro_unexpected_tool")
                            self.assertEqual(final["output"], [])
                            self.assertNotIn(b"event: response.output_item.added", raw)
                            self.assertNotIn(b"event: response.function_call_arguments", raw)
                    self.assertEqual(len(peer.requests), 1)

    def test_actual_eight_mib_block_rejects_three_valid_multibyte_frames(self):
        def utf8_event(kind, **payload):
            return ("event: " + kind + "\ndata: " + json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n\n").encode("utf-8")
        fragment = "\u20ac" * (1024 * 1024)
        frame = utf8_event("content_block_delta", index=0, delta={"type": "text_delta", "text": fragment})
        self.assertLess(len(frame), 4 * 1024 * 1024, "fixture must not trip the separate frame limit")
        self.assertEqual(3 * len(fragment.encode("utf-8")), 9437184)
        wire = (KiroMessages.START + event("content_block_start", index=0, content_block={"type": "text", "text": ""})
                + frame * 3 + event("content_block_stop", index=0) + event("message_delta", delta={"stop_reason": "end_turn"}) + event("message_stop"))
        peer = MessagesPeer([wire])
        base = self.start(peer)
        _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "stream": True})
        final = consume(raw)
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"]["code"], "kiro_invalid_response")
        self.assertIn("text block exceeded", final["error"]["message"])
        # Tool input must also be bounded by UTF-8 bytes, not character count.
        wire = (KiroMessages.START + event("content_block_start", index=0,
                content_block={"type": "tool_use", "id": "call_limit", "name": "echo", "input": {"text": "\u20ac" * 20}})
                + event("content_block_stop", index=0) + event("message_delta", delta={"stop_reason": "tool_use"}) + event("message_stop"))
        base = self.start(MessagesPeer([wire]))
        with patch.object(muse_kiro, "_BLOCK_LIMIT", 40):
            _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "tools": [function_tool()], "stream": True})
        final = consume(raw)
        self.assertEqual(final["status"], "failed")
        self.assertIn("tool block exceeded", final["error"]["message"])
        self.assertNotIn(b"event: response.function_call_arguments.done", raw)

    def test_reserved_and_long_client_tool_names_round_trip_without_server_ownership(self):
        originals = ["web_search", "long_" + "x" * 80]
        def upstream_calls(body):
            blocks = [{"type": "tool_use", "id": "call_" + str(index), "name": tool["name"], "input": {}}
                      for index, tool in enumerate(body["tools"])]
            if not body["stream"]:
                return (200, "application/json", json.dumps({"type": "message", "id": "msg_alias", "role": "assistant",
                        "content": blocks, "stop_reason": "tool_use"}).encode())
            wire = KiroMessages.START
            for index, block in enumerate(blocks):
                wire += event("content_block_start", index=index, content_block=block) + event("content_block_stop", index=index)
            return wire + event("message_delta", delta={"stop_reason": "tool_use"}) + event("message_stop")
        peer = MessagesPeer([upstream_calls, anthropic_text("TOOL_OWNERSHIP_OK"), upstream_calls, anthropic_text("TOOL_OWNERSHIP_OK")])
        base = self.start(peer)
        tools = [{"type": "function", "name": name, "parameters": {"type": "object"}} for name in originals]
        for stream in (True, False):
            with self.subTest(stream=stream):
                history = [{"role": "user", "content": "Use client tools"}]
                _, raw = post(base, {"model": "kiro/haiku", "input": history, "tools": tools, "stream": stream})
                first = consume(raw) if stream else json.loads(raw)
                self.assertEqual([item["name"] for item in first["output"]], originals)
                wire_names = [tool["name"] for tool in peer.requests[-1][2]["tools"]]
                self.assertEqual(len(set(wire_names)), 2)
                self.assertNotIn("web_search", wire_names)
                self.assertTrue(all(len(name) <= 64 and name.replace("_", "").isalnum() for name in wire_names))
                history += first["output"] + [{"type": "function_call_output", "call_id": item["call_id"], "output": "client-owned result"}
                                               for item in first["output"]]
                _, raw = post(base, {"model": "kiro/haiku", "input": history, "tools": tools, "stream": True})
                self.assertEqual(consume(raw)["status"], "completed")
                replayed = [part["name"] for message in peer.requests[-1][2]["messages"] for part in message["content"] if part["type"] == "tool_use"]
                self.assertEqual(replayed, wire_names)

    def test_tool_name_alias_collision_preserves_distinct_call_ids_through_http_continuation(self):
        def upstream_calls(body):
            alias = body["tools"][0]["name"]
            self.assertNotEqual(alias, "web_search")
            blocks = [{"type": "tool_use", "id": cid, "name": alias, "input": {"index": index}}
                      for index, cid in enumerate((alias, "web_search"))]
            if not body["stream"]:
                return (200, "application/json", json.dumps({"type": "message", "id": "msg_alias_ids", "role": "assistant",
                        "content": blocks, "stop_reason": "tool_use"}).encode())
            wire = KiroMessages.START
            for index, block in enumerate(blocks):
                wire += event("content_block_start", index=index, content_block={**block, "input": {}})
                wire += event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": json.dumps(block["input"])})
                wire += event("content_block_stop", index=index)
            return wire + event("message_delta", delta={"stop_reason": "tool_use"}) + event("message_stop")
        tools = [{"type": "function", "name": "web_search", "parameters": {"type": "object"}}]
        for stream in (True, False):
            with self.subTest(stream=stream):
                peer = MessagesPeer([upstream_calls, anthropic_text("CONTINUED")])
                base = self.start(peer)
                history = [{"role": "user", "content": "Use both calls"}]
                body = {"model": "kiro/haiku", "input": history, "tools": tools, "stream": stream}
                if stream:
                    body["parallel_tool_calls"] = True
                _, raw = post(base, body)
                first = consume(raw) if stream else json.loads(raw)
                self.assertEqual(first["status"], "completed")
                history += first["output"] + [{"type": "function_call_output", "call_id": item["call_id"],
                                               "output": "result:" + str(index)} for index, item in enumerate(first["output"])]
                try:
                    _, raw = post(base, {**body, "input": history, "stream": True})
                except urllib.error.HTTPError as error:
                    with error:
                        self.fail(f"Completed alias calls could not be replayed: HTTP {error.code}: {error.read()!r}")
                self.assertEqual(consume(raw)["status"], "completed")
                self.assertEqual(len(peer.requests), 2)
                alias = peer.requests[0][2]["tools"][0]["name"]
                expected_ids = [alias, "web_search"]
                self.assertEqual([item["call_id"] for item in first["output"]], expected_ids)
                self.assertEqual([item["name"] for item in first["output"]], ["web_search"] * 2)
                calls = [part for message in peer.requests[1][2]["messages"] for part in message["content"] if part["type"] == "tool_use"]
                results = [part for message in peer.requests[1][2]["messages"] for part in message["content"] if part["type"] == "tool_result"]
                self.assertEqual([part["id"] for part in calls], expected_ids)
                self.assertEqual([part["name"] for part in calls], [alias] * 2)
                self.assertEqual([part["input"] for part in calls], [{"index": 0}, {"index": 1}])
                self.assertEqual([(part["tool_use_id"], part["content"]) for part in results], list(zip(expected_ids, ["result:0", "result:1"])))

    def test_precontent_service_retry_keeps_one_lifecycle_but_postcontent_restart_fails(self):
        restarted = KiroMessages.START + anthropic_text("RETRY_OK")
        after_content = anthropic_text("visible").split(b"event: content_block_stop")[0] + KiroMessages.START
        peer = MessagesPeer([restarted, after_content])
        base = self.start(peer)
        for expected in ("completed", "failed"):
            _, raw = post(base, {"model": "kiro/haiku", "input": "hello", "stream": True})
            self.assertEqual(consume(raw)["status"], expected)
            self.assertEqual(raw.count(b"event: response.created\n"), 1)


class KiroPreparation(unittest.TestCase):
    def test_new_prompt_cannot_substitute_for_missing_response_state(self):
        for supplied in (None, "new prompt", [{"role": "user", "content": "new prompt"}]):
            with self.subTest(input=supplied), self.assertRaises(ProviderError) as caught:
                muse_kiro.prepare_request({"model": "haiku", "input": supplied, "previous_response_id": "resp_missing"})
            self.assertEqual(caught.exception.code, "kiro_previous_response_unsupported" if supplied else "kiro_invalid_request")
        payload, _ = muse_kiro.prepare_request({"model": "haiku", "input": "new prompt", "previous_response_id": None})
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "new prompt")

    def test_required_references_and_incomplete_tool_history_fail_closed(self):
        call = {"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": '{"value":1}'}
        result = {"type": "function_call_output", "call_id": "call_1", "output": "result"}
        invalid = [
            ([{"type": "item_reference", "id": "missing"}], "kiro_unresolved_reference"),
            ([call], "kiro_invalid_history"),
            ([result], "kiro_invalid_history"),
            ([call, result, result], "kiro_invalid_history"),
            ([{**call, "arguments": '{"invalid":'} , result], "kiro_invalid_history"),
            ([call, {"role": "user", "content": "continued without tool result"}, result], "kiro_invalid_history"),
        ]
        for items, code in invalid:
            with self.subTest(items=items), self.assertRaises(ProviderError) as caught:
                muse_kiro.prepare_request({"model": "haiku", "input": items})
            self.assertEqual(caught.exception.code, code)

    def test_unsupported_controls_and_media_are_not_silently_dropped(self):
        options = [{"tool_choice": {"type": "function", "name": "echo"}}, {"temperature": 0},
                   {"parallel_tool_calls": False},
                   {"tools": [{"type": "web_search"}]},
                   {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}]}]}]
        for option in options:
            with self.subTest(option=option):
                with self.assertRaises(ProviderError) as caught:
                    muse_kiro.prepare_request({"model": "haiku", "input": "hello", **option})
                self.assertEqual(caught.exception.status, 400)
                self.assertTrue(caught.exception.code.startswith("kiro_unsupported"))
        for option in ({}, {"parallel_tool_calls": True}, {"parallel_tool_calls": None}):
            payload, _ = muse_kiro.prepare_request({"model": "haiku", "input": "hello", **option})
            self.assertEqual(payload["messages"][0]["content"][0]["text"], "hello")


class KiroTransport(unittest.TestCase):
    def test_upstream_and_proxy_key_security_policy(self):
        for supplied, expected in (("http://localhost:8000", "http://localhost:8000/v1"),
                                   ("http://[::1]:8000/v1/messages/", "http://[::1]:8000/v1"),
                                   ("https://example.com/proxy/v1/", "https://example.com/proxy/v1")):
            self.assertEqual(muse_kiro.normalize_upstream(supplied), expected)
        invalid = ("http://example.com/v1", "https://user:private-url-secret@example.com/v1",
                   "http://127.0.0.1:0/v1", "https://example.com/v1?key=private-url-secret",
                   "https://example.com/v1#fragment", "http://127.0.0.1/v1\r\nX: x",
                   "http://127.0.0.1/v1%0a", "ftp://127.0.0.1/v1", "http://127.0.0.1/elsewhere")
        for supplied in invalid:
            with self.subTest(url=supplied), self.assertRaises(ProviderError) as caught:
                muse_kiro.normalize_upstream(supplied)
            self.assertEqual(caught.exception.code, "kiro_invalid_upstream")
            self.assertNotIn("private-url-secret", str(caught.exception))
        for supplied in (None, "", " ", "key\r\nX: leak", "key\twith-tab", "non-ascii-\u03bb"):
            with self.subTest(key=supplied), self.assertRaises(ProviderError):
                muse_kiro.validate_proxy_key(supplied)

    def test_private_opener_bypasses_loopback_proxy_and_preserves_global_opener(self):
        peer = MessagesPeer()
        with serving(ThreadingHTTPServer(("127.0.0.1", 0), peer.handler())) as base:
            prior = urllib.request._opener
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}):
                self.assertEqual(muse_kiro.discover_models(base, "synthetic-key"), ["claude-haiku-4.5", "glm-5"])
            self.assertIs(urllib.request._opener, prior)
        self.assertEqual(peer.catalog_requests[0][1]["Authorization"], "Bearer synthetic-key")

    def test_every_redirect_is_refused_even_on_same_origin(self):
        requests = []
        class RedirectPeer(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.reply()

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.reply()

            def reply(self):
                requests.append((self.command, self.path))
                self.send_response(int(self.path.rsplit("/", 1)[-1]) if self.path != "/credential-sink" else 200)
                self.send_header("Location", "/credential-sink")
                self.send_header("Content-Length", "0")
                self.end_headers()
        with serving(ThreadingHTTPServer(("127.0.0.1", 0), RedirectPeer)) as base:
            for code in (302, 307):
                for body in (None, b"{}"):
                    with self.subTest(code=code, method="GET" if body is None else "POST"), self.assertRaises(urllib.error.HTTPError) as caught:
                        muse_kiro.open_request(base + "/" + str(code), "synthetic-key", body=body)
                    caught.exception.close()
                    self.assertEqual(caught.exception.code, code)
        self.assertEqual(len(requests), 4)
        self.assertFalse(any(path == "/credential-sink" for _, path in requests), "proxy key followed a redirect")


class KiroMessages(unittest.TestCase):
    START = event("message_start", message={"id": "msg_contract", "role": "assistant", "content": []})

    def converted(self, wire, source=None, *, tools=()):
        _, policy = muse_kiro.prepare_request({"model": "haiku", "input": "test", "tools": [function_tool(name) for name in tools]})
        items = list(muse_kiro.stream_events(source or io.BytesIO(wire), model="haiku", tool_names=policy))
        return consume(b"".join(event(kind, **payload) for kind, payload in items))

    def test_read_error_after_output_is_a_terminal_failure(self):
        class BrokenStream(io.BytesIO):
            def readline(self, limit=-1):
                result = super().readline(limit)
                if not result:
                    raise IncompleteRead(b"", 100)
                return result
        final = self.converted(b"", source=BrokenStream(self.START))
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"]["type"], "kiro_upstream_error")
        self.assertEqual(final["error"]["metadata"]["source"], "transport")
        self.assertEqual(final["error"]["metadata"]["category"], "transport")
        self.assertIsNone(final["error"]["metadata"]["upstream_status"])
        self.assertIsNone(final["error"]["metadata"]["transient"])
        self.assertEqual(final["error"]["metadata"]["output_exposure"], "lifecycle")

    def test_total_response_budget_fails_after_several_valid_frames(self):
        wire = anthropic_text("some text")
        with patch.object(muse_kiro, "_RESPONSE_LIMIT", len(wire) - 1):
            final = self.converted(wire)
        self.assertEqual(final["status"], "failed")
        self.assertIn("total read limit", final["error"]["message"])

    def test_capture_failure_is_visible_but_cannot_break_inference(self):
        class BrokenCapture:
            errors = []

            def __getattr__(self, name):
                def unavailable(*args, **kwargs):
                    raise RuntimeError("private-recorder-detail")
                return unavailable
        recorder = BrokenCapture()
        with patch.object(muse_kiro, "current_capture", return_value=recorder):
            final = self.converted(anthropic_text("still succeeds"))
        self.assertEqual(final["status"], "completed")
        self.assertIn("kiro_capture_bytes_failed", recorder.errors)
        self.assertEqual(len(recorder.errors), len(set(recorder.errors)))
        self.assertNotIn("private-recorder-detail", json.dumps(recorder.errors))

    def test_capture_records_wire_and_timing_without_an_arbitrary_proxy_key(self):
        key = 'arbitrary-synthetic-quote"backslash\\credential'
        wire = anthropic_text("echo " + key)
        peer = MessagesPeer([wire])
        with tempfile.TemporaryDirectory() as directory, serving(ThreadingHTTPServer(("127.0.0.1", 0), peer.handler())) as base:
            capture = RequestCapture(Path(directory) / "request", "synthetic-request")
            try:
                with muse_kiro.open_request(base + "/v1/messages", key, body=b'{"model":"haiku","stream":true}') as response:
                    final = self.converted(b"", source=response)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["output"][0]["content"][0]["text"], "echo " + key, "recording must not modify transport")
            finally:
                capture.finish(status=200)
            root = Path(directory) / "request"
            for path in root.iterdir():
                self.assertNotIn(key.encode(), path.read_bytes(), "proxy key leaked into capture")
            saved_wire = (root / "kiro.upstream.body").read_bytes()
            self.assertEqual(saved_wire, anthropic_text("echo [REDACTED]"))
            saved_data = [json.loads(line[5:]) for line in saved_wire.splitlines() if line.startswith(b"data:")]
            self.assertNotIn(key, "".join(value.get("delta", {}).get("text", "") for value in saved_data))
            timeline = [json.loads(line) for line in (root / "timeline.jsonl").read_text().splitlines()]
            self.assertTrue(any(item["event"] == "kiro.upstream.read" for item in timeline))
            self.assertTrue(any(item["event"] == "kiro.upstream.semantic_stop" for item in timeline))
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["streams"]["kiro.upstream.body"]["observed_bytes"], len(wire))
            self.assertEqual(manifest["capture_errors"], [])


if __name__ == "__main__":
    unittest.main()
