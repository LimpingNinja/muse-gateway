"""Public HTTP behavior. Real sockets + real routing, wire decoder and converters.

Provider protocol peers are synthetic. Passing proves offline contracts,
not live entitlement, model quality, or acceptance by an installed Muse.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from tests.contracts import consume
from tests.peers import Peer, TRAILER, anthropic_text, chat, envelope, field, gateway, post, request, responses_text, text_turn, tool_turn


class GatewayContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        # Do not inherit real provider credentials/proxies into offline tests.
        clean = {key: value for key, value in os.environ.items()
                 if not key.upper().startswith(("MUSE_", "MUSE_GATEWAY_", "MUSE_SHIM_", "META_", "DEVIN_", "CODEX_", "OPENAI_", "ANTHROPIC_", "CLAUDE_"))
                 and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
        clean["NO_PROXY"] = "127.0.0.1,localhost"
        self.stack.enter_context(patch.dict(os.environ, clean, clear=True))

    def start(self, peer, **options):
        base = self.stack.enter_context(gateway(self.directory, peer, **options))
        self.addCleanup(lambda: self.assertEqual(peer.errors, [], "protocol peer setup failed"))
        return base

    def test_text_and_nonstream_have_equivalent_complete_output(self):
        peer = Peer([text_turn(), text_turn()])
        base = self.start(peer)
        mime, wire = post(base, {"model": "devin/swe-1-6", "input": "hello", "stream": True})
        self.assertEqual(mime, "text/event-stream")
        streamed = consume(wire)
        mime, raw = post(base, {"model": "devin/swe-1-6", "input": "hello", "stream": False})
        self.assertEqual(mime, "application/json")
        final = json.loads(raw)
        self.assertEqual(streamed, final)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["output"][0]["content"][0]["text"], "Hello, \u03bb \U0001f44b")
        self.assertEqual(len(peer.requests), 2)

    def test_tool_loop_replays_exact_arguments_ids_and_large_result(self):
        peer = Peer([tool_turn(), text_turn("Tool result received")])
        base = self.start(peer)
        history = [{"role": "developer", "content": "Keep the sentinel."},
                   {"role": "user", "content": "Echo OK"}]
        tools = [{"type": "namespace", "name": "muse", "tools": [{
            "type": "function", "name": "echo", "parameters": {
                "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}]
        _, wire = post(base, {"model": "devin/swe-1-6", "input": history, "tools": tools, "stream": True})
        first = consume(wire)
        self.assertEqual(first["status"], "completed", "TOOL_REPLAY: tool turn failed")
        self.assertEqual(len(first["output"]), 1)
        tool = first["output"][0]
        self.assertEqual(tool["name"], "echo", "TOOL_REPLAY: late name lost")
        self.assertEqual(json.loads(tool["arguments"]), {"text": "OK"}, "TOOL_REPLAY: cumulative/omitted-ID arguments corrupted")
        giant = "RESULT_START:" + "x" * 340000 + ":RESULT_END"
        history += first["output"] + [{"type": "function_call_output", "call_id": tool["call_id"], "output": giant},
                                      {"role": "user", "content": "Confirm result"}]
        _, wire = post(base, {"model": "devin/swe-1-6", "input": history, "tools": tools, "stream": True})
        self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "Tool result received")
        first_request, second = peer.requests
        self.assertEqual(first_request.cascade_id, second.cascade_id)
        self.assertNotEqual(first_request.execution_id, second.execution_id)
        prompts = list(second.chat_message_prompts)
        replayed = next(call for prompt in prompts for call in prompt.tool_calls)
        result = next(prompt for prompt in prompts if prompt.tool_call_id)
        self.assertEqual(replayed.id, result.tool_call_id, "TOOL_REPLAY: call/result identity broken")
        self.assertEqual(json.loads(replayed.arguments_json), {"text": "OK"})
        self.assertEqual(result.prompt, giant, "HISTORY: tool result was truncated")
        self.assertTrue(any("Keep the sentinel." in prompt.prompt for prompt in prompts), "HISTORY: developer text lost")

    def test_repaired_tool_arguments_match_every_stream_snapshot(self):
        peer = Peer([tool_turn(arguments="not JSON")])
        base = self.start(peer)
        _, wire = post(base, {"model": "devin/swe-1-6", "input": "tool", "stream": True})
        final = consume(wire)
        self.assertEqual(json.loads(final["output"][0]["arguments"]), {"__parseError": "not JSON"})

    def test_claude_http_translation_has_the_same_consumer_contract(self):
        complete = anthropic_text()
        truncated = complete[:complete.index(b"event: message_stop")]
        rejected = b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
        peer = Peer([complete, truncated, rejected])
        base = self.start(peer, claude_transport="http", access_token="fixture-claude-token", allow_oauth_host_override=True)
        mime, wire = post(base, {"model": "claude/claude-test", "input": "hello", "stream": True})
        self.assertEqual(mime, "text/event-stream")
        final = consume(wire)
        self.assertEqual(final["output"][0]["content"][0]["text"], "CLAUDE_ACCEPTED")
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["usage"]["input_tokens"], 4)
        for label in ("truncated before message_stop", "provider error"):
            with self.subTest(label=label):
                _, wire = post(base, {"model": "claude/claude-test", "input": "hello", "stream": True})
                self.assertEqual(consume(wire)["status"], "failed", "FAILURE_SURFACE: Claude failure reported success")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(base, {"model": "claude/claude-test", "input": "continue", "previous_response_id": "resp_missing"})
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("durable response-state", json.load(response)["error"]["message"])
        self.assertEqual(len(peer.requests), 3, "Unsupported native state must fail before opening the upstream")

    def test_small_frame_is_visible_while_upstream_is_still_blocked(self):
        first = chat("READY")
        native = responses_text("READY")
        scenarios = [("devin/swe-1-6", first + chat(" DONE", stop=2) + TRAILER, len(first)),
                     ("meta/muse-spark-1.3", native, native.index(b"event: response.output_item.done"))]
        for model, wire, hold_after in scenarios:
            with self.subTest(model=model):
                peer = Peer([wire], hold_after=hold_after)
                base = self.start(peer)
                try:
                    with urllib.request.urlopen(request(base, {"model": model, "input": "hi", "stream": True}), timeout=2) as response:
                        while True:
                            line = response.readline()
                            if not line:
                                self.fail("STREAM_LIVENESS: EOF before first text delta")
                            if line.startswith(b"data:"):
                                event = json.loads(line[5:])
                                if event.get("type") == "response.output_text.delta":
                                    self.assertEqual(event["delta"], "READY")
                                    break
                        self.assertFalse(peer.release.is_set(), "STREAM_LIVENESS: peer already released")
                        peer.release.set()
                        # Full-stream tests check semantic completion; here
                        # the client observes bytes before allowing upstream EOF.
                        response.read()
                except (TimeoutError, OSError) as error:
                    self.fail(f"STREAM_LIVENESS: first frame withheld: {type(error).__name__}")
                finally:
                    peer.release.set()

    def test_incomplete_transport_or_provider_error_never_claims_success(self):
        scenarios = {
            "truncated envelope": chat("partial") + b"\x00\x00\x00\x00\x10short",
            "EOF without completion": chat("partial"),
            "incomplete stop": chat("partial", stop=1) + TRAILER,
            "partial stop": chat("partial", stop=9) + TRAILER,
            "error trailer": chat("partial") + envelope(b'{"error":{"code":"internal","message":"fixture failure"}}', 2),
            "max tokens": chat("partial", stop=3) + TRAILER,
        }
        peer = Peer(list(scenarios.values()))
        base = self.start(peer)
        for label in scenarios:
            with self.subTest(label=label):
                _, wire = post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
                final = consume(wire)
                self.assertEqual(final["status"], "incomplete" if label == "max tokens" else "failed",
                                 "FAILURE_SURFACE: partial stream reported success")
                if label == "max tokens":
                    self.assertEqual(final["incomplete_details"], {"reason": "max_output_tokens"})
                else:
                    self.assertTrue(final["error"]["message"])

    def test_empty_and_http_error_are_json_errors_before_stream_headers(self):
        peer = Peer([b"", (503, b'{"error":{"message":"unavailable"}}')])
        base = self.start(peer)
        for label in ("empty stream", "upstream HTTP error"):
            with self.subTest(label=label), self.assertRaises(urllib.error.HTTPError) as caught:
                post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
            with caught.exception as response:
                self.assertEqual(response.code, 502)
                self.assertEqual(response.headers.get_content_type(), "application/json")
                self.assertTrue(json.load(response)["error"]["message"])

    def test_catalog_has_usable_context_not_context_minus_fallback(self):
        catalog = b"".join(field(1, field(1, uid) + field(18, size) + field(22, uid))
                           for uid, size in (("small", 200000), ("large", 1000000), ("unknown", 0)))
        base = self.start(Peer(catalog=catalog), provider="devin", model_discovery=True)
        with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
            rows = json.load(response)["data"]
        contexts = {row["id"]: row["metadata"]["muse-code"]["limit"]["context"] for row in rows if row["id"].startswith("devin/")}
        self.assertEqual(contexts, {"devin/small": 200000, "devin/large": 1000000,
                                    "devin/unknown": 200000}, "CATALOG_CONTEXT: unusable or understated context budget")

    def test_codex_400_is_readable_even_when_error_has_its_own_code(self):
        peer = Peer([(400, b'{"error":{"type":"invalid_request_error","code":"invalid_value","message":"bad item"}}')])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        try:
            post(base, {"model": "codex/gpt-5.4", "input": "hi", "stream": True})
        except urllib.error.HTTPError as error:
            with error:
                self.assertEqual(error.code, 400)
                self.assertEqual(error.headers.get_content_type(), "application/json")
                self.assertEqual(json.load(error)["error"]["type"], "codex_upstream_error")
        except Exception as error:
            self.fail(f"HTTP_ERROR_RESPONSE: disconnected instead of readable error: {type(error).__name__}")
        else:
            self.fail("HTTP_ERROR_RESPONSE: upstream rejection became success")
        events = [json.loads(line[line.index("{"):]) for line in (Path(self.directory) / "gateway.log").read_text().splitlines()]
        diagnostic = next(event for event in events if event.get("event") == "provider_error")
        self.assertEqual(diagnostic["upstream_code"], "invalid_value")

    def test_cross_provider_history_survives_routes_and_bounded_codex_recovery(self):
        meta = responses_text("Remember the sentinel", "resp_meta")
        codex = responses_text("REPLAY_ACCEPTED", "resp_codex")
        rejection = (400, b'{"error":{"code":"invalid_encrypted_content","message":"foreign reasoning"}}')
        peer = Peer([meta, tool_turn(), rejection, codex])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        history = [{"type": "message", "role": "user", "content": "Remember the sentinel"}]
        _, wire = post(base, {"model": "meta/muse-spark-1.3", "input": history, "stream": True})
        self.assertEqual(wire, meta)
        history += consume(wire)["output"] + [{"role": "user", "content": "Echo OK"}]
        _, wire = post(base, {"model": "devin/swe-1-6", "input": history, "stream": True})
        tool = consume(wire)["output"][0]
        reasoning = {"type": "reasoning", "id": "rs_meta:rs_item", "encrypted_content": "foreign-state",
                     "summary": [{"type": "summary_text", "text": "Visible portable summary"}]}
        history += [reasoning, {"type": "item_reference", "id": reasoning["id"]}, tool,
                    {"type": "function_call_output", "call_id": tool["call_id"], "output": "RESULT_SENTINEL"},
                    {"role": "user", "content": "Continue"}]
        original = json.loads(json.dumps(history))
        controls = {"reasoning": {"effort": "xhigh"}, "include": ["reasoning.encrypted_content"],
                    "prompt_cache_key": "cache-sentinel", "max_output_tokens": 1234,
                    "previous_response_id": "resp_foreign"}
        _, wire = post(base, {"model": "codex/gpt-5.4", "input": history, "stream": True, **controls})
        self.assertEqual(wire, codex)
        self.assertEqual(consume(wire)["status"], "completed")
        self.assertEqual(len(peer.requests), 4, "CODEX_RECOVERY: expected exactly one retry")
        first, recovered = peer.requests[-2:]
        for outgoing in (first, recovered):
            self.assertFalse(outgoing["store"])
            self.assertNotIn("max_output_tokens", outgoing)
            self.assertNotIn("previous_response_id", outgoing)
            for key in ("reasoning", "include", "prompt_cache_key"):
                self.assertEqual(outgoing[key], controls[key])
            ids = [item["id"] for item in outgoing["input"] if "id" in item]
            self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in ids), "CODEX_ID: invalid replay ID")
            replayed = next(item for item in outgoing["input"] if item.get("type") == "function_call")
            result = next(item for item in outgoing["input"] if item.get("type") == "function_call_output")
            self.assertEqual(replayed["call_id"], result["call_id"])
            self.assertEqual(result["output"], "RESULT_SENTINEL")
            self.assertEqual(json.loads(replayed["arguments"]), {"text": "OK"})
            self.assertEqual(outgoing["input"][0], history[0])
        self.assertTrue(any(item.get("encrypted_content") == "foreign-state" for item in first["input"]))
        self.assertFalse(any(item.get("encrypted_content") or item.get("type") == "item_reference" for item in recovered["input"]),
                         "CODEX_RECOVERY: rejected state or dangling reference replayed")
        self.assertTrue(any(part.get("text") == "Visible portable summary" for item in recovered["input"]
                            for part in item.get("content", []) if isinstance(part, dict)), "CODEX_RECOVERY: visible summary lost")
        self.assertEqual(history, original)

    def test_codex_recovery_does_not_retry_indefinitely(self):
        rejection = (400, b'{"error":{"code":"invalid_encrypted_content","message":"still rejected"}}')
        peer = Peer([rejection, rejection])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(base, {"model": "codex/gpt-5.4", "stream": True, "input": [{
                "type": "reasoning", "encrypted_content": "foreign", "summary": []}]})
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertEqual(json.load(response)["error"]["type"], "codex_upstream_error")
        self.assertEqual(len(peer.requests), 2, "CODEX_RECOVERY: unbounded recovery")


if __name__ == "__main__":
    unittest.main()
