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

    def _post_error(self, base, body=None, *, url=None, request_obj=None):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            if request_obj is not None:
                urllib.request.urlopen(request_obj, timeout=5)
            elif url is not None:
                urllib.request.urlopen(url, timeout=5)
            else:
                post(base, body)
        with caught.exception as response:
            return response.code, response.headers, json.load(response)

    def test_text_and_nonstream_have_equivalent_complete_output(self):
        peer = Peer([text_turn(), text_turn()])
        base = self.start(peer)
        mime, wire = post(base, {"model": "devin/swe-1-6", "input": "hello", "stream": True})
        self.assertEqual(mime, "text/event-stream")
        streamed = consume(wire)
        try:
            mime, raw = post(base, {"model": "devin/swe-1-6", "input": "hello", "stream": False})
        except urllib.error.HTTPError as error:
            self.fail(f"TERMINAL: nonstream Devin turn returned HTTP {error.code} instead of a terminal response")
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
        # P0-06 rewrite: unrecoverable arguments fail the turn (no fabricated
        # __parseError call); recoverable input still publishes final
        # arguments matching every streamed snapshot (consume() asserts it).
        nested = (
            chat(call=("fixture-call", "echo", '{"x":'))
            + chat(call=("fixture-call", "echo", '{"x":1}}'), stop=10)
            + TRAILER
        )
        peer = Peer([tool_turn(arguments="not JSON"), tool_turn(arguments='{"text":"OK"}'), nested])
        base = self.start(peer)
        _, wire = post(base, {"model": "devin/swe-1-6", "input": "tool", "stream": True})
        final = consume(wire)
        self.assertEqual(final["status"], "failed")
        self.assertNotIn("__parseError", json.dumps(final))
        _, wire = post(base, {"model": "devin/swe-1-6", "input": "tool", "stream": True})
        final = consume(wire)
        self.assertEqual(json.loads(final["output"][0]["arguments"]), {"text": "OK"})
        # P1-04 nested-prefix fragments: concatenation wins, delta == done.
        _, wire = post(base, {"model": "devin/swe-1-6", "input": "tool", "stream": True})
        final = consume(wire)
        self.assertEqual(json.loads(final["output"][0]["arguments"]), {"x": {"x": 1}})


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
        status, _, payload = self._post_error(base, {
            "model": "claude/claude-test", "input": "continue",
            "previous_response_id": "resp_missing",
        })
        self.assertEqual(status, 400)
        self.assertIn("durable response-state", payload["error"]["message"])
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
            with self.subTest(label=label):
                status, headers, payload = self._post_error(
                    base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
                self.assertEqual(status, 502)
                self.assertEqual(headers.get_content_type(), "application/json")
                self.assertTrue(payload["error"]["message"])

    def test_catalog_has_usable_context_not_context_minus_fallback(self):
        catalog = b"".join(field(1, field(1, uid) + field(18, size) + field(22, uid) + field(23, field(13, 64000)))
                           for uid, size in (("small", 200000), ("large", 1000000), ("unknown", 0)))
        base = self.start(Peer(catalog=catalog), provider="devin", model_discovery=True)
        with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
            rows = json.load(response)["data"]
        contexts = {row["id"]: row["metadata"]["muse-code"]["limit"]["context"] for row in rows if row["id"].startswith("devin/")}
        self.assertEqual(contexts, {"devin/small": 200000, "devin/large": 1000000},
                         "CATALOG_CONTEXT: unusable or understated context budget")
        _, _, payload = self._post_error(
            base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
        self.assertEqual(payload["error"]["type"], "devin_unknown_model")

    def test_menu_allowlist_does_not_block_hidden_model_requests(self):
        with patch.dict(os.environ, {"MUSE_GATEWAY_MODELS": "meta/muse-spark-1.3"}):
            peer = Peer([text_turn()])
            base = self.start(peer)
            with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
                rows = json.load(response)["data"]
            self.assertEqual([row["id"] for row in rows], ["meta/muse-spark-1.3"])
            self.assertEqual(rows[0]["metadata"]["muse-code"]["limit"],
                             {"context": 1048576, "output": 32768})
            _, wire = post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
            self.assertEqual(consume(wire)["status"], "completed")
            self.assertEqual(len(peer.requests), 1)

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
        self.assertEqual(diagnostic["upstream"]["code"], "invalid_value")

    def test_cross_provider_history_survives_routes_and_bounded_codex_recovery(self):
        meta = responses_text("Remember the sentinel", "resp_meta")
        codex = responses_text("REPLAY_ACCEPTED", "resp_codex")
        rejection = (400, b'{"error":{"code":"invalid_encrypted_content","message":"foreign reasoning"}}')
        peer = Peer([meta, tool_turn(), rejection, codex])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        history = [{"type": "message", "role": "user", "content": "Remember the sentinel"}]
        _, wire = post(base, {"model": "meta/muse-spark-1.3", "input": history, "stream": True})
        self.assertEqual(consume(wire), consume(meta))
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
        self.assertEqual(consume(wire), consume(codex))
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

    def test_codex_replay_ids_match_item_types_and_keep_references(self):
        peer = Peer([responses_text("PREFIX_ACCEPTED"), responses_text("REPLAY_ACCEPTED")])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        history = [
            {"type": "message", "id": "msg_native", "role": "assistant",
             "content": [{"type": "output_text", "text": "native message"}]},
            {"type": "message", "id": "rs_foreign_message", "role": "assistant",
             "content": [{"type": "output_text", "text": "portable message"}]},
            {"type": "item_reference", "id": "rs_foreign_message"},
            {"type": "message", "id": "resp_foreign:msg_native", "role": "assistant",
             "content": [{"type": "output_text", "text": "distinct qualified message"}]},
            {"type": "item_reference", "id": "resp_foreign:msg_native"},
            {"type": "reasoning", "id": "msg_foreign_reasoning", "summary": []},
            {"type": "item_reference", "id": "msg_foreign_reasoning"},
            {"type": "function_call", "id": "msg_foreign_call", "call_id": "keep-call-id",
             "name": "echo", "arguments": '{"text":"OK"}'},
            {"type": "item_reference", "id": "msg_foreign_call"},
            {"type": "function_call_output", "call_id": "keep-call-id", "output": "RESULT"},
            {"type": "message", "id": "unqualified", "role": "assistant", "content": []},
            {"type": "message", "id": "resp_foreign:rs_message", "role": "assistant", "content": []},
            {"type": "message", "role": "user", "content": "Testing."},
        ]
        original = json.loads(json.dumps(history))
        _, wire = post(base, {"model": "codex/gpt-6.1-sol", "input": history,
                              "reasoning": {"effort": "low"}, "stream": True})
        self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "PREFIX_ACCEPTED")
        outgoing = peer.requests[0]["input"]
        self.assertEqual(len(outgoing), len(history))
        for before, after in zip(history, outgoing):
            self.assertEqual({key: value for key, value in before.items() if key != "id"},
                             {key: value for key, value in after.items() if key != "id"})
            if "id" in after and after["type"] != "item_reference":
                prefix = {"message": "msg_", "reasoning": "rs_", "function_call": "fc_"}[after["type"]]
                self.assertRegex(after["id"], "^" + prefix + r"[A-Za-z0-9_-]+$", "CODEX_ID_PREFIX")
        self.assertEqual(outgoing[0]["id"], "msg_native", "Native IDs must remain unchanged")
        self.assertNotEqual(outgoing[0]["id"], outgoing[3]["id"], "ID normalization must not merge messages")
        for item, reference in ((1, 2), (3, 4), (5, 6), (7, 8)):
            self.assertEqual(outgoing[item]["id"], outgoing[reference]["id"])
        self.assertEqual(outgoing[7]["call_id"], outgoing[9]["call_id"])
        self.assertEqual(history, original, "Caller history must not be mutated")
        _, wire = post(base, {"model": "codex/gpt-6.1-sol", "input": outgoing, "stream": True})
        self.assertEqual(consume(wire)["status"], "completed")
        self.assertEqual(peer.requests[1]["input"], outgoing, "Replayed normalization must be idempotent")

    def test_codex_recovery_does_not_retry_indefinitely(self):
        rejection = (400, b'{"error":{"code":"invalid_encrypted_content","message":"still rejected"}}')
        peer = Peer([rejection, rejection])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        status, _, payload = self._post_error(base, {
            "model": "codex/gpt-5.4", "stream": True,
            "input": [{"type": "reasoning", "encrypted_content": "foreign", "summary": []}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["type"], "codex_upstream_error")
        self.assertEqual(len(peer.requests), 2, "CODEX_RECOVERY: unbounded recovery")

    def test_codex_quota_falls_back_to_reserve_once(self):
        quota = (429, b'{"error":{"code":"insufficient_quota","message":"quota out"}}')
        peer = Peer([quota, responses_text("RESERVE_OK")])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        _, wire = post(base, {"model": "codex/gpt-5.4", "input": "hi", "stream": True})
        self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "RESERVE_OK")
        self.assertEqual(len(peer.requests), 2, "CODEX_RESERVE: must retry exactly once")
        self.assertEqual(peer.requests[0]["model"], "gpt-5.4")
        self.assertEqual(peer.requests[1]["model"], "gpt-reserve")

    def test_codex_nonstream_forces_upstream_stream_and_returns_single_response(self):
        peer = Peer([responses_text("OMITTED_OK"), responses_text("EXPLICIT_OK")])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        bodies = ({"model": "codex/gpt-5.4", "input": "hi"},
                  {"model": "codex/gpt-5.4", "input": "hi", "stream": False})
        for body, text in zip(bodies, ("OMITTED_OK", "EXPLICIT_OK")):
            with self.subTest(body=body):
                mime, raw = post(base, body)
                self.assertEqual(mime, "application/json")
                final = json.loads(raw)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["output"][0]["content"][0]["text"], text)
        self.assertEqual(len(peer.requests), 2)
        for outgoing in peer.requests:
            self.assertTrue(outgoing["stream"], "CODEX_STREAM: upstream requires stream:true")

    def test_codex_nonstream_collapse_outcomes_across_content_type_labels(self):
        # Bare Peer bytes are labeled text/event-stream; tuple turns are
        # labeled application/json. Buffered non-streaming must collapse
        # SSE in both cases and keep failure/truncation outcomes typed.
        failed = (b'event: response.failed\n'
                  b'data: {"type":"response.failed","response":{"status":"failed",'
                  b'"error":{"code":"server_error","message":"boom"}}}\n\n')
        truncated = responses_text("cut")[:60]
        peer = Peer([failed, (200, responses_text("MISLABELED_OK")), (200, failed),
                     (200, truncated), (200, responses_text("STREAMED_OK"))])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)

        def expect_error(expected):
            status, _, payload = self._post_error(
                base, {"model": "codex/gpt-5.4", "input": "hi", "stream": False})
            self.assertEqual(status, 502)
            self.assertEqual(payload["error"]["type"], expected)

        expect_error("codex_upstream_error")
        mime, raw = post(base, {"model": "codex/gpt-5.4", "input": "hi", "stream": False})
        self.assertEqual(mime, "application/json")
        final = json.loads(raw)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["output"][0]["content"][0]["text"], "MISLABELED_OK")
        for expected in ("codex_upstream_error", "codex_truncated_stream"):
            with self.subTest(expected=expected):
                expect_error(expected)
        _, wire = post(base, {"model": "codex/gpt-5.4", "input": "hi", "stream": True})
        self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "STREAMED_OK")
        self.assertEqual(len(peer.requests), 5)

    def test_devin_stop_reason_matrix_and_trailer_strictness(self):
        scenarios = {
            "nonfinite-7": (chat("partial", stop=7) + TRAILER, "failed"),
            "filter-11": (chat("partial", stop=11) + TRAILER, "failed"),
            "error-13": (chat("partial", stop=13) + TRAILER, "failed"),
            "noninsertion-12": (chat("partial", stop=12) + TRAILER, "incomplete"),
            "minlogprob-4": (chat("partial", stop=4) + TRAILER, "completed"),
            "unknown-99": (chat("partial", stop=99) + TRAILER, "failed"),
            "bad-trailer-json": (chat("partial", stop=2) + envelope(b"{not-json", 2), "failed"),
            "bad-trailer-shape": (chat("partial", stop=2) + envelope(b'{"error":42}', 2), "failed"),
            "empty-trailer": (chat("partial", stop=2) + envelope(b"", 2), "failed"),
            "frame-after-envelope": (chat("partial", stop=2) + TRAILER + chat("late"), "failed"),
            "missing-trailer": (chat("partial", stop=2), "failed"),
            "clean-empty-trailer": (chat("done", stop=2) + TRAILER, "completed"),
        }
        peer = Peer([turn for turn, _ in scenarios.values()])
        base = self.start(peer)
        for label, (_, expected) in scenarios.items():
            with self.subTest(label=label):
                _, wire = post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
                self.assertEqual(consume(wire)["status"], expected, label)

    def test_passthrough_truncation_synthesizes_numbered_failed(self):
        truncated = responses_text("hi")[:60]
        peer = Peer([truncated])
        base = self.start(peer)
        _, wire = post(base, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True})
        terminal = consume(wire)
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"]["code"], "upstream_truncated")
        logs = (Path(self.directory) / "gateway.log").read_text()
        self.assertIn("upstream_stream_truncated", logs)
        self.assertIn('"stream_failed": true', logs)

    def test_upstream_reset_is_answered_not_dropped(self):
        import socket as socket_module
        import struct
        import threading

        listener = socket_module.socket()
        listener.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def reset_once():
            conn, _ = listener.accept()
            conn.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_LINGER,
                            struct.pack("ii", 1, 0))
            conn.close()
            listener.close()

        threading.Thread(target=reset_once, daemon=True).start()
        peer = Peer([])
        base = self.start(peer, meta_upstream=f"http://127.0.0.1:{port}")
        status, _, payload = self._post_error(
            base, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": False})
        self.assertEqual(status, 502)
        self.assertIn(payload["error"]["type"], {"upstream_reset", "upstream_unreachable"})

    def test_deeply_nested_body_is_a_400_not_a_drop(self):
        peer = Peer([])
        base = self.start(peer)
        req = urllib.request.Request(
            base + "/responses", data=b"[" * 10000,
            headers={"Content-Type": "application/json"},
        )
        status, _, _ = self._post_error(base, request_obj=req)
        self.assertEqual(status, 400)

    def test_previous_response_id_policy_is_identical_across_stateless_routes(self):
        # P1-06 (inverts R21): null/"" proceed; full history drops + logs;
        # delta-only is rejected 400 before any upstream call, on every route.
        full = [{"role": "user", "content": "hi"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
                {"role": "user", "content": "again"}]
        turns = {
            "claude/claude-test": lambda: anthropic_text("OK"),
            "codex/gpt-5.4": lambda: responses_text("OK"),
            "devin/swe-1-6": lambda: text_turn("OK"),
            "kiro/haiku": lambda: anthropic_text("OK"),
        }
        from http.server import ThreadingHTTPServer
        from tests.peers import serving
        peer = Peer([make() for make in turns.values() for _ in range(3)])
        kiro = self.stack.enter_context(serving(ThreadingHTTPServer(("127.0.0.1", 0), peer.handler())))
        base = self.start(peer, claude_transport="http", access_token="fixture-token",
                          kiro_upstream=kiro + "/v1", kiro_api_key="fixture-kiro-proxy-token")
        log = Path(self.directory) / "gateway.log"
        for model in turns:
            for previous, history, outcome in ((None, "hi", "accept"), ("", "hi", "accept"),
                                               ("resp_earlier", full, "drop"),
                                               ("resp_earlier", "only the new prompt", "reject")):
                with self.subTest(model=model, previous=previous, outcome=outcome):
                    before = len(peer.requests)
                    logged = log.read_text().count("previous_response_id_dropped") if log.exists() else 0
                    body = {"model": model, "input": history, "stream": True, "previous_response_id": previous}
                    if outcome == "reject":
                        status, _, payload = self._post_error(base, body)
                        self.assertEqual(status, 400)
                        self.assertTrue(payload["error"]["type"].endswith("_previous_response_unsupported"))
                        self.assertEqual(len(peer.requests), before, "delta-only reached inference")
                        continue
                    _, wire = post(base, body)
                    self.assertEqual(consume(wire)["status"], "completed")
                    self.assertEqual(len(peer.requests), before + 1)
                    sent = peer.requests[-1]
                    if isinstance(sent, dict):
                        self.assertNotIn("previous_response_id", sent)
                    drops = log.read_text().count("previous_response_id_dropped") - logged
                    self.assertEqual(drops, 1 if outcome == "drop" else 0)

    def test_route_by_control_matrix(self):
        import argparse
        import muse_kiro
        from muse_devin import DevinModel, build_chat_request, build_tools
        from muse_gateway import Config, run_claude_cli
        from muse_providers import ProviderError, responses_to_anthropic

        tool = {"type": "function", "name": "echo", "parameters": {"type": "object", "properties": {}}}
        controls = {
            "tool_choice": {"type": "function", "name": "echo"},
            "parallel_tool_calls": False,
            "stop": ["END"],
        }
        body = {"model": "fixture", "input": "hi", "tools": [tool], **controls}

        # Claude HTTP: Anthropic payload carries all three controls.
        claude_payload, _ = responses_to_anthropic(body)
        self.assertEqual(claude_payload["tool_choice"], {
            "type": "tool", "name": "echo", "disable_parallel_tool_use": True})
        self.assertEqual(claude_payload["stop_sequences"], ["END"])

        # Claude CLI: choice/parallel use the same Anthropic payload policy;
        # stop is explicitly unsupported and rejected before spawn.
        cli_payload, _ = responses_to_anthropic({
            key: value for key, value in body.items() if key != "stop"})
        self.assertEqual(cli_payload["tool_choice"], claude_payload["tool_choice"])
        with patch("muse_gateway.subprocess.Popen", side_effect=AssertionError("child spawned")):
            with self.assertRaises(ProviderError) as caught:
                run_claude_cli(body, Config(argparse.Namespace(provider="claude", model_discovery=False)))
        self.assertEqual(caught.exception.code, "claude_cli_stop_unsupported")

        # Devin: protobuf request carries all three controls.
        model = DevinModel(
            uid="fixture", logical_id="fixture", label="Fixture",
            context_limit=1000, output_limit=1000, supports_parallel=True)
        devin_request = build_chat_request(
            "", body, model, model.uid, "k", "jwt", "cascade", [],
            build_tools([tool], False))
        self.assertEqual(devin_request.tool_choice.tool_name, "echo")
        self.assertTrue(devin_request.disable_parallel_tool_calls)
        self.assertEqual(list(devin_request.configuration.stop_patterns), ["END"])

        # Kiro: each unrepresentable control is rejected by name.
        for option, code in (
            ({"tool_choice": controls["tool_choice"]}, "kiro_unsupported_tool_choice"),
            ({"parallel_tool_calls": False}, "kiro_unsupported_option"),
            ({"stop": ["END"]}, "kiro_unsupported_option"),
        ):
            with self.subTest(route="kiro", option=option):
                with self.assertRaises(ProviderError) as caught:
                    muse_kiro.prepare_request({"model": "haiku", "input": "hi", "tools": [tool], **option})
                self.assertEqual(caught.exception.code, code)

        # Codex passthrough: the upstream request preserves all three.
        peer = Peer([responses_text("ok")])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        _, wire = post(base, {"model": "codex/gpt-5.4", "input": "hi", "tools": [tool],
                              "stream": True, **controls})
        self.assertEqual(consume(wire)["status"], "completed")
        sent = peer.requests[-1]
        self.assertEqual(sent["tool_choice"], controls["tool_choice"])
        self.assertIs(sent["parallel_tool_calls"], False)
        self.assertEqual(sent["stop"], ["END"])

    def test_split_timeouts_idle_connect_and_inbound_body(self):
        # P1-09: idle budget after headers -> numbered response.failed;
        # slow headers -> 502 before any byte; stalled body -> 408.
        import socket as socket_module
        import threading

        peer = Peer([responses_text("idle")], hold_after=len(responses_text("idle")) // 2)
        base = self.start(peer, stream_idle_timeout=1.0, timeout=4)
        started = __import__("time").monotonic()
        _, wire = post(base, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True})
        elapsed = __import__("time").monotonic() - started
        peer.release.set()
        self.assertLess(elapsed, 3.5, "idle budget did not fire before the connect budget")
        events = [json.loads(line[6:]) for line in wire.decode().splitlines()
                  if line.startswith("data: ") and line.endswith("}")]
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIsInstance(events[-1]["sequence_number"], int)

        listener = socket_module.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        silent = []
        threading.Thread(target=lambda: silent.append(listener.accept()), daemon=True).start()
        self.addCleanup(lambda: [conn.close() for conn, _ in silent])
        port = listener.getsockname()[1]
        slow = self.start(Peer([]), meta_upstream=f"http://127.0.0.1:{port}", timeout=1)
        status, _, _ = self._post_error(
            slow, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True})
        self.assertEqual(status, 502)

        # The slow gateway runs with timeout=1, so the body deadline is
        # max(floor, 1) once the 60 s floor is lowered for the test.
        self.stack.enter_context(patch("muse_gateway.MuseShimHandler.body_timeout_floor", 0.5))
        host, port = slow.rsplit(":", 1)
        client = socket_module.create_connection((host.split("//")[1], int(port)), timeout=10)
        self.addCleanup(client.close)
        client.sendall(b"POST /responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                       b"Content-Length: 100\r\n\r\n{\"model\":")
        started = __import__("time").monotonic()
        reply = b""
        while b"\r\n\r\n" not in reply:
            chunk = client.recv(4096)
            if not chunk:
                break
            reply += chunk
        self.assertTrue(reply.startswith(b"HTTP/1.1 408"), reply[:64])
        self.assertLess(__import__("time").monotonic() - started, 5)

    def test_client_cancel_closes_the_upstream_and_shutdown_closes_inflight(self):
        # P1-10: a client that hangs up mid-stream stops the upstream read at
        # the next upstream chunk; server shutdown closes tracked upstreams.
        import socket as socket_module
        import threading
        import time as time_module
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from tests.peers import serving

        upstream_gone = threading.Event()
        frame = b'event: response.in_progress\ndata: {"type":"response.in_progress","sequence_number":0}\n\n'

        class Dribble(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                # One frame, then silence (a model thinking): no later write
                # can trip the BrokenPipe backstop, so only the cancel watch
                # can release the upstream.
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.flush()
                self.connection.settimeout(8)
                try:
                    if self.connection.recv(1) == b"":
                        upstream_gone.set()
                except OSError:
                    upstream_gone.set()

        upstream = self.stack.enter_context(serving(ThreadingHTTPServer(("127.0.0.1", 0), Dribble)))
        base = self.start(Peer([]), meta_upstream=upstream, stream_idle_timeout=30)
        host, port = base.split("//")[1].split(":")
        body = json.dumps({"model": "meta/muse-spark-1.3", "input": "hi", "stream": True}).encode()
        client = socket_module.create_connection((host, int(port)), timeout=5)
        client.sendall(b"POST /responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                       + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        self.assertIn(b"200", client.recv(4096))
        closed_at = time_module.monotonic()
        client.close()
        self.assertTrue(upstream_gone.wait(5), "upstream was never closed after the client left")
        self.assertLess(time_module.monotonic() - closed_at, 3)

        from muse_gateway import MuseShimServer

        class Closable:
            closed = False

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            from muse_gateway import Config
            import argparse
            server = MuseShimServer(("127.0.0.1", 0), Config(argparse.Namespace(
                provider="auto", model_discovery=False, log_file=str(Path(directory) / "g.log"))))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            resource = Closable()
            server.track(resource)
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
            self.assertTrue(resource.closed)

    def test_inference_admission_is_bounded_and_turn_deadline_fails_stream(self):
        import threading
        import time as time_module

        held = responses_text("held")
        peer = Peer([held], hold_after=len(held) // 2)
        base = self.start(peer, max_concurrent_requests=1, stream_idle_timeout=10, timeout=4)
        result = {}

        def first_request():
            try:
                result["wire"] = post(base, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True})[1]
            except Exception as error:  # reported by the assertions below
                result["error"] = error

        worker = threading.Thread(target=first_request, daemon=True)
        worker.start()
        deadline = time_module.monotonic() + 5
        while not peer.requests and time_module.monotonic() < deadline:
            time_module.sleep(0.02)
        self.assertEqual(len(peer.requests), 1)
        status, headers, payload = self._post_error(
            base, {"model": "meta/muse-spark-1.3", "input": "second", "stream": True})
        self.assertEqual(status, 503)
        self.assertEqual(headers.get("Retry-After"), "1")
        self.assertEqual(payload["error"]["type"], "gateway_busy")
        peer.release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertNotIn("error", result)
        self.assertEqual(consume(result["wire"])["status"], "completed")

        held = responses_text("deadline")
        peer = Peer([held], hold_after=len(held) // 2)
        base = self.start(peer, stream_idle_timeout=10, timeout=4, turn_timeout=0.5)
        started = time_module.monotonic()
        _, wire = post(base, {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True})
        elapsed = time_module.monotonic() - started
        peer.release.set()
        self.assertLess(elapsed, 3)
        self.assertIn(b"response.failed", wire)
        self.assertIn(b"turn_timeout", wire)

    def test_inbound_host_origin_and_token_checks_spend_no_turn(self):
        import socket as socket_module

        body = {"model": "meta/muse-spark-1.3", "input": "hi", "stream": True}
        peer = Peer([responses_text("allowed")])
        base = self.start(peer)
        origin_request = request(base, body)
        origin_request.add_header("Origin", "https://evil.example")
        status, _headers, payload = self._post_error(base, body, request_obj=origin_request)
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["type"], "browser_origin_rejected")
        self.assertEqual(peer.requests, [])

        host, port = base.rsplit(":", 1)
        raw_body = json.dumps(body).encode()
        with socket_module.create_connection((host.split("//")[1], int(port)), timeout=5) as client:
            client.sendall(
                b"POST /responses HTTP/1.1\r\nHost: evil.example\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(raw_body)}\r\n\r\n".encode() + raw_body
            )
            reply = client.recv(4096)
        self.assertTrue(reply.startswith(b"HTTP/1.1 403"), reply[:64])
        self.assertEqual(peer.requests, [])

        peer = Peer([responses_text("token-ok")])
        base = self.start(
            peer, require_inbound_token=True, inbound_token="fixture-inbound-token")
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            self.assertEqual(response.status, 200)
        status, _headers, payload = self._post_error(base, body)
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"]["type"], "inbound_token_required")
        wrong = request(base, body)
        wrong.add_header("x-muse-gateway-token", "wrong-token")
        status, _headers, _payload = self._post_error(base, body, request_obj=wrong)
        self.assertEqual(status, 401)
        self.assertEqual(peer.requests, [])
        allowed = request(base, body)
        allowed.add_header("x-muse-gateway-token", "fixture-inbound-token")
        with urllib.request.urlopen(allowed, timeout=5) as response:
            self.assertEqual(consume(response.read())["status"], "completed")
        self.assertEqual(len(peer.requests), 1)

    def test_slow_drip_body_uses_absolute_deadline(self):
        import socket as socket_module
        import threading
        import time as time_module

        base = self.start(Peer([]), timeout=1)
        self.stack.enter_context(patch("muse_gateway.MuseShimHandler.body_timeout_floor", 0.5))
        host, port = base.rsplit(":", 1)
        client = socket_module.create_connection((host.split("//")[1], int(port)), timeout=10)
        self.addCleanup(client.close)
        client.sendall(b"POST /responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                       b"Content-Length: 100\r\n\r\n{")
        stop = threading.Event()

        def drip():
            while not stop.wait(0.15):
                try:
                    client.sendall(b"x")
                except OSError:
                    return

        worker = threading.Thread(target=drip, daemon=True)
        worker.start()
        started = time_module.monotonic()
        reply = b""
        try:
            while b"\r\n\r\n" not in reply:
                chunk = client.recv(4096)
                if not chunk:
                    break
                reply += chunk
        finally:
            stop.set()
            worker.join(timeout=2)
        self.assertTrue(reply.startswith(b"HTTP/1.1 408"), reply[:64])
        self.assertLess(time_module.monotonic() - started, 2.5)

    def test_get_failures_answer_and_failed_nonstream_devin_is_marked(self):
        # P2-13: discovery raising mid-GET is a JSON 500, not a dropped socket;
        # a failed non-stream Devin turn is 200 + failed and logged as failed.
        peer = Peer([chat("partial", stop=13) + TRAILER])
        base = self.start(peer)
        with patch("muse_gateway.provider_model_catalog", side_effect=RuntimeError("boom")):
            status, _, payload = self._post_error(base, url=base + "/muse-code/models")
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"]["type"], "internal_error")
        _, raw = post(base, {"model": "devin/swe-1-6", "input": "hi", "stream": False})
        self.assertEqual(json.loads(raw)["status"], "failed")
        logs = (Path(self.directory) / "gateway.log").read_text()
        self.assertIn('"stream_failed": true', logs)
        self.assertNotIn("Traceback", logs.split("internal_error", 1)[0])

    def test_upstream_429_keeps_retry_after_and_request_id_on_every_route(self):
        # P1-08: Retry-After and one bounded request ID survive generic
        # (meta ladder), Codex, and Devin; Devin 429 is 429, not 502.
        sentinel = "prose-sentinel-must-not-leak"
        body = json.dumps({"error": {"message": sentinel}}).encode()
        limited = (429, body, {"Retry-After": "7", "x-request-id": "req_fixture_7"})
        peer = Peer([limited, limited, limited])
        base = self.start(peer, access_token="fixture-codex-token", allow_oauth_host_override=True)
        with patch("muse_gateway.CODEX_RESERVE_MODEL", "gpt-5.4"):
            for model in ("meta/muse-spark-1.3", "codex/gpt-5.4", "devin/swe-1-6"):
                with self.subTest(model=model):
                    status, headers, payload = self._post_error(
                        base, {"model": model, "input": "hi", "stream": True})
                    self.assertEqual(status, 429)
                    self.assertEqual(headers.get("Retry-After"), "7")
                    error = payload["error"]
                    self.assertEqual(error["upstream_request_id"], "req_fixture_7")
                    if model.startswith("devin/"):
                        self.assertNotIn(sentinel, json.dumps(error))

    def test_devin_401_is_an_auth_error_with_quarantine_and_no_prose(self):
        from muse_devin import _AUTH_FAILURE_MARKS, _token_fingerprint, auth_failure_mark

        sentinel = "devin-prose-sentinel"
        token = "fixture-devin-token"
        self.addCleanup(_AUTH_FAILURE_MARKS.pop, _token_fingerprint(token), None)
        peer = Peer([(401, json.dumps({"error": {"message": sentinel + " " + token}}).encode())])
        base = self.start(peer, devin_api_key=token)
        status, _, payload = self._post_error(
            base, {"model": "devin/swe-1-6", "input": "hi", "stream": True})
        self.assertEqual(status, 401)
        raw = json.dumps(payload)
        self.assertEqual(payload["error"]["type"], "devin_upstream_unauthorized")
        self.assertNotIn(sentinel, raw)
        self.assertNotIn(token, raw)
        self.assertIsNotNone(auth_failure_mark(token))

    def test_sse_line_and_event_caps_fail_typed_just_over_four_mib(self):
        # P1-13 remainder: one cap (4 MiB, Kiro _EVENT_LIMIT) on every line
        # reader; just under passes, just over is upstream_protocol_error.
        import io as io_module

        from muse_gateway import _buffered_codex_response, _iter_upstream_chunks
        from muse_providers import ProviderError, _iter_sse_text

        cap = 4 * 1024 * 1024
        under = b"data: " + b"x" * (cap - 7) + b"\n"
        over = b"data: " + b"x" * cap
        self.assertEqual(len(under), cap)
        readers = {
            "passthrough": lambda raw: list(_iter_upstream_chunks(
                io_module.BytesIO(raw), content_type="text/event-stream")),
            "claude": lambda raw: list(_iter_sse_text(io_module.BytesIO(raw))),
        }
        for name, read in readers.items():
            with self.subTest(reader=name):
                read(under + b"\n")
                with self.assertRaises(ProviderError) as caught:
                    read(over)
                self.assertEqual(caught.exception.code, "upstream_protocol_error")
        split = b"".join(b"data: " + b"y" * (1024 * 1024) + b"\n" for _ in range(5))
        with self.assertRaises(ProviderError) as caught:
            _buffered_codex_response(io_module.BytesIO(split + b"\n"), "text/event-stream")
        self.assertEqual(caught.exception.code, "upstream_protocol_error")

    def test_write_upstream_body_split_terminal_and_bounded_reads(self):
        import io as io_module

        from muse_gateway import _read_bounded_upstream_body, _write_upstream_body
        from muse_providers import ProviderError

        # (a) Parsed relay: split frames are buffered, terminals are
        # de-duplicated, missing sequence numbers are replaced locally, and
        # a missing lifecycle opening is synthesized.
        wire = responses_text("split-ok")

        class Dribble:
            readline = None

            def __init__(self, data):
                self._data = data
                self._pos = 0

            def read1(self, _size):
                chunk = self._data[self._pos:self._pos + 3]
                self._pos += len(chunk)
                return chunk

        class FakeHandler:
            def __init__(self):
                self.wfile = io_module.BytesIO()
                self._stream_failed = False
                self.events = []

            def _log_event(self, event, **fields):
                self.events.append((event, fields))

        def relay(data):
            handler = FakeHandler()
            _write_upstream_body(handler, Dribble(data), stream=True,
                                 content_type="text/event-stream")
            return handler, handler.wfile.getvalue()

        handler, relayed = relay(wire)
        self.assertEqual(consume(relayed)["status"], "completed")
        self.assertFalse(handler._stream_failed)

        frames = wire.split(b"\n\n")
        without_opening = b"\n\n".join(frames[2:])
        handler, relayed = relay(without_opening)
        self.assertEqual(consume(relayed)["status"], "completed")

        import re as re_module
        unnumbered = re_module.sub(rb',\s*"sequence_number":\s*\d+', b"", wire)
        handler, relayed = relay(unnumbered)
        self.assertEqual(consume(relayed)["status"], "completed")
        self.assertIn("upstream_missing_sequence_number", [event for event, _ in handler.events])

        duplicated_terminal = wire + wire.split(b"\n\n")[-2] + b"\n\n"
        handler, relayed = relay(duplicated_terminal)
        self.assertEqual(consume(relayed)["status"], "completed")
        self.assertEqual(relayed.count(b"event: response.completed"), 1)

        handler, relayed = relay(responses_text("x" * 70000))
        self.assertEqual(consume(relayed)["status"], "completed")

        malformed_tail = b"\n\n".join(frames[:3]) + b"\n\nevent: broken\ndata: {not-json\n\n"
        handler, relayed = relay(malformed_tail)
        terminal = consume(relayed)
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["error"]["code"], "upstream_protocol_error")
        self.assertNotIn(b"not-json", relayed)

        # (b) Distinct bounded-read codes for oversize vs truncation.
        class FakeUpstream:
            def __init__(self, data=b"", length=None):
                self._data = data
                self.length = length

            def read(self, size=-1):
                return self._data if size < 0 else self._data[:size]

        with self.assertRaises(ProviderError) as caught:
            _read_bounded_upstream_body(FakeUpstream(b"x" * 64), limit=8)
        self.assertEqual(caught.exception.code, "upstream_response_too_large")
        with self.assertRaises(ProviderError) as caught:
            _read_bounded_upstream_body(FakeUpstream(b"short", length=99), limit=64)
        self.assertEqual(caught.exception.code, "upstream_truncated")
        self.assertEqual(_read_bounded_upstream_body(FakeUpstream(b"fine"), limit=64), b"fine")
        from muse_gateway import _read_bounded_json
        with self.assertRaises(ProviderError) as caught:
            _read_bounded_json(io_module.BytesIO(b"x" * 65), limit=64)
        self.assertEqual(caught.exception.code, "upstream_response_too_large")
        self.assertEqual(_read_bounded_json(io_module.BytesIO(b'{"ok": true}'), limit=64), {"ok": True})


if __name__ == "__main__":
    unittest.main()
