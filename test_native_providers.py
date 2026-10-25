import argparse
import base64
import io
import json
import os
import stat
import subprocess
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from muse_providers import AuthError, CLAUDE_TOKEN_URLS, CLAUDE_OAUTH_CLIENT_ID, CODEX_OAUTH_CLIENT_ID, CredentialState, DEFAULT_CLAUDE_ENDPOINT, DEFAULT_CODEX_ENDPOINT, DEFAULT_CODEX_TOKEN_URL, ProviderError, StreamState, ToolNameMap, _atomic_write_json, anthropic_json_to_responses, claude_headers, convert_anthropic_event, discover_codex_credentials, refresh_claude_oauth, refresh_codex_oauth, resolve_claude_credentials, resolve_codex_credentials, responses_to_anthropic
from contextlib import redirect_stderr

from muse_gateway import Config, MuseShimHandler, ShimError, _redact_bytes, parse_args, run_claude_cli, run_doctor, run_muse
from tests.peers import serving


class FakeResponse:
    def __init__(self, payload, *, headers=None, status=200):
        self.payload = payload if isinstance(payload, bytes) else payload.encode()
        self.headers = headers or {"Content-Type": "application/json"}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, amount=-1):
        if amount == -1:
            payload, self.payload = self.payload, b""
            return payload
        payload, self.payload = self.payload[:amount], self.payload[amount:]
        return payload


class RecordingOpener:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        return self.response


class FakeClaudePopen:
    """Stand-in for the owned `claude -p` child (P1-10: tracked, stoppable).

    Class attributes set the scripted outcome per test; ``spawned``
    records every constructed child in order.
    """

    spawned: list = []
    stdout = ""
    stderr = ""
    script_returncode = 0

    @classmethod
    def script(cls, result, returncode=0):
        cls.spawned = []
        cls.stdout = json.dumps(result)
        cls.stderr = ""
        cls.script_returncode = returncode

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.stdin_text = None
        self.returncode: int | None = None
        self.terminated = False
        FakeClaudePopen.spawned.append(self)

    def communicate(self, input=None, timeout=None):
        self.stdin_text = input
        self.returncode = FakeClaudePopen.script_returncode
        return FakeClaudePopen.stdout, FakeClaudePopen.stderr

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class DummyForwardHandler:
    server_version = "muse-shim-test"

    def __init__(self, config):
        self.config = config
        self.wfile = io.BytesIO()
        self.headers = {}
        self._request_secrets = []

    def _register_secret(self, value):
        if isinstance(value, str) and value and value not in self._request_secrets:
            self._request_secrets.append(value)

    def _all_secrets(self):
        return [*self.config.secret_values(), *self._request_secrets, ""]

    _note_upstream_error_headers = MuseShimHandler._note_upstream_error_headers
    _track_upstream = MuseShimHandler._track_upstream

    def _send_upstream_headers(self, *args, **kwargs):
        return None

    def send_response(self, *args, **kwargs):
        return None

    def send_header(self, *args, **kwargs):
        return None

    def end_headers(self):
        return None

    def send_json(self, status, value, **kwargs):
        self.wfile.write(json.dumps(value).encode("utf-8"))


def _unauthorized_response():
    return urllib.error.HTTPError(
        "https://native.example/responses",
        401,
        "unauthorized",
        {},
        io.BytesIO(b'{"error":"unauthorized"}'),
    )


def _args(**overrides):
    values = {
        "provider": "generic",
        "mode": "generic",
        "host": "127.0.0.1",
        "port": 0,
        "upstream": "http://127.0.0.1:1",
        "api_key": None,
        "access_token": None,
        "model": None,
        "portable": False,
        "max_output_tokens": None,
        "insecure_upstream": False,
        "allow_oauth_host_override": False,
        "allow_non_loopback": False,
        "verbose": False,
        "timeout": 5.0,
        "max_body_bytes": 32 * 1024 * 1024,
        "refresh_skew": 120,
        "codex_home": None,
        "codex_auth_path": None,
        "codex_token_url": None,
        "claude_credentials": None,
        "claude_token_url": None,
        "claude_transport": "cli",
        "claude_cli": "claude",
    }
    values.update(overrides)
    if "upstream" not in overrides:
        if values["provider"] == "codex":
            values["upstream"] = DEFAULT_CODEX_ENDPOINT
        elif values["provider"] == "claude":
            values["upstream"] = DEFAULT_CLAUDE_ENDPOINT
    return argparse.Namespace(**values)


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"eyJhbGciOiJub25lIn0.{encoded}.signature"


class NoCredentialEnvTest(unittest.TestCase):
    def setUp(self):
        self._credential_env = patch.dict(
            os.environ,
            {
                "CLAUDE_CODE_OAUTH_TOKEN": "",
                "CLAUDE_CODE_REFRESH_TOKEN": "",
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_CREDENTIALS_PATH": "",
                "CLAUDE_CONFIG_DIR": "",
            },
            clear=False,
        )
        self._credential_env.start()

    def tearDown(self):
        self._credential_env.stop()


class CodexAuthTests(NoCredentialEnvTest):
    def test_native_refresh_refuses_unapproved_token_endpoints(self):
        for refresh, options in ((refresh_codex_oauth, {"token_url": "https://fake.test/token"}),
                                  (refresh_claude_oauth, {"token_urls": ("https://fake.test/token",)})):
            opener = RecordingOpener(FakeResponse("{}"))
            with self.subTest(provider=refresh.__name__), self.assertRaises(ProviderError):
                refresh("refresh-secret", opener=opener, **options)
            self.assertEqual(opener.requests, [], "Refresh token escaped the native endpoint allowlist")

    def test_codex_cli_file_refresh_is_atomic_and_preserves_unrelated_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            original = {
                "auth_mode": "chatgpt",
                "account_id": "unrelated-account-field",
                "other": {"keep": True},
                "tokens": {
                    "access_token": _jwt({"exp": int(time.time()) - 10}),
                    "refresh_token": "old-refresh",
                    "last_refresh": "old",
                },
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            os.chmod(path, 0o644)
            config = Config(
                _args(
                    provider="codex",
                    codex_auth_path=str(path),
                )
            )
            opener = RecordingOpener(FakeResponse(json.dumps({"access_token": "rotated-access", "refresh_token": "rotated-refresh", "expires_in": 3600})))
            state = resolve_codex_credentials(config, opener=opener)
            self.assertEqual(state.access_token, "rotated-access")
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(opener.requests[0].full_url, DEFAULT_CODEX_TOKEN_URL)
            self.assertIn("client_id=" + CODEX_OAUTH_CLIENT_ID, opener.requests[0].data.decode())
            self.assertIn("grant_type=refresh_token", opener.requests[0].data.decode())
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["other"], {"keep": True})
            self.assertEqual(updated["account_id"], "unrelated-account-field")
            self.assertEqual(updated["tokens"]["refresh_token"], "rotated-refresh")
            if os.name == "posix":
                # Fork: Windows has no Unix mode bits; the atomicity and
                # preservation assertions above still run everywhere.
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertNotIn("rotated-access", " ".join(opener.requests[0].headers.values()))

    def test_file_refresh_adopts_newer_snapshot_before_spending_refresh_token(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(
                json.dumps(
                    {
                        "tokens": {
                            "access_token": "new-access",
                            "refresh_token": "new-refresh",
                        }
                    }
                ),
                encoding="utf-8",
            )
            stale = discover_codex_credentials(
                Config(_args(provider="codex", codex_auth_path=str(path)))
            )
            self.assertIsNotNone(stale)
            stale.access_token = "old-access"
            stale.refresh_token = "old-refresh"
            stale.expires_at = time.time() - 1
            opener = RecordingOpener(FakeResponse("{}"))
            with patch("muse_providers.discover_codex_credentials", return_value=stale):
                state = resolve_codex_credentials(
                    Config(_args(provider="codex", codex_auth_path=str(path))),
                    force_refresh=True,
                    opener=opener,
                )
            self.assertEqual(state.access_token, "new-access")
            self.assertEqual(state.refresh_token, "new-refresh")
            self.assertEqual(opener.requests, [])

    def test_atomic_write_refuses_symlink_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "real.json"
            target.write_text("{}", encoding="utf-8")
            link = Path(directory) / "auth.json"
            try:
                link.symlink_to(target)
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaises(ProviderError):
                _atomic_write_json(link, {"tokens": {"access_token": "x"}})
            self.assertEqual(target.read_text(encoding="utf-8"), "{}")


class NativeRetryTests(NoCredentialEnvTest):
    def test_native_401_refresh_is_bounded_and_closes_each_denial(self):
        for provider in ("codex", "claude"):
            for outcome in ("success", "repeated denial", "refresh failure"):
                with self.subTest(provider=provider, outcome=outcome):
                    old = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}}) if provider == "codex" else "old-access"
                    states = [CredentialState(provider=provider, access_token=old, refresh_token="old-refresh"),
                              AuthError("refresh failed") if outcome == "refresh failure" else
                              CredentialState(provider=provider, access_token="new-access", refresh_token="new-refresh")]
                    denied = [_unauthorized_response() for _ in range(2 if outcome == "repeated denial" else 1)]
                    bodies = [response.fp for response in denied]
                    for response in denied:
                        self.addCleanup(response.close)
                    replies = list(denied)
                    if outcome == "success":
                        replies.append(FakeResponse(json.dumps({"id": "msg-ok", "type": "message", "content": [{"type": "text", "text": "OK"}]})))
                    handler = DummyForwardHandler(Config(_args(provider=provider)))
                    forward = getattr(MuseShimHandler, "forward_" + provider)
                    with patch("muse_gateway.resolve_" + provider + "_credentials", side_effect=states) as resolve, patch(
                        "muse_gateway._credentialed_urlopen", side_effect=replies,
                    ) as opened:
                        if outcome == "success":
                            forward(handler, {"model": "fixture-model", "input": "hello", "stream": False})
                        else:
                            with self.assertRaisesRegex(AuthError, provider + " (login|setup-token)"):
                                forward(handler, {"model": "fixture-model", "input": "hello", "stream": False})
                    self.assertEqual(resolve.call_count, 2, "Native denial must refresh at most once")
                    self.assertTrue(resolve.call_args.kwargs["force_refresh"])
                    self.assertEqual(opened.call_count, 1 if outcome == "refresh failure" else 2)
                    self.assertEqual(opened.call_args_list[0].args[0].get_header("Authorization"), "Bearer " + old)
                    if opened.call_count == 2:
                        self.assertEqual(opened.call_args_list[1].args[0].get_header("Authorization"), "Bearer new-access")
                    if provider == "codex":
                        headers = opened.call_args_list[0].args[0].headers
                        self.assertEqual(headers["Originator"], "codex_cli_rs")
                        self.assertIn("codex_cli_rs", headers["User-agent"])
                        self.assertEqual(headers["Chatgpt-account-id"], "acct-test")
                    self.assertTrue(all(body.closed for body in bodies), "Every denied HTTP response must close")


class ClaudeAuthAndConversionTests(NoCredentialEnvTest):
    def test_claude_file_refresh_preserves_file_and_uses_oauth_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".credentials.json"
            path.write_text(
                json.dumps(
                    {
                        "other": "preserve-me",
                        "claudeAiOauth": {
                            "accessToken": "old-access",
                            "refreshToken": "old-refresh",
                            "expiresAt": int((time.time() - 1) * 1000),
                            "scopes": ["user:inference"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = Config(
                _args(
                    provider="claude",
                    claude_credentials=str(path),
                )
            )
            opener = RecordingOpener(FakeResponse(json.dumps({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})))
            state = resolve_claude_credentials(config, opener=opener)
            self.assertEqual(state.access_token, "new-access")
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(opener.requests[0].full_url, CLAUDE_TOKEN_URLS[0])
            self.assertEqual(state.auth_mode, "oauth")
            headers = claude_headers(state, stream=True)
            self.assertEqual(headers["Authorization"], "Bearer new-access")
            self.assertEqual(headers["x-app"], "cli")
            self.assertIn("claude-code-20250219", headers["anthropic-beta"])
            self.assertIn("oauth-2025-04-20", headers["anthropic-beta"])
            self.assertIn("claude-code/", headers["user-agent"])
            request_form = opener.requests[0].data.decode()
            self.assertIn(f"client_id={CLAUDE_OAUTH_CLIENT_ID}", request_form)
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["other"], "preserve-me")
            self.assertEqual(updated["claudeAiOauth"]["refreshToken"], "new-refresh")
            if os.name == "posix":
                # Fork: Windows has no Unix mode bits; the preservation
                # assertions above still run everywhere.
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_responses_request_maps_tools_calls_results_limits_and_reasoning(self):
        body = {
            "model": "claude-sonnet-4-5",
            "instructions": "Use the tools",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "read"}]},
                {"type": "function_call", "call_id": "call-1", "name": "read_file", "arguments": '{"path":"a"}'},
                {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
            ],
            "tools": [
                {"type": "namespace", "name": "muse", "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]}
            ],
            "tool_choice": {"type": "function", "function": {"name": "read_file"}},
            "max_output_tokens": 1000,
            "reasoning": {"effort": "high"},
        }
        payload, names = responses_to_anthropic(body, oauth=True)
        self.assertEqual(payload["max_tokens"], 1000)
        self.assertEqual(payload["tools"][0]["name"], "mcp__read_file")
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "mcp__read_file"})
        self.assertNotIn("thinking", payload)
        self.assertNotIn("output_config", payload)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][1]["role"], "assistant")
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(payload["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(names.decode("mcp__read_file"), "read_file")


class FlattenWireShapeTests(NoCredentialEnvTest):
    """P2-07 contract: namespace-flatten error wire shapes are pinned.

    The gateway transform path reports ``shim_error``; the provider codec
    reports ``duplicate_tool``/``invalid_tools``. The flatten
    consolidation must share logic without changing these wire types.
    """

    def _shapes(self):
        from muse_gateway import ShimError, flatten_namespace_tools
        from muse_providers import ProviderError as CodecError, _flatten_tools

        dup = [
            {"type": "namespace", "name": "ns", "tools": [
                {"type": "function", "name": "same"},
                {"type": "function", "name": "same"},
            ]},
        ]
        no_list = [{"type": "namespace", "name": "ns"}]
        unnamed = [{"type": "namespace", "name": "ns", "tools": [{"type": "function"}]}]
        shapes = {}
        for key, tools in (("duplicate", dup), ("no_list", no_list), ("unnamed", unnamed)):
            with self.assertRaises(ShimError, msg=f"gateway {key}"):
                flatten_namespace_tools(tools)
            shapes[f"gateway_{key}"] = "shim_error"
            with self.assertRaises(CodecError, msg=f"provider {key}") as caught:
                _flatten_tools(tools)
            shapes[f"provider_{key}"] = caught.exception.code
        return shapes

    def test_flatten_error_wire_shapes(self):
        self.assertEqual(
            self._shapes(),
            {
                "gateway_duplicate": "shim_error",
                "gateway_no_list": "shim_error",
                "gateway_unnamed": "shim_error",
                "provider_duplicate": "duplicate_tool",
                "provider_no_list": "invalid_tools",
                "provider_unnamed": "invalid_tools",
            },
        )


class ClaudeStreamTests(NoCredentialEnvTest):
    def _converted_calls(self, names, ids, wire_name):
        content = [{"type": "tool_use", "id": call_id, "name": wire_name, "input": {"index": index}}
                   for index, call_id in enumerate(ids)]
        converted = anthropic_json_to_responses({"id": "msg_ids", "type": "message", "content": content}, tool_names=names)
        state = StreamState("fixture", names)
        convert_anthropic_event("message_start", {"message": {"id": "msg_ids"}}, state)
        for index, block in enumerate(content):
            convert_anthropic_event("content_block_start", {"index": index, "content_block": block}, state)
            convert_anthropic_event("content_block_delta", {"index": index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"])}}, state)
            convert_anthropic_event("content_block_stop", {"index": index}, state)
        final = convert_anthropic_event("message_stop", {}, state)[-1][1]["response"]
        return converted["output"], final["output"]

    def test_claude_oauth_legacy_prefix_mapping_is_unchanged(self):
        for name in ("read_file", "mcp_read_file", "mcp__read_file"):
            names = ToolNameMap(oauth=True)
            wire_name = names.add(name)
            ids = ["toolu_original", wire_name, "mcp__unregistered"]
            for output in self._converted_calls(names, ids, wire_name):
                self.assertEqual([item["call_id"] for item in output], ["toolu_original", name, "mcp__unregistered"])
                self.assertEqual([item["name"] for item in output], [name] * 3)

    def test_identity_maps_are_injective(self):
        # P1-03 (inverts R09): no tool name, call ID, or item ID collapses.
        names = ToolNameMap(oauth=True)
        first, second = names.add("_foo"), names.add("mcp__foo")
        self.assertNotEqual(first, second)
        self.assertEqual((names.decode(first), names.decode(second)), ("_foo", "mcp__foo"))
        history = [
            {"type": "function_call", "call_id": "a:b", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "muse_call_YTpi", "name": "f", "arguments": "{}"},
            {"type": "function_call", "name": "f", "arguments": "{}"},
            {"type": "function_call", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a:b", "output": "first"},
            {"type": "function_call_output", "call_id": "muse_call_YTpi", "output": "second"},
            {"type": "function_call_output", "call_id": "orphan:id", "output": "orphan"},
        ]
        payload, _ = responses_to_anthropic({"model": "claude-test", "input": history})
        parts = [part for message in payload["messages"] for part in message["content"] if isinstance(part, dict)]
        call_ids = [part["id"] for part in parts if part["type"] == "tool_use"]
        result_ids = [part["tool_use_id"] for part in parts if part["type"] == "tool_result"]
        self.assertEqual(len(set(call_ids)), 4)
        self.assertEqual(result_ids[:2], call_ids[:2])
        self.assertRegex(result_ids[2], r"^muse_call_[A-Za-z0-9_-]{1,54}$")
        for ids in (["call_x", "fc_x"], ["fc_x", "call_x"]):
            with self.subTest(ids=ids):
                for output in self._converted_calls(ToolNameMap(oauth=False), ids, "read_file"):
                    self.assertEqual(len({item["id"] for item in output}), 2)
                    self.assertEqual([item["call_id"] for item in output], ids)

    def test_text_tool_text_item_ids_are_distinct_and_equal_across_modes(self):
        content = [
            {"type": "text", "text": "before"},
            {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}},
            {"type": "text", "text": "after"},
        ]
        for coalesce in (False, True):
            with self.subTest(coalesce_text=coalesce):
                converted = anthropic_json_to_responses({
                    "id": "msg-dummy", "type": "message", "stop_reason": "tool_use", "content": content,
                }, coalesce_text=coalesce)
                state = StreamState("fixture", ToolNameMap(oauth=False), coalesce_text=coalesce)
                convert_anthropic_event("message_start", {"message": {"id": "msg-dummy"}}, state)
                for index, block in enumerate(content):
                    start = dict(block, text="") if block["type"] == "text" else dict(block, input={})
                    convert_anthropic_event("content_block_start", {"index": index, "content_block": start}, state)
                    delta = ({"type": "text_delta", "text": block["text"]} if block["type"] == "text"
                             else {"type": "input_json_delta", "partial_json": "{}"})
                    convert_anthropic_event("content_block_delta", {"index": index, "delta": delta}, state)
                    convert_anthropic_event("content_block_stop", {"index": index}, state)
                convert_anthropic_event("message_delta", {"delta": {"stop_reason": "tool_use"}}, state)
                streamed = convert_anthropic_event("message_stop", {}, state)[-1][1]["response"]
                json_ids = [item["id"] for item in converted["output"]]
                stream_ids = [item["id"] for item in streamed["output"]]
                self.assertEqual(len(json_ids), 2 if coalesce else 3)
                self.assertEqual(len(set(json_ids)), len(json_ids))
                self.assertEqual(stream_ids, json_ids)
                self.assertEqual([item["type"] for item in streamed["output"]],
                                 [item["type"] for item in converted["output"]])

    def test_native_usage_counts_cache_and_summary_arrays_replay(self):
        # P2-03 (inverts R20 and R10); Kiro's caller-estimated total unchanged.
        from muse_providers import _anthropic_usage
        usage = {"input_tokens": 50, "output_tokens": 1,
                 "cache_read_input_tokens": 100_000, "cache_creation_input_tokens": 2_000}
        self.assertEqual(_anthropic_usage(usage, native=True)["input_tokens"], 102_050)
        self.assertEqual(_anthropic_usage(usage)["input_tokens"], 50)
        payload, _ = responses_to_anthropic({"model": "claude-test", "input": [
            {"role": "user", "content": "first"},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "SUMMARY_SENTINEL"}],
             "encrypted_content": "ENCRYPTED_SENTINEL"},
            {"role": "user", "content": "next"}]})
        self.assertIn("SUMMARY_SENTINEL", json.dumps(payload))
        self.assertNotIn("ENCRYPTED_SENTINEL", json.dumps(payload))

    def test_cache_usage_is_preserved_across_modes_without_adding_to_totals(self):
        for cached, created in ((31, 17), (0, 0)):
            with self.subTest(cached=cached, created=created):
                usage = {"input_tokens": 101, "output_tokens": 7,
                         "cache_read_input_tokens": cached, "cache_creation_input_tokens": created}
                converted = anthropic_json_to_responses({"type": "message", "content": [], "usage": usage})
                state = StreamState("fixture", ToolNameMap(oauth=False))
                convert_anthropic_event("message_start", {"message": {"usage": {**usage, "output_tokens": 0}}}, state)
                convert_anthropic_event("message_delta", {"usage": {"output_tokens": 7}}, state)
                self.assertEqual(state.response()["usage"], converted["usage"])
                self.assertEqual(converted["usage"], {
                    "input_tokens": 101, "output_tokens": 7, "total_tokens": 108,
                    "input_tokens_details": {"cached_tokens": cached, "anthropic_cache_creation_input_tokens": created},
                })
                convert_anthropic_event("message_delta", {"usage": {"cache_read_input_tokens": 9, "cache_creation_input_tokens": 2}}, state)
                self.assertEqual(state.response()["usage"]["input_tokens_details"], {
                    "cached_tokens": 9, "anthropic_cache_creation_input_tokens": 2})
        ordinary = anthropic_json_to_responses({"type": "message", "content": [], "usage": {"input_tokens": 1}})
        self.assertNotIn("input_tokens_details", ordinary["usage"])

    def test_coalescing_message_stop_does_not_finish_partial_tool_arguments(self):
        # P0-07: an unclosed block at message_stop fails the turn (never a
        # finished call); partial arguments stay in_progress, not done.
        state = StreamState("fixture", ToolNameMap(oauth=False), coalesce_text=True)
        convert_anthropic_event("message_start", {"message": {"id": "msg_partial"}}, state)
        convert_anthropic_event("content_block_start", {"index": 0, "content_block": {
            "type": "tool_use", "id": "partial", "name": "read_file"}}, state)
        convert_anthropic_event("content_block_delta", {"index": 0, "delta": {
            "type": "input_json_delta", "partial_json": '{"path":'}}, state)
        convert_anthropic_event("message_delta", {"delta": {"stop_reason": "max_tokens"}}, state)
        events = convert_anthropic_event("message_stop", {}, state)
        self.assertEqual([kind for kind, _ in events], ["response.failed"])
        self.assertEqual(events[0][1]["response"]["output"][0]["status"], "in_progress")
        self.assertEqual(events[0][1]["response"]["output"][0]["arguments"], '{"path":')


class MuseAcceptanceAssertionTests(unittest.TestCase):
    def test_acceptance_rejects_direct_script_invocation_before_provider_imports(self):
        import sys
        root = Path(__file__).resolve().parent
        environment = {**os.environ, "PYTHONPATH": str(root)}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", str(root / "tests" / "check_muse.py"), "--help"],
                                    cwd=directory, env=environment, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("python -m tests.check_muse", result.stderr)
        self.assertIn("Direct script invocation", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_acceptance_checks_all_module_origins_against_its_checkout(self):
        from tests.check_muse import assert_checkout_imports
        origins = assert_checkout_imports()
        root = Path(__file__).resolve().parent
        self.assertEqual(Path(origins["muse_gateway"]), root / "muse_gateway.py")
        self.assertEqual(Path(origins["tests.peers"]), root / "tests" / "peers.py")
        for name in origins:
            if name == "tests.check_muse":
                continue
            with self.subTest(module=name), patch.object(__import__("sys").modules[name], "__file__", str(root.parent / "other-checkout" / "wrong.py")):
                with self.assertRaisesRegex(RuntimeError, "must come from this checkout"):
                    assert_checkout_imports()

    def test_cache_acceptance_requires_downstream_counters_and_unchanged_totals(self):
        from tests.check_muse import assert_downstream_cache_usage
        usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
                 "input_tokens_details": {"cached_tokens": 9, "anthropic_cache_creation_input_tokens": 3}}
        assert_downstream_cache_usage([{"type": "response.completed", "response": {"usage": usage}}])
        for path in (("input_tokens_details",), ("input_tokens_details", "cached_tokens"),
                     ("input_tokens_details", "anthropic_cache_creation_input_tokens"), ("input_tokens",), ("total_tokens",)):
            broken = self._replace_path(usage, path, None)
            with self.subTest(missing=path), self.assertRaisesRegex(AssertionError, "cache counters or unchanged token totals"):
                assert_downstream_cache_usage([{"type": "response.completed", "response": {"usage": broken}}])
        doubled = {**usage, "input_tokens": 16, "total_tokens": 18}
        with self.assertRaises(AssertionError):
            assert_downstream_cache_usage([{"type": "response.completed", "response": {"usage": doubled}}])

    def _records(self, wire_events):
        records = []
        for wire_event in wire_events:
            for phase in ("opening_stream", "stream_succeeded"):
                records.append({"payload_type": "task.lifecycle.status", "payload": {"event": {"details": {
                    "phase": phase, "facets": [
                        {"kind": "external_attempt", "operation": "model.response", "attempt": 1},
                        {"kind": "producer", "detail": {"stream": {"last_wire_event_type": wire_event}}},
                    ],
                }}}})
        records.append({"payload_type": "run.terminal.completed", "payload": {
            "terminal": "completed", "text": "PARTIAL"}})
        return records

    def _clone(self, value):
        return json.loads(json.dumps(value))

    def _replace_path(self, value, path, replacement):
        broken = self._clone(value)
        parent = broken
        for key in path[:-1]:
            parent = parent[key]
        if replacement is None:
            parent.pop(path[-1])
        else:
            parent[path[-1]] = replacement
        return broken

    def _details(self, record):
        return record["payload"]["event"]["details"]

    def _kiro_http_response(self, *, status, kind, code, category, transient, retry_after=None):
        from tests.check_muse import KIRO_ERROR_MESSAGE

        upstream = {"type": kind, "code": code}
        metadata = {
            "provider": "kiro", "layer": "kiro_gateway", "source": "http",
            "phase": "before_response", "upstream_status": status,
            "category": category, "transient": transient,
            "output_exposure": "none", "upstream": upstream,
        }
        headers = {"content-type": "application/json"}
        if retry_after is not None:
            metadata["retry_after"] = {"value": retry_after, "kind": "delay-seconds"}
            headers["retry-after"] = retry_after
        return upstream, {
            "status": status, "headers": headers, "events": [],
            "json": {"error": {
                "type": "kiro_upstream_error", "code": code,
                "request_id": "gateway_id", "message": KIRO_ERROR_MESSAGE,
                "metadata": metadata,
            }},
        }

    def test_acceptance_requires_completed_without_retry(self):
        from tests.check_muse import assert_accepted

        assert_accepted(self._records(["response.completed", "response.completed"]), "PARTIAL")
        for events in (["response.incomplete"], ["response.incomplete", "response.completed"],
                       ["response.completed", "response.incomplete"], [None]):
            with self.subTest(events=events), self.assertRaisesRegex(AssertionError, "requires response.completed"):
                assert_accepted(self._records(events), "PARTIAL")
        records = self._records(["response.completed"])
        retry = self._clone(records[0])
        self._details(retry)["phase"] = "retry_scheduled"
        records.insert(1, retry)
        with self.assertRaisesRegex(AssertionError, "without retries"):
            assert_accepted(records, "PARTIAL")

    def test_failure_assertion_rejects_decode_or_setup_errors(self):
        from tests.check_muse import assert_failed
        for error_kind, wire_event in (("decode", "response.failed"), ("stream_error", None)):
            records = self._records([wire_event])
            details = self._details(records[1])
            details["phase"] = "stream_failed"
            details["facets"][0]["error_kind"] = error_kind
            records[-1] = {"payload_type": "run.terminal.failed", "payload": {
                "terminal": "failed", "text": "", "reason": "synthetic failure"}}
            with self.subTest(error_kind=error_kind, wire_event=wire_event), self.assertRaisesRegex(AssertionError, "never decode/setup"):
                assert_failed(records, type("Peer", (), {"requests": [object()]})(), "synthetic failure", "")

    def test_http_capture_preserves_pairs_and_rejects_ambiguous_or_truncated_responses(self):
        from tests.check_muse import capture_response
        raw = b'HTTP/1.0 429 Busy\r\nContent-Type: application/json\r\nContent-Length: 2\r\nRetry-After: 1\r\nX-Trace: a\r\nX-Trace: b\r\n\r\n{}'
        response = capture_response(raw)
        self.assertEqual(response["json"], {})
        self.assertEqual(response["header_pairs"][-2:], [("x-trace", "a"), ("x-trace", "b")])
        broken = [raw[:-1], raw + b"x", raw.replace(b"Content-Length: 2", b"Content-Length: nope")]
        for name, value in ((b"Content-Length", b"2"), (b"Content-Type", b"text/event-stream"),
                            (b"Retry-After", b"2"), (b"Transfer-Encoding", b"chunked")):
            broken.append(raw.replace(b"\r\n\r\n", b"\r\n" + name + b": " + value + b"\r\n\r\n"))
        for wire in broken:
            with self.subTest(wire=wire), self.assertRaisesRegex(AssertionError, "HTTP_CAPTURE"):
                capture_response(wire)

    def test_native_evidence_survives_a_rejected_capture_before_result_parsing(self):
        import urllib.request
        from tests.check_muse import capture_response, run_case
        from tests.peers import Peer, anthropic_text, request
        records = [{"payload_type": "run.terminal.completed", "payload": {"terminal": "completed", "text": "OBSERVED"}}]
        observed = []
        def offline_consumer(args, directory, env, *, reject):
            base = args[args.index("--base-url") + 1]
            for _ in range(2):
                with urllib.request.urlopen(request(base, {"input": "hi", "stream": True}), timeout=3) as response:
                    observed.append(response.read())
            return records, 0, ""
        peer = Peer([anthropic_text("OBSERVED"), anthropic_text("OBSERVED")])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Only replace the external process and inject the parser fault;
            # retain the real HTTP path, writers and evidence persistence.
            with patch("tests.check_muse.drive", offline_consumer), patch(
                    "tests.check_muse.capture_response", side_effect=AssertionError("synthetic capture rejection")):
                with self.assertRaisesRegex(AssertionError, "synthetic capture rejection"):
                    run_case(Path("not-launched"), root, peer, "hi", route="kiro")
            self.assertEqual([json.loads(line) for line in (root / "consumer.jsonl").read_text().splitlines()], records)
            captured = sorted(root.glob("downstream-*.http"))
            self.assertEqual(len(captured), 2)
            self.assertEqual([capture_response(path.read_bytes())["body"].encode() for path in captured], observed)
            self.assertEqual(len(peer.requests), 2)
            self.assertFalse(peer.errors)

    def test_acceptance_counts_responses_posts_and_retains_the_catalog_get(self):
        import urllib.request
        from tests.check_muse import run_case
        from tests.peers import Peer, anthropic_text, request
        def offline_consumer(args, directory, env, *, reject):
            base = args[args.index("--base-url") + 1]
            with urllib.request.urlopen(base + "/muse-code/models", timeout=3) as response:
                self.assertEqual(json.load(response)["object"], "list")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request(base, {"input": "hi", "stream": True}), timeout=3)
            with caught.exception as response:
                self.assertEqual(response.status, 429)
                self.assertEqual(json.load(response)["error"]["code"], "rate_limit_error")
            with urllib.request.urlopen(request(base, {"input": "hi", "stream": True}), timeout=3) as response:
                response.read()
            return [], 0, ""
        peer = Peer([(429, b'{"error":{"type":"rate_limit_error"}}'), anthropic_text("OBSERVED")])
        with tempfile.TemporaryDirectory() as directory, patch("tests.check_muse.drive", offline_consumer):
            root = Path(directory)
            run_case(Path("not-launched"), root, peer, "hi", route="kiro")
            result = json.loads((root / "result.json").read_text())
            self.assertEqual([(response["method"], response["path"]) for response in result["downstream_http_all"]],
                             [("GET", "/muse-code/models"), ("POST", "/responses"), ("POST", "/responses")])
            self.assertEqual([response["status"] for response in result["downstream_http"]], [429, 200])
            self.assertEqual(len(list(root.glob("downstream-*.http"))), 3)
            self.assertEqual(len(result["incoming"]), 2)
            self.assertEqual(len(peer.requests), 2)
            self.assertEqual(result["post_counts"], {"responses_incoming": 2, "upstream_inference": 2, "responses_captured": 2})

    def test_http_error_acceptance_rejects_flattened_or_missing_wire_evidence(self):
        from tests.check_muse import assert_http_error

        upstream, response = self._kiro_http_response(
            status=429, kind="rate_limit_error", code="rate_limit_exceeded",
            category="rate_limit", transient=True, retry_after="1")
        def check(value):
            assert_http_error(value, status=429, kind="rate_limit_error", code="rate_limit_exceeded",
                              category="rate_limit", transient=True, upstream=upstream, retry_after="1")
        check(response)
        changes = [(('json', 'error', 'code'), None), (('json', 'error', 'code'), 'kiro_upstream_error'),
                   (('json', 'error', 'metadata', 'category'), 'unknown'),
                   (('json', 'error', 'metadata', 'transient'), 1),
                   (('json', 'error', 'metadata', 'upstream_status'), None),
                   (('json', 'error', 'metadata', 'retry_after'), None), (('headers', 'retry-after'), None)]
        for path, replacement in changes:
            broken = self._replace_path(response, path, replacement)
            with self.subTest(path=path, replacement=replacement), self.assertRaisesRegex(AssertionError, "KIRO_"):
                check(broken)

    def test_http_failure_acceptance_rejects_setup_decode_and_retry(self):
        from tests.check_muse import assert_http_failed
        records = self._records([None])
        details = self._details(records[1])
        details["phase"] = "stream_failed"
        details["facets"][0]["error_kind"] = "http_status"
        records[-1] = {"payload_type": "run.terminal.failed", "payload": {
            "terminal": "failed", "text": "", "reason": "HTTP401 fixture"}}
        peer = type("Peer", (), {"requests": [object()]})()
        response = {"status": 401, "events": [], "json": {"error": {"code": "authentication_error", "metadata": {
            "source": "http", "upstream_status": 401, "category": "authentication"}}}}
        assert_http_failed(records, peer, response)
        # The observed native HTTP401 path has no stream_failed status.
        assert_http_failed([records[0], records[-1]], peer, response)
        for kind in (None, "decode", "setup", "configuration"):
            broken = self._clone(records)
            self._details(broken[1])["facets"][0]["error_kind"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(AssertionError, "decode/setup"):
                assert_http_failed(broken, peer, response)
        retry = self._clone(records[0])
        self._details(retry)["phase"] = "retry_scheduled"
        with self.assertRaisesRegex(AssertionError, "without retry"):
            assert_http_failed(records[:-1] + [retry, records[-1]], peer, response)
        with self.assertRaisesRegex(AssertionError, "observed typed HTTP401"):
            assert_http_failed([records[0], records[-1]], peer, {**response, "status": 200})
        with self.assertRaisesRegex(AssertionError, "observed typed HTTP401"):
            assert_http_failed([records[0], records[-1]], peer, {})
        with self.assertRaisesRegex(AssertionError, "natural HTTP failure"):
            assert_http_failed(records[1:], peer, response)
        with self.assertRaisesRegex(AssertionError, "natural HTTP failure"):
            assert_http_failed([records[0], records[-1]], type("Peer", (), {"requests": []})(), response)

    def test_context_http_failure_acceptance_rejects_missing_code_metadata_and_retry(self):
        from tests.check_muse import assert_http_error, assert_http_failed

        upstream, response = self._kiro_http_response(
            status=400, kind="invalid_request_error", code="context_length_exceeded",
            category="context", transient=False)
        records = self._records([None])
        records.pop(1)  # Natural HTTP rejection need not emit stream_failed.
        records[-1] = {"payload_type": "run.terminal.failed", "payload": {
            "terminal": "failed", "text": "", "reason": "HTTP400 context fixture"}}
        peer = type("Peer", (), {"requests": [object()]})()
        def check(observed, wire):
            assert_http_error(wire, status=400, kind="invalid_request_error", code="context_length_exceeded",
                              category="context", transient=False, upstream=upstream)
            assert_http_failed(observed, peer, wire, status=400, code="context_length_exceeded", category="context")
        check(records, response)
        for path, value in ((('json', 'error', 'code'), None),
                            (('json', 'error', 'code'), 'invalid_request_error'),
                            (('json', 'error', 'metadata', 'category'), 'request'),
                            (('json', 'error', 'metadata', 'transient'), None)):
            broken = self._replace_path(response, path, value)
            with self.subTest(path=path, value=value), self.assertRaisesRegex(AssertionError, "KIRO_"):
                check(records, broken)
        for phase, error in (("retry_scheduled", None), ("stream_succeeded", None),
                             ("stream_failed", "decode"), ("stream_failed", "setup")):
            extra = self._clone(records[0])
            details = self._details(extra)
            details['phase'] = phase
            details['facets'][0]['error_kind'] = error
            with self.subTest(phase=phase, error=error), self.assertRaisesRegex(AssertionError, "KIRO_NATIVE_HTTP"):
                check([records[0], extra, records[-1]], response)

    def test_native_retry_acceptance_requires_two_observed_attempts_and_one_retry(self):
        from tests.check_muse import assert_retry_completed
        records = []
        for phase, attempt, wire in (("opening_stream", 1, None), ("retry_scheduled", 1, None),
                                     ("opening_stream", 2, None), ("stream_succeeded", 2, "response.completed")):
            records.append({"payload_type": "task.lifecycle.status", "payload": {"event": {"details": {
                "phase": phase, "facets": [{"kind": "external_attempt", "operation": "model.response", "attempt": attempt},
                                           {"kind": "producer", "detail": {"stream": {"last_wire_event_type": wire}}}]}}}})
        records.append({"payload_type": "run.terminal.completed", "payload": {"terminal": "completed", "text": "RETRIED"}})
        peer = type("Peer", (), {"requests": [object(), object()]})()
        assert_retry_completed(records, peer, "RETRIED")
        for change in ("missing_retry", "extra_retry", "wrong_attempt", "decode", "incomplete"):
            broken = self._clone(records)
            if change == "missing_retry":
                broken.pop(1)
            elif change == "extra_retry":
                broken.insert(1, broken[1])
            elif change == "wrong_attempt":
                self._details(broken[2])["facets"][0]["attempt"] = 1
            elif change == "decode":
                self._details(broken[1])["facets"][0]["error_kind"] = "decode"
            else:
                self._details(broken[3])["facets"][1]["detail"]["stream"]["last_wire_event_type"] = "response.incomplete"
            with self.subTest(change=change), self.assertRaisesRegex(AssertionError, "KIRO_NATIVE_RETRY"):
                assert_retry_completed(broken, peer, "RETRIED")
        self._details(records[1])["facets"][0]["retry_delay_ms"] = 4000
        peer.request_times = [10.0, 14.0]
        assert_retry_completed(records, peer, "RETRIED", retry_delay_ms=4000, min_retry_spacing=3.8)
        for change in ("default_delay", "default_spacing"):
            broken = self._clone(records)
            peer.request_times = [10.0, 11.0 if change == "default_spacing" else 14.0]
            if change == "default_delay":
                self._details(broken[1])["facets"][0]["retry_delay_ms"] = 1000
            with self.subTest(change=change), self.assertRaisesRegex(AssertionError, "honor Retry-After"):
                assert_retry_completed(broken, peer, "RETRIED", retry_delay_ms=4000, min_retry_spacing=3.8)


class CliAndSafetyTests(NoCredentialEnvTest):
    def assert_cli_error(self, args, message):
        diagnostic = io.StringIO()
        with redirect_stderr(diagnostic), self.assertRaises(SystemExit) as caught:
            parse_args(args)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(message, diagnostic.getvalue())

    def test_claude_cli_transport_maps_structured_tool_call(self):
        body = {
            "model": "claude-sonnet-5",
            "instructions": "Use tools when needed.",
            "input": "Look up item 7",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up an item",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                }
            ],
            "reasoning": {"effort": "high"},
        }
        cli_result = {
            "is_error": False,
            "structured_output": {
                "text": "",
                "tool_calls": [
                    {"name": "lookup", "arguments_json": '{"id":7}'}
                ],
            },
            "usage": {"input_tokens": 11, "output_tokens": 5},
        }
        FakeClaudePopen.script(cli_result)
        config = Config(_args(provider="auto"))
        with patch.dict(os.environ, {"CLAUDECODE": "1"}, clear=False):
            with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
                payload, tool_names = run_claude_cli(body, config)
        child = FakeClaudePopen.spawned[0]
        command = child.command
        self.assertEqual(command[:2], ["claude", "-p"])
        self.assertIn("--safe-mode", command)
        self.assertIn("--json-schema", command)
        self.assertIn("--effort", command)
        self.assertNotIn("Look up item 7", command)
        self.assertIn("Look up item 7", child.stdin_text)
        self.assertNotIn("CLAUDECODE", child.kwargs["env"])
        converted = anthropic_json_to_responses(
            payload, model="claude-sonnet-5", tool_names=tool_names
        )
        self.assertEqual(converted["output"][0]["type"], "function_call")
        self.assertEqual(converted["output"][0]["name"], "lookup")
        self.assertEqual(converted["output"][0]["arguments"], '{"id":7}')

    def test_request_tool_policy_reaches_claude_http_and_cli(self):
        # P1-05 (inverts R08 Claude halves): serial and forced choices are
        # sent or enforced, and unrepresentable controls are rejected.
        tool = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
        body = {"model": "claude-sonnet-5", "input": "go", "tools": [tool]}
        native, _ = responses_to_anthropic(dict(body, parallel_tool_calls=False))
        self.assertEqual(native["tool_choice"], {"type": "auto", "disable_parallel_tool_use": True})
        native, _ = responses_to_anthropic(dict(body, tool_choice="required", parallel_tool_calls=False))
        self.assertEqual(native["tool_choice"], {"type": "any", "disable_parallel_tool_use": True})
        for bad, code in (({"tool_choice": {"type": "web_search"}}, "claude_unsupported_tool_choice"),
                          ({"parallel_tool_calls": "no"}, "claude_invalid_request")):
            with self.subTest(bad=bad), self.assertRaises(ProviderError) as caught:
                responses_to_anthropic(dict(body, **bad))
            self.assertEqual(caught.exception.code, code)

        def cli(request, calls):
            FakeClaudePopen.script({
                "is_error": False, "structured_output": {"text": "plain", "tool_calls": calls}})
            with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
                result = run_claude_cli(dict(body, **request), Config(_args(provider="auto")))
            command = FakeClaudePopen.spawned[-1].command
            return result, json.loads(command[command.index("--json-schema") + 1])

        one = [{"name": "lookup", "arguments_json": "{}"}]
        for request, calls in (({"tool_choice": "required"}, []),
                               ({"tool_choice": {"type": "function", "name": "lookup"}}, []),
                               ({"parallel_tool_calls": False}, one * 2)):
            with self.subTest(request=request), self.assertRaises(ProviderError) as caught:
                cli(request, calls)
            self.assertEqual(caught.exception.code, "claude_cli_invalid_response")
        (_, _), schema = cli({"tool_choice": "required", "parallel_tool_calls": False}, one)
        self.assertEqual((schema["properties"]["tool_calls"]["minItems"],
                          schema["properties"]["tool_calls"]["maxItems"]), (1, 1))
        (payload, _), schema = cli({"tool_choice": "none"}, [])
        self.assertEqual(schema["properties"]["tool_calls"]["maxItems"], 0)
        self.assertEqual(payload["stop_reason"], "end_turn")

    def test_claude_cli_rejects_images_and_unenforceable_output_caps_before_spawn(self):
        # P2-02 (inverts R22): no silent no-op cap, no base64 images in stdin.
        cases = (
            ({"max_output_tokens": 100}, "claude_cli_output_limit_unsupported"),
            ({"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]},
             "claude_cli_images_unsupported"),
        )
        for extra, code in cases:
            with self.subTest(code=code), patch(
                "muse_gateway.subprocess.Popen", side_effect=AssertionError("child spawned")
            ), self.assertRaises(ProviderError) as caught:
                run_claude_cli({"model": "claude-sonnet-5", "input": "hi", **extra}, Config(_args(provider="auto")))
            self.assertEqual(caught.exception.code, code)
        FakeClaudePopen.script({
            "is_error": False, "structured_output": {"text": "ok", "tool_calls": []}})
        with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
            run_claude_cli({"model": "claude-sonnet-5", "input": "hi", "max_output_tokens": 10_000_000},
                           Config(_args(provider="auto")))
        child_kwargs = FakeClaudePopen.spawned[-1].kwargs
        self.assertEqual(child_kwargs["encoding"], "utf-8")
        self.assertEqual(child_kwargs["errors"], "replace")

    def test_claude_cli_transport_surfaces_subscription_limit(self):
        cli_result = {
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your session limit",
        }
        FakeClaudePopen.script(cli_result, returncode=1)
        config = Config(_args(provider="auto"))
        with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
            with self.assertRaises(ProviderError) as raised:
                run_claude_cli(
                    {"model": "claude-haiku-4-5-20251001", "input": "hello"},
                    config,
                )
        self.assertEqual(raised.exception.status, 429)
        self.assertIn("session limit", str(raised.exception))

    def test_run_mode_launch_wiring_and_interrupt_lifecycle(self):
        config = Config(_args(provider="auto", port=0))
        with patch("muse_gateway.subprocess.Popen") as launched:
            launched.return_value.wait.return_value = 7
            launched.return_value.poll.return_value = 7
            result = run_muse(config, ["--resume", "session-id"], output=io.StringIO())
        self.assertEqual(result, 7)
        command = launched.call_args.args[0]
        environment = launched.call_args.kwargs["env"]
        self.assertTrue(command[0].endswith("muse") or command[0].endswith("muse.cmd"))
        self.assertEqual(command[1:4], ["--provider", "meta", "--base-url"])
        self.assertTrue(command[4].startswith("http://127.0.0.1:"))
        self.assertEqual(command[5:], ["--resume", "session-id"])
        self.assertEqual(environment["META_API_KEY"], "local-gateway-placeholder")

        with patch("muse_gateway.subprocess.Popen") as launched:
            process = launched.return_value
            process.wait.side_effect = [KeyboardInterrupt(), None]
            process.poll.return_value = None
            result = run_muse(config, [], output=io.StringIO())
        self.assertEqual(result, 130)
        process.terminate.assert_called_once_with()

    def test_native_hosts_and_tls_are_rejected_without_explicit_development_opt_in(self):
        for args, message in (
            (["codex", "--upstream", "https://evil.example/responses"], "not an approved native URL"),
            (["codex", "--codex-token-url", "https://evil.example/token"], "not an approved native URL"),
            (["claude", "--claude-token-url", "https://evil.example/token"], "not an approved native URL"),
            (["codex", "--insecure-upstream"], "not permitted"),
            (["claude", "--insecure-upstream"], "not permitted"),
        ):
            with self.subTest(args=args):
                self.assert_cli_error(args, message)

    def test_environment_typo_cannot_implicitly_enable_oauth_host_override(self):
        with patch.dict(
            os.environ,
            {
                "MUSE_SHIM_CODEX_UPSTREAM": "https://evil.example/responses",
                "MUSE_SHIM_ALLOW_OAUTH_HOST_OVERRIDE": "1",
            },
            clear=False,
        ):
            with redirect_stderr(io.StringIO()) as output:
                args = parse_args(["codex"])
            self.assertEqual(args.upstream, DEFAULT_CODEX_ENDPOINT)
            self.assertFalse(args.allow_oauth_host_override)
            self.assertIn("MUSE_SHIM_CODEX_UPSTREAM is removed and ignored", output.getvalue())
            self.assertNotIn("https://evil.example", output.getvalue())

    def test_non_loopback_binding_requires_explicit_opt_in(self):
        self.assert_cli_error(["generic", "--host", "0.0.0.0"], "refusing non-loopback")
        args = parse_args(["generic", "--host", "0.0.0.0", "--allow-non-loopback"])
        self.assertTrue(args.allow_non_loopback)

    def test_body_limit_rejects_before_reading(self):
        class NeverRead:
            def read(self, amount=-1):
                raise AssertionError("body was read before size rejection")

        handler = object.__new__(MuseShimHandler)
        handler.server = argparse.Namespace(
            config=Config(_args(provider="generic", max_body_bytes=4))
        )
        handler.headers = {"Content-Length": "5"}
        handler.rfile = NeverRead()
        with self.assertRaisesRegex(ShimError, "exceeds"):
            MuseShimHandler._read_request_body(handler)

    def test_doctor_does_not_print_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"tokens": {"access_token": "doctor-secret", "refresh_token": "refresh-secret"}}), encoding="utf-8")
            config = Config(_args(provider="codex", codex_auth_path=str(path)))
            output = io.StringIO()
            result = run_doctor(config, output=output)
            self.assertEqual(result, 0)
            rendered = output.getvalue()
            self.assertNotIn("doctor-secret", rendered)
            self.assertNotIn("refresh-secret", rendered)
            self.assertIn("secrets: not displayed", rendered)

    def test_error_redaction_is_byte_safe(self):
        import http.client
        import threading
        from muse_gateway import MuseShimServer

        self.assertEqual(_redact_bytes(b"token=secret-value", ["secret-value"]), b"token=[REDACTED]")
        self.assertEqual(_redact_bytes(b"secret-value", ["Bearer secret-value"]), b"[REDACTED]")
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "gateway.log"
            config = Config(_args(api_key="route-secret", log_file=str(log_path)))
            server = MuseShimServer(("127.0.0.1", 0), config)
            stderr = io.StringIO()
            upstream_body = json.dumps({"error": {
                "message": "route-secret incoming-secret private-prompt",
                "authentication_credential": "unknown-secret",
                "param": "reasoning.effort", "code": "unsupported_value",
            }}).encode()
            failure = urllib.error.HTTPError(
                "http://127.0.0.1/responses", 400, "bad request", {}, io.BytesIO(upstream_body)
            )
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                with patch("muse_gateway._credentialed_urlopen", side_effect=failure), redirect_stderr(stderr):
                    connection.request("POST", "/responses", json.dumps({
                        "model": "audit-model", "input": "private-prompt",
                        "reasoning": {"effort": "minimal"},
                    }), {"Authorization": "Bearer incoming-secret", "Content-Type": "application/json"})
                    response = connection.getresponse()
                    payload = json.loads(response.read())
                    self.assertEqual(response.status, 400)
                    self.assertEqual(payload["error"]["type"], "upstream_http_error")
                    self.assertTrue(payload["error"]["request_id"])
                stream_bytes = b'event: error\ndata: {"type":"error","error":{"message":"Unsupported effort","param":"reasoning.effort","code":"unsupported_value"}}\n\n'
                stream = FakeResponse(stream_bytes, headers={"Content-Type": "text/event-stream"})
                stream.readline = io.BytesIO(stream_bytes).readline
                with patch("muse_gateway._credentialed_urlopen", return_value=stream), redirect_stderr(stderr):
                    connection.request("POST", "/responses", json.dumps({
                        "model": "audit-model", "input": "private-prompt", "stream": True,
                    }), {"Authorization": "Bearer incoming-secret", "Content-Type": "application/json"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    relayed = response.read()
                    self.assertTrue(relayed.startswith(stream_bytes))
                    self.assertIn(b"response.failed", relayed)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
            logged = log_path.read_text(encoding="utf-8")
            combined = logged + json.dumps(payload) + stderr.getvalue()
            for private in ("route-secret", "incoming-secret", "private-prompt", "unknown-secret"):
                self.assertNotIn(private, combined)
            self.assertIn(payload["error"]["request_id"], logged)
            self.assertIn("unsupported_reasoning_effort", logged)
            self.assertIn('"effort": "minimal"', logged)
            self.assertIn('"stream_failed": true', logged)
            self.assertIn("upstream_stream_error", logged)
            self.assertEqual(stderr.getvalue(), "")


class RedirectAndHermesTests(unittest.TestCase):
    """P0-04 scoped opener matrix + P1-14 prompt fidelity."""

    def test_redirect_matrix_through_real_opener(self):
        import urllib.request
        from contextlib import ExitStack
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from muse_providers import build_gateway_opener

        hits = {"target": []}

        class Target(BaseHTTPRequestHandler):
            def do_GET(self):
                hits["target"].append(dict(self.headers))
                body = b"target"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
        target_port = target.server_port

        class Source(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/cross-port":
                    location = f"http://127.0.0.1:{target_port}/stolen"
                elif self.path == "/bad-port":
                    location = "http://127.0.0.1:notaport/x"
                elif self.path == "/bad-bracket":
                    location = "http://[::1/x"
                else:
                    location = "/landed"
                if self.path == "/landed":
                    body = b"landed"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
        with ExitStack() as stack:
            stack.enter_context(serving(target))
            base = stack.enter_context(serving(source))
            for path in ("/cross-port", "/bad-port", "/bad-bracket"):
                with self.subTest(path=path):
                    request = urllib.request.Request(base + path)
                    request.add_header("x-api-key", "sentinel-key")
                    opener = build_gateway_opener(base + path)
                    with self.assertRaises(urllib.error.HTTPError):
                        opener.open(request, timeout=5)
            self.assertEqual(hits["target"], [])
            request = urllib.request.Request(base + "/same-origin")
            with build_gateway_opener(base).open(request, timeout=5) as response:
                self.assertEqual(response.read(), b"landed")
            # Unit-level origin checks for scheme/port tuples.
            from muse_providers import _SameOriginRedirectHandler

            origin = _SameOriginRedirectHandler._origin
            self.assertEqual(origin("https://Example.COM/x"), ("https", "example.com", 443))
            self.assertNotEqual(origin("https://example.com"), origin("http://example.com"))
            self.assertIsNone(origin("https://example.com:notaport/"))
            # The global opener is never replaced by building ours.
            before = urllib.request._opener  # noqa: SLF001
            build_gateway_opener(base)
            self.assertIs(urllib.request._opener, before)  # noqa: SLF001

    def test_hermes_nous_text_is_not_rewritten(self):
        needles = "Hermes Agent hermes-agent Hermes agent Nous Research"
        payload, _ = responses_to_anthropic(
            {"model": "claude-test", "instructions": needles, "input": "hi"},
            oauth=True,
        )
        system_text = " ".join(
            str(part.get("text") or "") for part in payload.get("system", [])
        )
        self.assertIn(needles, system_text)


class ClaudeLifecycleTruthfulnessTests(unittest.TestCase):
    """P0-07/P1-12/P2-01: no false completions from malformed lifecycles."""

    def _run(self, raw: bytes) -> bytes:
        from muse_providers import convert_anthropic_stream

        return b"".join(convert_anthropic_stream(
            io.BytesIO(raw), model="claude-test", tool_names=ToolNameMap(oauth=False)))

    def _sse(self, *events) -> bytes:
        out = b""
        for kind, payload in events:
            out += f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n".encode()
        return out

    def _valid_tail(self):
        return (
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {}),
        )

    def test_pre_start_event_policy(self):
        raw = self._sse(
            ("ping", {}),
            ("future_extension", {"anything": 1}),
            ("message_start", {"message": {"id": "msg_x"}}),
            *self._valid_tail(),
        )
        wire = self._run(raw)
        self.assertIn(b"response.completed", wire)
        self.assertNotIn(b"response.failed", wire)

        for kind, payload in (
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "x"}}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {}),
        ):
            with self.subTest(kind=kind):
                wire = self._run(self._sse((kind, payload)))
                self.assertIn(b"response.failed", wire)
                self.assertNotIn(b"response.completed", wire)

    def test_malformed_json_and_bad_index_and_midstream_raise_fail(self):
        malformed = b'event: content_block_delta\ndata: {not-json\n\n'
        raw = self._sse(("message_start", {"message": {"id": "m"}})) + malformed
        wire = self._run(raw)
        self.assertIn(b"response.failed", wire)
        self.assertNotIn(b"response.completed", wire)

        raw = self._sse(
            ("message_start", {"message": {"id": "m"}}),
            ("content_block_start", {"index": "zero", "content_block": {"type": "text", "text": ""}}),
        )
        wire = self._run(raw)
        self.assertIn(b"response.failed", wire)
        self.assertIn(b"claude_invalid_sse", wire)

        class Exploding:
            def __init__(self, prefix):
                self._stream = io.BytesIO(prefix)

            def readline(self, limit=-1):
                line = self._stream.readline(limit)
                if not line:
                    raise ConnectionResetError("boom")
                return line

        from muse_providers import convert_anthropic_stream

        prefix = self._sse(("message_start", {"message": {"id": "m"}}))
        wire = b"".join(convert_anthropic_stream(
            Exploding(prefix), model="claude-test", tool_names=ToolNameMap(oauth=False)))
        self.assertIn(b"response.failed", wire)
        self.assertNotIn(b"response.completed", wire)

    def test_no_read_after_message_stop(self):
        raw = self._sse(("message_start", {"message": {"id": "m"}}), *self._valid_tail())

        class Guarded:
            def __init__(self, data):
                self._stream = io.BytesIO(data)

            def readline(self, limit=-1):
                line = self._stream.readline(limit)
                if not line:
                    raise AssertionError("read past message_stop")
                return line

        from muse_providers import convert_anthropic_stream

        wire = b"".join(convert_anthropic_stream(
            Guarded(raw), model="claude-test", tool_names=ToolNameMap(oauth=False)))
        self.assertIn(b"response.completed", wire)

    def test_invalid_tool_arguments_fail_without_done(self):
        raw = self._sse(
            ("message_start", {"message": {"id": "m"}}),
            ("content_block_start", {"index": 0, "content_block": {
                "type": "tool_use", "id": "t1", "name": "read_file", "input": {}}}),
            ("content_block_delta", {"index": 0, "delta": {
                "type": "input_json_delta", "partial_json": "{bad"}}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "tool_use"}}),
            ("message_stop", {}),
        )
        wire = self._run(raw)
        self.assertIn(b"response.failed", wire)
        self.assertNotIn(b"function_call_arguments.done", wire)

    def test_stop_reason_matrix_and_empty_json(self):
        cases = {
            "max_tokens": "incomplete",
            "model_context_window_exceeded": "incomplete",
            "refusal": "failed",
            "sensitive": "failed",
            "end_turn": "completed",
            "pause_turn": "completed",
            "some_future_reason": "failed",
        }
        for reason, expected in cases.items():
            with self.subTest(reason=reason):
                events = list(self._valid_tail())
                events[-2] = ("message_delta", {"delta": {"stop_reason": reason}})
                raw = self._sse(("message_start", {"message": {"id": "m"}}), *events)
                wire = self._run(raw)
                self.assertIn(f"response.{expected}".encode(), wire)
                converted = anthropic_json_to_responses({
                    "type": "message",
                    "content": [{"type": "text", "text": "hi"}],
                    "stop_reason": reason,
                })
                self.assertEqual(converted["status"], expected)
        with self.assertRaises(ProviderError) as caught:
            anthropic_json_to_responses({})
        self.assertEqual(caught.exception.code, "claude_invalid_response")


class NamedCoverageGapTests(NoCredentialEnvTest):
    """P1-16: CLI stream synthesis, effort correction, expiry arithmetic."""

    def test_claude_cli_stream_synthesis_satisfies_the_consumer_oracle(self):
        from muse_gateway import _anthropic_message_sse
        from muse_providers import convert_anthropic_stream
        from tests.contracts import consume

        for content, terminal_type in (
            ([{"type": "text", "text": "hi \u03bb"}], "message"),
            ([{"type": "text", "text": "calling"},
              {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"id": 7}}], "function_call"),
        ):
            with self.subTest(last=terminal_type):
                payload = {"id": "msg_cli", "type": "message", "role": "assistant", "model": "claude-test",
                           "content": content, "stop_reason": "tool_use" if terminal_type == "function_call" else "end_turn",
                           "usage": {"input_tokens": 3, "output_tokens": 2}}
                wire = b"".join(convert_anthropic_stream(
                    _anthropic_message_sse(payload), model="claude-test",
                    tool_names=ToolNameMap(oauth=False), coalesce_text=True, native_usage=True))
                final = consume(wire)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["output"][-1]["type"], terminal_type)
                if terminal_type == "function_call":
                    self.assertEqual(json.loads(final["output"][-1]["arguments"]), {"id": 7})

    def test_forward_claude_cli_streams_on_the_wire(self):
        from tests.contracts import consume

        FakeClaudePopen.script({
            "is_error": False, "structured_output": {"text": "CLI_WIRE", "tool_calls": []},
            "usage": {"input_tokens": 1, "output_tokens": 1}})
        handler = DummyForwardHandler(Config(_args(provider="auto")))
        handler.wfile = io.BytesIO()
        handler._log_event = lambda *args, **kwargs: None
        with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
            MuseShimHandler.forward_claude_cli(handler, {"model": "claude-test", "input": "hi", "stream": True})
        final = consume(handler.wfile.getvalue())
        self.assertEqual(final["output"][0]["content"][0]["text"], "CLI_WIRE")

    def test_claude_cli_child_is_tracked_for_shutdown(self):
        # P1-10: the owned `claude -p` child is handed to the tracker so
        # server shutdown can terminate it; the request-end release then
        # untracks it.
        FakeClaudePopen.script({
            "is_error": False, "structured_output": {"text": "hi", "tool_calls": []}})
        tracked = []
        config = Config(_args(provider="auto"))
        with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
            run_claude_cli({"model": "claude-test", "input": "hi"}, config,
                           track=tracked.append)
        self.assertEqual(len(FakeClaudePopen.spawned), 1)
        self.assertEqual(tracked, FakeClaudePopen.spawned)

    def test_correct_codex_effort_all_branches(self):
        from muse_providers import correct_codex_effort

        self.assertEqual(correct_codex_effort("xhigh", []), ("xhigh", False))
        self.assertEqual(correct_codex_effort("high", ["low", "high"]), ("high", False))
        self.assertEqual(correct_codex_effort("HIGH", ["low", "high"]), ("high", True))
        self.assertEqual(correct_codex_effort("minimal", ["low", "high"]), ("low", True))
        self.assertEqual(correct_codex_effort("ultra", ["max"]), ("max", True))
        with self.assertRaises(ProviderError) as caught:
            correct_codex_effort("minimal", ["medium"])
        self.assertEqual(caught.exception.code, "codex_unsupported_reasoning_effort")
        self.assertIn("medium", str(caught.exception))

    def test_expiry_parsing_jwt_claims_and_skew_boundaries(self):
        from muse_providers import _parse_expiry, jwt_claims, token_is_expiring

        self.assertEqual(_parse_expiry(1_800_000_000), 1_800_000_000.0)
        self.assertEqual(_parse_expiry(1_800_000_000_000), 1_800_000_000.0)
        self.assertEqual(_parse_expiry("1800000000000"), 1_800_000_000.0)
        self.assertEqual(_parse_expiry("2027-01-15T08:00:00Z"), 1_800_000_000.0)
        for value in (None, True, 0, -5, "", "  ", "not-a-date"):
            self.assertIsNone(_parse_expiry(value), value)
        self.assertEqual(jwt_claims(_jwt({"exp": 5, "sub": "x"}))["exp"], 5)
        for opaque in ("opaque-token", "a.!!!.c", "a." + base64.urlsafe_b64encode(b"[1]").decode() + ".c", None):
            self.assertEqual(jwt_claims(opaque), {})
        now = 1_000_000.0
        with patch("muse_providers._now", return_value=now):
            self.assertTrue(token_is_expiring("x", now + 120, skew_seconds=120))
            self.assertFalse(token_is_expiring("x", now + 121, skew_seconds=120))
            self.assertTrue(token_is_expiring("x", now, skew_seconds=-30))
            self.assertFalse(token_is_expiring("opaque", None))
            self.assertTrue(token_is_expiring(_jwt({"exp": int(now) + 60}), None))


class NitTests(NoCredentialEnvTest):
    """P3-01 / P3-03 / P3-04 one-line fixes."""

    def test_cli_model_token_usage_ints_and_endpoint_boundary(self):
        from muse_providers import endpoint_for

        for model in ("--dangerously-skip-permissions", "claude sonnet", "x" * 129):
            with self.subTest(model=model), patch("muse_gateway.subprocess.Popen",
                                                  side_effect=AssertionError("child spawned")):
                with self.assertRaises(ProviderError) as caught:
                    run_claude_cli({"model": model, "input": "hi"}, Config(_args(provider="auto")))
                self.assertEqual(caught.exception.code, "claude_cli_invalid_model")
        FakeClaudePopen.script({
            "is_error": False, "structured_output": {"text": "ok", "tool_calls": []},
            "usage": {"input_tokens": "many", "output_tokens": [1]}})
        with patch("muse_gateway.subprocess.Popen", FakeClaudePopen):
            payload, _ = run_claude_cli({"model": "claude-test", "input": "hi"}, Config(_args(provider="auto")))
        self.assertEqual(payload["usage"], {"input_tokens": 0, "output_tokens": 0})
        self.assertEqual(endpoint_for("https://x/v1/responses", "responses"), "https://x/v1/responses")
        self.assertEqual(endpoint_for("https://x/myresponses", "responses"), "https://x/myresponses/responses")

    def test_capture_marks_account_id_and_auth_marks_are_bounded(self):
        import muse_capture
        import muse_devin

        self.assertTrue(muse_capture._secret_field("ChatGPT-Account-ID"))
        saved = dict(muse_devin._AUTH_FAILURE_MARKS)
        self.addCleanup(lambda: (muse_devin._AUTH_FAILURE_MARKS.clear(), muse_devin._AUTH_FAILURE_MARKS.update(saved)))
        for index in range(80):
            muse_devin.mark_auth_failure(f"token-{index}")
        self.assertEqual(len(muse_devin._AUTH_FAILURE_MARKS), 64)
        self.assertIsNotNone(muse_devin.auth_failure_mark("token-79"))
        self.assertIsNone(muse_devin.auth_failure_mark("token-0"))


class RefreshLockingTests(NoCredentialEnvTest):
    """P1-01: one R0 spend under concurrency; retained fileless rotation."""

    def setUp(self):
        super().setUp()
        import muse_providers

        muse_providers._RETAINED_FILELESS.clear()
        self.addCleanup(muse_providers._RETAINED_FILELESS.clear)

    def _fileless_config(self):
        return Config(_args(provider="claude"))

    def test_fileless_rotation_is_retained_and_never_respents_r0(self):
        expired = _jwt({"exp": int(time.time()) - 10})
        env = {"CLAUDE_CODE_OAUTH_TOKEN": expired, "CLAUDE_CODE_REFRESH_TOKEN": "r0-refresh"}
        calls = []

        def opener(request, timeout=None):
            calls.append(request.data.decode())
            return FakeResponse(json.dumps({
                "access_token": "r1-access", "refresh_token": "r1-refresh",
                "expires_in": 3600}))

        with patch.dict(os.environ, env, clear=False):
            first = resolve_claude_credentials(self._fileless_config(), opener=opener)
            second = resolve_claude_credentials(self._fileless_config(), opener=opener)
        self.assertEqual(first.access_token, "r1-access")
        self.assertEqual(second.access_token, "r1-access")
        self.assertEqual(len(calls), 1, "second resolve must adopt retained R1, not re-spend R0")
        self.assertIn("r0-refresh", calls[0])

    def test_concurrent_fileless_refresh_spends_grant_once_without_fcntl(self):
        import threading as threading_module

        import muse_providers

        expired = _jwt({"exp": int(time.time()) - 10})
        env = {"CLAUDE_CODE_OAUTH_TOKEN": expired, "CLAUDE_CODE_REFRESH_TOKEN": "r0-refresh"}
        calls = []
        active = {"now": 0, "max": 0}
        guard = threading_module.Lock()

        def opener(request, timeout=None):
            with guard:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
                calls.append(request.data.decode())
            time.sleep(0.25)
            with guard:
                active["now"] -= 1
            return FakeResponse(json.dumps({
                "access_token": "r1-access", "refresh_token": "r1-refresh",
                "expires_in": 3600}))

        results = []

        def worker():
            with patch.dict(os.environ, env, clear=False):
                results.append(
                    resolve_claude_credentials(self._fileless_config(), opener=opener).access_token)

        with patch.object(muse_providers, "fcntl", None):
            threads = [threading_module.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(sorted(results), ["r1-access", "r1-access"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(active["max"], 1)


if __name__ == "__main__":
    unittest.main()
