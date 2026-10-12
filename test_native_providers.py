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

from muse_providers import (
    AuthError,
    CLAUDE_TOKEN_URLS,
    CLAUDE_OAUTH_CLIENT_ID,
    CODEX_OAUTH_CLIENT_ID,
    CredentialState,
    DEFAULT_CLAUDE_ENDPOINT,
    DEFAULT_CODEX_ENDPOINT,
    DEFAULT_CODEX_TOKEN_URL,
    ProviderError,
    ToolNameMap,
    _atomic_write_json,
    anthropic_json_to_responses,
    claude_headers,
    codex_headers,
    discover_claude_credentials,
    discover_codex_credentials,
    prepare_codex_request,
    refresh_claude_oauth,
    refresh_codex_oauth,
    resolve_claude_credentials,
    resolve_codex_credentials,
    responses_to_anthropic,
    translate_anthropic_sse,
)
from contextlib import redirect_stderr

from muse_shim import (
    Config,
    MuseShimHandler,
    ShimError,
    _existing_shim_is_compatible,
    _forwarding_warnings,
    _iter_upstream_chunks,
    _redact_bytes,
    _run_command_index,
    _warn_credential_forwarding_once,
    main,
    parse_args,
    run_claude_cli,
    run_doctor,
    run_muse,
)


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


class DummyForwardHandler:
    server_version = "muse-shim-test"

    def __init__(self, config):
        self.config = config
        self.wfile = io.BytesIO()

    def _send_upstream_headers(self, *args, **kwargs):
        return None

    def send_response(self, *args, **kwargs):
        return None

    def send_header(self, *args, **kwargs):
        return None

    def end_headers(self):
        return None


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


def _event_names(rendered):
    return [line[7:] for line in rendered.splitlines() if line.startswith("event: ")]


class NoCredentialEnvTest(unittest.TestCase):
    def setUp(self):
        self._credential_env = patch.dict(
            os.environ,
            {
                "MUSE_SHIM_CODEX_ACCESS_TOKEN": "",
                "CODEX_ACCESS_TOKEN": "",
                "OPENAI_CODEX_ACCESS_TOKEN": "",
                "MUSE_SHIM_CODEX_REFRESH_TOKEN": "",
                "CODEX_REFRESH_TOKEN": "",
                "CLAUDE_CODE_OAUTH_TOKEN": "",
                "ANTHROPIC_TOKEN": "",
                "MUSE_SHIM_CLAUDE_ACCESS_TOKEN": "",
                "CLAUDE_CODE_REFRESH_TOKEN": "",
                "MUSE_SHIM_CLAUDE_REFRESH_TOKEN": "",
                "ANTHROPIC_API_KEY": "",
                "MUSE_SHIM_CLAUDE_CREDENTIALS": "",
                "CLAUDE_CODE_CREDENTIALS_PATH": "",
                "CLAUDE_CONFIG_DIR": "",
            },
            clear=False,
        )
        self._credential_env.start()

    def tearDown(self):
        self._credential_env.stop()


class CodexAuthTests(NoCredentialEnvTest):
    def test_headers_include_first_party_identity_and_account_claim(self):
        token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}})
        headers = codex_headers(token)
        self.assertEqual(headers["Authorization"], f"Bearer {token}")
        self.assertEqual(headers["originator"], "codex_cli_rs")
        self.assertIn("codex_cli_rs", headers["User-Agent"])
        self.assertEqual(headers["ChatGPT-Account-ID"], "acct-test")

    def test_codex_preflight_is_backend_safe_without_dropping_valid_fields(self):
        body = {
            "model": "gpt-5-codex",
            "input": "hello",
            "store": True,
            "max_output_tokens": 1234,
            "tools": [],
            "reasoning": {"effort": "xhigh"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "cache-key",
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        prepared = prepare_codex_request(body)
        self.assertFalse(prepared["store"])
        self.assertNotIn("max_output_tokens", prepared)
        self.assertNotIn("tools", prepared)
        for key in (
            "model",
            "reasoning",
            "include",
            "prompt_cache_key",
            "tool_choice",
            "parallel_tool_calls",
        ):
            self.assertEqual(prepared[key], body[key])
        # String input is normalized: Codex 400s it ("Input must be a list").
        self.assertEqual(
            prepared["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        )
        history = {
            "model": "gpt-5-codex", "instructions": "original instructions",
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": "  saved context  "}]},
                {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            ],
        }
        original = json.dumps(history)
        prepared = prepare_codex_request(history)
        self.assertEqual(prepared["instructions"], "original instructions\n\n  saved context  ")
        self.assertEqual(prepared["input"], history["input"][1:])
        self.assertEqual(json.dumps(history), original)
        dropped = prepare_codex_request(
            dict(history, previous_response_id="previous")
        )
        self.assertNotIn("previous_response_id", dropped)
        self.assertEqual(dropped["input"], history["input"][1:])
        with self.assertRaises(ProviderError) as rejected:
            prepare_codex_request(
                {"model": "gpt-5-codex", "previous_response_id": "previous"}
            )
        self.assertEqual(rejected.exception.code, "codex_previous_response_unsupported")
        history["input"][0]["content"][0]["type"] = []
        with self.assertRaises(ProviderError) as rejected:
            prepare_codex_request(history)
        self.assertEqual(rejected.exception.code, "codex_system_content_unsupported")

    def test_refresh_uses_public_client_and_redacts_token(self):
        refresh = "refresh-secret"
        opener = RecordingOpener(FakeResponse(json.dumps({"access_token": "new-access", "expires_in": 3600})))
        result = refresh_codex_oauth(
            refresh,
            token_url="https://fake.test/token",
            opener=opener,
            allow_host_override=True,
        )
        self.assertEqual(result["access_token"], "new-access")
        request = opener.requests[0]
        form = request.data.decode()
        self.assertIn(f"client_id={CODEX_OAUTH_CLIENT_ID}", form)
        self.assertIn("grant_type=refresh_token", form)
        # The refresh token is necessarily sent to the token endpoint; safety
        # is checked at the logging/doctor surface, not by hiding it from the
        # in-memory result used to make the next request.
        self.assertEqual(result["refresh_token"], refresh)

    def test_refresh_rejects_non_native_endpoint_without_explicit_override(self):
        with self.assertRaises(ProviderError):
            refresh_codex_oauth(
                "refresh-secret",
                token_url="https://fake.test/token",
                opener=RecordingOpener(FakeResponse("{}")),
            )

    def test_codex_refresh_defaults_to_native_token_endpoint(self):
        opener = RecordingOpener(
            FakeResponse(json.dumps({"access_token": "new-access", "expires_in": 3600}))
        )
        refresh_codex_oauth("refresh-secret", opener=opener)
        self.assertEqual(opener.requests[0].full_url, DEFAULT_CODEX_TOKEN_URL)

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
                    codex_token_url="https://fake.test/token",
                    allow_oauth_host_override=True,
                )
            )
            opener = RecordingOpener(FakeResponse(json.dumps({"access_token": "rotated-access", "refresh_token": "rotated-refresh", "expires_in": 3600})))
            state = resolve_codex_credentials(config, opener=opener)
            self.assertEqual(state.access_token, "rotated-access")
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

    def test_codex_home_shape_is_discovered_without_printing_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"tokens": {"accessToken": "access-secret", "refreshToken": "refresh-secret"}}), encoding="utf-8")
            state = discover_codex_credentials(Config(_args(provider="codex", codex_home=directory)))
            self.assertIsNotNone(state)
            self.assertEqual(state.access_token, "access-secret")


class NativeRetryTests(NoCredentialEnvTest):
    def test_codex_401_refreshes_and_retries_once(self):
        first = CredentialState(provider="codex", access_token="old-access", refresh_token="old-refresh")
        second = CredentialState(provider="codex", access_token="new-access", refresh_token="new-refresh")
        success = FakeResponse(json.dumps({"id": "resp-ok"}), headers={"Content-Type": "application/json"})
        config = Config(_args(provider="codex"))
        handler = DummyForwardHandler(config)
        with patch(
            "muse_shim.resolve_codex_credentials",
            side_effect=[first, second],
        ), patch(
            "muse_shim.urllib.request.urlopen",
            side_effect=[_unauthorized_response(), success],
        ) as urlopen:
            MuseShimHandler.forward_codex(
                handler,
                {
                    "model": "gpt-5-codex",
                    "input": "hello",
                    "stream": False,
                },
            )
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[0].args[0].headers["Authorization"], "Bearer old-access")
        self.assertEqual(urlopen.call_args_list[1].args[0].headers["Authorization"], "Bearer new-access")

    def test_codex_refresh_failure_surfaces_login_action(self):
        first = CredentialState(provider="codex", access_token="old-access", refresh_token="old-refresh")
        config = Config(_args(provider="codex"))
        handler = DummyForwardHandler(config)
        with patch(
            "muse_shim.resolve_codex_credentials",
            side_effect=[first, AuthError("refresh failed")],
        ), patch(
            "muse_shim.urllib.request.urlopen",
            side_effect=[_unauthorized_response()],
        ):
            with self.assertRaisesRegex(AuthError, "codex login"):
                MuseShimHandler.forward_codex(
                    handler,
                    {"model": "gpt-5-codex", "input": "hello"},
                )

    def test_claude_oauth_401_refreshes_and_retries_once(self):
        first = CredentialState(provider="claude", access_token="old-access", refresh_token="old-refresh")
        second = CredentialState(provider="claude", access_token="new-access", refresh_token="new-refresh")
        success = FakeResponse(
            json.dumps(
                {
                    "id": "msg-ok",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "ok"}],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        config = Config(_args(provider="claude"))
        handler = DummyForwardHandler(config)
        with patch(
            "muse_shim.resolve_claude_credentials",
            side_effect=[first, second],
        ), patch(
            "muse_shim.urllib.request.urlopen",
            side_effect=[_unauthorized_response(), success],
        ) as urlopen:
            MuseShimHandler.forward_claude(
                handler,
                {"model": "claude-sonnet-4-6", "input": "hello"},
            )
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[0].args[0].headers["Authorization"], "Bearer old-access")
        self.assertEqual(urlopen.call_args_list[1].args[0].headers["Authorization"], "Bearer new-access")


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
                    claude_token_url="https://fake.test/token",
                    allow_oauth_host_override=True,
                )
            )
            opener = RecordingOpener(FakeResponse(json.dumps({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})))
            state = resolve_claude_credentials(config, opener=opener)
            self.assertEqual(state.access_token, "new-access")
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

    def test_claude_refresh_rejects_non_native_endpoint_without_explicit_override(self):
        with self.assertRaises(ProviderError):
            refresh_claude_oauth(
                "refresh-secret",
                token_urls=("https://fake.test/token",),
                opener=RecordingOpener(FakeResponse("{}")),
            )

    def test_claude_refresh_defaults_to_a_known_native_token_endpoint(self):
        opener = RecordingOpener(
            FakeResponse(json.dumps({"access_token": "new-access", "expires_in": 3600}))
        )
        refresh_claude_oauth("refresh-secret", opener=opener)
        self.assertEqual(opener.requests[0].full_url, CLAUDE_TOKEN_URLS[0])

    def test_previous_response_id_is_rejected_locally(self):
        with self.assertRaisesRegex(ProviderError, "durable response-state"):
            responses_to_anthropic(
                {
                    "model": "claude-sonnet-4-6",
                    "input": "continue",
                    "previous_response_id": "resp_previous",
                },
                oauth=True,
            )

    def test_non_stream_response_round_trips_tool_name_and_usage(self):
        names = ToolNameMap(oauth=True)
        names.add("read_file")
        response = anthropic_json_to_responses(
            {
                "id": "msg-1",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "tool_use", "id": "toolu-1", "name": "mcp__read_file", "input": {"path": "a"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
            tool_names=names,
        )
        self.assertTrue(response["id"].startswith("resp_"))
        call = response["output"][0]
        self.assertEqual(call["type"], "function_call")
        self.assertEqual(call["name"], "read_file")
        self.assertEqual(response["usage"]["total_tokens"], 7)


class ClaudeStreamTests(NoCredentialEnvTest):
    def test_stream_converts_text_tool_usage_and_completion(self):
        raw = """event: message_start
data: {"type":"message_start","message":{"id":"msg-stream","model":"claude-sonnet-4-6","usage":{"input_tokens":2}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu-1","name":"mcp__read_file","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"a\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

"""
        names = ToolNameMap(oauth=True)
        names.add("read_file")
        converted = translate_anthropic_sse(raw, model="claude-sonnet-4-6", tool_names=names)
        self.assertEqual(
            _event_names(converted),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        self.assertIn("response.created", converted)
        self.assertIn("response.output_text.delta", converted)
        self.assertIn("response.function_call_arguments.delta", converted)
        self.assertIn('"name":"read_file"', converted)
        self.assertIn('"input_tokens":2', converted)
        self.assertIn("response.completed", converted)

    def test_stream_error_does_not_claim_success(self):
        raw = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"try later\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        converted = translate_anthropic_sse(raw, model="claude-sonnet-4-6")
        self.assertIn("event: error", converted)
        self.assertIn("response.failed", converted)
        self.assertNotIn("event: response.completed", converted)

    def test_truncated_stream_fails_without_fabricating_completion(self):
        raw = """event: message_start
data: {"type":"message_start","message":{"id":"msg-truncated","model":"claude-sonnet-4-6"}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}

"""
        converted = translate_anthropic_sse(raw, model="claude-sonnet-4-6")
        names = _event_names(converted)
        self.assertEqual(names[-1], "response.failed")
        self.assertEqual(sum(name in {"response.completed", "response.incomplete", "response.failed"} for name in names), 1)
        self.assertNotIn("response.completed", names)
        self.assertIn("before message_stop", converted)


class CliAndSafetyTests(NoCredentialEnvTest):
    def test_cli_modes_and_provider_endpoints(self):
        self.assertEqual(parse_args([]).provider, "auto")
        self.assertEqual(parse_args(["codex"]).provider, "codex")
        self.assertEqual(parse_args(["codex"]).upstream, DEFAULT_CODEX_ENDPOINT)
        self.assertEqual(parse_args(["http"]).provider, "generic")
        self.assertEqual(parse_args(["claude"]).provider, "claude")
        self.assertEqual(parse_args(["claude"]).upstream, DEFAULT_CLAUDE_ENDPOINT)
        self.assertEqual(parse_args(["claude"]).claude_transport, "cli")

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
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(cli_result), stderr=""
        )
        config = Config(_args(provider="auto"))
        with patch.dict(os.environ, {"CLAUDECODE": "1"}, clear=False):
            with patch("muse_shim.subprocess.run", return_value=completed) as launched:
                payload, tool_names = run_claude_cli(body, config)
        command = launched.call_args.args[0]
        self.assertEqual(command[:2], ["claude", "-p"])
        self.assertIn("--safe-mode", command)
        self.assertIn("--json-schema", command)
        self.assertIn("--effort", command)
        self.assertNotIn("Look up item 7", command)
        self.assertIn("Look up item 7", launched.call_args.kwargs["input"])
        self.assertNotIn("CLAUDECODE", launched.call_args.kwargs["env"])
        converted = anthropic_json_to_responses(
            payload, model="claude-sonnet-5", tool_names=tool_names
        )
        self.assertEqual(converted["output"][0]["type"], "function_call")
        self.assertEqual(converted["output"][0]["name"], "lookup")
        self.assertEqual(converted["output"][0]["arguments"], '{"id":7}')

    def test_claude_cli_transport_surfaces_subscription_limit(self):
        cli_result = {
            "is_error": True,
            "api_error_status": 429,
            "result": "You've hit your session limit",
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=json.dumps(cli_result), stderr=""
        )
        config = Config(_args(provider="auto"))
        with patch("muse_shim.subprocess.run", return_value=completed):
            with self.assertRaises(ProviderError) as raised:
                run_claude_cli(
                    {"model": "claude-haiku-4-5-20251001", "input": "hello"},
                    config,
                )
        self.assertEqual(raised.exception.status, 429)
        self.assertIn("session limit", str(raised.exception))

    def test_run_mode_launches_muse_with_local_wiring_and_passthrough_args(self):
        config = Config(_args(provider="auto", port=0))
        with patch("muse_shim.subprocess.Popen") as launched:
            launched.return_value.wait.return_value = 7
            launched.return_value.poll.return_value = 7
            result = run_muse(config, ["--resume", "session-id"], output=io.StringIO())
        self.assertEqual(result, 7)
        command = launched.call_args.args[0]
        environment = launched.call_args.kwargs["env"]
        self.assertEqual(command[:2], ["muse", "--provider"])
        self.assertEqual(command[2], "meta")
        self.assertEqual(command[3], "--base-url")
        self.assertTrue(command[4].startswith("http://127.0.0.1:"))
        self.assertEqual(command[5:], ["--resume", "session-id"])
        self.assertEqual(environment["META_API_KEY"], "local-shim-placeholder")

    def test_run_mode_stops_muse_when_launcher_is_interrupted(self):
        config = Config(_args(provider="auto", port=0))
        with patch("muse_shim.subprocess.Popen") as launched:
            process = launched.return_value
            process.wait.side_effect = [KeyboardInterrupt(), None]
            process.poll.return_value = None
            result = run_muse(config, [], output=io.StringIO())
        self.assertEqual(result, 130)
        process.terminate.assert_called_once_with()

    def test_existing_server_must_identify_matching_shim_provider(self):
        auto = json.dumps(
            {
                "ok": True,
                "service": "muse-shim",
                "provider": "auto",
                "claude_transport": "cli",
            }
        )
        with patch("muse_shim.urllib.request.urlopen", return_value=FakeResponse(auto)):
            self.assertTrue(_existing_shim_is_compatible("http://127.0.0.1:8787", "auto"))
            self.assertFalse(_existing_shim_is_compatible("http://127.0.0.1:8787", "codex"))
        with patch(
            "muse_shim.urllib.request.urlopen",
            return_value=FakeResponse('{"ok":true}'),
        ):
            self.assertFalse(_existing_shim_is_compatible("http://127.0.0.1:8787", "auto"))
        with patch(
            "muse_shim.urllib.request.urlopen",
            return_value=FakeResponse("[]"),
        ):
            self.assertFalse(_existing_shim_is_compatible("http://127.0.0.1:8787", "auto"))

    def test_main_run_intercepts_muse_args_before_shim_argparse(self):
        with patch("muse_shim.run_muse", return_value=9) as launch:
            result = main(["run", "--model", "codex/gpt-test"])
        self.assertEqual(result, 9)
        self.assertEqual(launch.call_args.args[1], ["--model", "codex/gpt-test"])
        self.assertEqual(launch.call_args.args[0].provider, "auto")

    def test_main_accepts_shim_options_before_run(self):
        with patch("muse_shim.run_muse", return_value=0) as launch:
            result = main(["--port", "9876", "run", "--resume"])
        self.assertEqual(result, 0)
        self.assertEqual(launch.call_args.args[0].port, 9876)
        self.assertEqual(launch.call_args.args[1], ["--resume"])

    def test_run_option_value_is_not_mistaken_for_subcommand(self):
        self.assertIsNone(_run_command_index(["generic", "--model", "run"]))
        self.assertEqual(_run_command_index(["--port", "9876", "run"]), 2)

    def test_auto_rejects_global_model_pin(self):
        with self.assertRaises(SystemExit):
            parse_args(["auto", "--model", "gpt-5"])

    def test_native_hosts_and_tls_are_rejected_without_explicit_development_opt_in(self):
        with self.assertRaises(SystemExit):
            parse_args(["codex", "--upstream", "https://evil.example/responses"])
        with self.assertRaises(SystemExit):
            parse_args(["codex", "--codex-token-url", "https://evil.example/token"])
        with self.assertRaises(SystemExit):
            parse_args(["claude", "--claude-token-url", "https://evil.example/token"])
        with self.assertRaises(SystemExit):
            parse_args(["codex", "--insecure-upstream"])
        with self.assertRaises(SystemExit):
            parse_args(["claude", "--insecure-upstream"])

    def test_environment_typo_cannot_implicitly_enable_oauth_host_override(self):
        with patch.dict(
            os.environ,
            {
                "MUSE_SHIM_CODEX_UPSTREAM": "https://evil.example/responses",
                "MUSE_SHIM_ALLOW_OAUTH_HOST_OVERRIDE": "1",
            },
            clear=False,
        ):
            with self.assertRaises(SystemExit):
                parse_args(["codex"])

    def test_explicit_oauth_host_override_is_required_for_local_fake_endpoints(self):
        args = parse_args(
            [
                "codex",
                "--upstream",
                "http://127.0.0.1:9999/responses",
                "--codex-token-url",
                "http://127.0.0.1:9999/token",
                "--allow-oauth-host-override",
            ]
        )
        self.assertTrue(args.allow_oauth_host_override)
        self.assertEqual(args.upstream, "http://127.0.0.1:9999/responses")

    def test_non_loopback_binding_requires_explicit_opt_in(self):
        with self.assertRaises(SystemExit):
            parse_args(["generic", "--host", "0.0.0.0"])
        args = parse_args(["generic", "--host", "0.0.0.0", "--allow-non-loopback"])
        self.assertTrue(args.allow_non_loopback)

    def test_stream_reader_is_prompt_and_does_not_use_large_read(self):
        class LineReader:
            def __init__(self):
                self.lines = [b"event: one\n", b"data: {}\n", b"\n"]
                self.calls = []

            def readline(self):
                self.calls.append("readline")
                return self.lines.pop(0) if self.lines else b""

            def read(self, amount=-1):
                self.calls.append(("read", amount))
                raise AssertionError("large read path used")

        reader = LineReader()
        self.assertEqual(
            list(_iter_upstream_chunks(reader, content_type="text/event-stream")),
            [b"event: one\n", b"data: {}\n", b"\n"],
        )
        self.assertEqual(reader.calls[:3], ["readline", "readline", "readline"])

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

    def test_doctor_uses_gateway_branding(self):
        config = Config(_args(provider="generic", api_key="test-key"))
        config.doctor_provider = "generic"
        output = io.StringIO()
        self.assertEqual(run_doctor(config, output=output), 0)
        self.assertTrue(output.getvalue().startswith("muse-gateway doctor\n"))
        self.assertEqual(MuseShimHandler.server_version, "muse-gateway/0.5.0")

    def test_local_doctor_reports_unusable_discovery(self):
        responses = (
            FakeResponse('{"data":[]}'),
            FakeResponse('{"unexpected":"shape"}'),
            FakeResponse('not json'),
            urllib.error.URLError("unreachable"),
        )
        for response in responses:
            with self.subTest(response=type(response).__name__):
                output = io.StringIO()
                kwargs = ({"side_effect": response} if isinstance(response, Exception)
                          else {"return_value": response})
                with patch.dict(os.environ, {}, clear=True), patch(
                    "muse_shim.urllib.request.urlopen", **kwargs
                ), patch("sys.stdout", output):
                    result = main([
                        "doctor", "--doctor-provider", "local",
                        "--local-upstream", "http://127.0.0.1:1/v1",
                        "--local-api-key", "test-key",
                    ])
                self.assertEqual(result, 1)
                self.assertIn("local: not ready", output.getvalue())
                self.assertIn("no usable models", output.getvalue())
                self.assertNotIn("test-key", output.getvalue())

    def test_local_doctor_accepts_discovered_or_explicit_models(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                config = Config(_args(
                    provider="local", local_upstream="http://127.0.0.1:1",
                    local_api_key="test-key", local_models="local-test" if explicit else None,
                ))
                output = io.StringIO()
                response = FakeResponse('{"data":[{"id":"local-test"}]}')
                with patch("muse_shim.urllib.request.urlopen", return_value=response):
                    self.assertEqual(run_doctor(config, output=output), 0)
                self.assertIn("local: ready", output.getvalue())

    def test_auto_doctor_checks_discovery_only_local_route(self):
        config = Config(_args(
            provider="auto", local_upstream="http://127.0.0.1:1",
            local_api_key="test-key",
        ))
        output = io.StringIO()
        with patch("muse_shim._credential_status", return_value=(True, "configured")) as status:
            self.assertEqual(run_doctor(config, output=output), 0)
        self.assertIn("local", [call.args[1] for call in status.call_args_list])

    def test_credential_forwarding_warning_fires_once_for_custom_upstreams(self):
        _forwarding_warnings.clear()
        try:
            output = io.StringIO()
            with redirect_stderr(output):
                _warn_credential_forwarding_once(
                    "meta", "https://custom.example/v1", custom=True
                )
                _warn_credential_forwarding_once(
                    "meta", "https://custom.example/v1", custom=True
                )
                _warn_credential_forwarding_once(
                    "local", "http://127.0.0.1:4000/v1", custom=False
                )
            rendered = output.getvalue()
            self.assertIn("custom.example", rendered)
            self.assertEqual(rendered.count("WARNING"), 1)
        finally:
            _forwarding_warnings.clear()

    def test_error_redaction_is_byte_safe(self):
        import http.client
        import threading
        from muse_shim import MuseShimServer

        self.assertEqual(_redact_bytes(b"token=secret-value", ["secret-value"]), b"token=[REDACTED]")
        self.assertEqual(_redact_bytes(b"secret-value", ["Bearer secret-value"]), b"[REDACTED]")
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "gateway.log"
            config = Config(_args(api_key="route-secret", log_file=str(log_path)))
            server = MuseShimServer(("127.0.0.1", 0), config)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            stderr = io.StringIO()
            upstream_body = json.dumps({"error": {
                "message": "route-secret incoming-secret private-prompt",
                "authentication_credential": "unknown-secret",
                "param": "reasoning.effort", "code": "unsupported_value",
            }}).encode()
            failure = urllib.error.HTTPError(
                "http://127.0.0.1/responses", 400, "bad request", {}, io.BytesIO(upstream_body)
            )
            try:
                with patch("muse_shim.urllib.request.urlopen", side_effect=failure), redirect_stderr(stderr):
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
                with patch("muse_shim.urllib.request.urlopen", return_value=stream), redirect_stderr(stderr):
                    connection.request("POST", "/responses", json.dumps({
                        "model": "audit-model", "input": "private-prompt", "stream": True,
                    }), {"Authorization": "Bearer incoming-secret", "Content-Type": "application/json"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), stream_bytes)
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

    def test_cli_inference_is_explicitly_limited_to_claude_print_mode(self):
        source = Path("muse_shim.py").read_text(encoding="utf-8") + Path("muse_providers.py").read_text(encoding="utf-8")
        self.assertNotIn("codex " + "exec", source)
        self.assertNotIn("shell=True", source)


if __name__ == "__main__":
    unittest.main()
