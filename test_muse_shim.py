import argparse
import io
import json
import socket
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from muse_gateway import (
    Config,
    MuseShimHandler,
    MuseShimServer,
    ProviderError,
    ShimError,
    _select_bearer,
    flatten_namespace_tools,
    muse_catalog_response,
    provider_model_catalog,
    transform_request,
)


def loopback_sockets_available():
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
        return True
    except OSError:
        return False


NETWORK_TEST = unittest.skipUnless(
    loopback_sockets_available(),
    "loopback sockets are unavailable in this restricted test environment",
)


def config(**overrides):
    values = {
        "host": "127.0.0.1",
        "port": 0,
        "upstream": "http://127.0.0.1:1/v1",
        "api_key": "test-key",
        "model": None,
        "portable": False,
        "max_output_tokens": None,
        "insecure_upstream": False,
        "verbose": False,
    }
    values.update(overrides)
    return Config(argparse.Namespace(**values))


class TransformTests(unittest.TestCase):
    def test_flattens_muse_namespace(self):
        tools = [
            {
                "type": "namespace",
                "name": "muse",
                "description": "Muse tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        ]
        self.assertEqual(flatten_namespace_tools(tools), [tools[0]["tools"][0]])

    def test_rejects_duplicate_names(self):
        tools = [
            {"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "x"}]},
            {"type": "namespace", "name": "b", "tools": [{"type": "function", "name": "x"}]},
        ]
        with self.assertRaises(ShimError):
            flatten_namespace_tools(tools)

    def test_model_cap_and_portable_mode(self):
        body = {
            "model": "muse-spark-1.2",
            "max_output_tokens": 128000,
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "abc",
        }
        result = transform_request(
            body, config(model="anthropic/claude-sonnet-4", portable=True, max_output_tokens=16000)
        )
        self.assertEqual(result["model"], "anthropic/claude-sonnet-4")
        self.assertEqual(result["max_output_tokens"], 16000)
        self.assertNotIn("reasoning", result)
        self.assertNotIn("include", result)
        self.assertNotIn("prompt_cache_key", result)

    def test_auto_routes_catalog_ids_and_unprefixed_native_models(self):
        auto = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            codex_models="gpt-test",
            claude_models="claude-test",
        )
        self.assertEqual(auto.resolve_model("codex/gpt-test"), ("codex", "gpt-test"))
        self.assertEqual(auto.resolve_model("claude/claude-test"), ("claude", "claude-test"))
        self.assertEqual(auto.resolve_model("gpt-5"), ("codex", "gpt-5"))
        self.assertEqual(auto.resolve_model("claude-sonnet-4"), ("claude", "claude-sonnet-4"))

    def test_auto_rejects_ambiguous_model(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        with self.assertRaisesRegex(Exception, "cannot infer a provider"):
            auto.resolve_model("mystery-model")

    def test_live_codex_catalog_replaces_default_fallbacks(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "models": [
                            {
                                "slug": "gpt-account-model",
                                "context_window": 123456,
                                "max_output_tokens": 7890,
                            }
                        ]
                    }
                ).encode()

        auto = config(provider="auto", upstream=None)
        with (
            patch(
                "muse_gateway.discover_codex_credentials",
                return_value=SimpleNamespace(access_token="test-token"),
            ),
            patch("muse_gateway.urllib.request.urlopen", return_value=Response()),
        ):
            catalog = provider_model_catalog(auto, "codex")
        self.assertEqual([item["model_id"] for item in catalog], ["codex/gpt-account-model"])
        self.assertEqual(catalog[0]["context_limit"], 123456)

    def test_current_claude_oauth_fallback_catalog_and_limits(self):
        auto = config(provider="auto", upstream=None, model_discovery=True)
        with (
            patch(
                "muse_gateway.discover_claude_credentials",
                side_effect=AssertionError("CLI catalog must not inspect HTTP credentials"),
            ),
            patch("muse_gateway.shutil.which", return_value="/bin/claude"),
        ):
            catalog = provider_model_catalog(auto, "claude")
        self.assertEqual(
            [item["model_id"] for item in catalog],
            [
                "claude/claude-fable-5",
                "claude/claude-opus-5",
                "claude/claude-sonnet-5",
                "claude/claude-opus-4-8",
                "claude/claude-haiku-4-5-20251001",
            ],
        )
        for item in catalog[:3]:
            self.assertEqual(item["context_limit"], 1_000_000)
            self.assertEqual(item["output_limit"], 128_000)
        self.assertEqual(catalog[-1]["output_limit"], 64_000)


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    request_body = None
    authorization = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        type(self).authorization = self.headers.get("Authorization")
        type(self).request_body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
        )
        payload = b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@NETWORK_TEST
class ProxyTests(unittest.TestCase):
    def test_auto_dispatches_each_selected_model_to_its_native_transport(self):
        shim_config = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
        )
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        routed = []

        def record(provider):
            def forward(handler, body):
                routed.append((provider, body["model"]))
                handler.send_json(200, {"ok": True})

            return forward

        with (
            patch("muse_gateway.MuseShimHandler.forward_codex", new=record("codex")),
            patch("muse_gateway.MuseShimHandler.forward_claude_cli", new=record("claude")),
        ):
            shim_thread.start()
            try:
                for model in ("codex/gpt-selected", "claude/claude-selected"):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{shim.server_port}/responses",
                        data=json.dumps({"model": model, "input": "hello"}).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
            finally:
                shim.shutdown()
                shim.server_close()
        self.assertEqual(
            routed,
            [("codex", "gpt-selected"), ("claude", "claude-selected")],
        )

    def test_auto_catalog_is_visible_to_muse(self):
        shim_config = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            codex_models="gpt-one,gpt-two",
            claude_models="claude-one",
        )
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        shim_thread.start()
        # Menu-shape test: simulate installed providers so availability gating
        # does not hide the asserted rows.
        availability = patch(
            "muse_gateway._provider_is_available", return_value=True
        )
        availability.start()
        # Devin rows are out of scope here (covered by test_devin.py) and
        # would otherwise reflect machine credential state.
        devin_branch = patch(
            "muse_gateway._devin_catalog_entries", return_value=[]
        )
        devin_branch.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{shim.server_port}/muse-code/models",
                timeout=5,
            ) as response:
                payload = json.loads(response.read())
            self.assertEqual(payload["object"], "list")
            self.assertEqual(
                [item["id"] for item in payload["data"]],
                [
                    "codex/gpt-one",
                    "codex/gpt-two",
                    "claude/claude-one",
                    # Gateway fork: auto always advertises Meta defaults too.
                    "meta/muse-spark-1.3",
                    "meta/muse-spark-1.2",
                ],
            )
            metadata = payload["data"][0]["metadata"]["muse-code"]
            self.assertEqual(metadata["name"], "Codex · gpt-one")
            self.assertEqual(
                metadata["description"],
                "(ctx 272k) OpenAI Codex subscription model",
            )
            self.assertNotIn("cost", metadata)
            self.assertFalse(metadata["is_hidden"])
            self.assertEqual(metadata["limit"]["context"], 272000)
            self.assertTrue(metadata["tool_call"])
            self.assertEqual(metadata["modalities"]["input"], ["text", "image"])
            with urllib.request.urlopen(
                f"http://127.0.0.1:{shim.server_port}/muse-code/models",
                timeout=5,
            ) as response:
                repeated = json.loads(response.read())
            self.assertEqual(repeated, payload)
        finally:
            availability.stop()
            devin_branch.stop()
            shim.shutdown()
            shim.server_close()

    def test_forwards_transformed_stream(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()

        shim_config = config(
            upstream=f"http://127.0.0.1:{upstream.server_port}/v1",
            model="openai/gpt-5",
        )
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        shim_thread.start()
        try:
            body = {
                "model": "muse-spark-1.2",
                "stream": True,
                "input": "hello",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "muse",
                        "tools": [{"type": "function", "name": "read_file"}],
                    }
                ],
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{shim.server_port}/responses",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = response.read().decode()

            self.assertIn("response.completed", payload)
            self.assertEqual(FakeUpstreamHandler.authorization, "Bearer test-key")
            self.assertEqual(FakeUpstreamHandler.request_body["model"], "openai/gpt-5")
            self.assertEqual(FakeUpstreamHandler.request_body["tools"][0]["name"], "read_file")
            self.assertEqual(FakeUpstreamHandler.request_body["tools"][0]["type"], "function")
        finally:
            shim.shutdown()
            shim.server_close()
            upstream.shutdown()
            upstream.server_close()

    def test_missing_key_is_a_json_error(self):
        shim_config = config(api_key=None)
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        shim_thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{shim.server_port}/responses",
                data=b'{"model":"test","input":"hello"}',
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, 400)
            error = json.loads(raised.exception.read())
            self.assertIn("no upstream API key", error["error"]["message"])
        finally:
            shim.shutdown()
            shim.server_close()


class GatewayRoutingTests(unittest.TestCase):
    """Gateway fork: meta/ + local/ routes with native Meta passthrough."""

    def test_auto_routes_prefixed_meta_and_local_models(self):
        auto = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            local_models="qwen-local,llama-local",
        )
        self.assertEqual(
            auto.resolve_model("meta/muse-spark-1.3"), ("meta", "muse-spark-1.3")
        )
        self.assertEqual(
            auto.resolve_model("meta:muse-spark-1.2"), ("meta", "muse-spark-1.2")
        )
        self.assertEqual(
            auto.resolve_model("local/qwen-local"), ("local", "qwen-local")
        )
        self.assertEqual(
            auto.resolve_model("local:llama-local"), ("local", "llama-local")
        )

    def test_auto_infers_bare_muse_and_configured_local_models(self):
        auto = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            local_models="my-alias",
        )
        self.assertEqual(
            auto.resolve_model("muse-spark-1.3"), ("meta", "muse-spark-1.3")
        )
        self.assertEqual(auto.resolve_model("my-alias"), ("local", "my-alias"))

    def test_single_modes_reject_foreign_prefixes(self):
        codex_only = config(provider="codex", upstream=None, model_discovery=False)
        with self.assertRaisesRegex(ProviderError, "belongs to Meta"):
            codex_only.resolve_model("meta/muse-spark-1.3")
        generic = config(model_discovery=False)
        with self.assertRaisesRegex(ProviderError, "belongs to the local upstream"):
            generic.resolve_model("local/my-alias")
        meta_only = config(provider="meta", upstream=None, model_discovery=False)
        with self.assertRaisesRegex(ProviderError, "belongs to Codex"):
            meta_only.resolve_model("codex/gpt-x")

    def test_single_meta_and_local_modes_pass_bare_names_through(self):
        meta_only = config(provider="meta", upstream=None, model_discovery=False)
        self.assertEqual(
            meta_only.resolve_model("muse-spark-1.3"), ("meta", "muse-spark-1.3")
        )
        local_only = config(provider="local", upstream=None, model_discovery=False)
        self.assertEqual(local_only.resolve_model("my-alias"), ("local", "my-alias"))

    def test_transform_native_preserves_meta_wire_format(self):
        body = {
            "model": "meta/muse-spark-1.3",
            "max_output_tokens": 128000,
            "reasoning": {"effort": "high"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "abc",
            "tools": [
                {
                    "type": "namespace",
                    "name": "muse",
                    "tools": [{"type": "function", "name": "read_file"}],
                }
            ],
        }
        result = transform_request(
            body, config(portable=True, max_output_tokens=16000), native=True
        )
        self.assertEqual(result["tools"][0]["type"], "namespace")
        self.assertEqual(result["tools"][0]["tools"][0]["name"], "read_file")
        self.assertEqual(result["include"], ["reasoning.encrypted_content"])
        self.assertIn("reasoning", result)
        self.assertIn("prompt_cache_key", result)
        # Explicit caps still apply on the native route.
        self.assertEqual(result["max_output_tokens"], 16000)

    def test_transform_default_still_flattens(self):
        body = {
            "model": "local/my-alias",
            "tools": [
                {
                    "type": "namespace",
                    "name": "muse",
                    "tools": [{"type": "function", "name": "read_file"}],
                }
            ],
        }
        result = transform_request(body, config())
        self.assertEqual(result["tools"][0]["type"], "function")
        self.assertEqual(result["tools"][0]["name"], "read_file")

    def test_meta_catalog_entries(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        catalog = provider_model_catalog(auto, "meta")
        self.assertEqual(
            [item["model_id"] for item in catalog],
            ["meta/muse-spark-1.3", "meta/muse-spark-1.2"],
        )
        self.assertEqual(catalog[0]["display_label"], "Meta · muse-spark-1.3")
        self.assertEqual(catalog[0]["context_limit"], 1_048_576)
        self.assertEqual(
            catalog[0]["description"], "(ctx 1M) Muse Spark model · limits unverified"
        )
        self.assertFalse(catalog[0]["limits_verified"])
        self.assertTrue(catalog[0]["supports_images"])
        self.assertTrue(catalog[0]["is_default"])

    def test_local_catalog_is_opt_in(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        self.assertEqual(provider_model_catalog(auto, "local"), [])
        with_aliases = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            local_models="qwen,llama",
        )
        catalog = provider_model_catalog(with_aliases, "local")
        self.assertEqual(
            [item["model_id"] for item in catalog], ["local/qwen", "local/llama"]
        )
        self.assertEqual(catalog[0]["display_label"], "Local · qwen")
        self.assertEqual(
            catalog[0]["description"],
            "(ctx 128k) Local model through muse-gateway · limits unverified",
        )
        self.assertFalse(catalog[0]["limits_verified"])
        self.assertFalse(catalog[0]["supports_images"])

    def test_unknown_catalog_provider_raises(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        with self.assertRaisesRegex(ProviderError, "unknown provider"):
            provider_model_catalog(auto, "generic")

    def test_select_bearer_prefers_static_key_then_incoming(self):
        static, incoming = "static-key", "incoming-key"
        self.assertEqual(
            _select_bearer(static, f"Bearer {incoming}", missing="nope"),
            f"Bearer {static}",
        )
        self.assertEqual(
            _select_bearer(None, f"Bearer {incoming}", missing="nope"),
            f"Bearer {incoming}",
        )
        with self.assertRaisesRegex(ShimError, "nope"):
            _select_bearer(None, None, missing="nope")


class GatewayMetaDiscoveryTests(unittest.TestCase):
    """meta/ entries are discovered live with a static key, else bundled."""

    def _meta_config(self, **overrides):
        values = {
            "provider": "auto",
            "upstream": None,
            "meta_api_key": "test-meta-key",
        }
        values.update(overrides)
        return config(**values)

    def test_live_discovery_replaces_bundled_fallbacks(self):
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            return _ModelsResponse(
                {"data": [
                    {"id": "muse-spark-9", "context_window": 2000000,
                     "max_output_tokens": 64000},
                ]}
            )

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._meta_config(), "meta")
        self.assertEqual(seen["url"], "https://api.meta.ai/v1/models")
        self.assertEqual(seen["auth"], "Bearer test-meta-key")
        self.assertEqual(
            [entry["model_id"] for entry in catalog], ["meta/muse-spark-9"]
        )
        self.assertEqual(catalog[0]["context_limit"], 2000000)
        self.assertEqual(catalog[0]["output_limit"], 64000)
        self.assertTrue(catalog[0]["limits_verified"])
        self.assertEqual(catalog[0]["description"], "(ctx 2M) Muse Spark model")

    def test_no_static_key_keeps_bundled_catalog_without_request(self):
        def fail_urlopen(request, timeout=None):
            raise AssertionError("subscription setups must not probe Meta")

        cfg = config(provider="auto", upstream=None)
        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            catalog = provider_model_catalog(cfg, "meta")
        self.assertEqual(
            [entry["model_id"] for entry in catalog],
            ["meta/muse-spark-1.3", "meta/muse-spark-1.2"],
        )

    def test_discovery_failure_keeps_bundled_fallbacks(self):
        def fail_urlopen(request, timeout=None):
            raise urllib.error.URLError("down")

        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            catalog = provider_model_catalog(self._meta_config(), "meta")
        self.assertEqual(
            [entry["model_id"] for entry in catalog],
            ["meta/muse-spark-1.3", "meta/muse-spark-1.2"],
        )

    def test_explicit_list_is_enriched_not_replaced(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse(
                {"data": [
                    {"id": "muse-spark-1.3", "context_window": 2000000,
                     "max_output_tokens": 64000},
                    {"id": "unlisted-model", "context_window": 1,
                     "max_output_tokens": 1},
                ]}
            )

        cfg = self._meta_config(meta_models="muse-spark-1.3")
        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(cfg, "meta")
        self.assertEqual(
            [(entry["model_id"], entry["context_limit"]) for entry in catalog],
            [("meta/muse-spark-1.3", 2000000)],
        )


class GatewayForwardTests(unittest.TestCase):
    """Exercise the forward methods without binding sockets."""

    def _run_forward(self, method_name, cfg, body, incoming_auth=None):
        handler = SimpleNamespace(
            config=cfg,
            headers={"Authorization": incoming_auth} if incoming_auth else {},
            server_version="muse-shim-test",
            wfile=io.BytesIO(),
            _send_upstream_headers=lambda *args, **kwargs: None,
        )
        handler._post_responses = lambda url, payload, auth: (
            MuseShimHandler._post_responses(handler, url, payload, auth)
        )
        seen: dict = {}

        class FakeUpstream:
            headers = {"Content-Type": "application/json"}
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, *args):
                return b'{"ok":true}'

        def fake_urlopen(request, timeout=None, context=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeUpstream()

        with patch("muse_gateway.urllib.request.urlopen", side_effect=fake_urlopen):
            getattr(MuseShimHandler, method_name)(handler, body)
        return seen

    def test_forward_meta_posts_to_meta_upstream_with_static_key(self):
        cfg = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            api_key=None,
            meta_upstream="http://127.0.0.1:9/v1",
            meta_api_key="meta-key",
        )
        body = {
            "model": "muse-spark-1.3",
            "input": "hi",
            "tools": [{"type": "namespace", "name": "m", "tools": []}],
        }
        seen = self._run_forward(
            "forward_meta", cfg, body, incoming_auth=f"Bearer {'placeholder'}"
        )
        self.assertEqual(seen["url"], "http://127.0.0.1:9/v1/responses")
        self.assertEqual(seen["auth"], f"Bearer {'meta-key'}")
        self.assertEqual(seen["body"]["tools"][0]["type"], "namespace")

    def test_forward_meta_falls_back_to_incoming_bearer(self):
        cfg = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            api_key=None,
            meta_upstream="http://127.0.0.1:9/v1",
            meta_api_key=None,
        )
        seen = self._run_forward(
            "forward_meta",
            cfg,
            {"model": "muse-spark-1.3", "input": "hi"},
            incoming_auth=f"Bearer {'placeholder'}",
        )
        self.assertEqual(seen["auth"], f"Bearer {'placeholder'}")

    def test_forward_meta_missing_key_is_a_shim_error(self):
        cfg = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            api_key=None,
            meta_api_key=None,
        )
        with self.assertRaisesRegex(ShimError, "no Meta API key"):
            self._run_forward(
                "forward_meta", cfg, {"model": "muse-spark-1.3", "input": "hi"}
            )

    def test_forward_local_prefers_local_key_then_generic_key(self):
        body = {"model": "my-alias", "input": "hi"}
        local_first = config(
            local_upstream="http://127.0.0.1:9/v1",
            model_discovery=False,
            api_key="generic-key",
            local_api_key="local-key",
        )
        seen = self._run_forward("forward_local", local_first, body)
        self.assertEqual(seen["auth"], f"Bearer {'local-key'}")
        self.assertEqual(seen["url"], "http://127.0.0.1:9/v1/responses")
        generic_fallback = config(
            local_upstream="http://127.0.0.1:9/v1",
            model_discovery=False,
            api_key="generic-key",
            local_api_key=None,
        )
        seen = self._run_forward("forward_local", generic_fallback, body)
        self.assertEqual(seen["auth"], f"Bearer {'generic-key'}")

    def test_forward_generic_posts_to_configured_upstream(self):
        cfg = config(
            upstream="http://127.0.0.1:9/v1",
            model_discovery=False,
            api_key="generic-key",
        )
        seen = self._run_forward(
            "forward_generic",
            cfg,
            {"model": "anything", "input": "hi"},
            incoming_auth=f"Bearer {'placeholder'}",
        )
        self.assertEqual(seen["url"], "http://127.0.0.1:9/v1/responses")
        self.assertEqual(seen["auth"], f"Bearer {'generic-key'}")


@NETWORK_TEST
class GatewayProxyTests(unittest.TestCase):
    """End-to-end route checks. Need loopback binds (see ProxyTests)."""

    def test_auto_catalog_lists_all_routes(self):
        shim_config = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
            codex_models="gpt-one",
            claude_models="claude-one",
            meta_models="muse-spark-1.3",
            local_models="my-alias",
        )
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        shim_thread.start()
        # Menu-shape test: simulate installed providers so availability gating
        # does not hide the asserted rows.
        availability = patch(
            "muse_gateway._provider_is_available", return_value=True
        )
        availability.start()
        # Devin rows are out of scope here (covered by test_devin.py) and
        # would otherwise reflect machine credential state.
        devin_branch = patch(
            "muse_gateway._devin_catalog_entries", return_value=[]
        )
        devin_branch.start()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{shim.server_port}/muse-code/models",
                timeout=5,
            ) as response:
                payload = json.loads(response.read())
            self.assertEqual(
                [item["id"] for item in payload["data"]],
                [
                    "codex/gpt-one",
                    "claude/claude-one",
                    "meta/muse-spark-1.3",
                    "local/my-alias",
                ],
            )
            meta = payload["data"][2]["metadata"]["muse-code"]
            self.assertEqual(meta["name"], "Meta · muse-spark-1.3")
            self.assertEqual(meta["limit"]["context"], 1048576)
        finally:
            availability.stop()
            devin_branch.stop()
            shim.shutdown()
            shim.server_close()

    def test_meta_dispatch_preserves_namespaces(self):
        shim_config = config(
            provider="auto",
            upstream=None,
            model_discovery=False,
        )
        shim = MuseShimServer(("127.0.0.1", 0), shim_config)
        shim_thread = threading.Thread(target=shim.serve_forever, daemon=True)
        routed = []

        def record(provider):
            def forward(handler, body):
                routed.append((provider, body["model"], body.get("tools")))
                handler.send_json(200, {"ok": True})

            return forward

        with (
            patch("muse_gateway.MuseShimHandler.forward_meta", new=record("meta")),
            patch("muse_gateway.MuseShimHandler.forward_local", new=record("local")),
        ):
            shim_thread.start()
            try:
                for model in ("meta/muse-spark-1.3", "local/my-alias"):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{shim.server_port}/responses",
                        data=json.dumps(
                            {
                                "model": model,
                                "input": "hello",
                                "tools": [
                                    {
                                        "type": "namespace",
                                        "name": "muse",
                                        "tools": [
                                            {
                                                "type": "function",
                                                "name": "read_file",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
            finally:
                shim.shutdown()
                shim.server_close()
        self.assertEqual(
            [(provider, model) for provider, model, _ in routed],
            [("meta", "muse-spark-1.3"), ("local", "my-alias")],
        )
        self.assertEqual(routed[0][2][0]["type"], "namespace")
        self.assertEqual(routed[1][2][0]["type"], "function")


class GatewayLocalHistoryTests(unittest.TestCase):
    """Local route must flatten history call names alongside tool schemas.

    Regression test for the GLM stall: tools were flattened (``muse`` group
    -> bare ``search``) while history ``function_call`` names kept their
    ``muse.search`` qualification, so from turn 2 the upstream model saw
    calls to tools absent from its schema list and degenerated into
    native-tag text.
    """

    NAMESPACED_TOOLS = [
        {
            "type": "namespace",
            "name": "muse",
            "tools": [
                {"type": "function", "name": "search",
                 "parameters": {"type": "object"}},
                {"type": "function", "name": "read_file",
                 "parameters": {"type": "object"}},
            ],
        }
    ]

    def _turn_two_body(self):
        return {
            "model": "local/glm",
            "tools": [dict(group, tools=[dict(child) for child in group["tools"]])
                      for group in self.NAMESPACED_TOOLS],
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "call_1",
                 "name": "muse.search", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1",
                 "output": [{"type": "input_text", "text": "ok"}]},
            ],
        }

    def test_flattens_qualified_history_call_names(self):
        result = transform_request(self._turn_two_body(), config())
        calls = [item for item in result["input"]
                 if item.get("type") == "function_call"]
        self.assertEqual([call["name"] for call in calls], ["search"])
        self.assertEqual(
            sorted(tool["name"] for tool in result["tools"]),
            ["read_file", "search"],
        )

    def test_leaves_outputs_and_unknown_names_verbatim(self):
        body = self._turn_two_body()
        body["input"].extend([
            {"type": "function_call", "call_id": "call_2",
             "name": "search", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_3",
             "name": "otherns.search", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_4",
             "name": ["search"], "arguments": "{}"},
        ])
        result = transform_request(body, config())
        calls = [item for item in result["input"]
                 if item.get("type") == "function_call"]
        self.assertEqual(
            [(call["call_id"], call["name"]) for call in calls],
            [("call_1", "search"), ("call_2", "search"),
             ("call_3", "otherns.search"), ("call_4", ["search"])],
        )
        outputs = [item for item in result["input"]
                   if item.get("type") == "function_call_output"]
        self.assertEqual(outputs[0]["call_id"], "call_1")

    def test_does_not_mutate_original_body(self):
        body = self._turn_two_body()
        transform_request(body, config())
        calls = [item for item in body["input"]
                 if item.get("type") == "function_call"]
        self.assertEqual([call["name"] for call in calls], ["muse.search"])

    def test_native_route_leaves_history_verbatim(self):
        body = self._turn_two_body()
        result = transform_request(body, config(), native=True)
        calls = [item for item in result["input"]
                 if item.get("type") == "function_call"]
        self.assertEqual([call["name"] for call in calls], ["muse.search"])


class _ModelsResponse:
    """Minimal urlopen stand-in serving one JSON payload."""

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


class GatewayLocalDiscoveryTests(unittest.TestCase):
    """local/ entries come from the upstream /models list unless curated."""

    def _local_config(self, **overrides):
        values = {
            "provider": "auto",
            "local_upstream": "http://127.0.0.1:9/v1",
            "local_api_key": "test-local-key",
        }
        values.update(overrides)
        return config(**values)

    def test_discovers_aliases_from_upstream_models(self):
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["auth"] = request.get_header("Authorization")
            return _ModelsResponse(
                {"data": [{"id": "alpha"}, {"id": "beta"}, {"id": "  "}, {}]}
            )

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._local_config(), "local")
        self.assertEqual(seen["url"], "http://127.0.0.1:9/v1/models")
        # (split literal: the tooling redacts "Bearer <word>" in file writes)
        self.assertEqual(seen["auth"], "Bearer " + "test-local-key")
        self.assertEqual(
            [entry["model_id"] for entry in catalog],
            ["local/alpha", "local/beta"],
        )

    def test_explicit_list_curates_names_with_discovered_limits(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse(
                {"data": [
                    {"id": "x", "max_input_tokens": 999,
                     "max_output_tokens": 111},
                    {"id": "extra", "max_input_tokens": 999,
                     "max_output_tokens": 111},
                ]}
            )

        cfg = self._local_config(local_models="x,y")
        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(cfg, "local")
        limits = {
            entry["model_id"]: (entry["context_limit"], entry["output_limit"])
            for entry in catalog
        }
        self.assertEqual(
            limits, {"local/x": (999, 111), "local/y": (128000, 32768)}
        )

    def test_explicit_list_keeps_defaults_when_discovery_down(self):
        def fail_urlopen(request, timeout=None):
            raise urllib.error.URLError("down")

        cfg = self._local_config(local_models="x")
        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            catalog = provider_model_catalog(cfg, "local")
        self.assertEqual(
            [(entry["model_id"], entry["context_limit"])
             for entry in catalog],
            [("local/x", 128000)],
        )

    def test_unconfigured_upstream_makes_no_request(self):
        def fail_urlopen(request, timeout=None):
            raise AssertionError("discovery must not run when unconfigured")

        cfg = config(provider="auto")
        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            self.assertEqual(provider_model_catalog(cfg, "local"), [])

    def test_discovery_failure_degrades_to_empty(self):
        def fail_urlopen(request, timeout=None):
            raise urllib.error.URLError("down")

        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            self.assertEqual(
                provider_model_catalog(self._local_config(), "local"), []
            )

    def test_malformed_payload_degrades_to_empty(self):
        with patch(
            "muse_gateway.urllib.request.urlopen",
            new=lambda request, timeout=None: _ModelsResponse({"nope": True}),
        ):
            self.assertEqual(
                provider_model_catalog(self._local_config(), "local"), []
            )

    def test_no_model_discovery_flag_disables(self):
        def fail_urlopen(request, timeout=None):
            raise AssertionError("discovery must honor the flag")

        cfg = self._local_config(model_discovery=False)
        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            self.assertEqual(provider_model_catalog(cfg, "local"), [])

    def test_discovery_carries_upstream_limits(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse(
                {
                    "data": [
                        {"id": "big", "max_input_tokens": 524288,
                         "max_output_tokens": 262144},
                        {"id": "bare"},
                        {"id": "bogus", "max_input_tokens": "nope",
                         "max_output_tokens": -5},
                    ]
                }
            )

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._local_config(), "local")
        limits = {
            entry["model_id"]: (entry["context_limit"], entry["output_limit"])
            for entry in catalog
        }
        self.assertEqual(
            limits,
            {
                "local/big": (524288, 262144),
                "local/bare": (128000, 32768),
                "local/bogus": (128000, 32768),
            },
        )

    def test_bare_discovered_alias_routes_to_local(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse({"data": [{"id": "beta"}]})

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            cfg = self._local_config()
            self.assertEqual(cfg.resolve_model("beta"), ("local", "beta"))
            with self.assertRaises(ProviderError):
                cfg.resolve_model("not-a-model-anywhere")

    def test_discovery_reads_limit_aliases_and_image_evidence(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse(
                {"data": [
                    {"id": "m", "context_length": 500000,
                     "max_completion_tokens": 8000,
                     "capabilities": ["vision"]},
                ]}
            )

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._local_config(), "local")
        self.assertEqual(catalog[0]["context_limit"], 500000)
        self.assertEqual(catalog[0]["output_limit"], 8000)
        self.assertTrue(catalog[0]["limits_verified"])
        self.assertTrue(catalog[0]["supports_images"])
        self.assertEqual(
            catalog[0]["description"],
            "(ctx 500k) Local model through muse-gateway",
        )

    def test_ollama_tags_answer_when_models_is_empty(self):
        seen = []

        def fake_urlopen(request, timeout=None):
            seen.append(request.full_url)
            if request.full_url.endswith("/api/tags"):
                return _ModelsResponse({"models": [{"name": "qwen:32b"}]})
            return _ModelsResponse({"data": []})

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._local_config(), "local")
        self.assertEqual(
            seen,
            ["http://127.0.0.1:9/v1/models", "http://127.0.0.1:9/api/tags"],
        )
        self.assertEqual(
            [entry["model_id"] for entry in catalog], ["local/qwen:32b"]
        )
        self.assertFalse(catalog[0]["limits_verified"])

    def test_wire_modalities_follow_image_evidence(self):
        def fake_urlopen(request, timeout=None):
            return _ModelsResponse(
                {"data": [{"id": "plain"}, {"id": "seer", "supports_vision": True}]}
            )

        with patch("muse_gateway.urllib.request.urlopen", new=fake_urlopen):
            catalog = provider_model_catalog(self._local_config(), "local")
        wire = muse_catalog_response(catalog)["data"]
        modalities = {
            item["id"]: item["metadata"]["muse-code"]["modalities"]["input"]
            for item in wire
        }
        self.assertEqual(
            modalities, {"local/plain": ["text"], "local/seer": ["text", "image"]}
        )


if __name__ == "__main__":
    unittest.main()