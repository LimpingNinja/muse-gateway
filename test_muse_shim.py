import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import contextmanager, redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

from tests.contracts import consume
from tests.peers import responses_text

from muse_gateway import Config, MuseShimServer, ProviderError, muse_catalog_response, provider_model_catalog, parse_args, run_doctor


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
    def test_mocked_codex_discovery_replaces_bundled_fallbacks(self):
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
        login = SimpleNamespace(access_token="cli-login-token", refresh_token=None,
                                expires_at=None, auth_mode="oauth")
        with (
            patch("muse_gateway.discover_claude_credentials", return_value=login),
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

    def test_unusable_credentials_hide_codex_and_claude_models(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        dead_oauth = SimpleNamespace(access_token="dead", refresh_token=None,
                                     expires_at=time.time() - 3600, auth_mode="oauth")
        with (
            patch("muse_gateway.discover_codex_credentials", return_value=dead_oauth),
            patch("muse_gateway.discover_claude_credentials", return_value=None),
            patch("muse_gateway.shutil.which", return_value="/bin/claude"),
        ):
            self.assertEqual(provider_model_catalog(auto, "codex"), [])
            self.assertEqual(provider_model_catalog(auto, "claude"), [])

    def test_api_key_and_refreshable_oauth_still_list(self):
        auto = config(provider="auto", upstream=None, model_discovery=False)
        api_key = SimpleNamespace(access_token="sk-ant-api-live", refresh_token=None,
                                  expires_at=None, auth_mode="api_key")
        stale_but_refreshable = SimpleNamespace(
            access_token="stale", refresh_token="refresh-live",
            expires_at=time.time() - 10, auth_mode="oauth")
        with (
            patch("muse_gateway.discover_codex_credentials",
                  return_value=stale_but_refreshable),
            patch("muse_gateway.discover_claude_credentials", return_value=api_key),
            patch("muse_gateway.shutil.which", return_value="/bin/claude"),
        ):
            self.assertTrue(provider_model_catalog(auto, "codex"))
            self.assertTrue(provider_model_catalog(auto, "claude"))


class FakeUpstreamHandler(BaseHTTPRequestHandler):
    wire = responses_text("FORWARDED")
    request_body = None
    authorization = None
    path_seen = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        type(self).path_seen = self.path
        type(self).authorization = self.headers.get("Authorization")
        type(self).request_body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
        )
        payload = self.wire
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ProxyTests(unittest.TestCase):
    def test_public_routes_preserve_history_namespaces_and_key_precedence(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        try:
            endpoint = f"http://127.0.0.1:{upstream.server_port}/v1"
            for route in ("generic", "meta", "local"):
                with self.subTest(route=route):
                    cfg = config(provider=route, upstream=endpoint, meta_upstream=endpoint, local_upstream=endpoint,
                                 model_discovery=False, meta_api_key="meta-key", local_api_key="local-key",
                                 model="pinned-model" if route == "generic" else None,
                                 portable=True, max_output_tokens=16000)
                    body = {"model": "requested-model", "stream": True, "max_output_tokens": 128000,
                            "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"],
                            "prompt_cache_key": "cache-sentinel",
                            "tools": [{"type": "namespace", "name": "muse", "tools": [{
                                "type": "function", "name": "read_file", "parameters": {"type": "object"}}]}],
                            "input": [{"role": "user", "content": "Read it"},
                                      {"type": "function_call", "call_id": "call_original", "name": "muse.read_file", "arguments": '{"path":"C:/fixture.txt"}'},
                                      {"type": "function_call_output", "call_id": "call_original", "output": '1|C:/fixture.txt\n{"exit_code":7,"is_error":true}\n' + "x" * 10000}]}
                    original = json.loads(json.dumps(body))
                    shim = MuseShimServer(("127.0.0.1", 0), cfg)
                    thread = threading.Thread(target=shim.serve_forever, daemon=True)
                    thread.start()
                    try:
                        keys = {"generic": ["test-key"], "meta": ["meta-key", "incoming-key"],
                                "local": ["local-key", "test-key"]}[route]
                        for index, expected_key in enumerate(keys):
                            if index:
                                setattr(cfg, route + "_api_key", None)
                            req = urllib.request.Request(f"http://127.0.0.1:{shim.server_port}/responses",
                                data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer incoming-key"})
                            with urllib.request.urlopen(req, timeout=5) as response:
                                wire = response.read()
                            self.assertEqual(wire, FakeUpstreamHandler.wire)
                            self.assertEqual(consume(wire)["output"][0]["content"][0]["text"], "FORWARDED")
                            self.assertEqual(FakeUpstreamHandler.path_seen, "/v1/responses")
                            self.assertEqual(FakeUpstreamHandler.authorization, "Bearer " + expected_key)
                            sent = FakeUpstreamHandler.request_body
                            self.assertEqual(sent["model"], "pinned-model" if route == "generic" else "requested-model")
                            self.assertEqual(sent["max_output_tokens"], 16000)
                            if route == "meta":
                                self.assertEqual(sent["tools"], original["tools"])
                                self.assertEqual(sent["input"], original["input"])
                                for key in ("reasoning", "include", "prompt_cache_key"):
                                    self.assertEqual(sent[key], original[key])
                            else:
                                self.assertEqual(sent["tools"], original["tools"][0]["tools"])
                                self.assertEqual(sent["input"][1], {**original["input"][1], "name": "read_file"})
                                self.assertEqual(sent["input"][2], original["input"][2])
                                self.assertFalse(set(sent) & {"reasoning", "include", "prompt_cache_key"})
                            self.assertEqual(body, original)
                    finally:
                        shim.shutdown()
                        shim.server_close()
                        thread.join(timeout=5)
        finally:
            upstream.shutdown()
            upstream.server_close()
            worker.join(timeout=5)

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
            with raised.exception as response:
                error = json.load(response)
            self.assertIn("no upstream API key", error["error"]["message"])
        finally:
            shim.shutdown()
            shim.server_close()


class GatewayRoutingTests(unittest.TestCase):
    """Gateway fork: meta/ + local/ routes with native Meta passthrough."""

    def test_kiro_prefixes_do_not_steal_native_bare_model_names(self):
        auto = config(
            provider="auto", upstream=None, model_discovery=False,
            kiro_api_key="synthetic-proxy-key", kiro_models="claude-haiku-4.5,auto",
        )
        for name in ("kiro/claude-haiku-4.5", "kiro:claude-haiku-4.5"):
            self.assertEqual(auto.resolve_model(name), ("kiro", "claude-haiku-4.5"))
        self.assertEqual(auto.resolve_model("kiro/auto"), ("kiro", "auto"))
        self.assertEqual(auto.resolve_model("claude-haiku-4.5"), ("claude", "claude-haiku-4.5"))
        with self.assertRaisesRegex(ProviderError, "cannot infer"):
            auto.resolve_model("auto")
        with self.assertRaisesRegex(ProviderError, "belongs to Kiro"):
            config(provider="claude", upstream=None).resolve_model("kiro/auto")
        with self.assertRaisesRegex(ProviderError, "empty"):
            auto.resolve_model("kiro/")
        # An explicitly selected single-provider mode intentionally accepts
        # bare upstream IDs, just like Codex/Claude/local modes.
        single = config(provider="kiro", upstream=None)
        self.assertEqual(single.resolve_model("auto"), ("kiro", "auto"))
        self.assertEqual(single.resolve_model("claude-haiku-4.5"), ("kiro", "claude-haiku-4.5"))

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

    def test_curated_catalog_preserves_positive_limits_and_requires_image_evidence(self):
        payload = {"data": [{"id": "x", "max_input_tokens": 999, "max_output_tokens": 111, "supports_vision": True},
                             {"id": "bad", "max_input_tokens": "nope", "max_output_tokens": -5},
                             {"id": "unlisted", "max_input_tokens": 1, "max_output_tokens": 1}]}
        with patch("muse_gateway.urllib.request.urlopen", return_value=_ModelsResponse(payload)):
            catalog = provider_model_catalog(self._local_config(local_models="x,bad,missing"), "local")
        self.assertEqual({entry["model_id"]: (entry["context_limit"], entry["output_limit"]) for entry in catalog},
                         {"local/x": (999, 111), "local/bad": (128000, 32768), "local/missing": (128000, 32768)})
        wire = muse_catalog_response(catalog)["data"]
        self.assertEqual({entry["id"]: entry["metadata"]["muse-code"]["modalities"]["input"] for entry in wire},
                         {"local/x": ["text", "image"], "local/bad": ["text"], "local/missing": ["text"]})
        self.assertTrue(catalog[0]["limits_verified"])
        self.assertTrue(all(not entry["limits_verified"] for entry in catalog[1:]))

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

    def test_no_model_discovery_flag_disables(self):
        def fail_urlopen(request, timeout=None):
            raise AssertionError("discovery must honor the flag")

        cfg = self._local_config(model_discovery=False)
        with patch("muse_gateway.urllib.request.urlopen", new=fail_urlopen):
            self.assertEqual(provider_model_catalog(cfg, "local"), [])

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


class KiroConfigurationTests(unittest.TestCase):
    def test_env_legacy_and_cli_precedence_and_single_provider_upstream(self):
        with patch.dict(os.environ, {
            "MUSE_SHIM_KIRO_API_KEY": "legacy-proxy-key",
            "MUSE_SHIM_KIRO_MODELS": "legacy-model",
            "MUSE_GATEWAY_KIRO_UPSTREAM": "http://127.0.0.1:1/v1/messages",
            "MUSE_GATEWAY_KIRO_API_KEY": "current-proxy-key",
            "MUSE_GATEWAY_KIRO_CONTEXT_LIMIT": "32768",
            "MUSE_GATEWAY_KIRO_OUTPUT_LIMIT": "4096",
        }, clear=True):
            cfg = Config(parse_args(["kiro", "--kiro-models", "cli-model", "--no-model-discovery"]))
            self.assertEqual(cfg.kiro_api_key, "current-proxy-key")
            self.assertEqual(cfg.kiro_models, ["cli-model"])
            self.assertEqual((cfg.kiro_context_limit, cfg.kiro_output_limit), (32768, 4096))
            self.assertEqual(cfg.provider_url("kiro"), "http://127.0.0.1:1/v1/messages")
            self.assertIn("current-proxy-key", cfg.secret_values())
            override = Config(parse_args([
                "kiro", "--upstream", "http://127.0.0.1:2", "--no-model-discovery",
            ]))
            self.assertEqual(override.provider_url(), "http://127.0.0.1:2/v1/messages")
            del os.environ["MUSE_GATEWAY_KIRO_API_KEY"]
            self.assertEqual(Config(parse_args(["auto"])).kiro_api_key, "legacy-proxy-key")

    def test_invalid_kiro_settings_fail_before_serving(self):
        for flags in (
            ["--kiro-context-limit", "0"], ["--kiro-output-limit", "-1"],
            ["--kiro-upstream", "http://remote.invalid/v1"],
            ["--kiro-upstream", "https://user:password@remote.invalid/v1"],
            ["--kiro-api-key", "key\r\ninjected"],
        ):
            with self.subTest(flags=flags[0]), patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["auto", *flags])


class _KiroRoutingHandler(BaseHTTPRequestHandler):
    """Independent Messages JSON fixture; no production conversion in the peer."""
    def log_message(self, *args):
        pass

    def _reply(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._reply_bytes(raw, status)

    def _reply_bytes(self, raw, status=200):
        if self.server.chunked_trickle:
            self._reply_chunked_trickle(status)
            return
        self.send_response(status)
        self.send_header("Content-Type", self.server.content_type)
        if not self.server.hold_body_open:
            self.send_header("Content-Length", str(len(raw) + self.server.declared_extra))
        self.end_headers()
        try:
            self.wfile.write(raw)
            self.wfile.flush()
            self.server.body_written.set()
            if self.server.observe_client_close:
                self.connection.settimeout(3)
                try:
                    closed = not self.connection.recv(1)
                except ConnectionResetError:
                    closed = True  # Closing with unread error bytes may send RST, not FIN.
                if closed:
                    self.server.client_closed.set()
            if self.server.hold_body_open:
                self.server.release_body.wait(timeout=5)
        except (ConnectionError, OSError):
            pass  # Rejected/oversized bodies deliberately close this peer early.
        self.close_connection = True

    def _reply_chunked_trickle(self, status):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if self.server.chunked_trickle == "size":
                # read1 must finish the size/extension line before any data
                # is returned. Each fragment renews an inactivity timeout.
                initial, ending = b"1;", b"\r\nx\r\n"
            else:
                # Zero-size chunk handling internally reads all trailer lines.
                initial, ending = b"0\r\nX-Trace: ", b"\r\n\r\n"
            self.wfile.write(initial)
            self.wfile.flush()
            for fragment in [b"x"] * 22 + [ending]:
                if self.server.release_body.wait(timeout=0.05):
                    return
                self.wfile.write(fragment)
                self.wfile.flush()
            self.server.chunked_framing_finished.set()
            self.server.release_body.wait(timeout=5)
        except (ConnectionError, OSError):
            self.server.client_closed.set()
        finally:
            self.close_connection = True

    def do_GET(self):
        self.server.observed.append((self.path, self.headers.get("Authorization"), None))
        if self.server.catalog_bytes is not None:
            self._reply_bytes(self.server.catalog_bytes)
        elif self.server.catalog_status != 200:
            self._reply({"error": {"message": "synthetic unavailable"}}, self.server.catalog_status)
        else:
            self._reply({"data": [{"id": "claude-haiku-4.5"}, {"id": "auto"}]})

    def do_POST(self):
        body = json.load(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
        self.server.observed.append((self.path, self.headers.get("Authorization"), body))
        if self.server.response_bytes is not None:
            self._reply_bytes(self.server.response_bytes, self.server.response_status)
            return
        self._reply({
            "type": "message", "id": "msg_kiro_routing", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": "Hello \U0001f332"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 3},
        })


class KiroPublicRoutingTests(unittest.TestCase):
    @contextmanager
    def _servers(self, *, capture=False, **overrides):
        peer = ThreadingHTTPServer(("127.0.0.1", 0), _KiroRoutingHandler)
        peer.observed = []
        peer.catalog_status = 200
        peer.catalog_bytes = None
        peer.response_bytes = None
        peer.response_status = 200
        peer.declared_extra = 0
        peer.content_type = "application/json"
        peer.hold_body_open = False
        peer.release_body = threading.Event()
        peer.body_written = threading.Event()
        peer.observe_client_close = False
        peer.client_closed = threading.Event()
        peer.chunked_trickle = None
        peer.chunked_framing_finished = threading.Event()
        peer_thread = threading.Thread(target=peer.serve_forever, daemon=True)
        peer_thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                values = dict(
                    provider="auto", upstream=None, model_discovery=False,
                    kiro_upstream=f"http://127.0.0.1:{peer.server_port}/v1",
                    log_file=str(Path(directory) / "gateway.log"),
                    capture_dir=str(Path(directory) / "captures") if capture else None,
                )
                values.update(overrides)
                cfg = config(**values)
                server = MuseShimServer(("127.0.0.1", 0), cfg)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    yield cfg, f"http://127.0.0.1:{server.server_port}", peer
                finally:
                    peer.release_body.set()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)
        finally:
            peer.shutdown()
            peer.server_close()
            peer_thread.join(timeout=5)

    def _manifest(self, cfg):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            manifests = list(Path(cfg.capture_dir).glob("gateway-*/*/manifest.json"))
            if manifests:
                try:
                    return manifests[0], json.loads(manifests[0].read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            time.sleep(0.01)
        self.fail("Request capture was not finalized")

    def test_deeply_nested_optional_catalog_preserves_public_auto_menu_and_short_retry(self):
        with self._servers(kiro_api_key="synthetic-proxy-key", model_discovery=True) as (cfg, base, peer):
            peer.catalog_bytes = b"[" * 20000 + b"]" * 20000
            # Only credential availability is replaced. No test may inspect
            # private native stores or query an unrelated remote provider.
            with patch("muse_gateway.discover_codex_credentials", return_value=None), patch(
                "muse_gateway.discover_claude_credentials", return_value=None,
            ), patch("muse_gateway._devin_provider_ready", return_value=False), patch(
                "muse_gateway.shutil.which", return_value=None,
            ):
                for _ in range(2):
                    with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
                        ids = [row["id"] for row in json.load(response)["data"]]
                    self.assertIn("meta/muse-spark-1.3", ids, "KIRO_CATALOG_ISOLATION")
                    self.assertFalse(any(name.startswith("kiro/") for name in ids))
                self.assertEqual(len(peer.observed), 1, "Failed catalogs should have a short cache")
                peer.catalog_bytes = None
                from muse_gateway import MODEL_CATALOG_RETRY_SECONDS
                with patch("muse_gateway.time.time", return_value=time.time() + MODEL_CATALOG_RETRY_SECONDS + 1):
                    with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
                        ids = [row["id"] for row in json.load(response)["data"]]
                self.assertIn("kiro/auto", ids)
                self.assertEqual(len(peer.observed), 2)

    def test_pre_forward_rejection_marks_capture_failed(self):
        with self._servers(capture=True, kiro_api_key="synthetic-proxy-key") as (cfg, base, peer):
            req = urllib.request.Request(base + "/responses", data=json.dumps({
                "model": "kiro/auto", "input": "hello", "stream": True,
                "tools": [{"type": "namespace", "name": group, "tools": [{
                    "type": "function", "name": "duplicate", "parameters": {"type": "object"},
                }]} for group in ("a", "b")],
            }).encode(), headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=5)
            with caught.exception as error:
                self.assertEqual(error.code, 400)
                self.assertIn("duplicate", json.load(error)["error"]["message"])
            _, manifest = self._manifest(cfg)
            self.assertTrue(manifest["outcome"]["stream_failed"], "KIRO_FAILURE_OUTCOME: pre-forward rejection is not success")
            self.assertEqual(peer.observed, [])

    def test_wrong_mime_capture_is_bounded_scrubbed_and_does_not_wait_for_open_body(self):
        key = "arbitrary-proxy-key-sentinel"
        with self._servers(capture=True, kiro_api_key=key) as (cfg, base, peer):
            peer.content_type = "text/plain"
            peer.response_bytes = b"synthetic rejection " + key.encode() + b"!" * (200 * 1024)
            peer.hold_body_open = True
            req = urllib.request.Request(base + "/responses", data=json.dumps({
                "model": "kiro/auto", "input": "hello", "stream": True,
            }).encode(), headers={"Content-Type": "application/json"})
            started = time.monotonic()
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=3)
            with caught.exception as error:
                self.assertEqual(error.code, 502)
                self.assertEqual(json.load(error)["error"]["type"], "kiro_invalid_response")
            self.assertLess(time.monotonic() - started, 2, "KIRO_REJECT_CAPTURE_LIVENESS")
            self.assertFalse(peer.release_body.is_set(), "The peer must still be held open")
            path, manifest = self._manifest(cfg)
            self.assertIn("upstream.body", manifest["streams"], "KIRO_REJECT_EVIDENCE: retain a bounded body prefix")
            observed = manifest["streams"]["upstream.body"]["observed_bytes"]
            self.assertGreater(observed, 0)
            self.assertLessEqual(observed, 65537)
            self.assertIn("kiro_rejected_body_incomplete", manifest["capture_errors"])
            captured = (path.parent / "upstream.body").read_bytes()
            self.assertIn(b"synthetic rejection", captured)
            self.assertNotIn(key.encode(), captured)
            self.assertNotIn(key, Path(cfg.log_file).read_text(encoding="utf-8"))
            self.assertTrue(manifest["outcome"]["stream_failed"])

    def test_http_401_flushed_prefix_does_not_wait_for_eof_and_closes_upstream(self):
        key = 'http"error\\proxy-key'
        wire = json.dumps({"error": {"type": "authentication_error", "code": "unauthorized",
                                     "message": "Synthetic rejection " + key}}).encode()
        for capture in (False, True):
            with self.subTest(capture=capture), self._servers(capture=capture, kiro_api_key=key, timeout=1.2) as (cfg, base, peer):
                peer.response_status, peer.response_bytes, peer.hold_body_open = 401, wire, True
                peer.observe_client_close = True
                request = urllib.request.Request(base + "/responses", data=json.dumps({
                    "model": "kiro/auto", "input": "hello", "stream": True,
                }).encode(), headers={"Content-Type": "application/json"})
                started = time.monotonic()
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=3)
                with caught.exception as error:
                    body = json.load(error)
                    status = error.code
                self.assertLess(time.monotonic() - started, 0.9, "KIRO_HTTP_ERROR_LIVENESS: known errors must not await body EOF")
                self.assertEqual(status, 401, "KIRO_HTTP_ERROR_STATUS")
                self.assertEqual(body["error"]["type"], "kiro_upstream_error")
                self.assertTrue(peer.body_written.is_set())
                self.assertTrue(peer.client_closed.wait(timeout=1), "KIRO_HTTP_ERROR_CLOSE: release the upstream even while its body is open")
                self.assertFalse(peer.release_body.is_set())
                if capture:
                    path, manifest = self._manifest(cfg)
                    self.assertTrue(manifest["outcome"]["stream_failed"])
                    self.assertEqual(manifest["outcome"]["status"], 401)
                    self.assertIn("kiro_http_error_body_incomplete", manifest["capture_errors"])
                    raw = (path.parent / "upstream.error.body").read_bytes()
                    decoded = json.loads(raw)
                    self.assertNotIn(key, decoded["error"]["message"])
                    self.assertIn("[REDACTED]", decoded["error"]["message"])
                    self.assertEqual(manifest["streams"]["upstream.error.body"]["observed_bytes"], len(wire))
                    self.assertNotIn(key, Path(cfg.log_file).read_text(encoding="utf-8"))

    def test_chunked_error_size_line_and_trailers_are_omitted_without_waiting(self):
        for framing, status in (("size", 401), ("trailers", 503)):
            for capture in (False, True):
                with self.subTest(framing=framing, status=status, capture=capture), self._servers(
                    capture=capture, kiro_api_key="synthetic-proxy-key", timeout=3,
                ) as (cfg, base, peer):
                    peer.response_status, peer.response_bytes, peer.chunked_trickle = status, b"", framing
                    request = urllib.request.Request(base + "/responses", data=json.dumps({
                        "model": "kiro/auto", "input": "hello", "stream": True,
                    }).encode(), headers={"Content-Type": "application/json"})
                    started = time.monotonic()
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request, timeout=3)
                    with caught.exception as error:
                        result = json.load(error)
                        self.assertEqual(error.code, status, "KIRO_HTTP_ERROR_STATUS")
                        self.assertEqual(result["error"]["type"], "kiro_upstream_error")
                    self.assertLess(time.monotonic() - started, 0.9,
                                    "KIRO_CHUNKED_ERROR_LIVENESS: inactivity timeout is not an absolute framing deadline")
                    self.assertFalse(peer.chunked_framing_finished.is_set(),
                                     "Known HTTP status must arrive while the peer's framing is unfinished")
                    self.assertTrue(peer.client_closed.wait(timeout=1), "KIRO_HTTP_ERROR_CLOSE")
                    self.assertFalse(peer.release_body.is_set())
                    if capture:
                        path, manifest = self._manifest(cfg)
                        self.assertEqual(manifest["outcome"]["status"], status)
                        self.assertTrue(manifest["outcome"]["stream_failed"])
                        headers = json.loads((path.parent / "upstream.error.headers.json").read_text(encoding="utf-8"))
                        self.assertEqual(headers["status"], status)
                        self.assertEqual(headers["headers"]["Transfer-Encoding"], "chunked")
                        self.assertIn("kiro_http_error_body_incomplete", manifest["capture_errors"])
                        self.assertNotIn("upstream.error.body", manifest["streams"])
                        timeline = [json.loads(line) for line in (path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
                        omitted = next(row for row in timeline if row["event"] == "upstream.error.body.omitted")
                        self.assertEqual(omitted["reason"], "chunked_framing_has_no_absolute_read_bound")
                        self.assertFalse(omitted["complete"])

    def test_http_error_silent_body_timeout_preserves_status_and_missing_evidence(self):
        for status in (401, 429, 503):
            with self.subTest(status=status), self._servers(capture=True, kiro_api_key="synthetic-proxy-key", timeout=0.05) as (cfg, base, peer):
                peer.response_status, peer.response_bytes, peer.hold_body_open = status, b"", True
                peer.observe_client_close = True
                request = urllib.request.Request(base + "/responses", data=json.dumps({
                    "model": "kiro/auto", "input": "hello", "stream": True,
                }).encode(), headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(request, timeout=3)
                with caught.exception as error:
                    self.assertEqual(error.code, status, "KIRO_HTTP_ERROR_STATUS: body-read failure cannot replace known HTTP status")
                    self.assertEqual(json.load(error)["error"]["type"], "kiro_upstream_error")
                self.assertTrue(peer.client_closed.wait(timeout=1), "KIRO_HTTP_ERROR_CLOSE")
                path, manifest = self._manifest(cfg)
                self.assertTrue(manifest["outcome"]["stream_failed"])
                self.assertIn("kiro_http_error_body_incomplete", manifest["capture_errors"])
                timeline = [json.loads(line) for line in (path.parent / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]
                omitted = next(row for row in timeline if row["event"] == "upstream.error.body.omitted")
                self.assertFalse(omitted["complete"])
                self.assertEqual(omitted["reason"], "TimeoutError")

    def test_error_capture_trusts_only_a_fulfilled_content_length(self):
        wire = b'{"error":{"code":"unauthorized"}}'
        for extra in (0, 17):
            with self.subTest(declared_extra=extra), self._servers(capture=True, kiro_api_key="synthetic-proxy-key") as (cfg, base, peer):
                peer.response_status, peer.response_bytes, peer.declared_extra = 401, wire, extra
                req = urllib.request.Request(base + "/responses", data=json.dumps({"model": "kiro/auto", "input": "hi", "stream": True}).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=3)
                with caught.exception as error:
                    self.assertEqual(error.code, 401)
                    json.load(error)
                path, manifest = self._manifest(cfg)
                timeline = [json.loads(line) for line in (path.parent / "timeline.jsonl").read_text().splitlines()]
                evidence = next(row for row in timeline if row["event"] == "upstream.error.body.rejected")
                self.assertEqual(evidence["complete"], not extra, "Socket EOF alone cannot fulfill Content-Length")
                self.assertEqual("kiro_http_error_body_incomplete" in manifest["capture_errors"], bool(extra))
                self.assertEqual((path.parent / "upstream.error.body").read_bytes(), wire)

    def test_nonstandard_json_constants_never_become_completed_responses(self):
        with self._servers(kiro_api_key="synthetic-proxy-key") as (_, base, peer):
            for constant in ("NaN", "Infinity", "-Infinity"):
                peer.response_bytes = (
                    '{"type":"message","id":"msg_constant","role":"assistant",'
                    '"content":[{"type":"text","text":"apparently valid"}],'
                    '"stop_reason":"end_turn","unrecognized":' + constant + "}"
                ).encode()
                with self.subTest(constant=constant):
                    req = urllib.request.Request(base + "/responses", data=json.dumps({
                        "model": "kiro/auto", "input": "hello", "stream": False,
                    }).encode(), headers={"Content-Type": "application/json"})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req, timeout=5)
                    with caught.exception as error:
                        self.assertEqual(error.code, 502)
                        self.assertEqual(json.load(error)["error"]["type"], "kiro_invalid_response")

    def test_missing_kiro_key_never_discovers_or_falls_back_to_muse_or_generic_key(self):
        with self._servers(kiro_models="auto") as (cfg, base, peer):
            cfg.model_discovery = True
            self.assertEqual(provider_model_catalog(cfg, "kiro"), [])
            req = urllib.request.Request(base + "/responses", data=json.dumps({
                "model": "kiro/auto", "input": "hello", "stream": False,
            }).encode(), headers={"Authorization": "Bearer incoming-Muse-key", "Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(req, timeout=5)
            with caught.exception as error:
                result = json.load(error)
                self.assertEqual(error.code, 400)
            self.assertEqual(result["error"]["type"], "kiro_not_configured")
            self.assertEqual(peer.observed, [])

    def test_json_errors_overflow_and_http_truncation_are_readable_failures_with_honest_capture(self):
        complete_json = json.dumps({
            "type": "message", "id": "msg_truncated", "role": "assistant",
            "content": [{"type": "text", "text": "apparently valid"}], "stop_reason": "end_turn",
        }).encode()
        for label, raw, limit, extra in (
            ("invalid JSON", b"not-json", 1024, 0),
            ("invalid Messages shape", b"{}", 1024, 0),
            ("oversized JSON", b"x" * 2048, 1024, 0),
            ("truncated HTTP", complete_json, 1024, 17),
        ):
            with self.subTest(case=label), self._servers(capture=True, kiro_api_key="synthetic-proxy-key") as (cfg, base, peer):
                peer.response_bytes, peer.declared_extra = raw, extra
                req = urllib.request.Request(base + "/responses", data=json.dumps({
                    "model": "kiro/auto", "input": "hello", "stream": False,
                }).encode(), headers={"Content-Type": "application/json"})
                with patch("muse_gateway.KIRO_JSON_RESPONSE_LIMIT", limit):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req, timeout=5)
                    with caught.exception as error:
                        result = json.load(error)
                        self.assertEqual(error.code, 502)
                self.assertEqual(result["error"]["type"], "kiro_invalid_response")
                deadline = time.monotonic() + 2
                manifests = []
                manifest = None
                while time.monotonic() < deadline and manifest is None:
                    manifests = list(Path(cfg.capture_dir).glob("gateway-*/*/manifest.json"))
                    if manifests:
                        try:
                            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            pass  # The handler may still be closing the manifest.
                    if manifest is None:
                        time.sleep(0.01)
                self.assertEqual(len(manifests), 1)
                self.assertIsNotNone(manifest)
                self.assertTrue(manifest["outcome"]["stream_failed"])
                self.assertEqual(manifest["outcome"]["status"], 502)
                if label == "oversized JSON":
                    self.assertEqual(manifest["streams"]["upstream.body"]["observed_bytes"], limit + 1)
                    self.assertIn("kiro_response_body_limit", manifest["capture_errors"])
                if label == "truncated HTTP":
                    self.assertIn("kiro_response_body_incomplete", manifest["capture_errors"])

    def test_authenticated_catalog_curates_names_and_never_invents_verified_capabilities(self):
        with self._servers(provider="kiro", kiro_api_key="synthetic-proxy-key", kiro_models="auto",
                           model="kiro/auto", max_output_tokens=123) as (cfg, base, peer):
            cfg.model_discovery = True
            rows = provider_model_catalog(cfg, "kiro")
            self.assertEqual([row["model_id"] for row in rows], ["kiro/auto"])
            self.assertEqual([(path, auth) for path, auth, _ in peer.observed], [
                ("/v1/models", "Bearer synthetic-proxy-key"),
            ])
            self.assertEqual((rows[0]["context_limit"], rows[0]["output_limit"]), (65536, 123))
            with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
                row = json.load(response)["data"][0]
            self.assertEqual(row["metadata"]["muse-code"]["limit"], {"context": 65536, "output": 123})
            self.assertIn("unverified", row["metadata"]["muse-code"]["description"])
            self.assertFalse(rows[0]["limits_verified"])
            self.assertFalse(rows[0]["supports_images"])
            self.assertFalse(rows[0].get("reasoning_efforts"))
            out = io.StringIO()
            cfg.doctor_provider = "kiro"
            self.assertEqual(run_doctor(cfg, output=out), 0)
            self.assertIn("authenticated catalog reachable", out.getvalue())
            self.assertIn("inference not validated", out.getvalue())
            self.assertNotIn("synthetic-proxy-key", out.getvalue())
            self.assertEqual(len(peer.observed), 1, "doctor must use cached catalog, not inference")


if __name__ == "__main__":
    unittest.main()
