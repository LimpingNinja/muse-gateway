"""Focused contracts for bundled provider data and its loader."""

import json
import unittest
from unittest.mock import patch

from muse_models import PROVIDERS, _load_catalog, _parse_catalog, provider_catalog, resolve_limits


class ProviderCatalogTests(unittest.TestCase):
    def test_all_provider_resources_load_and_preserve_route_specific_values(self):
        self.assertEqual(set(PROVIDERS), {"meta", "claude", "codex", "kiro", "devin", "local"})
        for provider in PROVIDERS:
            with self.subTest(provider=provider):
                self.assertEqual(_load_catalog(provider), provider_catalog(provider))
        self.assertEqual(provider_catalog("local"), {"fallbacks": [], "models": {}})
        self.assertEqual(provider_catalog("kiro")["fallbacks"], [])
        self.assertEqual(resolve_limits("codex/gpt-5.6-sol")["context_limit"], 1050000)
        self.assertEqual(resolve_limits("kiro/gpt-5.6-sol")["context_limit"], 1000000)
        wonder = resolve_limits("meta/muse-wonder-1.0", context=2000000)
        self.assertEqual((wonder["context_limit"], wonder["output_limit"]), (2000000, 128000))
        self.assertEqual((wonder["context_source"], wonder["output_source"]), ("live", "user-approved"))

    def test_invalid_catalogs_fail_with_resource_name(self):
        valid = {"fallbacks": ["model"], "models": {"model": {
            "context": 200000, "output": 128000,
            "sources": {"context": "documented", "output": "documented"},
        }}}
        cases = ["{", "[]", '{"models":{},"models":{},"fallbacks":[]}']
        for field, bad_value in (("context", True), ("output", 0), ("output", "128000"),
                                 ("sources", {"context": "live", "output": "documented"})):
            data = json.loads(json.dumps(valid))
            data["models"]["model"][field] = bad_value
            cases.append(json.dumps(data))
        cases.extend([json.dumps({"fallbacks": ["missing"], "models": {}}),
                      json.dumps({**valid, "fallbacks": ["model", "model"]})])
        for text in cases:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"providers/local\.json"):
                _parse_catalog("local", text)
        with patch("muse_models.files", side_effect=FileNotFoundError), self.assertRaisesRegex(
            ValueError, r"Cannot read provider catalog providers/local\.json"
        ):
            _load_catalog("local")

    def test_devin_fallbacks_use_catalog_capabilities_and_order(self):
        from muse_devin import seed_models

        models = seed_models()
        self.assertEqual([model.uid for model in models], ["swe-1-6-fast", "swe-1-6"])
        self.assertEqual([model.is_default for model in models], [False, True])
        for model in models:
            self.assertEqual((model.context_limit, model.output_limit), (200000, 128000))
            self.assertTrue(model.supports_tools and model.supports_parallel and model.reasoning)
            self.assertFalse(model.supports_images)
            self.assertTrue(model.unverified)


class CatalogSnapshotTests(unittest.TestCase):
    """P1-02: coherent snapshots, single-flight, honest retry TTL."""

    def _config(self):
        import argparse

        from muse_gateway import Config

        return Config(argparse.Namespace(
            provider="auto", model_discovery=True,
            log_file="/tmp/muse-gateway-catalog-test.log"))

    def test_menu_limit_always_matches_enforced_limit(self):
        config = self._config()
        config._publish_catalog(
            "codex", "codex",
            entries=[{"model_id": "codex/m", "output_limit": 1000}],
            raw={"m": {"model": "m", "context_limit": 10000, "output_limit": 1000}},
            stamped=0.0, generation=config._catalog_generation.get("codex", 0))
        self.assertEqual(config.model_output_limit("codex", "m"), 1000)
        config._publish_catalog(
            "codex", "codex",
            entries=[{"model_id": "codex/m", "output_limit": 50}],
            raw={"m": {"model": "m", "context_limit": 10000, "output_limit": 50}},
            stamped=0.0, generation=config._catalog_generation.get("codex", 0))
        self.assertEqual(config.model_output_limit("codex", "m"), 50)
        snapshot = config._catalog_snapshots["codex"]
        self.assertEqual(snapshot.entries[0]["output_limit"],
                         config.model_output_limit("codex", "m"))

    def test_stale_generation_cannot_publish(self):
        config = self._config()
        config._catalog_generation["codex"] = 5
        published = config._publish_catalog(
            "codex", "codex", entries=[{"model_id": "codex/old"}],
            raw={}, stamped=0.0, generation=4)
        self.assertFalse(published)
        self.assertNotIn("codex", config._catalog_snapshots)

    def test_single_flight_one_discovery_for_concurrent_cold_fetches(self):
        import threading
        import time

        from muse_gateway import provider_model_catalog

        config = self._config()
        calls = []

        def discover(_config, _provider, authorization=None):
            calls.append(1)
            time.sleep(0.25)
            return [{"model": "m", "context_limit": 10000, "output_limit": 1000}]

        results = []
        with patch("muse_gateway._discover_provider_models", side_effect=discover), \
                patch("muse_gateway._provider_is_available", return_value=True):
            threads = [threading.Thread(
                target=lambda: results.append(provider_model_catalog(config, "codex")))
                for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])

    def test_failed_discovery_retries_after_short_ttl(self):
        import time

        from muse_gateway import MODEL_CATALOG_TTL_SECONDS, provider_model_catalog

        config = self._config()
        with patch("muse_gateway._discover_provider_models", return_value=[]), \
                patch("muse_gateway._provider_is_available", return_value=True):
            provider_model_catalog(config, "codex")
        stamped, _ = config._model_catalog_cache["codex"]
        age = time.time() - stamped
        self.assertGreater(age, MODEL_CATALOG_TTL_SECONDS - 61)
        self.assertLess(age, MODEL_CATALOG_TTL_SECONDS)


class NoCacheRateLimitTests(unittest.TestCase):
    def test_two_no_cache_gets_a_second_apart_discover_once(self):
        # P3-02: unauthenticated no-cache refreshes are throttled to one per 10 s.
        import argparse
        import tempfile
        import threading
        import time
        import urllib.request
        from pathlib import Path
        from muse_gateway import Config, MuseShimServer

        calls = []

        def discover(config, provider, **kwargs):
            calls.append(provider)
            return []

        with tempfile.TemporaryDirectory() as directory:
            config = Config(argparse.Namespace(provider="local", upstream="http://127.0.0.1:9/v1",
                                               model_discovery=True,
                                               log_file=str(Path(directory) / "g.log")))
            server = MuseShimServer(("127.0.0.1", 0), config)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch("muse_gateway._discover_provider_models", side_effect=discover):
                    for _ in range(2):
                        request = urllib.request.Request(
                            f"http://127.0.0.1:{server.server_port}/muse-code/models",
                            headers={"Cache-Control": "no-cache"})
                        urllib.request.urlopen(request, timeout=5).close()
                        time.sleep(1)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)
        self.assertEqual(len(calls), 1)


class ResolveAndPresentationTests(unittest.TestCase):
    """P2-07: uniform prefix handling; Devin blurb and maintained label."""

    def test_empty_prefixed_names_are_model_missing_on_every_route(self):
        import argparse
        from muse_gateway import Config, ProviderError

        config = Config(argparse.Namespace(provider="auto", model_discovery=False))
        for value in ("meta/", "codex/ ", "claude:", "local/  ", "devin/", "kiro/ "):
            with self.subTest(value=value), self.assertRaises(ProviderError) as caught:
                config.resolve_model(value)
            self.assertEqual(caught.exception.code, "model_missing")
        self.assertEqual(config.resolve_model("codex/ gpt-5.4 "), ("codex", "gpt-5.4"))
        pinned = Config(argparse.Namespace(provider="codex", model_discovery=False))
        with self.assertRaises(ProviderError) as caught:
            pinned.resolve_model("kiro/haiku")
        self.assertEqual(caught.exception.code, "model_provider_mismatch")

    def test_devin_rows_use_catalog_label_and_devin_blurb(self):
        from muse_gateway import _catalog_entry

        row = _catalog_entry("devin", "swe-1-6", context_limit=200000, output_limit=64000,
                             is_default=True, label="SWE 1.6")
        self.assertEqual(row["display_label"], "Devin · SWE 1.6")
        self.assertIn("Devin model via Connect RPC", json.dumps(row))
        self.assertEqual(_catalog_entry("codex", "gpt-5.4", context_limit=1, output_limit=1,
                                        is_default=False)["display_label"], "Codex · gpt-5.4")


if __name__ == "__main__":
    unittest.main()
