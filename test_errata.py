"""Durable checks for the shipped Kiro configuration alternative."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


class KiroErrataTest(unittest.TestCase):
    def test_static_models_have_valid_ids_and_existing_alias_shape(self):
        # Mirror upstream resolver's existing public alias/hiding behavior.
        namespace = {}
        example = ROOT / "errata" / "models.py.example"
        exec(compile(example.read_text(), str(example), "exec"), {}, namespace)
        rows = namespace["FALLBACK_MODELS"]
        ids = [row["modelId"] for row in rows]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(name and not any(char.isspace() for char in name) for name in ids))
        # Current upstream keeps auto hidden and adds auto-kiro as an alias.
        exposed = (set(ids) - {"auto"}) | {"auto-kiro"}
        self.assertEqual(exposed, set(ids))
        self.assertIn("claude-opus-5.5", exposed)
        self.assertIn("gpt-5.6-sol", exposed)

    def test_static_models_stay_in_lockstep_with_the_packaged_kiro_catalog(self):
        # P2-06: every example ID is a packaged Kiro model (no standing warning).
        import json
        namespace = {}
        example = ROOT / "errata" / "models.py.example"
        exec(compile(example.read_text(), str(example), "exec"), {}, namespace)
        ids = {row["modelId"] for row in namespace["FALLBACK_MODELS"]} - {"auto"}
        packaged = set(json.loads((ROOT / "providers" / "kiro.json").read_text())["models"])
        self.assertLessEqual(ids, packaged)

    def test_patch_does_not_modify_inference_search_or_auth_modules(self):
        patch = (ROOT / "errata" / "kiro-model-discovery.patch").read_text()
        files = [line.split()[2].removeprefix("a/") for line in patch.splitlines()
                 if line.startswith("diff --git ")]
        self.assertEqual(set(files), {
            "kiro/account_manager.py", "kiro/http_client.py",
            "tests/unit/test_account_manager.py", "tests/unit/test_http_client.py",
        })
        self.assertIn("AmazonCodeWhispererService.ListAvailableModels", patch)
        self.assertIn("management.", patch)
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", (ROOT / "errata" / "KIRO-LICENSE").read_text())


if __name__ == "__main__":
    unittest.main()
