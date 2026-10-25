"""Durable checks for the shipped Kiro configuration alternative."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


def _fallback_ids():
    namespace = {}
    example = ROOT / "errata" / "models.py.example"
    exec(compile(example.read_text(), str(example), "exec"), {}, namespace)
    return [row["modelId"] for row in namespace["FALLBACK_MODELS"]]


class KiroErrataTest(unittest.TestCase):
    def test_static_models_are_valid_and_in_the_packaged_catalog(self):
        ids = _fallback_ids()
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(name and not any(char.isspace() for char in name) for name in ids))
        # Mirror upstream's public alias/hiding behavior: auto is hidden and
        # auto-kiro is its exposed alias.
        exposed = (set(ids) - {"auto"}) | {"auto-kiro"}
        self.assertEqual(exposed, set(ids))
        self.assertIn("claude-opus-5.5", exposed)
        self.assertIn("gpt-5.6-sol", exposed)
        packaged = set(json.loads((ROOT / "providers" / "kiro.json").read_text())["models"])
        self.assertLessEqual(set(ids) - {"auto"}, packaged)

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
