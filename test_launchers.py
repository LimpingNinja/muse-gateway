"""Tests for the unified subscription launcher.

The platform wrapper (PowerShell on Windows, Bash elsewhere) forwards to
`python muse_shim.py launch`, so both entries are exercised against the same
clean-room startup contract. Pure parsing units are tested in-process; the
startup path runs as a subprocess so server shutdown is always observed.
"""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

from muse_shim import ProviderError, _split_launch_args, load_env_file


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _clean_environment(explicit_models=False):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MUSE_SHIM_") and key != "META_API_KEY"
    }
    environment.update({
        "MUSE_SHIM_NO_MODEL_DISCOVERY": "1",
        "MUSE_SHIM_CODEX_ACCESS_TOKEN": "launcher-test-token",
    })
    if explicit_models:
        environment["MUSE_SHIM_LOCAL_MODELS"] = "launcher-test"
    return environment


def _stage_checkout(directory):
    root = Path(__file__).resolve().parent
    work = Path(directory)
    launcher = "muse-gateway.ps1" if os.name == "nt" else "muse-gateway"
    for name in (launcher, "muse_shim.py", "muse_providers.py"):
        shutil.copyfile(root / name, work / name)
    return work, launcher


def _run_check_only(command, work, environment, port):
    result = subprocess.run(
        command, cwd=work, env=environment, capture_output=True,
        text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "skipping local-backend preflight" in output, output
    assert "catalog (" in output, output
    assert "models are disabled" not in output, output
    assert not (work / ".env").exists()
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    return output


class LauncherTests(unittest.TestCase):
    def test_keyless_startup_reports_preflight_without_claiming_routes_disabled(self):
        for explicit_models in (False, True):
            with self.subTest(explicit_models=explicit_models):
                with tempfile.TemporaryDirectory() as directory:
                    work, launcher = _stage_checkout(directory)
                    environment = _clean_environment(explicit_models)
                    port = _free_port()
                    if os.name == "nt":
                        command = [
                            "powershell", "-NoProfile", "-File",
                            str(work / launcher), "-CheckOnly", "-Port",
                            str(port),
                        ]
                    else:
                        command = [
                            "bash", str(work / launcher), "--check-only",
                            "--port", str(port),
                        ]
                    output = _run_check_only(command, work, environment, port)
                    self.assertEqual(
                        "local/launcher-test" in output, explicit_models
                    )

    def test_direct_launch_matches_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            work, _ = _stage_checkout(directory)
            environment = _clean_environment(explicit_models=True)
            port = _free_port()
            command = [
                sys.executable, str(work / "muse_shim.py"), "launch",
                "--env-file", str(work / ".env"),
                "--check-only", "--port", str(port),
            ]
            output = _run_check_only(command, work, environment, port)
            self.assertIn("local/launcher-test", output)


class LaunchArgTests(unittest.TestCase):
    def test_leading_options_split_from_muse_args(self):
        opts, muse_args = _split_launch_args(
            ["--port", "9000", "--check-only", "resume", "abc123"]
        )
        self.assertEqual(opts.port, 9000)
        self.assertTrue(opts.check_only)
        self.assertEqual(muse_args, ["resume", "abc123"])

    def test_equals_forms_and_short_port(self):
        opts, muse_args = _split_launch_args(
            ["-p9000", "--local-models=a,b"]
        )
        # Glued shorts are not launcher options: they pass through to Muse.
        self.assertIsNone(opts.port)
        self.assertEqual(muse_args, ["-p9000", "--local-models=a,b"])
        opts, muse_args = _split_launch_args(
            ["-p", "9000", "--port=9001", "--local-models=a,b"]
        )
        self.assertEqual(opts.port, 9001)
        self.assertEqual(opts.local_models, "a,b")
        self.assertEqual(muse_args, [])

    def test_double_dash_and_help_passthrough(self):
        opts, muse_args = _split_launch_args(["--", "--yolo"])
        self.assertEqual(muse_args, ["--yolo"])
        opts, muse_args = _split_launch_args(["--help"])
        self.assertFalse(opts.help)
        self.assertEqual(muse_args, ["--help"])
        opts, muse_args = _split_launch_args(["-h"])
        self.assertTrue(opts.help)
        self.assertEqual(muse_args, [])

    def test_missing_value_is_an_error(self):
        with self.assertRaises(ProviderError):
            _split_launch_args(["--port"])
        with self.assertRaises(ProviderError):
            _split_launch_args(["--port", "not-a-port"])

    def test_env_file_loading_respects_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'MUSE_SHIM_LOCAL_UPSTREAM="http://from-file:4000/v1"\n'
                "export MUSE_SHIM_LOCAL_API_KEY='file-key'\n"
                "# comment\n"
                "not a valid line\n"
                "MUSE_SHIM_LOCAL_MODELS=\n",
                encoding="utf-8",
            )
            saved = {
                key: os.environ.pop(key)
                for key in (
                    "MUSE_SHIM_LOCAL_UPSTREAM", "MUSE_SHIM_LOCAL_API_KEY",
                )
                if key in os.environ
            }
            os.environ["MUSE_SHIM_LOCAL_API_KEY"] = "env-key"
            try:
                loaded = load_env_file(str(path))
            finally:
                os.environ.pop("MUSE_SHIM_LOCAL_API_KEY", None)
                os.environ.pop("MUSE_SHIM_LOCAL_UPSTREAM", None)
                os.environ.update(saved)
            # Only the upstream is new; the key keeps its environment value
            # and the empty models list is skipped.
            self.assertEqual(loaded, 1)

    def test_missing_env_file_loads_nothing(self):
        self.assertEqual(load_env_file(str(Path("does-not-exist.env"))), 0)


if __name__ == "__main__":
    unittest.main()
