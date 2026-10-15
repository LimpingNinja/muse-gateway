"""Tests for the unified subscription launcher.

The platform wrapper (PowerShell on Windows, Bash elsewhere) forwards to
`python muse_gateway.py launch`, so both entries are exercised against the same
clean-room startup contract. Pure parsing units are tested in-process; the
startup path runs as a subprocess so server shutdown is always observed.
"""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from unittest.mock import patch

from muse_gateway import Config, MuseShimServer, _split_launch_args, launch_main, load_env_file, main, parse_args, run_muse, _port_is_busy


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _clean_environment(explicit_models=False):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MUSE_GATEWAY_", "MUSE_SHIM_"))
        and key != "META_API_KEY"
    }
    # Staged subprocesses must resolve the same interpreter that runs the
    # suite: the gateway has a hard protobuf dependency, so whatever `python`
    # the wrapper finds needs the package deps installed.
    bin_dir = os.path.dirname(sys.executable)
    environment["PATH"] = (
        bin_dir + os.pathsep + environment.get("PATH", "")
        if bin_dir
        else environment.get("PATH", "")
    )
    environment.update({
        "MUSE_GATEWAY_NO_MODEL_DISCOVERY": "1",
        "MUSE_GATEWAY_CODEX_ACCESS_TOKEN": "launcher-test-token",
        # A synthetic explicit token bypasses both native Devin credential
        # stores; discovery is off, so only bundled public names are read.
        "MUSE_GATEWAY_DEVIN_API_KEY": "launcher-test-token",
    })
    if explicit_models:
        environment["MUSE_GATEWAY_LOCAL_MODELS"] = "launcher-test"
    return environment


def _stage_checkout(directory):
    root = Path(__file__).resolve().parent
    work = Path(directory)
    launcher = "muse-gateway.ps1" if os.name == "nt" else "muse-gateway"
    for name in (
        launcher,
        "muse_gateway.py",
        "muse_providers.py",
        "muse_capture.py",
        "muse_devin.py",
        "muse_devin_wire.py",
        "muse_kiro.py",
        "devin_pb2.py",
    ):
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
                sys.executable, str(work / "muse_gateway.py"), "launch",
                "--env-file", str(work / ".env"),
                "--check-only", "--port", str(port),
            ]
            output = _run_check_only(command, work, environment, port)
            self.assertIn("local/launcher-test", output)


class ModelsAndReuseTests(unittest.TestCase):
    """In-process coverage for --models and second-terminal reuse.

    Neither path spawns Muse (--models exits before serving; reuse is
    exercised with --check-only), so no subprocess is needed. Discovery
    stays off and credentials stay fake: both tests are fully offline.
    """

    def setUp(self):
        # Availability is the only auth boundary this file needs. Do not
        # inspect a contributor's Claude Code login during launcher tests.
        no_claude_login = patch("muse_gateway.discover_claude_credentials", return_value=None)
        no_claude_login.start()
        self.addCleanup(no_claude_login.stop)

    def _offline_environment(self):
        environment = _clean_environment()
        # Claude CLI may or may not exist on the test machine; keyless HTTP
        # transport is unavailable everywhere, keeping the listing exact.
        environment["MUSE_GATEWAY_CLAUDE_TRANSPORT"] = "http"
        return environment

    def test_launch_attaches_to_compatible_gateway(self):
        saved = os.environ.copy()
        environment = self._offline_environment()
        os.environ.clear()
        os.environ.update(environment)
        try:
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                port = _free_port()
                server_config = Config(parse_args([
                    "auto", "--port", str(port),
                    "--log-file", str(work / "server.log"),
                ]))
                server = MuseShimServer(
                    (server_config.host, port), server_config
                )
                thread = threading.Thread(
                    target=server.serve_forever, daemon=True
                )
                thread.start()
                try:
                    out, err = io.StringIO(), io.StringIO()
                    with redirect_stdout(out), redirect_stderr(err):
                        code = launch_main([], [
                            "--port", str(port), "--check-only",
                            "--env-file", str(work / ".env"),
                            "--log-file", str(work / "launch.log"),
                        ])
                    output = out.getvalue()
                    self.assertEqual(code, 0)
                    self.assertIn("reusing existing gateway", output)
                    self.assertIn("catalog (", output)
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=5
                    ) as response:
                        health = json.loads(response.read().decode("utf-8"))
                    self.assertTrue(health.get("ok"))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_kiro_configured_launch_and_run_refuse_occupied_gateway(self):
        """Do not silently discard new Kiro settings or bind over a Windows server."""
        with patch.dict(os.environ, self._offline_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                original = Config(parse_args([
                    "auto", "--port", "0", "--log-file", str(work / "original.log"),
                ]))
                server = MuseShimServer(("127.0.0.1", 0), original)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                port = server.server_port
                try:
                    for settings in (
                        ["--kiro-api-key", "synthetic-proxy-key"],
                        ["--kiro-upstream", "http://127.0.0.1:1/v1"],
                        ["--kiro-models", "claude-haiku-4.5"],
                        ["--kiro-context-limit", "32768"],
                        ["--kiro-output-limit", "4096"],
                    ):
                        with self.subTest(settings=settings[0]):
                            out, err = io.StringIO(), io.StringIO()
                            with redirect_stdout(out), redirect_stderr(err), patch(
                                "muse_gateway.subprocess.Popen",
                                side_effect=AssertionError("Must not start Muse on the old server"),
                            ):
                                code = launch_main([], [
                                    "--port", str(port), "--check-only", *settings,
                                    "--env-file", str(work / "missing.env"),
                                    "--log-file", str(work / "launch.log"),
                                ])
                            self.assertEqual(code, 1, out.getvalue() + err.getvalue())
                            self.assertIn("require a new", err.getvalue())
                            configured = Config(parse_args([
                                "auto", "--port", str(port), *settings,
                                "--log-file", str(work / "run.log"),
                            ]))
                            output = io.StringIO()
                            with patch("muse_gateway.subprocess.Popen", side_effect=AssertionError("Must not start Muse")):
                                self.assertEqual(run_muse(configured, [], output=output), 1)
                            self.assertIn("require a new", output.getvalue())
                    # The original server remains alive and unchanged.
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                        health = json.load(response)
                    self.assertTrue(health["ok"])
                    self.assertEqual(health["provider"], "auto")
                    self.assertEqual(health["supported_routes"]["version"], 1)
                    self.assertIn("kiro", health["supported_routes"]["providers"])
                    self.assertNotIn("synthetic-proxy-key", json.dumps(health))
                    self.assertFalse((work / "launch.log").exists())
                    self.assertFalse((work / "run.log").exists())
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=5)

    def test_launch_catalog_applies_all_kiro_options_without_contacting_provider(self):
        with patch.dict(os.environ, self._offline_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                output, error = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(error), patch(
                    "muse_gateway._devin_provider_ready", return_value=False,
                ), patch(
                    "muse_gateway.muse_devin.credential_state", return_value={"state": "missing"},
                ):
                    code = launch_main([], [
                        "--port", "0", "--check-only",
                        "--env-file", str(work / "missing.env"),
                        "--log-file", str(work / "gateway.log"),
                        "--kiro-upstream", "http://127.0.0.1:1/v1",
                        "--kiro-api-key", "synthetic-proxy-key",
                        "--kiro-models", "claude-haiku-4.5,auto",
                        "--kiro-context-limit", "32768", "--kiro-output-limit=4096",
                    ])
                self.assertEqual(code, 0, output.getvalue() + error.getvalue())
                self.assertIn("kiro/claude-haiku-4.5", output.getvalue())
                self.assertIn("kiro/auto", output.getvalue())
                self.assertNotIn("synthetic-proxy-key", output.getvalue() + error.getvalue())

    def test_models_reads_kiro_env_file_and_explicit_flags_still_win(self):
        with patch.dict(os.environ, self._offline_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                env_file = Path(directory) / "catalog.env"
                env_file.write_text(
                    "MUSE_GATEWAY_KIRO_API_KEY=synthetic-proxy-key\n"
                    "MUSE_GATEWAY_KIRO_MODELS=from-file\n",
                    encoding="utf-8",
                )
                for extra, expected in (([], "kiro/from-file"), (["--kiro-models", "from-cli"], "kiro/from-cli")):
                    with self.subTest(extra=extra), redirect_stdout(io.StringIO()) as output:
                        code = main(["--models", "--models-provider", "kiro", "--env-file", str(env_file), *extra])
                    self.assertEqual(code, 0)
                    self.assertEqual(output.getvalue().strip(), expected)

    def test_post_launch_flags_override_invalid_environment_before_validation(self):
        alternatives = (
            ("MUSE_GATEWAY_KIRO_OUTPUT_LIMIT", "bad", "--kiro-output-limit", "4096"),
            ("MUSE_GATEWAY_KIRO_CONTEXT_LIMIT", "bad", "--kiro-context-limit", "32768"),
            ("MUSE_GATEWAY_KIRO_API_KEY", "invalid key", "--kiro-api-key", "synthetic-proxy-key"),
            ("MUSE_GATEWAY_KIRO_UPSTREAM", "http://remote.invalid", "--kiro-upstream", "http://127.0.0.1:1/v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, invalid, flag, valid in alternatives:
                environment = self._offline_environment()
                environment.update({
                    name: invalid, "MUSE_GATEWAY_KIRO_API_KEY": "synthetic-proxy-key",
                    "MUSE_GATEWAY_KIRO_MODELS": "auto",
                })
                environment[name] = invalid
                output, error = io.StringIO(), io.StringIO()
                with self.subTest(option=flag), patch.dict(os.environ, environment, clear=True):
                    with redirect_stdout(output), redirect_stderr(error):
                        code = launch_main([], [
                            "--port", "0", "--check-only", flag, valid,
                            "--env-file", str(Path(directory) / "missing.env"),
                            "--log-file", str(Path(directory) / "gateway.log"),
                        ])
                    self.assertEqual(code, 0, output.getvalue() + error.getvalue())
                    self.assertIn("kiro/auto", output.getvalue())
                    self.assertNotIn("synthetic-proxy-key", output.getvalue() + error.getvalue())


class KiroSocketOwnershipTests(unittest.TestCase):
    def _config(self, directory, name):
        return Config(argparse.Namespace(
            provider="auto", host="127.0.0.1", port=0, model_discovery=False,
            kiro_api_key="synthetic-proxy-key",
            log_file=str(Path(directory) / (name + ".log")),
            capture_dir=str(Path(directory) / "captures"),
        ))

    def test_bound_not_listening_socket_cannot_be_taken_over(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as owner:
            owner.bind(("127.0.0.1", 0))
            address = owner.getsockname()
            self.assertFalse(_port_is_busy(*address), "The old probe cannot see a bound-only owner")
            created = None
            try:
                with self.assertRaises(OSError, msg="KIRO_BIND_OWNERSHIP: do not take a bound socket"):
                    created = MuseShimServer(address, self._config(directory, "loser"))
            finally:
                if created is not None:
                    created.server_close()
            self.assertFalse((Path(directory) / "loser.log").exists())
            self.assertFalse((Path(directory) / "captures").exists())

    def test_simultaneous_real_free_port_probes_have_only_one_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            probes = threading.Barrier(2)
            results, failures = [], []

            def synchronized_probe(host, target):
                busy = _port_is_busy(host, target)
                probes.wait(timeout=5)
                return busy

            def start(name):
                try:
                    results.append((name, MuseShimServer(("127.0.0.1", port), self._config(directory, name))))
                except OSError as error:
                    failures.append((name, error))
                except BaseException as error:
                    failures.append((name, error))

            threads = [threading.Thread(target=start, args=(name,)) for name in ("first", "second")]
            try:
                with patch("muse_gateway._port_is_busy", side_effect=synchronized_probe):
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=8)
                self.assertTrue(all(not thread.is_alive() for thread in threads))
                self.assertEqual(len(results), 1, "KIRO_BIND_OWNERSHIP: simultaneous starts must not share the listener")
                self.assertEqual(len(failures), 1)
                self.assertIsInstance(failures[0][1], OSError)
                self.assertFalse((Path(directory) / (failures[0][0] + ".log")).exists())
                self.assertEqual(len(list((Path(directory) / "captures").glob("gateway-*"))), 1)
            finally:
                for _, server in results:
                    server.server_close()

    def test_capture_initialization_failure_releases_the_new_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            for boundary in ("CaptureSession", "_PrivateRotatingFileHandler", "logging.Logger"):
                with self.subTest(boundary=boundary):
                    address = ("127.0.0.1", _free_port())
                    with patch("muse_gateway." + boundary, side_effect=OSError("synthetic setup failure")):
                        with self.assertRaises(OSError):
                            MuseShimServer(address, self._config(directory, boundary))
                    with socket.socket() as replacement:
                        replacement.bind(address)


class LaunchArgTests(unittest.TestCase):
    def test_double_dash_and_help_passthrough(self):
        opts, muse_args = _split_launch_args(["--", "--yolo"])
        self.assertEqual(muse_args, ["--yolo"])
        opts, muse_args = _split_launch_args(["--help"])
        self.assertFalse(opts.help)
        self.assertEqual(muse_args, ["--help"])
        opts, muse_args = _split_launch_args(["-h"])
        self.assertTrue(opts.help)
        self.assertEqual(muse_args, [])
        with ExitStack() as stack:
            for boundary in ("load_env_file", "Config", "MuseShimServer", "subprocess.Popen",
                             "discover_codex_credentials", "discover_claude_credentials"):
                stack.enter_context(patch("muse_gateway." + boundary,
                    side_effect=AssertionError("Launcher -h must exit before initialization")))
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(launch_main([], ["-h"]), 0)
            self.assertIn("launch", output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
