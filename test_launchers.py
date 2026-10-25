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
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from unittest.mock import patch

from muse_gateway import Config, MuseShimServer, _split_launch_args, ensure_kiro_gateway, launch_main as _launch_main, load_env_file, main, parse_args, run_muse, _kiro_server_python, _port_is_busy, _stop_child
from tests.peers import serving


_EMPTY_ENV_FILE = Path(tempfile.mkdtemp(prefix="muse-gateway-test-env-")) / "empty.env"
_EMPTY_ENV_FILE.write_text("", encoding="utf-8")


def launch_main(pre_argv, post_argv):
    # Explicit CLI-only discovery switch keeps synthetic credentials offline.
    # An explicit empty env file isolates in-process launches from any
    # beside-module .env in the checkout (P0-02: named files must exist).
    options = [] if "--kiro-autostart" in post_argv else ["--no-model-discovery"]
    post_argv = list(post_argv)
    if not any(str(a).startswith("--env-file") for a in [*pre_argv, *post_argv]):
        post_argv = ["--env-file", str(_EMPTY_ENV_FILE), *post_argv]
    return _launch_main([*options, *pre_argv], post_argv)


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
        "CODEX_HOME": ".codex",
        "ANTHROPIC_API_KEY": "launcher-test-token",
        # A synthetic explicit token bypasses the native Devin CLI login;
        # discovery is off, so only bundled public names are read.
        "DEVIN_API_KEY": "launcher-test-token",
    })
    if explicit_models:
        environment["MUSE_GATEWAY_MODELS"] = "meta/muse-spark-1.2"
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
        "muse_models.py",
        "devin_pb2.py",
    ):
        shutil.copyfile(root / name, work / name)
    shutil.copytree(root / "providers", work / "providers", ignore=shutil.ignore_patterns("__pycache__"))
    (work / ".codex").mkdir()
    (work / ".codex" / "auth.json").write_text('{"tokens":{"access_token":"synthetic-launcher-token"}}')
    return work, launcher


def _run_check_only(command, work, environment, port):
    result = subprocess.run(
        [*command, "--no-model-discovery"], cwd=work, env=environment, capture_output=True,
        text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "skipping local-backend preflight" in output, output
    assert "config: no env file" in output, output
    assert "catalog (" in output, output
    assert "models are disabled" not in output, output
    assert not (work / ".env").exists()
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    return output


class RemovedDevinConfigTests(unittest.TestCase):
    def test_retired_flags_never_reach_muse_or_start_a_server(self):
        cases = (
            ["--devin-auth-db", "private-value", "run"],
            ["devin", "--devin-credential-id=private-value"],
            ["launch", "--devin-auth-db=private-value"],
            ["launch", "resume", "--devin-credential-id", "private-value"],
            ["launch", "--", "--devin-default-file=private-value"],
            ["run", "--devin-credential-id=private-value"],
        )
        with patch.dict(os.environ, _clean_environment(), clear=True), patch(
            "muse_gateway.subprocess.Popen", side_effect=AssertionError("Must not start Muse"),
        ), patch("muse_gateway.MuseShimServer", side_effect=AssertionError("Must not serve")):
            for argv in cases:
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(main(argv), 2)
                self.assertIn("removed Devin credential selection", error.getvalue())
                self.assertNotIn("private-value", error.getvalue())
            with redirect_stderr(io.StringIO()) as error:
                self.assertEqual(launch_main([], ["exec", "--devin-auth-db=private-value"]), 2)
            self.assertIn("removed Devin credential selection", error.getvalue())
            self.assertNotIn("private-value", error.getvalue())

    def test_retired_environment_settings_fail_without_showing_values(self):
        environment = _clean_environment()
        for suffix in ("AUTH_DB", "CREDENTIAL_ID", "DEFAULT_FILE"):
            environment["MUSE_GATEWAY_DEVIN_" + suffix] = "private-value"
        environment["MUSE_SHIM_DEVIN_AUTH_DB"] = "legacy-private-value"
        with patch.dict(os.environ, environment, clear=True), redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                parse_args(["doctor"])
        self.assertEqual(caught.exception.code, 2)
        for suffix in ("AUTH_DB", "CREDENTIAL_ID", "DEFAULT_FILE"):
            self.assertIn("MUSE_GATEWAY_DEVIN_" + suffix, error.getvalue())
        self.assertIn("MUSE_SHIM_DEVIN_AUTH_DB", error.getvalue())
        self.assertNotIn("private-value", error.getvalue())
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, _clean_environment(), clear=True,
        ):
            env_file = Path(directory) / "retired.env"
            env_file.write_text("MUSE_GATEWAY_DEVIN_DEFAULT_FILE=private-value\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()) as error, patch(
                "muse_gateway.MuseShimServer", side_effect=AssertionError("Must not serve"),
            ):
                self.assertEqual(launch_main([], ["--env-file", str(env_file), "--check-only"]), 2)
            self.assertIn("MUSE_GATEWAY_DEVIN_DEFAULT_FILE", error.getvalue())
            self.assertNotIn("private-value", error.getvalue())
        environment = _clean_environment()
        environment.update(MUSE_GATEWAY_KIRO_MODEL_BUDGETS="private-value",
                           MUSE_GATEWAY_MAX_OUTPUT_TOKENS="private-value")
        with patch.dict(os.environ, environment, clear=True), redirect_stderr(io.StringIO()) as error:
            cfg = Config(parse_args(["auto"]))
            self.assertFalse(hasattr(cfg, "kiro_model_budgets"))
            self.assertFalse(hasattr(cfg, "max_output_tokens"))
        self.assertIn("MUSE_GATEWAY_KIRO_MODEL_BUDGETS", error.getvalue())
        self.assertNotIn("private-value", error.getvalue())
        environment.update(MUSE_SHIM_PORT="private-value", MUSE_GATEWAY_PORT="12345")
        with patch.dict(os.environ, environment, clear=True), redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                parse_args(["auto"])
        self.assertIn("conflicting retired setting MUSE_SHIM_PORT", error.getvalue())
        self.assertNotIn("private-value", error.getvalue())
        with patch.dict(os.environ, _clean_environment(), clear=True), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(["launch", "--kiro-output-limit", "private-value"]), 2)
        self.assertNotIn("private-value", error.getvalue())

    def test_credential_environment_policy(self):
        environment = _clean_environment()
        environment["META_API_KEY"] = "private-billing-token"
        with patch.dict(os.environ, environment, clear=True), patch(
            "muse_gateway.MuseShimServer", side_effect=AssertionError("Must not serve"),
        ), redirect_stderr(io.StringIO()) as error:
            self.assertEqual(launch_main([], ["--check-only"]), 2)
        self.assertIn("conflicts with subscription launch", error.getvalue())
        self.assertNotIn("private-billing-token", error.getvalue())

        environment = _clean_environment()
        environment["DEVIN_API_KEY"] = "ordinary-explicit-token"
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(Config(parse_args(["devin"])).devin_api_key, "ordinary-explicit-token")
            self.assertEqual(Config(parse_args(["devin", "--devin-api-key", "flag-token"])).devin_api_key,
                             "flag-token")


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
                        "meta/muse-spark-1.3" not in output, explicit_models
                    )

    def test_direct_launch_matches_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            work, _ = _stage_checkout(directory)
            environment = _clean_environment(explicit_models=True)
            port = _free_port()
            command = [
                sys.executable, str(work / "muse_gateway.py"), "launch",
                "--check-only", "--port", str(port),
            ]
            output = _run_check_only(command, work, environment, port)
            self.assertIn("meta/muse-spark-1.2", output)
            self.assertNotIn("local/launcher-test", output)


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
        no_codex_login = patch("muse_gateway.discover_codex_credentials", return_value=None)
        no_codex_login.start()
        self.addCleanup(no_codex_login.stop)

    def _offline_environment(self):
        environment = _clean_environment()
        # Claude CLI may or may not exist on the test machine; keyless HTTP
        # transport is unavailable everywhere, keeping the listing exact.
        environment.pop("ANTHROPIC_API_KEY", None)
        return environment

    def test_launch_attaches_to_compatible_gateway(self):
        with patch.dict(os.environ, self._offline_environment(), clear=True), \
                tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            port = _free_port()
            server_config = Config(parse_args([
                "auto", "--port", str(port), "--log-file", str(work / "server.log"),
            ]))
            server = MuseShimServer((server_config.host, port), server_config)
            with serving(server):
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = launch_main([], [
                        "--port", str(port), "--check-only",
                        "--log-file", str(work / "launch.log"),
                    ])
                output = out.getvalue()
                self.assertEqual(code, 0)
                self.assertIn("reusing existing gateway", output)
                self.assertIn("owned by the running gateway", output)
                self.assertIn("catalog (", output)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    health = json.load(response)
                self.assertTrue(health.get("ok"))

    def test_check_only_reports_per_provider_auth_status(self):
        with patch.dict(os.environ, self._offline_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = launch_main([], [
                        "--port", "0", "--check-only",
                        "--log-file", str(work / "launch.log"),
                    ])
                output = out.getvalue()
                self.assertEqual(code, 0, output + err.getvalue())
                self.assertIn("Checking Codex... No auth", output)
                self.assertIn("Checking Claude... No auth", output)
                self.assertIn("Checking Meta... Subscription login", output)
                self.assertIn("Checking Devin... Found Auth!", output)
                self.assertIn("Checking Kiro... No auth", output)
                self.assertIn("MUSE_GATEWAY_KIRO_API_KEY", output)

    def test_kiro_configured_launch_and_run_refuse_occupied_gateway(self):
        """Do not silently discard new Kiro settings or bind over a Windows server."""
        with patch.dict(os.environ, self._offline_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                original = Config(parse_args([
                    "auto", "--port", "0", "--log-file", str(work / "original.log"),
                ]))
                server = MuseShimServer(("127.0.0.1", 0), original)
                port = server.server_port
                with serving(server):
                    for settings in (
                        ["--kiro-api-key", "synthetic-proxy-key"],
                        ["--kiro-upstream", "http://127.0.0.1:1/v1"],
                    ):
                        with self.subTest(settings=settings[0]):
                            out, err = io.StringIO(), io.StringIO()
                            with redirect_stdout(out), redirect_stderr(err), patch(
                                "muse_gateway.subprocess.Popen",
                                side_effect=AssertionError("Must not start Muse on the old server"),
                            ):
                                code = launch_main([], [
                                    "--port", str(port), "--check-only", *settings,
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

    def test_launch_disabled_discovery_never_fabricates_allowlist_models(self):
        environment = self._offline_environment()
        environment["MUSE_GATEWAY_MODELS"] = "kiro/claude-haiku-4.5"
        with patch.dict(os.environ, environment, clear=True), tempfile.TemporaryDirectory() as directory:
            output, error = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                code = launch_main([], ["--port", "0", "--check-only",
                    "--log-file", str(Path(directory) / "gateway.log"),
                    "--kiro-api-key", "synthetic-proxy-key"])
            self.assertEqual(code, 0, output.getvalue() + error.getvalue())
            self.assertIn("catalog (0 entries)", output.getvalue())
            self.assertIn("unresolved MUSE_GATEWAY_MODELS", error.getvalue())
            self.assertNotIn("synthetic-proxy-key", output.getvalue() + error.getvalue())

    def test_models_reads_current_filter_from_env_file(self):
        with patch.dict(os.environ, self._offline_environment(), clear=True), tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "catalog.env"
            env_file.write_text("MUSE_GATEWAY_MODELS=meta/muse-spark-1.2\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                code = main(["--models", "--models-provider", "meta", "--no-model-discovery", "--env-file", str(env_file)])
            self.assertEqual(code, 0)
            self.assertEqual(output.getvalue().strip(), "meta/muse-spark-1.2")

    def test_post_launch_flags_override_invalid_environment_before_validation(self):
        alternatives = (
            ("MUSE_GATEWAY_KIRO_API_KEY", "invalid key", "--kiro-api-key", "synthetic-proxy-key"),
            ("MUSE_GATEWAY_KIRO_UPSTREAM", "http://remote.invalid", "--kiro-upstream", "http://127.0.0.1:1/v1"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, invalid, flag, valid in alternatives:
                environment = self._offline_environment()
                environment.update({
                    name: invalid, "MUSE_GATEWAY_KIRO_API_KEY": "synthetic-proxy-key",
                })
                environment[name] = invalid
                output, error = io.StringIO(), io.StringIO()
                with self.subTest(option=flag), patch.dict(os.environ, environment, clear=True):
                    with redirect_stdout(output), redirect_stderr(error):
                        code = launch_main([], [
                            "--port", "0", "--check-only", flag, valid,
                            "--log-file", str(Path(directory) / "gateway.log"),
                        ])
                    self.assertEqual(code, 0, output.getvalue() + error.getvalue())
                    self.assertIn("catalog discovery disabled", output.getvalue())
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

    def test_env_file_loading_precedence_and_parser_edges(self):
        import muse_gateway

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                'MUSE_GATEWAY_LOCAL_UPSTREAM="http://from-file:4000/v1"\n'
                "export MUSE_GATEWAY_LOCAL_API_KEY='file-key'\n"
                "# comment\nnot a valid line\nMUSE_GATEWAY_MODELS=\n", encoding="utf-8",
            )
            with patch.dict(os.environ, {"MUSE_GATEWAY_LOCAL_API_KEY": ""}, clear=True):
                self.assertEqual(load_env_file(str(path)), 2)
                self.assertEqual(os.environ["MUSE_GATEWAY_LOCAL_API_KEY"], "")
                self.assertEqual(os.environ["MUSE_GATEWAY_LOCAL_UPSTREAM"], "http://from-file:4000/v1")
                self.assertEqual(os.environ["MUSE_GATEWAY_MODELS"], "")

            path.write_bytes(
                b"\xef\xbb\xbfFIRST_SETTING=one\n"
                b"export EXPORTED=two\n"
                b"exportable_T=three\n"
                b"INLINE=secret # a note\n"
                b'QUOTED="keep # inside" # trailing note\n'
                b"SINGLE='also # inside'\n"
            )
            expected = {
                "FIRST_SETTING": "one", "EXPORTED": "two", "exportable_T": "three",
                "INLINE": "secret", "QUOTED": "keep # inside", "SINGLE": "also # inside",
            }
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_env_file(str(path)), len(expected))
                for name, value in expected.items():
                    self.assertEqual(os.environ[name], value)
                    self.assertIn(name, muse_gateway._FILE_ENV_NAMES)


class KiroAutostartTest(unittest.TestCase):
    """Autostart spawns the local checkout only when enabled and needed."""

    stub_main = (
        "import json\n"
        "import os\n"
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "\n"
        "with open(os.path.join(os.getcwd(), 'stub.pid'), 'w') as handle:\n"
        "    handle.write(str(os.getpid()))\n"
        "\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "\n"
        "\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        if self.path.split('?', 1)[0] == '/v1/models':\n"
        "            body = json.dumps(\n"
        "                {'object': 'list', 'data': [{'id': 'claude-haiku-4.5'}]}\n"
        "            ).encode('utf-8')\n"
        "            self.send_response(200)\n"
        "            self.send_header('Content-Type', 'application/json')\n"
        "            self.send_header('Content-Length', str(len(body)))\n"
        "            self.end_headers()\n"
        "            self.wfile.write(body)\n"
        "        else:\n"
        "            self.send_response(404)\n"
        "            self.end_headers()\n"
        "\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "\n"
        "\n"
        "ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()\n"
    )
    sleep_main = (
        "import os\n"
        "import time\n"
        "with open(os.path.join(os.getcwd(), 'stub.pid'), 'w') as handle:\n"
        "    handle.write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )

    def _pid_alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _kiro_config(self, directory, port, **overrides):
        argv = [
            "auto",
            "--kiro-api-key", "synthetic-proxy-key",
            "--kiro-upstream", f"http://127.0.0.1:{port}/v1",
            "--kiro-server-dir", str(directory),
            "--log-file", str(Path(directory) / "gateway.log"),
        ]
        if overrides.pop("autostart", False):
            argv.append("--kiro-autostart")
        for flag, value in overrides.items():
            argv.extend([f"--{flag.replace('_', '-')}", value])
        return Config(parse_args(argv))

    @contextmanager
    def _ensure_stub(self, source, timeout):
        with patch.dict(os.environ, _clean_environment(), clear=True), \
                tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "main.py").write_text(source)
            config = self._kiro_config(directory, _free_port(), autostart=True)
            output = io.StringIO()
            with redirect_stdout(output):
                child = ensure_kiro_gateway(config, timeout=timeout)
            try:
                yield child, config, work, output
            finally:
                if child is not None:
                    _stop_child(child)

    def _assert_ensure_does_not_spawn(self, config, expected_note=None):
        with patch(
            "muse_gateway.subprocess.Popen",
            side_effect=AssertionError("must not spawn"),
        ):
            self.assertIsNone(ensure_kiro_gateway(config))
        if expected_note is None:
            self.assertIsNone(config._kiro_service_note)
        else:
            self.assertIn(expected_note, config._kiro_service_note)

    def test_config_autostart_is_cli_only_and_defaults_off(self):
        expected_dir = os.path.abspath(os.path.expanduser("~/kiro-gateway"))
        with patch.dict(os.environ, _clean_environment(), clear=True):
            config = Config(parse_args(["auto"]))
            self.assertFalse(config.kiro_autostart)
            self.assertEqual(config.kiro_server_dir, expected_dir)
            self.assertIsNone(config._kiro_service_note)
        environment = _clean_environment()
        environment["MUSE_GATEWAY_KIRO_AUTOSTART"] = "yes"
        environment["MUSE_GATEWAY_KIRO_SERVER_DIR"] = "~/custom-kiro"
        with patch.dict(os.environ, environment, clear=True):
            config = Config(parse_args(["auto"]))
            self.assertFalse(config.kiro_autostart)
            self.assertEqual(config.kiro_server_dir, expected_dir)

    def test_split_launch_args_parses_autostart_options(self):
        opts, rest = _split_launch_args([
            "--kiro-autostart", "--kiro-server-dir", "/tmp/kiro", "--check-only",
        ])
        self.assertTrue(opts.kiro_autostart)
        self.assertEqual(opts.kiro_server_dir, "/tmp/kiro")
        self.assertTrue(opts.check_only)
        self.assertEqual(rest, [])
        opts, _ = _split_launch_args(["--check-only"])
        self.assertFalse(opts.kiro_autostart)

    def test_server_python_prefers_checkout_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_kiro_server_python(directory), sys.executable)
            if os.name == "nt":
                venv_python = Path(directory) / ".venv" / "Scripts" / "python.exe"
            else:
                venv_python = Path(directory) / ".venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("stub")
            self.assertEqual(_kiro_server_python(directory), str(venv_python))

    def test_ensure_no_spawn_cases(self):
        def without_key(directory):
            return Config(parse_args([
                "auto", "--kiro-autostart",
                "--kiro-server-dir", directory,
                "--log-file", str(Path(directory) / "gateway.log"),
            ]))

        def non_loopback(directory):
            return Config(parse_args([
                "auto",
                "--kiro-api-key", "synthetic-proxy-key",
                "--kiro-upstream", "https://192.0.2.1:9/v1",
                "--kiro-autostart",
                "--kiro-server-dir", directory,
                "--log-file", str(Path(directory) / "gateway.log"),
            ]))

        cases = (
            ("without proxy key", without_key, None),
            ("autostart off", lambda directory: self._kiro_config(directory, _free_port()), "--kiro-autostart"),
            ("non-loopback upstream", non_loopback, "not loopback"),
            ("missing checkout", lambda directory: self._kiro_config(directory, _free_port(), autostart=True), "main.py not found"),
        )
        with patch.dict(os.environ, _clean_environment(), clear=True):
            for label, build, expected_note in cases:
                with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                    self._assert_ensure_does_not_spawn(build(directory), expected_note)

    def test_ensure_reports_already_running_service(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        with patch.dict(os.environ, _clean_environment(), clear=True):
            with tempfile.TemporaryDirectory() as directory:
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                with serving(server):
                    config = self._kiro_config(
                        directory, server.server_port, autostart=True,
                    )
                    self.assertIsNone(ensure_kiro_gateway(config))
                    self.assertEqual(
                        config._kiro_service_note, "service already running",
                    )

    def test_ensure_starts_and_detects_stub_service(self):
        with self._ensure_stub(self.stub_main, timeout=15.0) as (child, config, work, output):
            self.assertIsNotNone(child)
            assert child is not None
            self.assertIn("service autostarted", config._kiro_service_note or "")
            self.assertIn("autostarting from", output.getvalue())
            self.assertTrue((work / "autostart.log").exists())
            self.assertEqual(int((work / "stub.pid").read_text().strip()), child.pid)
            self.assertIsNone(child.poll())
            pid = child.pid
        self.assertFalse(self._pid_alive(pid))

    def test_ensure_timeout_kills_unresponsive_child(self):
        with self._ensure_stub(self.sleep_main, timeout=2.0) as (child, config, work, _output):
            self.assertIsNone(child)
            self.assertIn("did not answer", config._kiro_service_note or "")
            pid = int((work / "stub.pid").read_text().strip())
            self.assertFalse(self._pid_alive(pid))

    def test_ensure_exited_child_reports_failure(self):
        with self._ensure_stub("import sys\nsys.exit(3)\n", timeout=10.0) as (child, config, _work, _output):
            self.assertIsNone(child)
            self.assertIn("exited during startup", config._kiro_service_note or "")

    def test_launch_check_only_autostarts_and_stops_service(self):
        environment = _clean_environment()
        with patch.dict(os.environ, environment, clear=True):
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                (work / "main.py").write_text(self.stub_main)
                output, error = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(error), patch(
                    "muse_gateway._devin_provider_ready", return_value=False,
                ), patch(
                    "muse_gateway.muse_devin.credential_state",
                    return_value={"state": "missing"},
                ):
                    code = launch_main([], [
                        "--port", "0", "--check-only",
                        "--log-file", str(work / "gateway.log"),
                        "--kiro-upstream",
                        f"http://127.0.0.1:{_free_port()}/v1",
                        "--kiro-api-key", "synthetic-proxy-key",
                        "--kiro-autostart",
                        "--kiro-server-dir", str(work),
                    ])
                text = output.getvalue()
                self.assertEqual(code, 0, text + error.getvalue())
                self.assertIn("service autostarted", text)
                self.assertIn("kiro/claude-haiku-4.5", text)
                self.assertNotIn("synthetic-proxy-key", text + error.getvalue())
                pid = int((work / "stub.pid").read_text().strip())
                self.assertFalse(self._pid_alive(pid))


class TrustBoundaryTests(unittest.TestCase):
    """P0-02/P0-03/P2-10: no CWD trust, scrubbed children, env parser edges."""

    def test_cwd_env_is_never_loaded_implicitly(self):
        import muse_gateway

        with tempfile.TemporaryDirectory() as directory:
            hostile = Path(directory) / "project"
            module_dir = Path(directory) / "installed"
            hostile.mkdir()
            module_dir.mkdir()
            (hostile / ".env").write_text("MUSE_GATEWAY_HOSTILE_SENTINEL=pwned\n")
            saved = os.environ.copy()
            os.environ.pop("MUSE_GATEWAY_HOSTILE_SENTINEL", None)
            try:
                with patch.object(muse_gateway, "__file__", str(module_dir / "muse_gateway.py")):
                    previous = os.getcwd()
                    os.chdir(hostile)
                    try:
                        self.assertIsNone(muse_gateway._default_env_file())
                        path, count = muse_gateway._load_config_file([])
                    finally:
                        os.chdir(previous)
                self.assertEqual((path, count), (None, 0))
                self.assertNotIn("MUSE_GATEWAY_HOSTILE_SENTINEL", os.environ)
            finally:
                os.environ.clear()
                os.environ.update(saved)

    def test_named_missing_env_file_is_an_error_on_both_entry_points(self):
        import muse_gateway
        from muse_providers import ProviderError

        with self.assertRaises(ProviderError) as caught:
            muse_gateway._load_config_file(["--env-file", "/nonexistent/missing.env"])
        self.assertEqual(caught.exception.code, "env_file_missing")
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(_launch_main([], ["--env-file", "/nonexistent/missing.env", "--check-only"]), 2)
        self.assertNotIn("Traceback", err.getvalue())
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(main(["--env-file", "/nonexistent/missing.env"]), 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_child_environment_scrubs_every_brokered_secret(self):
        import muse_gateway

        sentinels = {
            "MUSE_GATEWAY_KIRO_API_KEY": "s1", "MUSE_SHIM_X": "s2",
            "META_API_KEY": "s3", "DEVIN_API_KEY": "s4", "CODEX_X": "s5",
            "OPENAI_API_KEY": "s6", "ANTHROPIC_API_KEY": "s7",
            "CLAUDE_CODE_OAUTH_TOKEN": "s8", "KIRO_X": "s9",
            "CLAUDE_CONFIG_DIR": "/tmp/claude-cfg", "PATH": "/usr/bin",
        }
        with patch.dict(os.environ, sentinels, clear=True):
            child = muse_gateway._child_environment()
            for name in sentinels:
                if name == "PATH":
                    self.assertEqual(child.get("PATH"), "/usr/bin")
                else:
                    self.assertNotIn(name, child)
            for policy, expected in (
                ("oauth", {"CLAUDE_CODE_OAUTH_TOKEN"}),
                ("api-key", {"ANTHROPIC_API_KEY"}),
                ("all", {"CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"}),
                ("none", set()),
            ):
                with self.subTest(policy=policy):
                    kept = muse_gateway._child_environment(
                        keep=muse_gateway._CLAUDE_PASSTHROUGH_SETS[policy] | muse_gateway._CLAUDE_HARMLESS_VARS
                    )
                    self.assertEqual(kept.get("CLAUDE_CONFIG_DIR"), "/tmp/claude-cfg")
                    self.assertNotIn("MUSE_GATEWAY_KIRO_API_KEY", kept)
                    for name in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
                        self.assertEqual(name in kept, name in expected)

    def test_passthrough_config_validation(self):
        from muse_providers import ProviderError

        with patch.dict(os.environ, _clean_environment(), clear=True):
            args = parse_args(["auto"])
            del args.claude_cli_passthrough
            self.assertEqual(Config(args).claude_cli_passthrough, "oauth")
            args = parse_args(["auto"])
            args.claude_cli_passthrough = None
            self.assertEqual(Config(args).claude_cli_passthrough, "oauth")
            for bad in ("bogus", ""):
                args = parse_args(["auto"])
                args.claude_cli_passthrough = bad
                with self.assertRaises(ProviderError) as caught:
                    Config(args)
                self.assertEqual(caught.exception.code, "invalid_claude_passthrough")

    def test_launch_argument_scanning_and_passthrough_routing(self):
        import muse_gateway

        for argv, expected, rest in (
            (["--claude-cli-passthrough", "api-key", "exec", "hi"], "api-key", ["exec", "hi"]),
            (["--claude-cli-passthrough=none", "exec"], "none", ["exec"]),
        ):
            with self.subTest(argv=argv):
                opts, actual_rest = _split_launch_args(argv)
                self.assertEqual(opts.claude_cli_passthrough, expected)
                self.assertEqual(actual_rest, rest)
        scan = muse_gateway._subcommand_index
        self.assertEqual(scan(["--kiro-autostart", "launch", "--check-only"], "launch"), 1)
        self.assertEqual(scan(["--models", "launch"], "launch"), 1)
        self.assertIsNone(scan(["--port", "launch"], "launch"))
        self.assertEqual(scan(["--port=8787", "run"], "run"), 1)
        self.assertIsNone(scan(["--", "launch"], "launch"))
        self.assertIsNone(scan(["run", "launch"], "launch"))

    def test_cli_surface_is_single_sourced_and_launch_shapes_hold(self):
        # P1-11 (inverts R23/R24): one parser feeds the scanner and the
        # launch table; auto+pin is rejected; explicit "" disables local.
        import muse_gateway

        parser_options = {option for action in muse_gateway._build_parser()._actions
                          for option in action.option_strings}
        self.assertLessEqual(set(muse_gateway._LAUNCH_VALUE_OPTIONS) - {"--muse-bin"}, parser_options)
        self.assertLessEqual(muse_gateway._LAUNCH_FLAG_OPTIONS, muse_gateway._gateway_flag_options())
        err = io.StringIO()
        with patch.dict(os.environ, _clean_environment(), clear=True), redirect_stderr(err):
            self.assertEqual(launch_main(["codex", "--model", "gpt-5.4"], ["--check-only"]), 2)
        self.assertIn("--model pins a provider-specific server mode", err.getvalue())
        with patch.dict(os.environ, {"MUSE_GATEWAY_LOCAL_UPSTREAM": "http://127.0.0.1:9/v1"}, clear=True):
            self.assertEqual(Config(parse_args(["--local-upstream="])).local_upstream, "")
            self.assertEqual(Config(parse_args([])).local_upstream, "http://127.0.0.1:9/v1")
        launched = []

        class FakeChild:
            def __init__(self, argv, env=None):
                launched.append(argv)

            def wait(self):
                return 0

            def poll(self):
                return 0

        with patch.dict(os.environ, _clean_environment(), clear=True), \
                patch.object(muse_gateway.subprocess, "Popen", FakeChild), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = launch_main([], ["--port", str(_free_port()), "--muse-bin", "muse", "--", "exec", "prompt"])
        self.assertEqual(code, 0)
        self.assertEqual(launched[0][:3], ["muse", "--provider", "meta"])
        self.assertEqual(launched[0][3], "--base-url")
        self.assertEqual(launched[0][5:], ["exec", "prompt"])


class LifecycleAndIdentityTests(unittest.TestCase):
    """P1-10 / P1-15: ownership on interrupt, operator identity, clean failures."""

    def test_interrupt_during_kiro_readiness_stops_the_child(self):
        # Inverts R17: a signal in the readiness poll cannot orphan the child.
        import muse_gateway

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "main.py").write_text("import time\ntime.sleep(60)\n")
            config = Config(parse_args([
                "auto", "--kiro-api-key", "synthetic-proxy-key",
                "--kiro-upstream", f"http://127.0.0.1:{_free_port()}/v1",
                "--kiro-server-dir", directory, "--kiro-autostart",
                "--log-file", str(Path(directory) / "gateway.log"),
            ]))
            stopped = []
            with patch.dict(os.environ, _clean_environment(), clear=True), \
                    patch.object(muse_gateway, "_kiro_service_healthy", side_effect=[False, KeyboardInterrupt]), \
                    patch.object(muse_gateway, "_stop_child", side_effect=lambda child: (stopped.append(child), _stop_child(child))), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    ensure_kiro_gateway(config, timeout=5.0)
            self.assertEqual(len(stopped), 1)
            self.assertIsNotNone(stopped[0].poll())

    def test_launch_signal_during_autostart_window_runs_cleanup(self):
        import muse_gateway

        class Owned:
            stopped = False

            def poll(self):
                return None if not self.stopped else 0

        owned = Owned()

        def lifecycle(config, opts, muse_args, muse_bin, box):
            # A signal right after autostart publishes ownership.
            box["kiro_child"] = owned
            raise muse_gateway._LauncherSignal(15)

        with patch.dict(os.environ, _clean_environment(), clear=True), \
                patch.object(muse_gateway, "_launch_lifecycle", side_effect=lifecycle), \
                patch.object(muse_gateway, "_stop_child", side_effect=lambda child: setattr(child, "stopped", True)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = launch_main([], ["--check-only", "--port", str(_free_port())])
        self.assertEqual(code, 128 + 15)
        self.assertTrue(owned.stopped)

    def test_health_identity_version_and_reuse_warning(self):
        import muse_gateway

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, _clean_environment(), clear=True):
            config = Config(parse_args(["auto", "--no-model-discovery", "--port", "0",
                                        "--log-file", str(Path(directory) / "gateway.log")]))
            server = MuseShimServer(("127.0.0.1", 0), config)
            with serving(server) as base:
                with urllib.request.urlopen(base + "/health", timeout=5) as response:
                    health = json.load(response)
                self.assertNotIn("capture_id", health)
                self.assertEqual(health["version"], muse_gateway._gateway_version())
                self.assertEqual(health["pid"], os.getpid())
                self.assertRegex(health["fingerprint"], r"^[0-9a-f]{64}$")
                err = io.StringIO()
                with redirect_stderr(err):
                    self.assertTrue(muse_gateway._existing_shim_is_compatible(base, "auto", config.claude_transport))
                self.assertEqual(err.getvalue(), "", "matching versions must not warn")
                stale = {**health, "version": "0.0.1"}
                with redirect_stderr(err):
                    muse_gateway._warn_version_mismatch(base, stale)
                self.assertIn("reporting version 0.0.1", err.getvalue())
                self.assertIn(f"pid {os.getpid()}", err.getvalue())
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as exited:
            parse_args(["--version"])
        self.assertEqual(exited.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), f"muse-gateway {muse_gateway._gateway_version()}")

    def test_bare_server_failures_exit_one_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**_clean_environment(), "PYTHONPATH": str(Path(__file__).resolve().parent)}
            blocker = socket.socket()
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            self.addCleanup(blocker.close)
            port = blocker.getsockname()[1]
            cases = {
                "second bind": ["meta", "--port", str(port), "--log-file", str(Path(directory) / "g.log")],
                "bad log file": ["meta", "--port", "0", "--log-file", str(Path(directory) / "missing" / "g.log")],
            }
            for label, argv in cases.items():
                with self.subTest(label=label):
                    result = subprocess.run(
                        [sys.executable, "-m", "muse_gateway", "--env-file", str(_EMPTY_ENV_FILE), *argv],
                        cwd=directory, env=env, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertIn("muse-gateway:", result.stderr)

    def test_missing_protobuf_degrades_to_one_devin_error(self):
        # P1-17 (de-wonder's method): block google.* in a child interpreter.
        root = Path(__file__).resolve().parent
        probe = (
            "import sys, json, threading, urllib.request, urllib.error, argparse, io\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'google' or name.startswith('google.'):\n"
            "            raise ImportError('blocked ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import muse_gateway as g\n"
            "from contextlib import redirect_stdout\n"
            "out = io.StringIO()\n"
            "try:\n"
            "    with redirect_stdout(out):\n"
            "        g.parse_args(['--help'])\n"
            "except SystemExit as e:\n"
            "    print('help', e.code)\n"
            "cfg = g.Config(argparse.Namespace(provider='auto', model_discovery=False, port=0,\n"
            "                devin_api_key='x', log_file=sys.argv[1]))\n"
            "print('doctor', g.run_doctor(cfg) in (0, 1))\n"
            "server = g.MuseShimServer(('127.0.0.1', 0), cfg)\n"
            "threading.Thread(target=server.serve_forever, daemon=True).start()\n"
            "req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/responses',\n"
            "    data=json.dumps({'model': 'devin/swe-1-6', 'input': 'hi'}).encode(),\n"
            "    headers={'Content-Type': 'application/json'})\n"
            "try:\n"
            "    urllib.request.urlopen(req, timeout=5)\n"
            "except urllib.error.HTTPError as e:\n"
            "    print('devin', e.code, json.load(e)['error']['type'])\n"
            "server.shutdown()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            env = {**_clean_environment(), "PYTHONPATH": str(root)}
            result = subprocess.run([sys.executable, "-c", probe, str(Path(directory) / "g.log")],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertIn("help 0", lines)
        self.assertIn("doctor True", lines)
        self.assertIn("devin 503 devin_unavailable", lines)
        self.assertNotIn("Traceback", result.stderr)

    def test_doctor_and_catalog_say_cli_turns_arrive_whole(self):
        # P2-12: the buffered transport is disclosed, never faked as streaming.
        import muse_gateway

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, _clean_environment(), clear=True):
            config = Config(parse_args(["claude", "--no-model-discovery", "--claude-cli", sys.executable,
                                        "--log-file", str(Path(directory) / "g.log")]))
            config.doctor_provider = "claude"
            output = io.StringIO()
            muse_gateway.run_doctor(config, output=output)
        self.assertIn("non-streaming: turns arrive whole", output.getvalue())
        self.assertIn("arrive whole", muse_gateway._catalog_description("claude", 200000))

    def test_bare_server_sigterm_shuts_down_with_a_final_line(self):
        if os.name == "nt":
            self.skipTest("SIGTERM delivery to a child is POSIX-only")
        import signal as signal_module
        import time as time_module

        with tempfile.TemporaryDirectory() as directory:
            port = _free_port()
            env = {**_clean_environment(), "PYTHONPATH": str(Path(__file__).resolve().parent)}
            process = subprocess.Popen(
                [sys.executable, "-m", "muse_gateway", "--env-file", str(_EMPTY_ENV_FILE), "meta",
                 "--port", str(port), "--log-file", str(Path(directory) / "g.log")],
                cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time_module.monotonic() + 20
                while time_module.monotonic() < deadline and not _port_is_busy("127.0.0.1", port):
                    time_module.sleep(0.05)
                process.send_signal(signal_module.SIGTERM)
                _, stderr = process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("stopped on SIGTERM", stderr)
            self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
