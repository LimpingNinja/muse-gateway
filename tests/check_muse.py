"""Explicit offline acceptance by an installed Muse executable, not a mock.

Run: python -m tests.check_muse --muse-bin /path/to/actual/muse-binary
No launcher auto-update, personal config, auth, or paid upstream is needed.
Missing binary, timeout, retries and inconclusive output FAIL this command;
it is intentionally not a silently skipped unittest/ordinary CI test.

Uses a known Muse model profile with gateway-pinned Devin/Claude HTTP routes:
checks the real Responses decoder, not TUI picking or provider entitlement.
"""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

from tests.peers import Peer, anthropic_text, gateway, text_turn, tool_turn


def isolated_environment(directory):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("MUSE_", "META_", "DEVIN_", "CODEX_", "OPENAI_", "ANTHROPIC_", "CLAUDE_"))
           and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "APPDATA", "LOCALAPPDATA"):
        path = directory / key.lower()
        path.mkdir()
        env[key] = str(path)
    env["MUSE_AUTH_PATH"] = str(directory / "auth.json")
    env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


def run_case(binary, directory, peer, prompt, *, reject=False, route="devin"):
    env = isolated_environment(directory)
    model = "swe-1-6" if route == "devin" else "claude-test"
    with gateway(directory, peer, provider=route, model=model, claude_transport="http",
                 access_token="fixture-claude-token", allow_oauth_host_override=True) as base:
        args = [str(binary), "exec", "--json", "--api-key-stdin", "--provider", "meta",
                "--model", "muse-spark-1.3", "--base-url", base, "--workspace", str(directory),
                "--contributor-mode", "off", "--disable-reminders", "--disable-web-tools",
                "--no-foreign-personal-context", "--no-session-log", "--disable-write", "--disable-shell",
                "--approval-judge", "off", "--max-model-steps", "3", prompt]
        records, code, stderr = drive(args, directory, env, reject=reject)
    if peer.errors:
        raise RuntimeError("protocol fixture failed: " + repr(peer.errors))
    if code != 0 and not reject:
        raise RuntimeError(f"Muse exited {code}: {stderr[-1500:]}")
    return records


def drive(args, directory, env, *, reject):
    lines = queue.Queue()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   cwd=directory, env=env, text=True, encoding="utf-8")
        def read_lines():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        records = []
        deadline = time.monotonic() + 25
        try:
            process.stdin.write("fixture-client-token\n")
            process.stdin.close()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Muse acceptance timed out (not a detected rejection)")
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError("Muse acceptance timed out (not a detected rejection)") from None
                if line is None:
                    break
                record = json.loads(line)
                records.append(record)
                if reject and decode_rejected(record):
                    break
            if reject and process.poll() is None:
                process.terminate()
            process.wait(timeout=3)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            reader.join(timeout=3)
            process.stdout.close()
        errors.seek(0)
        return records, process.returncode, errors.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muse-bin", required=True, type=Path, help="Actual executable, not an auto-updating shell wrapper")
    args = parser.parse_args()
    binary = args.muse_bin.resolve(strict=True)
    if binary.suffix.lower() in {".cmd", ".bat", ".ps1", ".sh"}:
        parser.error("supply the actual Muse executable, not its launcher")
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=5, check=True)
    print("Consumer:", version.stdout.strip())
    with tempfile.TemporaryDirectory(prefix="gateway-muse-acceptance-") as directory:
        peer = Peer([text_turn("OFFLINE_ACCEPTED")])
        records = run_case(binary, Path(directory), peer, "Say OFFLINE_ACCEPTED.")
        assert_accepted(records, "OFFLINE_ACCEPTED")
        print("Accepted: real Muse decoder, streamed text, successful run terminal")
    with tempfile.TemporaryDirectory(prefix="gateway-muse-claude-") as directory:
        peer = Peer([anthropic_text()])
        records = run_case(binary, Path(directory), peer, "Say CLAUDE_ACCEPTED.", route="claude")
        assert_accepted(records, "CLAUDE_ACCEPTED")
        if len(peer.requests) != 1:
            raise AssertionError("Claude fixture received extra requests")
        print("Accepted: actual Muse decoder with Claude HTTP converter output")
    with tempfile.TemporaryDirectory(prefix="gateway-muse-tool-") as directory:
        root = Path(directory)
        (root / "sentinel.txt").write_text("LOCAL_TOOL_SENTINEL\n", encoding="utf-8")
        peer = Peer([tool_turn("read_file", '{"path":"sentinel.txt"}'), text_turn("TOOL_ACCEPTED")])
        records = run_case(binary, root, peer, "Read sentinel.txt and confirm it.")
        assert_accepted(records, "TOOL_ACCEPTED")
        if len(peer.requests) != 2:
            raise AssertionError("Muse did not continue after the tool call")
        result = next((prompt for prompt in peer.requests[1].chat_message_prompts if prompt.tool_call_id), None)
        if result is None or result.tool_result_is_error or "LOCAL_TOOL_SENTINEL" not in result.prompt:
            raise AssertionError("Muse did not actually execute the file tool and replay its result")
        print("Accepted: actual Muse file tool execution and second model turn")
    import muse_devin
    original = muse_devin.DevinStreamState._number_events
    def without_numbers(state, events):
        events = original(state, events)
        for _, payload in events:
            payload.pop("sequence_number", None)
        return events
    with tempfile.TemporaryDirectory(prefix="gateway-muse-negative-") as directory:
        peer = Peer([text_turn("OFFLINE_ACCEPTED")])
        with patch.object(muse_devin.DevinStreamState, "_number_events", without_numbers):
            records = run_case(binary, Path(directory), peer, "Say OFFLINE_ACCEPTED.", reject=True)
        if not any(decode_rejected(record) for record in records):
            raise AssertionError("Missing-sequence stream did not produce an explicit Muse decode rejection")
        if len(peer.requests) != 1 or any(record.get("payload_type", "").startswith("run.terminal.completed") for record in records):
            raise AssertionError("Negative probe was accepted or continued unexpectedly")
        print("Rejected: same converter output without sequence_number (Muse error_kind=decode)")
    return 0


def assert_accepted(records, expected):
    terminals = [record["payload"] for record in records if record.get("payload_type", "").startswith("run.terminal.")]
    if len(terminals) != 1 or terminals[0].get("terminal") != "completed" or terminals[0].get("text") != expected:
        raise AssertionError("Muse did not complete with the exact expected output")
    statuses = [record["payload"].get("event", {}).get("details", {}) for record in records
                if record.get("payload_type") == "task.lifecycle.status"]
    attempts = [facet for status in statuses for facet in status.get("facets", []) if facet.get("kind") == "external_attempt"]
    if not attempts or any(attempt.get("attempt") != 1 for attempt in attempts):
        raise AssertionError("Muse retried or did not report its provider attempt")
    if not any(status.get("phase") == "stream_succeeded" for status in statuses):
        raise AssertionError("Muse did not report stream_succeeded")


def decode_rejected(record):
    if record.get("payload_type") != "task.lifecycle.status":
        return False
    details = record.get("payload", {}).get("event", {}).get("details", {})
    return details.get("phase") == "retry_scheduled" and any(
        facet.get("kind") == "external_attempt" and facet.get("error_kind") == "decode"
        for facet in details.get("facets", []))


if __name__ == "__main__":
    sys.exit(main())
