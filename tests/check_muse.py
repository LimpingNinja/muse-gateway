"""Explicit offline acceptance by an installed Muse executable, not a mock.

Run: python -m tests.check_muse --muse-bin /path/to/actual/muse-binary
No launcher auto-update, personal config, auth, or paid upstream is needed.
Missing binary, timeout, unexpected retries and inconclusive output FAIL this command;
it is intentionally not a silently skipped unittest/ordinary CI test.

Uses a known Muse model profile with gateway-pinned Devin/Claude/Kiro routes:
checks the real Responses decoder, not TUI picking or provider entitlement.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
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

if __name__ == "__main__" and not __package__:
    sys.exit("Direct script invocation is unsupported. Run from this checkout: python -m tests.check_muse")


def assert_checkout_imports():
    root = Path(__file__).resolve().parents[1]
    expected = {name: root / (name + ".py") for name in (
        "muse_gateway", "muse_providers", "muse_devin", "muse_devin_wire",
        "devin_pb2", "muse_capture", "muse_kiro")}
    expected.update({"tests": root / "tests" / "__init__.py", "tests.peers": root / "tests" / "peers.py",
                     "tests.contracts": root / "tests" / "contracts.py"})
    origins = {"tests.check_muse": str(Path(__file__).resolve())}
    for name, path in expected.items():
        module = sys.modules.get(name)
        spec = importlib.util.find_spec(name) if module is None else None
        origin = getattr(module, "__file__", None) if module is not None else getattr(spec, "origin", None)
        if origin is None or Path(origin).resolve() != path:
            raise RuntimeError(f"Acceptance import {name} must come from this checkout: expected {path}, got {origin}")
        origins[name] = str(Path(origin).resolve())
    return origins


assert_checkout_imports()

from muse_gateway import MuseShimHandler
from tests.contracts import consume
from tests.peers import Peer, anthropic_text, anthropic_tool, chat, envelope, gateway, text_turn, tool_turn


KIRO_ERROR_MESSAGE = "Kiro Gateway returned a provider error; check its configuration and service logs."


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py")) + sorted((root / "tests").glob("*.py"))}


def isolated_environment(directory):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("MUSE_", "META_", "DEVIN_", "CODEX_", "OPENAI_", "ANTHROPIC_", "CLAUDE_", "KIRO_"))
           and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}}
    for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOME"):
        path = directory / key.lower()
        path.mkdir()
        env[key] = str(path)
    env["MUSE_AUTH_PATH"] = str(directory / "auth.json")
    env["NO_PROXY"] = "127.0.0.1,localhost"
    return env


def run_case(binary, directory, peer, prompt, *, reject=False, failure=False, route="devin"):
    origins = assert_checkout_imports()
    env = isolated_environment(directory)
    incoming = []
    downstream = []
    read_body = MuseShimHandler._read_request_body
    def capture_body(handler):
        body = read_body(handler)
        incoming.append(body)
        return body
    setup = MuseShimHandler.setup
    class CaptureWriter:
        def __init__(self, wrapped, captured, handler):
            self.wrapped = wrapped
            self.captured = captured
            self.handler = handler
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
        def write(self, data):
            written = self.wrapped.write(data)
            self.captured["method"] = self.handler.command
            self.captured["path"] = self.handler.path
            self.captured["data"].extend(data[:written])
            return written
    def capture_setup(handler):
        setup(handler)
        captured = {"data": bytearray(), "method": None, "path": None}
        downstream.append(captured)
        handler.wfile = CaptureWriter(handler.wfile, captured, handler)
    model = {"devin": "swe-1-6", "claude": "claude-test", "kiro": "claude-haiku-4.5"}[route]
    with patch.object(MuseShimHandler, "_read_request_body", capture_body), patch.object(
                 MuseShimHandler, "setup", capture_setup), gateway(
                 directory, peer, provider=route, model=model, claude_transport="http",
                 access_token="fixture-claude-token", allow_oauth_host_override=True) as base:
        args = [str(binary), "exec", "--json", "--api-key-stdin", "--provider", "meta",
                "--model", "muse-spark-1.3", "--base-url", base, "--workspace", str(directory),
                "--contributor-mode", "off", "--disable-reminders", "--disable-web-tools",
                "--no-foreign-personal-context", "--no-session-log", "--disable-write", "--disable-shell",
                "--approval-judge", "off", "--max-model-steps", "3", prompt]
        records, code, stderr = drive(args, directory, env, reject=reject)
    (directory / "consumer.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    raw_responses = [(bytes(captured["data"]), captured["method"], captured["path"])
                     for captured in downstream if captured["data"]]
    for index, (raw, _, _) in enumerate(raw_responses, 1):
        (directory / f"downstream-{index}.http").write_bytes(raw)
    responses = [{**capture_response(raw), "method": method, "path": path} for raw, method, path in raw_responses]
    model_responses = [response for response in responses if response["method"] == "POST"
                       and response["path"].split("?", 1)[0] in {"/responses", "/v1/responses"}]
    post_counts = {"responses_incoming": len(incoming), "upstream_inference": len(peer.requests),
                   "responses_captured": len(model_responses)}
    (directory / "result.json").write_text(json.dumps({
        "exit_code": code, "stderr": stderr, "request_paths": peer.request_paths,
        "upstream_request_times": peer.request_times,
        "incoming": incoming, "source_origins": origins,
        "downstream_http": model_responses, "post_counts": post_counts,
        "downstream_http_all": responses,
        "downstream_events": [event for response in responses for event in response["events"]],
        "requests": [request if isinstance(request, dict) else str(request) for request in peer.requests],
    }, indent=2), encoding="utf-8")
    if peer.errors:
        raise RuntimeError("protocol fixture failed: " + repr(peer.errors))
    if len(set(post_counts.values())) != 1:
        raise AssertionError("POST_COUNTS: incoming Responses, captured Responses and upstream inference POSTs disagree")
    if failure and code == 0:
        raise AssertionError("Muse did not naturally exit with failure")
    if code != 0 and not (reject or failure):
        raise RuntimeError(f"Muse exited {code}: {stderr[-1500:]}")
    return records


def capture_response(raw):
    """Read bytes actually written to Muse, independently of the converter."""
    head, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        raise AssertionError("Downstream capture has no complete HTTP headers")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    pairs = [(name.lower(), value.strip()) for name, value in
             (line.split(":", 1) for line in lines[1:])]
    for name in ("content-length", "content-type", "transfer-encoding", "retry-after"):
        if sum(key == name for key, _ in pairs) > 1:
            raise AssertionError("HTTP_CAPTURE: duplicate " + name)
    headers = dict(pairs)
    if "transfer-encoding" in headers:
        raise AssertionError("HTTP_CAPTURE: unsupported transfer encoding")
    length = headers.get("content-length")
    if length is not None and (not length.isascii() or not length.isdigit() or int(length) != len(body)):
        raise AssertionError("HTTP_CAPTURE: Content-Length does not match captured body bytes")
    text = body.decode("utf-8")
    mime = headers.get("content-type", "").split(";", 1)[0].strip()
    return {"status": status, "headers": headers, "header_pairs": pairs, "body": text,
            "json": json.loads(text) if mime == "application/json" else None,
            "events": [json.loads(line[6:]) for line in body.splitlines()
                       if line.startswith(b"data: ") and line[6:] != b"[DONE]"]
                       if mime == "text/event-stream" else []}


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
    origins = assert_checkout_imports()
    before = source_hashes()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muse-bin", required=True, type=Path, help="Actual executable, not an auto-updating shell wrapper")
    parser.add_argument("--evidence-dir", type=Path, help="Retain synthetic captures outside the repository")
    args = parser.parse_args()
    binary = args.muse_bin.resolve(strict=True)
    if binary.suffix.lower() in {".cmd", ".bat", ".ps1", ".sh"}:
        parser.error("supply the actual Muse executable, not its launcher")
    if args.evidence_dir:
        args.evidence_dir = args.evidence_dir.resolve()
        if args.evidence_dir.is_relative_to(Path(__file__).resolve().parents[1]):
            parser.error("evidence must stay outside the repository")
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
        (args.evidence_dir / "source-hashes-before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="gateway-muse-version-") as directory:
        version = subprocess.run([str(binary), "--version"], cwd=directory,
                                 env=isolated_environment(Path(directory)), capture_output=True,
                                 text=True, timeout=5, check=True)
    print("Consumer:", version.stdout.strip())
    print("Checkout:", Path(origins["muse_gateway"]).parent)
    with case_directory(args, "text") as directory:
        peer = Peer([text_turn("OFFLINE_ACCEPTED")])
        records = run_case(binary, Path(directory), peer, "Say OFFLINE_ACCEPTED.")
        assert_accepted(records, "OFFLINE_ACCEPTED")
        if len(peer.requests) != 1:
            raise AssertionError("Devin fixture received extra requests")
        print("Accepted: real Muse decoder, streamed text, successful run terminal")
    with case_directory(args, "claude-text") as directory:
        peer = Peer([anthropic_text()])
        records = run_case(binary, Path(directory), peer, "Say CLAUDE_ACCEPTED.", route="claude")
        assert_accepted(records, "CLAUDE_ACCEPTED")
        if len(peer.requests) != 1:
            raise AssertionError("Claude fixture received extra requests")
        print("Accepted: actual Muse decoder with Claude HTTP converter output")
    with case_directory(args, "kiro-text") as directory:
        peer = Peer([anthropic_text("KIRO_ACCEPTED")])
        records = run_case(binary, Path(directory), peer, "Say KIRO_ACCEPTED.", route="kiro")
        assert_accepted(records, "KIRO_ACCEPTED")
        assert_kiro_requests(peer, 1)
        print("Accepted: actual Muse decoder with Kiro Messages route output (offline)")
    with case_directory(args, "kiro-tool") as directory:
        root = Path(directory)
        (root / "sentinel.txt").write_text("KIRO_LOCAL_TOOL_SENTINEL\n", encoding="utf-8")
        peer = Peer([anthropic_tool(), anthropic_text("KIRO_TOOL_ACCEPTED")])
        records = run_case(binary, root, peer, "Read sentinel.txt and confirm it.", route="kiro")
        assert_accepted(records, "KIRO_TOOL_ACCEPTED")
        assert_kiro_requests(peer, 2)
        first, second = peer.requests
        results = [part for message in second.get("messages", []) for part in message.get("content", [])
                   if isinstance(part, dict) and part.get("type") == "tool_result"]
        if (len(results) != 1 or results[0].get("tool_use_id") != "fixture-messages-call"
                or results[0].get("is_error") or "KIRO_LOCAL_TOOL_SENTINEL" not in json.dumps(results[0].get("content"))):
            raise AssertionError("Kiro route did not replay Muse's actual file-tool result with its original call ID")
        names = {tool["name"] for tool in first.get("tools", [])}
        if "read_file" not in names:
            raise AssertionError("Kiro route did not preserve Muse's file-tool schema")
        if "previous_response_id" in second:
            raise AssertionError("Kiro route forwarded redundant Responses-only state")
        print("Accepted: actual Muse file tool and inline continuation through Kiro (offline)")
    with case_directory(args, "devin-tool") as directory:
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
    with case_directory(args, "kiro-interleaved-tool") as directory:
        root = Path(directory)
        (root / "sentinel.txt").write_text("INTERLEAVED_FILE_RESULT\n", encoding="utf-8")
        turn = interleaved_turn()
        peer = Peer([turn, anthropic_text("INTERLEAVED_ACCEPTED")])
        records = run_case(binary, root, peer, "Read sentinel.txt and confirm it.", route="kiro")
        assert_accepted(records, "BEFORE_TOOLAFTER_TOOLINTERLEAVED_ACCEPTED")
        assert_kiro_requests(peer, 2)
        incoming = json.loads((root / "result.json").read_text(encoding="utf-8"))["incoming"]
        if len(incoming) != 2:
            raise AssertionError("Muse did not send exactly two Responses requests")
        replay = incoming[1].get("input", [])
        messages = [item for item in replay if isinstance(item, dict) and item.get("role") == "assistant"]
        if len(messages) != 1 or messages[0].get("content") != "BEFORE_TOOLAFTER_TOOL":
            raise AssertionError("Muse's actual Responses input did not retain both texts in one message")
        outputs = [item for item in replay if isinstance(item, dict) and item.get("type") == "function_call_output"]
        if (len(outputs) != 1 or outputs[0].get("call_id") != "fixture-messages-call"
                or "INTERLEAVED_FILE_RESULT" not in json.dumps(outputs[0].get("output"))):
            raise AssertionError("Muse's actual Responses input did not retain its file result")
        second = peer.requests[1]
        assistant = [part for message in second.get("messages", []) if message.get("role") == "assistant"
                     for part in message.get("content", []) if isinstance(part, dict)]
        texts = [part.get("text", "") for part in assistant if part.get("type") == "text"]
        if texts != ["BEFORE_TOOLAFTER_TOOL"]:
            raise AssertionError("Muse's actual second request lost or duplicated interleaved text: " + repr(texts))
        calls = [part for part in assistant if part.get("type") == "tool_use"]
        if len(calls) != 1 or calls[0].get("id") != "fixture-messages-call" or calls[0].get("input") != {"path": "sentinel.txt"}:
            raise AssertionError("Interleaved continuation changed its tool call")
        results = [part for message in second.get("messages", []) for part in message.get("content", [])
                   if isinstance(part, dict) and part.get("type") == "tool_result"]
        if (len(results) != 1 or results[0].get("tool_use_id") != "fixture-messages-call"
                or results[0].get("is_error") or "INTERLEAVED_FILE_RESULT" not in json.dumps(results[0].get("content"))):
            raise AssertionError("Interleaved continuation did not contain the actual file result")
        print("Accepted: actual Muse second request retains BEFORE_TOOL + AFTER_TOOL and actual file result")
    with case_directory(args, "kiro-cache-usage") as directory:
        body = anthropic_text("CACHE_ACCEPTED").replace(
            b'"input_tokens": 4, "output_tokens": 0',
            b'"input_tokens": 4, "output_tokens": 0, "cache_read_input_tokens": 9, "cache_creation_input_tokens": 3')
        peer = Peer([body])
        records = run_case(binary, Path(directory), peer, "Say CACHE_ACCEPTED.", route="kiro")
        assert_accepted(records, "CACHE_ACCEPTED")
        assert_kiro_requests(peer, 1)
        downstream = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))["downstream_events"]
        assert_downstream_cache_usage(downstream)
        print("Accepted: actual Muse decoder accepts preserved cache usage details")
    for partial in (False, True):
        for route in ("kiro", "devin", "claude"):
            with case_directory(args, f"{route}-failure-{'partial' if partial else 'start'}") as directory:
                if route == "kiro":
                    body = failure_turn(partial)
                    message = KIRO_ERROR_MESSAGE
                elif route == "claude":
                    # EOF exercises the shared failed terminal without the separate native error event.
                    body = anthropic_text("PARTIAL").split(b"event: message_delta")[0] if partial else anthropic_text().split(b"event: content_block_start")[0]
                    message = "Claude stream ended before message_stop."
                else:
                    body = (chat("PARTIAL") if partial else b"") + envelope(
                        b'{"error":{"code":"unavailable","message":"synthetic consumer failure"}}', 2)
                    message = "synthetic consumer failure"
                peer = Peer([body])
                records = run_case(binary, Path(directory), peer, "Say ACCEPTED.", failure=True, route=route)
                assert_failed(records, peer, message, "PARTIAL" if partial else "")
                if route == "kiro":
                    assert_kiro_requests(peer, 1)
                print(f"Rejected as stream_error with natural failed exit: {route}, partial={partial}")
    for route in ("kiro", "devin"):
        with case_directory(args, route + "-incomplete") as directory:
            body = anthropic_text("PARTIAL").replace(b'"end_turn"', b'"max_tokens"') if route == "kiro" else chat("PARTIAL", stop=3) + envelope(b"{}", 2)
            peer = Peer([body])
            records = run_case(binary, Path(directory), peer, "Say PARTIAL.", route=route)
            assert_incomplete(records, peer)
            print(f"Observed response.incomplete, rejected as full completion: {route}")
    with case_directory(args, "kiro-partial-tool") as directory:
        body = anthropic_tool(arguments='{"path":').replace(b'"stop_reason": "tool_use"', b'"stop_reason": "max_tokens"')
        peer = Peer([body])
        records = run_case(binary, Path(directory), peer, "Read sentinel.txt.", route="kiro")
        assert_incomplete(records, peer, expected="")
        assert_kiro_requests(peer, 1)
        if any(record.get("payload_type") == "tool.result" for record in records):
            raise AssertionError("Muse dispatched an incomplete tool call")
        print("Observed partial tool arguments: incomplete, no tool dispatch and no retry")
    with case_directory(args, "kiro-http429-once") as directory:
        root = Path(directory)
        payload = {"type": "error", "error": {"type": "rate_limit_error", "code": "rate_limit_exceeded",
                   "param": "messages.0.content", "message": "PRIVATE_HTTP_PROSE"}, "request_id": "req_body"}
        peer = Peer([(429, json.dumps(payload).encode(), {"Retry-After": "4", "request-id": "req_header"}),
                     anthropic_text("RETRY_ACCEPTED")])
        records = run_case(binary, root, peer, "Say RETRY_ACCEPTED.", route="kiro")
        result = read_result(root)
        if len(result["downstream_http"]) != 2 or len(result["incoming"]) != 2:
            raise AssertionError("429 must cause exactly two native Responses requests")
        assert_http_error(result["downstream_http"][0], status=429, kind="rate_limit_error", code="rate_limit_exceeded",
                          category="rate_limit", transient=True, retry_after="4", upstream={
                              "type": "rate_limit_error", "code": "rate_limit_exceeded", "param": "messages.0.content",
                              "body_request_id": "req_body", "header_request_id": "req_header", "request_id_conflict": True})
        final = consume(result["downstream_http"][1]["body"].encode())
        if final["status"] != "completed":
            raise AssertionError("429 retry did not end in response.completed")
        assert_retry_completed(records, peer, "RETRY_ACCEPTED", retry_delay_ms=4000, min_retry_spacing=3.8)
        assert_kiro_requests(peer, 2)
        if "PRIVATE_HTTP_PROSE" in result["downstream_http"][0]["body"]:
            raise AssertionError("HTTP error echoed upstream prose")
        print("Accepted: HTTP429 Retry-After: 4 honored (native 4000ms and >=3.8s POST spacing); one retry completes")
    with case_directory(args, "kiro-http401") as directory:
        root = Path(directory)
        payload = {"type": "error", "error": {"type": "authentication_error"}, "request_id": "req_auth"}
        peer = Peer([(401, json.dumps(payload).encode(), {"request-id": "req_auth"})])
        records = run_case(binary, root, peer, "Say ACCEPTED.", failure=True, route="kiro")
        result = read_result(root)
        if len(result["downstream_http"]) != 1:
            raise AssertionError("401 must produce exactly one downstream response")
        assert_http_error(result["downstream_http"][0], status=401, kind="authentication_error", category="authentication",
                          transient=False, upstream={"type": "authentication_error", "body_request_id": "req_auth",
                                                    "header_request_id": "req_auth"})
        assert_http_failed(records, peer, result["downstream_http"][0])
        assert_kiro_requests(peer, 1)
        print("Rejected: HTTP401 naturally fails without retry; native credential-label wording is not asserted")
    with case_directory(args, "kiro-http400-context") as directory:
        root = Path(directory)
        payload = {"type": "error", "error": {"type": "invalid_request_error", "code": "context_length_exceeded",
                   "message": "PRIVATE_HTTP_PROSE"}, "request_id": "req_context"}
        peer = Peer([(400, json.dumps(payload).encode(), {"request-id": "req_context"})])
        records = run_case(binary, root, peer, "Say ACCEPTED.", failure=True, route="kiro")
        result = read_result(root)
        if result["post_counts"] != {"responses_incoming": 1, "upstream_inference": 1, "responses_captured": 1}:
            raise AssertionError("Context rejection must produce exactly one model POST at all three boundaries")
        response = result["downstream_http"][0]
        assert_http_error(response, status=400, kind="invalid_request_error", code="context_length_exceeded",
                          category="context", transient=False, upstream={"type": "invalid_request_error",
                              "code": "context_length_exceeded", "body_request_id": "req_context", "header_request_id": "req_context"})
        assert_http_failed(records, peer, response, status=400, code="context_length_exceeded", category="context")
        assert_kiro_requests(peer, 1)
        if "PRIVATE_HTTP_PROSE" in response["body"]:
            raise AssertionError("Context rejection echoed upstream prose")
        print("Rejected: HTTP400 context_length_exceeded preserves evidence; natural native failure, one POST, no retry/decode")
    typed = [("authentication_error", None, "authentication", False), ("overloaded_error", None, "overload", True)]
    for kind, code, category, transient in typed + [("future_error", "vendor.future_17", "unknown", None)]:
        for partial in ((False, True) if category != "unknown" else (True,)):
            with case_directory(args, f"kiro-typed-{category}-{'partial' if partial else 'lifecycle'}") as directory:
                root = Path(directory)
                peer = Peer([failure_turn(partial, kind=kind, code=code)])
                records = run_case(binary, root, peer, "Say ACCEPTED.", failure=True, route="kiro")
                result = read_result(root)
                assert_typed_failed_wire(result, partial="PARTIAL" if partial else "", kind=kind, code=code,
                                         category=category, transient=transient)
                assert_failed(records, peer, KIRO_ERROR_MESSAGE, "PARTIAL" if partial else "")
                assert_kiro_requests(peer, 1)
                print(f"Rejected: typed SSE {category}, partial={partial}; preserved evidence, natural response.failed, no retry/decode")
    with case_directory(args, "kiro-typed-private-tool") as directory:
        root = Path(directory)
        (root / "sentinel.txt").write_text("MUST_NOT_READ\n", encoding="utf-8")
        peer = Peer([failure_turn(True, private_tool=True)])
        records = run_case(binary, root, peer, "Read sentinel.txt.", failure=True, route="kiro")
        result = read_result(root)
        assert_typed_failed_wire(result, partial="PARTIAL", kind="overloaded_error", category="overload", transient=True)
        assert_failed(records, peer, KIRO_ERROR_MESSAGE, "PARTIAL")
        assert_kiro_requests(peer, 1)
        if any(record.get("payload_type", "").startswith("tool.") for record in records):
            raise AssertionError("Muse dispatched a private unfinished tool call")
        if any(token in result["downstream_http"][0]["body"] for token in ("call_private", "PRIVATE_ARGUMENT")):
            raise AssertionError("Private unfinished tool bytes escaped downstream")
        print("Rejected: unfinished private tool + typed failure preserves partial text without tool dispatch or retry")
    import muse_kiro
    original_kiro = muse_kiro.stream_events
    def without_kiro_numbers(*args, **kwargs):
        for event, payload in original_kiro(*args, **kwargs):
            payload = dict(payload)
            payload.pop("sequence_number", None)
            yield event, payload
    with case_directory(args, "kiro-missing-sequence") as directory:
        peer = Peer([anthropic_text("KIRO_ACCEPTED")])
        with patch.object(muse_kiro, "stream_events", without_kiro_numbers):
            records = run_case(binary, Path(directory), peer, "Say KIRO_ACCEPTED.", reject=True, route="kiro")
        assert_rejected(records, peer)
        print("Rejected: same Kiro output without sequence_number (Muse error_kind=decode)")
    import muse_devin
    original = muse_devin.DevinStreamState._number_events
    def without_numbers(state, events):
        events = original(state, events)
        for _, payload in events:
            payload.pop("sequence_number", None)
        return events
    with case_directory(args, "devin-missing-sequence") as directory:
        peer = Peer([text_turn("OFFLINE_ACCEPTED")])
        with patch.object(muse_devin.DevinStreamState, "_number_events", without_numbers):
            records = run_case(binary, Path(directory), peer, "Say OFFLINE_ACCEPTED.", reject=True)
        assert_rejected(records, peer)
        print("Rejected: same converter output without sequence_number (Muse error_kind=decode)")
    after = source_hashes()
    if args.evidence_dir:
        (args.evidence_dir / "source-hashes-after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    if before != after:
        raise AssertionError("Acceptance source changed during the run; results require a stable rerun")
    print("All 27 native acceptance gates passed (26 existing + HTTP400 context-code gate); source hashes unchanged")
    return 0


@contextmanager
def case_directory(args, name):
    if args.evidence_dir:
        directory = args.evidence_dir / name
        directory.mkdir()  # Refuse to overwrite earlier evidence.
        yield directory
    else:
        with tempfile.TemporaryDirectory(prefix="gateway-muse-" + name + "-") as directory:
            yield Path(directory)


def interleaved_turn():
    frames = anthropic_tool().decode().strip().split("\n\n")
    events = []
    for frame in frames:
        value = json.loads(frame.split("\ndata: ", 1)[1])
        if "index" in value:
            value["index"] = 1
        events.append(value)
    def text(index, value):
        return [
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": value}},
            {"type": "content_block_stop", "index": index},
        ]
    events[1:1] = text(0, "BEFORE_TOOL")
    events[-2:-2] = text(2, "AFTER_TOOL")
    return b"".join(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode() for event in events)


def failure_turn(partial, *, kind="overloaded_error", code=None, private_tool=False):
    body = anthropic_text("PARTIAL")
    prefix = body.split(b"event: message_delta")[0] if partial else body.split(b"event: content_block_start")[0]
    if private_tool:
        # Keep only start + deltas, with no block stop or executable JSON.
        tool = anthropic_tool(arguments='{"path":"sentinel.txt","PRIVATE_ARGUMENT":')
        tool = b"event: content_block_start" + tool.split(b"event: content_block_start", 1)[1]
        tool = tool.split(b"event: content_block_stop", 1)[0]
        tool = tool.replace(b'"index": 0', b'"index": 1').replace(b"fixture-messages-call", b"call_private")
        prefix = body.split(b"event: message_delta")[0] + tool
    details = {"type": kind, "param": "messages.0.content", "message": "PRIVATE_SSE_PROSE"}
    if code is not None:
        details["code"] = code
    payload = {"type": "error", "error": details, "request_id": "req_sse"}
    return prefix + ("event: error\ndata: " + json.dumps(payload) + "\n\n").encode()


def read_result(directory):
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def assert_error_evidence(error, *, kind, category, transient, source, status, exposure, upstream, code=None, retry_after=None):
    if error.get("type") != "kiro_upstream_error" or error.get("code") != (code or kind):
        raise AssertionError("KIRO_ERROR_CODE: typed error was flattened or code precedence changed")
    if error.get("message") != KIRO_ERROR_MESSAGE:
        raise AssertionError("KIRO_ERROR_MESSAGE: compatible generic provider message changed")
    metadata = error.get("metadata", {})
    expected = {"provider": "kiro", "layer": "kiro_gateway", "source": source,
                "phase": "before_response" if exposure == "none" else "stream", "upstream_status": status,
                "category": category, "output_exposure": exposure, "upstream": upstream}
    if any(key not in metadata or metadata[key] != value for key, value in expected.items()) or (
            "transient" not in metadata or metadata["transient"] is not transient):
        raise AssertionError("KIRO_ERROR_METADATA: classification or bounded upstream evidence changed")
    retry = {"value": retry_after, "kind": "delay-seconds"} if retry_after is not None else None
    if metadata.get("retry_after") != retry:
        raise AssertionError("KIRO_RETRY_AFTER: retry metadata missing or changed")


def assert_http_error(response, *, status, kind, category, transient, upstream, code=None, retry_after=None):
    if (response.get("status") != status or response.get("events")
            or response.get("headers", {}).get("content-type", "").split(";", 1)[0] != "application/json"):
        raise AssertionError("KIRO_HTTP_ERROR: expected a readable error JSON response at the original status")
    if response["headers"].get("retry-after") != retry_after:
        raise AssertionError("KIRO_RETRY_AFTER: downstream HTTP retry header missing or changed")
    error = (response.get("json") or {}).get("error", {})
    if not isinstance(error.get("request_id"), str) or not error["request_id"] or error["request_id"] in {
            upstream.get("body_request_id"), upstream.get("header_request_id")}:
        raise AssertionError("KIRO_REQUEST_ID: gateway request ID missing or replaced with upstream provenance")
    assert_error_evidence(error, kind=kind, code=code, category=category, transient=transient, source="http",
                          status=status, exposure="none", upstream=upstream, retry_after=retry_after)


def assert_typed_failed_wire(result, *, partial, kind, category, transient, code=None):
    responses = result.get("downstream_http", [])
    if len(responses) != 1 or responses[0].get("status") != 200:
        raise AssertionError("KIRO_FAILED_WIRE: expected one already-open HTTP200 stream")
    response = responses[0]
    final = consume(response["body"].encode())  # Independent IDs, sequences and terminal checks.
    events = response["events"]
    if final["status"] != "failed":
        raise AssertionError("KIRO_FAILED_WIRE: failure became generic success or incomplete")
    deltas = "".join(event["delta"] for event in events if event.get("type") == "response.output_text.delta")
    text = "".join(part.get("text", "") for item in final["output"] for part in item.get("content", [])
                   if part.get("type") == "output_text")
    if deltas != partial or text != partial or any(item.get("type") == "function_call" for item in final["output"]):
        raise AssertionError("KIRO_FAILED_WIRE: partial output changed or private tool escaped")
    if any(event.get("type", "").startswith("response.function_call") or event.get("item", {}).get("type") == "function_call"
           for event in events) or "PRIVATE_SSE_PROSE" in response["body"]:
        raise AssertionError("KIRO_FAILED_WIRE: private arguments or upstream prose escaped")
    upstream = {"type": kind, "param": "messages.0.content", "body_request_id": "req_sse"}
    if code is not None:
        upstream["code"] = code
    assert_error_evidence(final.get("error", {}), kind=kind, code=code, category=category, transient=transient,
                          source="sse", status=200, exposure="content" if partial else "lifecycle", upstream=upstream)


def assert_retry_completed(records, peer, expected, *, retry_delay_ms=None, min_retry_spacing=None):
    terminals = [record["payload"] for record in records if record.get("payload_type", "").startswith("run.terminal.")]
    if len(terminals) != 1 or terminals[0].get("terminal") != "completed" or terminals[0].get("text") != expected:
        raise AssertionError("KIRO_NATIVE_RETRY: native retry did not naturally complete with the exact text")
    statuses = model_statuses(records)
    opened = [status for status in statuses if status.get("phase") == "opening_stream"]
    retries = [status for status in statuses if status.get("phase") == "retry_scheduled"]
    succeeded = [status for status in statuses if status.get("phase") == "stream_succeeded"]
    def attempts(statuses):
        return [facet.get("attempt") for status in statuses for facet in status.get("facets", [])
                if facet.get("kind") == "external_attempt"]
    if (len(peer.requests) != 2 or len(retries) != 1 or attempts(opened) != [1, 2]
            or attempts(retries) != [1] or attempts(succeeded) != [2]
            or any(facet.get("error_kind") in {"decode", "setup"} for status in statuses for facet in status.get("facets", []))
            or any(wire_terminals(status) != ["response.completed"] for status in succeeded)):
        raise AssertionError("KIRO_NATIVE_RETRY: expected exactly one native retry, two attempts and response.completed without decode/setup errors")
    reported = [facet.get("retry_delay_ms") for status in retries for facet in status.get("facets", [])
                if facet.get("kind") == "external_attempt"]
    if retry_delay_ms is not None and reported != [retry_delay_ms]:
        raise AssertionError("KIRO_NATIVE_RETRY: native reported delay does not honor Retry-After")
    times = getattr(peer, "request_times", [])
    if min_retry_spacing is not None and (len(times) != 2 or times[1] - times[0] < min_retry_spacing):
        raise AssertionError("KIRO_NATIVE_RETRY: upstream POST spacing does not honor Retry-After")


def assert_http_failed(records, peer, response, *, status=401, code="authentication_error", category="authentication"):
    # Muse's HTTP rejection path can exit directly after opening_stream; it does not always
    # emit stream_failed. Require the observed HTTP rejection separately.
    error = (response.get("json") or {}).get("error", {})
    metadata = error.get("metadata", {})
    if (response.get("status") != status or response.get("events") or error.get("code") != code
            or metadata.get("source") != "http" or metadata.get("upstream_status") != status
            or metadata.get("category") != category):
        raise AssertionError(f"KIRO_NATIVE_HTTP: missing observed typed HTTP{status} rejection")
    terminals = [record["payload"] for record in records if record.get("payload_type", "").startswith("run.terminal.")]
    statuses = model_statuses(records)
    failed = [status for status in statuses if status.get("phase") == "stream_failed"]
    opened = [status for status in statuses if status.get("phase") == "opening_stream"]
    errors = [facet.get("error_kind") for status in failed for facet in status.get("facets", [])
              if facet.get("kind") == "external_attempt"]
    if (len(terminals) != 1 or terminals[0].get("terminal") != "failed" or terminals[0].get("text") != ""
            or not terminals[0].get("reason") or len(peer.requests) != 1 or len(opened) != 1 or len(failed) > 1
            or any(status.get("phase") in {"retry_scheduled", "stream_succeeded"} for status in statuses)):
        raise AssertionError("KIRO_NATIVE_HTTP: expected natural HTTP failure without retry or success")
    attempts = [facet.get("attempt") for status in statuses for facet in status.get("facets", [])
                if facet.get("kind") == "external_attempt"]
    all_errors = [facet.get("error_kind") for status in statuses for facet in status.get("facets", [])
                  if facet.get("kind") == "external_attempt" and "error_kind" in facet]
    if (failed and not errors or any(error in {None, "decode", "setup", "configuration"} for error in all_errors)
            or not attempts or any(attempt != 1 for attempt in attempts)
            or any(wire_terminals(status) != [None] for status in failed)):
        raise AssertionError("KIRO_NATIVE_HTTP: decode/setup failure cannot count as an HTTP rejection")


def assert_downstream_cache_usage(events):
    completed = [event["response"] for event in events if event.get("type") == "response.completed"]
    expected = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
                "input_tokens_details": {"cached_tokens": 9, "anthropic_cache_creation_input_tokens": 3}}
    if len(completed) != 1 or completed[0].get("usage") != expected:
        raise AssertionError("Downstream cache counters or unchanged token totals were not preserved")


def assert_kiro_requests(peer, count):
    if len(peer.requests) != count or peer.request_paths != ["/v1/messages"] * count:
        raise AssertionError("Kiro acceptance did not use exactly the expected Messages requests")
    for headers, body in zip(peer.request_headers, peer.requests):
        if headers.get("Authorization") != "Bearer fixture-kiro-proxy-token":
            raise AssertionError("Kiro route did not use its independent proxy key")
        if body.get("model") != "claude-haiku-4.5" or body.get("thinking") != {"type": "disabled"}:
            raise AssertionError("Kiro route changed its selected model or did not disable synthetic thinking")


def assert_rejected(records, peer):
    if not any(decode_rejected(record) for record in records):
        raise AssertionError("Missing-sequence stream did not produce an explicit Muse decode rejection")
    if len(peer.requests) != 1 or any(record.get("payload_type", "").startswith("run.terminal.completed") for record in records):
        raise AssertionError("Negative probe was accepted or continued unexpectedly")


def assert_accepted(records, expected):
    terminals = [record["payload"] for record in records if record.get("payload_type", "").startswith("run.terminal.")]
    if len(terminals) != 1 or terminals[0].get("terminal") != "completed" or terminals[0].get("text") != expected:
        raise AssertionError("Muse did not complete with the exact expected output")
    statuses = model_statuses(records)
    attempts = [facet for status in statuses for facet in status.get("facets", []) if facet.get("kind") == "external_attempt"]
    if not attempts or any(attempt.get("attempt") != 1 for attempt in attempts):
        raise AssertionError("Muse retried or did not report its provider attempt")
    succeeded = [status for status in statuses if status.get("phase") == "stream_succeeded"]
    opened = [status for status in statuses if status.get("phase") == "opening_stream"]
    if not succeeded or len(succeeded) != len(opened) or any(
            status.get("phase") in {"retry_scheduled", "stream_failed"} for status in statuses):
        raise AssertionError("Muse did not succeed once for every model request without retries")
    for status in succeeded:
        if wire_terminals(status) != ["response.completed"]:
            raise AssertionError("Successful full completion requires response.completed for every model request")


def model_statuses(records):
    statuses = [record.get("payload", {}).get("event", {}).get("details", {}) for record in records
                if record.get("payload_type") == "task.lifecycle.status"]
    return [status for status in statuses if any(facet.get("kind") == "external_attempt"
            and facet.get("operation") == "model.response" for facet in status.get("facets", []))]


def wire_terminals(status):
    return [facet.get("detail", {}).get("stream", {}).get("last_wire_event_type")
            for facet in status.get("facets", []) if facet.get("kind") == "producer"]


def assert_failed(records, peer, message, partial):
    terminals = [record["payload"] for record in records if record.get("payload_type", "").startswith("run.terminal.")]
    if (len(terminals) != 1 or terminals[0].get("terminal") != "failed"
            or terminals[0].get("text") != partial or message not in str(terminals[0].get("reason"))):
        raise AssertionError("Muse did not naturally fail with the gateway message and expected partial output")
    statuses = model_statuses(records)
    failed = [status for status in statuses if status.get("phase") == "stream_failed"]
    attempts = [facet for status in statuses for facet in status.get("facets", []) if facet.get("kind") == "external_attempt"]
    if (len(peer.requests) != 1 or len([s for s in statuses if s.get("phase") == "opening_stream"]) != 1
            or len(failed) != 1 or any(s.get("phase") in {"retry_scheduled", "stream_succeeded"} for s in statuses)
            or any(attempt.get("attempt") != 1 for attempt in attempts)):
        raise AssertionError("Failure regression retried, skipped inference or reported success")
    errors = [facet.get("error_kind") for facet in failed[0].get("facets", []) if facet.get("kind") == "external_attempt"]
    if errors != ["stream_error"] or wire_terminals(failed[0]) != ["response.failed"]:
        raise AssertionError("Failure regression must be stream_error from response.failed, never decode/setup failure")


def assert_incomplete(records, peer, expected="PARTIAL"):
    statuses = model_statuses(records)
    succeeded = [status for status in statuses if status.get("phase") == "stream_succeeded"]
    if (len(peer.requests) != 1 or len(succeeded) != 1 or wire_terminals(succeeded[0]) != ["response.incomplete"]
            or any(s.get("phase") in {"retry_scheduled", "stream_failed"} for s in statuses)):
        raise AssertionError("Incomplete outcome was not observed exactly once without retries")
    try:
        assert_accepted(records, expected)
    except AssertionError as error:
        if "requires response.completed" not in str(error):
            raise AssertionError("Incomplete control failed for a reason other than its wire terminal") from error
    else:
        raise AssertionError("response.incomplete was counted as successful full completion")


def decode_rejected(record):
    if record.get("payload_type") != "task.lifecycle.status":
        return False
    details = record.get("payload", {}).get("event", {}).get("details", {})
    return details.get("phase") == "retry_scheduled" and any(
        facet.get("kind") == "external_attempt" and facet.get("error_kind") == "decode"
        for facet in details.get("facets", []))


if __name__ == "__main__":
    sys.exit(main())
