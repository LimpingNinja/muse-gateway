"""Challenge behavioral tests with selected historical bugs, in isolated copies.

Run: python -m tests.check_regressions
This is a small fault campaign, NOT a general mutation score. A baseline must
pass. Syntax/import/setup errors, skips and timeouts never count as detection.
A source refactor that makes a mutation inapplicable fails visibly for review.
"""

from dataclasses import dataclass
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PREFIX = "tests.test_gateway_contracts.GatewayContracts."
KIRO_PREFIX = "test_kiro.KiroContracts."


@dataclass(frozen=True)
class Fault:
    name: str
    module: str
    edits: tuple
    test: str
    diagnostic: str
    prefix: str = PREFIX

    @property
    def test_name(self):
        return self.prefix + self.test


FAULTS = (
    Fault("missing sequence numbers", "muse_devin.py", ((
        'payload["sequence_number"] = self.sequence_number', 'payload.pop("sequence_number", None)'),),
        "test_text_and_nonstream_have_equivalent_complete_output", "STREAM_SEQUENCE"),
    Fault("subtract context fallback", "muse_devin.py", ((
        'max_tokens if max_tokens > 0 else 200000', 'max(0, max_tokens - 200000)'),),
        "test_catalog_has_usable_context_not_context_minus_fallback", "CATALOG_CONTEXT"),
    Fault("lose omitted tool ID", "muse_devin.py", ((
        'wire_id = call.id or self.active_tool_id', 'wire_id = call.id'),),
        "test_tool_loop_replays_exact_arguments_ids_and_large_result", "TOOL_REPLAY"),
    Fault("append cumulative arguments", "muse_devin.py", ((
        'call.arguments_json if call.arguments_json.startswith(previous)', 'call.arguments_json if False'),),
        "test_tool_loop_replays_exact_arguments_ids_and_large_result", "TOOL_REPLAY"),
    Fault("stream unrepaired argument bytes", "muse_devin.py", ((
        '"item_id": item["id"], "output_index": index, "delta": arguments,',
        '"item_id": item["id"], "output_index": index, "delta": block["arguments"],'),),
        "test_repaired_tool_arguments_match_every_stream_snapshot", "TOOL_REPLAY"),
    Fault("filling read withholds flushed frame", "muse_devin_wire.py", ((
        'reader = getattr(response, "read1", None) or response.read', 'reader = response.read'),),
        "test_small_frame_is_visible_while_upstream_is_still_blocked", "STREAM_LIVENESS"),
    Fault("native SSE buffers until EOF", "muse_gateway.py", ((
        'reader = getattr(upstream, "readline", None)', 'reader = lambda: upstream.read(65536)'),),
        "test_small_frame_is_visible_while_upstream_is_still_blocked", "STREAM_LIVENESS"),
    Fault("missing terminal event", "muse_devin.py", ((
        'for event in state.finish():', 'for event in []:'),),
        "test_text_and_nonstream_have_equivalent_complete_output", "TERMINAL"),
    Fault("accept premature EOF", "muse_devin.py", ((
        '(not saw_end_stream and state.stop_reason is None)', 'False'),),
        "test_incomplete_transport_or_provider_error_never_claims_success", "FAILURE_SURFACE"),
    Fault("error-code collision drops HTTP response", "muse_gateway.py", ((
        'extra["upstream_code"] = extra.pop("code")', 'extra["upstream_code"] = extra["code"]'),),
        "test_codex_400_is_readable_even_when_error_has_its_own_code", "HTTP_ERROR_RESPONSE"),
    Fault("Claude missing sequence numbers", "muse_providers.py", ((
        'numbered = {**payload, "sequence_number": sequence}', 'numbered = dict(payload)'),),
        "test_claude_http_translation_has_the_same_consumer_contract", "STREAM_SEQUENCE"),
    Fault("leave composite Codex IDs", "muse_providers.py", ((
        'item["id"] = remapped[item_id]', 'item["id"] = item_id'),),
        "test_cross_provider_history_survives_routes_and_bounded_codex_recovery", "CODEX_ID"),
    Fault("replay dangling reasoning references", "muse_providers.py", ((
        'and item["id"] in demoted_ids', 'and False'),),
        "test_cross_provider_history_survives_routes_and_bounded_codex_recovery", "CODEX_RECOVERY"),
    Fault("Kiro borrows the generic credential", "muse_gateway.py", ((
        'key = muse_kiro.validate_proxy_key(self.config.kiro_api_key)',
        'key = muse_kiro.validate_proxy_key(self.config.api_key or self.config.kiro_api_key)'),),
        "test_route_uses_only_proxy_key_and_shared_conversion", "KIRO_AUTH", KIRO_PREFIX),
    Fault("Kiro missing sequence numbers", "muse_kiro.py", ((
        'yield name, {**data, "sequence_number": sequence}', 'yield name, dict(data)'),),
        "test_route_uses_only_proxy_key_and_shared_conversion", "STREAM_SEQUENCE", KIRO_PREFIX),
    Fault("Kiro loses replayed tool output", "muse_kiro.py", ((
        'elif item.get("type") == "function_call_output":\n            item["call_id"] = call_ids[item["call_id"]]',
        'elif item.get("type") == "function_call_output":\n            item["call_id"] = call_ids[item["call_id"]]\n            item["output"] = ""'),),
        "test_inline_tool_replay_keeps_history_and_drops_only_redundant_state", "KIRO_REPLAY", KIRO_PREFIX),
    Fault("Kiro strips literal line prefixes from tool output", "muse_kiro.py", ((
        'elif item.get("type") == "function_call_output":\n            item["call_id"] = call_ids[item["call_id"]]',
        'elif item.get("type") == "function_call_output":\n            item["call_id"] = call_ids[item["call_id"]]\n            item["output"] = re.sub(r"(?m)^\\d+\\|", "", item["output"])'),),
        "test_inline_tool_replay_keeps_history_and_drops_only_redundant_state", "KIRO_REPLAY", KIRO_PREFIX),
    Fault("Kiro waits for EOF after semantic completion", "muse_kiro.py", ((
        '            yield from convert("message_stop", payload)',
        '            upstream.read()\n            yield from convert("message_stop", payload)'),),
        "test_completion_does_not_wait_for_http_eof", "KIRO_COMPLETION", KIRO_PREFIX),
    Fault("Kiro accepts a stream without message_stop", "muse_kiro.py", ((
        'raise _invalid_response("Kiro Gateway stream ended before message_stop.")',
        'yield from convert("message_stop", {"type": "message_stop"})'),),
        "test_partial_malformed_and_error_streams_never_complete", "KIRO_FAILURE", KIRO_PREFIX),
    Fault("Kiro silently reuses an occupied gateway", "muse_gateway.py", ((
        'self.kiro_requested = bool(', 'self.kiro_requested = False and bool('),),
        "test_kiro_configured_launch_and_run_refuse_occupied_gateway", "0 != 1",
        "test_launchers.ModelsAndReuseTests."),
    Fault("Kiro flattens typed SSE and JSON error codes", "muse_kiro.py", ((
        '"code": upstream.get("code") or upstream.get("type") or error.code', '"code": error.code'),),
        "test_sse_and_json_typed_errors_preserve_category_and_observed_status", "kiro_upstream_error", KIRO_PREFIX),
    Fault("Kiro flattens evidence categories", "muse_kiro.py", ((
        '"category": category, "transient": transient', '"category": "unknown", "transient": transient'),),
        "test_http_error_evidence_survives_the_public_boundary_without_retry", "'unknown'", KIRO_PREFIX),
    Fault("Kiro omits the downstream Retry-After header", "muse_gateway.py", ((
        'headers={"Retry-After": retry["value"]} if retry else None,', 'headers=None,'),),
        "test_retry_after_dates_and_rejected_metadata_are_passive", "AssertionError: None !=", KIRO_PREFIX),
)


def run_selected(names):
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    report = {"run": result.testsRun, "failures": [trace for _, trace in result.failures],
              "errors": [trace for _, trace in result.errors], "skips": len(result.skipped),
              "unexpected_successes": len(result.unexpectedSuccesses),
              "expected_failures": len(result.expectedFailures)}
    print(json.dumps(report))
    return 0  # Parent classifies the structured result, not a nonzero exit.


def invoke(directory, names):
    completed = subprocess.run([sys.executable, "-B", "-m", "tests.check_regressions", "--run", *names],
                               cwd=directory, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError("fault runner crashed (not a detected behavioral regression): " + completed.stderr[-2000:])
    return json.loads(completed.stdout)


def usable(report):
    return (report["run"] > 0 and not report["errors"] and not report["skips"]
            and not report["unexpected_successes"] and not report["expected_failures"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        return run_selected(sys.argv[2:])
    source = Path(__file__).resolve().parents[1]
    names = sorted({fault.test_name for fault in FAULTS})
    with tempfile.TemporaryDirectory(prefix="gateway-faults-") as directory:
        copy = Path(directory)
        for name in ("muse_gateway.py", "muse_providers.py", "muse_devin.py", "muse_devin_wire.py",
                     "muse_capture.py", "devin_pb2.py", "muse_shim.py", "muse_kiro.py", "test_kiro.py",
                     "test_launchers.py", "muse-gateway.ps1", "muse-gateway"):
            shutil.copy2(source / name, copy / name)
        shutil.copytree(source / "tests", copy / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        originals = {fault.module: (copy / fault.module).read_text(encoding="utf-8") for fault in FAULTS}
        baseline = invoke(copy, names)
        if not usable(baseline) or baseline["failures"] or baseline["run"] != len(names):
            raise RuntimeError("baseline is not clean; fault results would be meaningless:\n" + json.dumps(baseline))
        print(f"Baseline: {baseline['run']} behavioral contracts passed")
        for fault in FAULTS:
            modified = originals[fault.module]
            for before, after in fault.edits:
                if modified.count(before) != 1:
                    raise RuntimeError(f"mutation needs review: {fault.name} (target not unique)")
                modified = modified.replace(before, after, 1)
            target = copy / fault.module
            try:
                target.write_text(modified, encoding="utf-8")
                report = invoke(copy, [fault.test_name])
                if not usable(report) or report["run"] != 1 or not report["failures"]:
                    raise RuntimeError(f"SURVIVED or harness error: {fault.name}\n" + json.dumps(report))
                if not all(fault.diagnostic in failure for failure in report["failures"]):
                    raise RuntimeError(f"wrong failure for {fault.name}:\n" + json.dumps(report))
                print(f"Detected: {fault.name} [{fault.diagnostic}]")
            finally:
                target.write_text(originals[fault.module], encoding="utf-8")
    print(f"All {len(FAULTS)} selected regressions detected; production files untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
