"""Devin provider tests. Fully offline.

Covers the Connect wire layer, credentials, discovery normalization, chat
translation, stream consumption, and gateway wiring. Network boundaries are
mocked at muse_devin.unary_rpc / open_chat_stream; byte vectors come from
junie's prost golden tests (independent oracle).
"""

import argparse
import base64
import gzip
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

from muse_gateway import Config, MuseShimHandler, MuseShimServer, ProviderError

import devin_pb2
from muse_devin import (
    DevinChatError,
    DevinCredentialError,
    DevinModel,
    DevinStreamState,
    assign_if_router,
    build_chat_request,
    build_history_prompts,
    build_router_prompt,
    build_tools,
    credential_state,
    deref_schema_refs,
    deterministic_uuid,
    discover_models,
    fetch_auth,
    history_to_prompts,
    is_gemini_uid,
    normalize_configs,
    normalize_schema_for_google,
    normalize_tool_call_id,
    read_store_credentials,
    resolve_access_token,
    run_turn_events,
    run_turn_final,
    seed_models,
    select_credential,
    supports_thinking,
)
from muse_devin_wire import (
    ConnectFrameDecoder,
    DevinFrameError,
    cli_metadata,
    decode_unary,
    discovery_metadata,
    frame_message,
    http_error_detail,
    normalize_base_url,
    normalize_session_token,
    parse_trailer_error,
)
from muse_gateway import (
    _credential_status,
    _devin_catalog_entries,
    _devin_provider_ready,
    provider_model_catalog,
)


def devin_config(**overrides):
    values = {
        "host": "127.0.0.1",
        "port": 0,
        "provider": "auto",
        "model_discovery": False,
        "devin_api_key": "test-devin-key",
    }
    values.update(overrides)
    return Config(argparse.Namespace(**values))


def make_jwt(exp=None):
    """Assemble (never literalize) a JWT for expiry tests."""
    def b64(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {}
    if exp is not None:
        payload["exp"] = exp
    return f"{b64({'alg': 'none'})}.{b64(payload)}.sig"


def make_auth_db(rows):
    """Create a synthetic OMP agent.db; rows are (provider, type, data, cause, identity)."""
    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " provider TEXT NOT NULL, credential_type TEXT NOT NULL, data TEXT NOT NULL,"
        " disabled_cause TEXT DEFAULT NULL, identity_key TEXT DEFAULT NULL,"
        " created_at INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0)"
    )
    for provider, cred_type, data, cause, identity in rows:
        connection.execute(
            "INSERT INTO auth_credentials (provider, credential_type, data,"
            " disabled_cause, identity_key) VALUES (?, ?, ?, ?, ?)",
            (provider, cred_type, json.dumps(data), cause, identity),
        )
    connection.commit()
    connection.close()
    return path


def devin_model(**overrides):
    values = {
        "uid": "swe-1-6",
        "logical_id": "swe-1-6",
        "label": "SWE",
        "context_limit": 200000,
        "output_limit": 128000,
    }
    values.update(overrides)
    return DevinModel(**values)


class DevinWireTests(unittest.TestCase):
    def test_golden_auth_request_bytes(self):
        message = devin_pb2.GetUserJwtRequest()
        message.metadata.api_key = "k"
        message.metadata.user_jwt = "j"
        self.assertEqual(
            message.SerializeToString(),
            bytes([0x0A, 0x07, 0x1A, 0x01, 0x6B, 0xAA, 0x01, 0x01, 0x6A]),
        )

    def test_golden_jwt_response_bytes(self):
        message = devin_pb2.GetUserJwtResponse(user_jwt="j", custom_api_server_url="u")
        self.assertEqual(
            message.SerializeToString(), bytes([0x0A, 0x01, 0x6A, 0x12, 0x01, 0x75])
        )

    def test_golden_assign_bytes(self):
        message = devin_pb2.AssignModelRequest(model_router_uid="r", cascade_id="c")
        message.metadata.api_key = "k"
        message.chat_message_prompt.prompt = "p"
        self.assertEqual(
            message.SerializeToString(),
            bytes([
                0x0A, 0x03, 0x1A, 0x01, 0x6B, 0x12, 0x01, 0x72,
                0x1A, 0x01, 0x63, 0x2A, 0x03, 0x1A, 0x01, 0x70,
            ]),
        )

    def test_golden_chat_request_bytes(self):
        message = devin_pb2.GetChatMessageRequest(
            prompt="p",
            chat_model_uid="m",
            request_type=devin_pb2.ChatMessageRequestType.Value(
                "CHAT_MESSAGE_REQUEST_TYPE_CASCADE"
            ),
            cascade_id="c",
            execution_id="e",
        )
        message.metadata.api_key = "k"
        message.metadata.user_jwt = "j"
        message.model_assignment_jwt = "jwt"
        self.assertEqual(
            message.SerializeToString(),
            bytes([
                0x0A, 0x07, 0x1A, 0x01, 0x6B, 0xAA, 0x01, 0x01, 0x6A,
                0x12, 0x01, 0x70, 0x38, 0x05, 0x82, 0x01, 0x01, 0x63,
                0xAA, 0x01, 0x01, 0x6D, 0xB2, 0x01, 0x01, 0x65,
                0xD2, 0x01, 0x03, 0x6A, 0x77, 0x74,
            ]),
        )

    def test_golden_discovery_response_bytes(self):
        message = devin_pb2.GetCliModelConfigsResponse()
        row = message.client_model_configs.add()
        row.label = "m"
        row.model_uid = "u"
        row.max_tokens = 64
        self.assertEqual(
            message.SerializeToString(),
            bytes([
                0x0A, 0x0A, 0x0A, 0x01, 0x6D, 0x90, 0x01, 0x40,
                0xB2, 0x01, 0x01, 0x75,
            ]),
        )

    def test_session_token_prefix(self):
        self.assertEqual(normalize_session_token("abc"), "devin-session-token$abc")
        self.assertEqual(
            normalize_session_token("devin-session-token$abc"), "devin-session-token$abc"
        )

    def test_cli_metadata_identity(self):
        metadata = cli_metadata("k")
        self.assertEqual(metadata.api_key, "devin-session-token$k")
        self.assertEqual(metadata.ide_name, "devin-cli")
        self.assertEqual(metadata.ide_type, "chisel")
        self.assertEqual(metadata.ide_version, "3000.11.3")
        self.assertEqual(metadata.locale, "en")

    def test_discovery_metadata_identity(self):
        metadata = discovery_metadata("k")
        self.assertEqual(metadata.ide_name, "chisel")
        self.assertEqual(metadata.ide_version, "0.0.0-dev")
        self.assertEqual(metadata.ide_type, "")
        self.assertEqual(
            list(metadata.supported_model_displays), [3, 4, 6, 7, 8]
        )

    def test_frames_reassemble_across_chunks(self):
        decoder = ConnectFrameDecoder()
        first = frame_message(b"hello", 0)
        second = frame_message(b"world!", 1)
        self.assertEqual(decoder.decode(first[:3]), [])
        self.assertEqual(
            decoder.decode(first[3:] + second), [(0, b"hello"), (1, b"world!")]
        )

    def test_oversize_length_prefix_fails_before_buffering(self):
        decoder = ConnectFrameDecoder(max_payload_bytes=4)
        with self.assertRaises(DevinFrameError):
            decoder.decode(frame_message(b"toolong", 0))

    def test_unary_decode_prefers_bare_protobuf(self):
        body = devin_pb2.GetUserJwtResponse(user_jwt="j").SerializeToString()
        self.assertEqual(
            decode_unary(devin_pb2.GetUserJwtResponse, body).user_jwt, "j"
        )

    def test_unary_decode_falls_back_to_gzip(self):
        body = devin_pb2.GetUserJwtResponse(user_jwt="j").SerializeToString()
        self.assertEqual(
            decode_unary(devin_pb2.GetUserJwtResponse, gzip.compress(body)).user_jwt,
            "j",
        )

    def test_unary_decode_rejects_garbage(self):
        self.assertIsNone(decode_unary(devin_pb2.GetUserJwtResponse, b"\xff\xff\xff"))

    def test_trailer_error_shape(self):
        error = parse_trailer_error(
            '{"error":{"code":"invalid_argument","message":"bad"}}'
        )
        self.assertEqual(error.code, "invalid_argument")
        self.assertEqual(error.formatted, "Devin stream error invalid_argument: bad")

    def test_trailer_leniency(self):
        self.assertIsNone(parse_trailer_error(""))
        self.assertIsNone(parse_trailer_error("{nope"))
        self.assertIsNone(parse_trailer_error("{}"))

    def test_trailer_details_summarized(self):
        error = parse_trailer_error(
            '{"error":{"code":"unavailable","message":"busy","details":['
            '{"type":"retryInfo","value":"later"},'
            '{"type":"debugInfo","debug":{"attempt":2}},'
            '{"type":"bare"},42,"junk"]}}'
        )
        self.assertIn("busy", error.formatted)
        self.assertIn("retryInfo: later", error.formatted)
        self.assertIn('debugInfo: {"attempt": 2}', error.formatted)
        self.assertIn("bare", error.formatted)

    def test_http_detail_suppresses_html_and_binary(self):
        self.assertIsNone(http_error_detail("text/html", b"<html>nope"))
        self.assertIsNone(http_error_detail("application/proto", b"\xff\xfe\x00bad"))

    def test_http_detail_extracts_envelope(self):
        self.assertEqual(
            http_error_detail(
                "application/json", b'{"error":{"message":"boom"}}'
            ),
            "boom",
        )

    def test_base_url_validation(self):
        self.assertEqual(
            normalize_base_url("https://x.example///"), "https://x.example"
        )
        with self.assertRaises(Exception):
            normalize_base_url("ftp://x.example")


class DevinCredentialTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("muse_devin.read_cli_token", return_value=(None, None))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_store_reads_active_rows_only(self):
        now = int(time.time())
        path = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now + 3600)}, None, "a"),
            ("devin", "oauth", {"access": make_jwt(now + 3600)}, "revoked", "b"),
            ("devin", "oauth", "not-json", None, "c"),
            ("other", "oauth", {"access": "x"}, None, "d"),
        ])
        try:
            active, disabled = read_store_credentials(path)
        finally:
            os.unlink(path)
        self.assertEqual([row["identity_key"] for row in active], ["a"])
        self.assertEqual([row["identity_key"] for row in disabled], ["b"])

    def test_missing_database_means_no_credentials(self):
        active, disabled = read_store_credentials("/nonexistent/agent.db")
        self.assertEqual((active, disabled), ([], []))

    def test_selection_requires_explicit_id_when_ambiguous(self):
        rows = [
            {"id": 1, "identity_key": "a", "access": "x"},
            {"id": 2, "identity_key": "b", "access": "y"},
        ]
        with self.assertRaises(DevinCredentialError):
            select_credential(rows)
        self.assertEqual(select_credential(rows, "b")["id"], 2)
        self.assertEqual(select_credential([rows[0]])["id"], 1)
        with self.assertRaises(DevinCredentialError):
            select_credential([])

    def test_credential_state_matrix(self):
        now = int(time.time())
        fresh = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now + 3600)}, None, "a"),
        ])
        stale = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now - 10)}, None, "a"),
        ])
        try:
            ready = credential_state(auth_db_path=fresh)
            self.assertEqual(ready["state"], "ready")
            self.assertEqual(ready["credential_id"], 1)
            expired = credential_state(auth_db_path=stale)
            self.assertEqual(expired["state"], "expired")
            missing = credential_state(auth_db_path="/nonexistent/agent.db")
            self.assertEqual(missing["state"], "unconfigured")
            explicit = credential_state(api_key="tok")
            self.assertEqual((explicit["state"], explicit["source"]), ("ready", "explicit"))
        finally:
            os.unlink(fresh)
            os.unlink(stale)

    def test_expiry_overflow_falls_back(self):
        path = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(2**40)}, None, "a"),
        ])
        try:
            state = credential_state(auth_db_path=path)
        finally:
            os.unlink(path)
        self.assertEqual(state["state"], "ready")
        self.assertGreater(state["expires_at"], int(time.time() * 1000))

    def test_sixty_second_use_skew(self):
        now = int(time.time())
        path = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now + 30)}, None, "a"),
        ])
        try:
            state = credential_state(auth_db_path=path)
        finally:
            os.unlink(path)
        self.assertEqual(state["state"], "expired")
        path = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now + 90)}, None, "a"),
        ])
        try:
            state = credential_state(auth_db_path=path)
        finally:
            os.unlink(path)
        self.assertEqual(state["state"], "ready")

    def test_auth_failure_quarantines_token(self):
        from muse_devin import auth_failure_mark, mark_auth_failure

        token = "quarantine-me-token"
        self.assertIsNone(auth_failure_mark(token))
        self.assertEqual(credential_state(api_key=token)["state"], "ready")
        mark_auth_failure(token, "GetUserJwt 401")
        self.assertIsNotNone(auth_failure_mark(token))
        state = credential_state(api_key=token)
        self.assertEqual(state["state"], "auth-failed")
        self.assertIn("sign-in again", state["note"])

    def test_fetch_auth_marks_persistent_401(self):
        from muse_devin import auth_failure_mark, fetch_auth
        from muse_devin_wire import DevinHttpError

        def fail_401(*args, **kwargs):
            raise DevinHttpError("request", 401, "unauthorized")

        with patch("muse_devin.unary_rpc", side_effect=fail_401):
            with self.assertRaises(Exception):
                fetch_auth("mark-on-401-token", "https://x")
        self.assertIsNotNone(auth_failure_mark("mark-on-401-token"))

        def fail_500(*args, **kwargs):
            raise DevinHttpError("request", 500, "oops")

        with patch("muse_devin.unary_rpc", side_effect=fail_500):
            with self.assertRaises(Exception):
                fetch_auth("no-mark-on-500-token", "https://x")
        self.assertIsNone(auth_failure_mark("no-mark-on-500-token"))


class DevinCliCredentialTests(unittest.TestCase):
    def _write_toml(self, directory, text):
        path = os.path.join(directory, "credentials.toml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_cli_token_reads_toml(self):
        from muse_devin import read_cli_token

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_toml(
                directory,
                'windsurf_api_key = "tok-123"\napi_server_url = "https://x"\n',
            )
            self.assertEqual(read_cli_token(path), ("tok-123", path))

    def test_cli_token_ignores_missing_and_keyless(self):
        from muse_devin import read_cli_token

        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                read_cli_token(os.path.join(directory, "nope.toml")), (None, None)
            )
            path = self._write_toml(directory, 'api_server_url = "https://x"\n')
            self.assertEqual(read_cli_token(path), (None, None))
            broken = self._write_toml(directory, "windsurf_api_key = \n")
            self.assertEqual(read_cli_token(broken), (None, None))

    def test_fallback_string_table_parser(self):
        from muse_devin import _parse_toml_string_table

        parsed = _parse_toml_string_table(
            '# comment\nwindsurf_api_key = "a\\"b"\ncount = 3\n'
        )
        self.assertEqual(parsed, {"windsurf_api_key": 'a"b'})

    def test_cli_source_precedence_over_store(self):
        now = int(time.time())
        db = make_auth_db([
            ("devin", "oauth", {"access": make_jwt(now + 3600)}, None, "a"),
        ])
        try:
            with patch(
                "muse_devin.read_cli_token", return_value=("cli-tok", "p")
            ):
                state = credential_state(auth_db_path=db)
                self.assertEqual(state["source"], "devin-cli")
                self.assertEqual(state["access_token"], "cli-tok")
                token, identity = resolve_access_token(auth_db_path=db)
                self.assertEqual((token, identity), ("cli-tok", "devin-cli:p"))
                explicit_id = credential_state(auth_db_path=db, credential_id="1")
                self.assertEqual(explicit_id["source"], "store")
        finally:
            os.unlink(db)

    def test_cli_quarantine_reports_auth_failed(self):
        from muse_devin import mark_auth_failure

        mark_auth_failure("cli-quarantine-tok")
        with patch(
            "muse_devin.read_cli_token",
            return_value=("cli-quarantine-tok", "p"),
        ):
            state = credential_state()
        self.assertEqual(state["state"], "auth-failed")
        self.assertIn("devin auth login", state["note"])


def devin_config_row(uid, label="M", **kwargs):
    row = devin_pb2.ClientModelConfig()
    row.model_uid = uid
    row.label = label
    row.max_tokens = kwargs.get("max_tokens", 264000)
    row.model_info.max_output_tokens = kwargs.get("max_output", 64000)
    for key, value in (kwargs.get("features") or {}).items():
        setattr(row.model_info.model_features, key, value)
    if kwargs.get("router"):
        row.model_info.is_model_router = True
    if kwargs.get("display") is not None:
        row.model_info.display_option = kwargs["display"]
    if kwargs.get("disabled"):
        row.disabled = True
    for name, order, fname in kwargs.get("family", []):
        entry = row.model_family_metadata.entries.add()
        entry.key = name
        entry.value.order = order
        entry.value.name = fname
    if kwargs.get("family_label"):
        row.model_family_metadata.model_family_label = kwargs["family_label"]
    if kwargs.get("family_default"):
        row.model_family_metadata.is_default_model_in_family = True
    return row


class DevinDiscoveryTests(unittest.TestCase):
    def test_thinking_matrix(self):
        self.assertFalse(supports_thinking(devin_config_row("x", "plain")))
        self.assertTrue(supports_thinking(devin_config_row("x", "Thinker High")))
        self.assertFalse(supports_thinking(devin_config_row("x", "no thinking here")))
        self.assertTrue(
            supports_thinking(
                devin_config_row("x", "plain", features={"supports_thinking": True})
            )
        )

    def test_normalize_filters_and_drops_fusion_pairings(self):
        from muse_devin import fusion_pairing_lead

        live = {"swe-1-6", "base-priority", "base"}
        self.assertEqual(
            fusion_pairing_lead("fusion-swe-1-6-sidekick-y", live), "swe-1-6"
        )
        self.assertIs(fusion_pairing_lead("fusion-nope-sidekick-y", live), False)
        self.assertIsNone(fusion_pairing_lead("swe-1-6", live))
        self.assertIsNone(fusion_pairing_lead("fusion-sidekick-x", live))
        self.assertEqual(
            fusion_pairing_lead("fusion-base-fast-sidekick-y", live), "base-priority"
        )
        response = devin_pb2.GetCliModelConfigsResponse()
        response.client_model_configs.append(
            devin_config_row(
                "swe-1-6",
                "SWE Thinking 1.6",
                features={
                    "supports_thinking": True,
                    "supports_tool_calls": True,
                    "supports_images": True,
                },
            )
        )
        response.client_model_configs.append(
            devin_config_row("fusion-swe-1-6-sidekick-y", "Fusion")
        )
        response.client_model_configs.append(devin_config_row("dead-1", disabled=True))
        response.client_model_configs.append(devin_config_row("q-1", display=4))
        response.client_model_configs.append(
            devin_config_row("adaptive", "Adaptive", router=True)
        )
        models, dropped = normalize_configs(response, "https://x")
        by_id = {model.logical_id: model for model in models}
        self.assertEqual(dropped, 1)
        self.assertEqual(sorted(by_id), ["adaptive", "swe-1-6"])
        self.assertTrue(by_id["adaptive"].is_router)
        self.assertFalse(by_id["swe-1-6"].supports_images)

    def test_lane_collapse_routes_efforts(self):
        response = devin_pb2.GetCliModelConfigsResponse()
        response.client_model_configs.append(
            devin_config_row(
                "m-high",
                "Claude High",
                family_label="Claude Family",
                family=[("effort", 0, "High"), ("thinking", 1, "yes")],
                family_default=True,
            )
        )
        response.client_model_configs.append(
            devin_config_row(
                "m-low",
                "Claude Low",
                family_label="Claude Family",
                family=[("effort", 0, "Low"), ("thinking", 1, "yes")],
            )
        )
        models, _ = normalize_configs(response, "https://x")
        self.assertEqual(len(models), 1)
        lane = models[0]
        self.assertEqual(lane.logical_id, "claude-family")
        self.assertEqual(lane.efforts, ["low", "high"])
        self.assertEqual(lane.default_effort, "high")
        self.assertEqual(lane.wire_uid("low"), "m-low")
        self.assertEqual(lane.wire_uid(), "m-high")
        response = devin_pb2.GetCliModelConfigsResponse()
        response.client_model_configs.extend([
            devin_config_row("off-uid", "Family Off", family_label="Switch Family",
                family=[("effort", 0, "High"), ("thinking", 0, "no")], family_default=True),
            devin_config_row("high-uid", "Family High", family_label="Switch Family",
                family=[("effort", 0, "High"), ("thinking", 1, "yes")]),
        ])
        lane = normalize_configs(response, "https://x")[0][0]
        self.assertEqual(lane.default_effort, "none")
        self.assertEqual(lane.wire_uid(), "off-uid")
        self.assertEqual(lane.wire_uid("none"), "off-uid")
        self.assertEqual(lane.wire_uid("off"), "off-uid")
        self.assertEqual(lane.wire_uid("high"), "high-uid")
        self.assertIn("none", lane.efforts)
        for spelling in ("None", "No Thinking", "Off"):
            with self.subTest(off_metadata=spelling):
                response = devin_pb2.GetCliModelConfigsResponse()
                response.client_model_configs.extend([
                    devin_config_row("high-uid", "High", family_label="Effort Family",
                        family=[("effort", 0, "High")], family_default=True),
                    devin_config_row("off-uid", "Off", family_label="Effort Family",
                        family=[("effort", 0, spelling)]),
                ])
                lane = normalize_configs(response, "https://x")[0][0]
                self.assertEqual(lane.wire_uid(), "high-uid")
                self.assertEqual(lane.wire_uid("none"), "off-uid")

    def test_account_default_wins(self):
        response = devin_pb2.GetCliModelConfigsResponse()
        response.client_model_configs.append(devin_config_row("a-1", "A"))
        response.client_model_configs.append(devin_config_row("b-2", "B"))
        response.default_override_model_config.model_uid = "b-2"
        models, _ = normalize_configs(response, "https://x")
        self.assertEqual(
            [model.is_default for model in models], [False, True]
        )

    def test_seeds_have_expected_shape(self):
        from muse_devin import seed_models

        seeds = seed_models()
        self.assertEqual([model.uid for model in seeds], ["swe-1-6-fast", "swe-1-6"])
        self.assertTrue(seeds[1].is_default)
        self.assertTrue(all(model.unverified for model in seeds))

    def test_discover_prefers_native_then_legacy(self):
        native = devin_pb2.GetCliModelConfigsResponse()
        native.client_model_configs.append(devin_config_row("n-1", "Native"))
        bodies = [native.SerializeToString()]

        def fake_unary(base_url, path, payload, timeout=5.0):
            return 200, "application/proto", bodies[0]

        with patch("muse_devin.unary_rpc", side_effect=fake_unary):
            models, dropped, warnings = discover_models("k", "https://x")
        self.assertEqual([model.uid for model in models], ["n-1"])
        self.assertEqual((dropped, warnings), (0, []))

    def test_discover_empty_warns(self):
        empty = devin_pb2.GetCliModelConfigsResponse().SerializeToString()

        def fake_unary(base_url, path, payload, timeout=5.0):
            return 200, "application/proto", empty

        with patch("muse_devin.unary_rpc", side_effect=fake_unary):
            models, dropped, warnings = discover_models("k", "https://x")
        self.assertEqual(models, [])
        self.assertEqual(len(warnings), 1)


class DevinTranslationTests(unittest.TestCase):
    def test_deterministic_uuid_shape(self):
        first = deterministic_uuid("a\x00b\x00c")
        self.assertEqual(len(first), 36)
        self.assertEqual(first[8] + first[13] + first[18] + first[23], "----")
        self.assertEqual(first, deterministic_uuid("a\x00b\x00c"))

    def test_call_id_mapping(self):
        from muse_devin import normalize_tool_call_id

        seen = set()
        self.assertEqual(normalize_tool_call_id("call_1|item_9", seen), "call_1")
        self.assertEqual(normalize_tool_call_id("a b.c!", seen), "a_b_c_")
        self.assertEqual(len(normalize_tool_call_id("x" * 60, seen)), 40)
        self.assertEqual(normalize_tool_call_id("dup", {"dup"}), "dup-2")

    def test_ref_deref_and_collision_guard(self):
        schema = {
            "$defs": {"N": {"type": "string"}},
            "properties": {"p": {"$ref": "#/$defs/N"}},
        }
        self.assertEqual(
            deref_schema_refs(schema)["properties"]["p"], {"type": "string"}
        )
        colliding = {
            "$defs": {"N": {"type": "string"}},
            "definitions": {"N": {"type": "number"}},
            "properties": {"p": {"$ref": "#/$defs/N"}},
        }
        self.assertEqual(
            deref_schema_refs(colliding)["properties"]["p"],
            {"$ref": "#/$defs/N"},
        )

    def test_google_normalize_strips_and_spills(self):
        schema = {
            "type": ["string", "null"],
            "minLength": 3,
            "description": "d",
            "properties": {"a": {"type": "string"}},
        }
        cleaned = normalize_schema_for_google(schema)
        self.assertEqual(cleaned["type"], "string")
        self.assertNotIn("minLength", cleaned)
        self.assertEqual(cleaned["description"], "d\n\n{minLength: 3}")

    def test_google_normalize_errors(self):
        self.assertEqual(
            normalize_schema_for_google({"enum": ["a", "b"]})["type"], "string"
        )
        nulled = normalize_schema_for_google({"type": "null"})
        self.assertNotIn("type", nulled)
        self.assertTrue(nulled["nullable"])
        with self.assertRaises(DevinChatError):
            normalize_schema_for_google({"type": ["string", "number"]})
        with self.assertRaises(DevinChatError):
            normalize_schema_for_google({"type": "weird"})

    def test_gemini_detection(self):
        self.assertTrue(is_gemini_uid("MODEL_GOOGLE_GEMINI_X"))
        self.assertFalse(is_gemini_uid("swe-1-6"))

    def test_history_walk(self):
        body = {
            "instructions": "sys1",
            "input": [
                {"type": "message", "role": "system", "content": "sys2"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {"type": "message", "role": "developer", "content": "current instruction"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "t",
                    "arguments": '{"a": 1}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
                {"type": "reasoning", "summary": "thought"},
            ],
        }
        system_text, staged, call_ids = history_to_prompts(body, "casc")
        self.assertEqual(system_text, "sys1\n\nsys2")
        self.assertEqual(
            [prompt["role"] for prompt in staged],
            ["user", "developer", "assistant", "tool", "assistant"],
        )
        self.assertEqual(call_ids, {"call_1": "call_1"})
        protos = build_history_prompts(staged, "casc")
        self.assertEqual(len({prompt.message_id for prompt in protos}), 5)
        self.assertEqual(protos[1].source, devin_pb2.CHAT_MESSAGE_SOURCE_USER)
        self.assertEqual(protos[3].tool_call_id, "call_1")

    def test_request_build(self):
        model = devin_model(supports_parallel=False)
        tools = build_tools(
            [{"name": "t", "description": "d", "parameters": {"type": "object"}}],
            False,
        )
        request = build_chat_request(
            "sys", {"temperature": 0.5}, model, "swe-1-6", "k", "jwt",
            "casc", [], tools,
        )
        self.assertEqual(request.chat_model_uid, "swe-1-6")
        self.assertEqual(request.request_type, 5)
        self.assertEqual(request.planner_mode, 1)
        self.assertEqual(request.tool_choice.option_name, "auto")
        self.assertTrue(request.disable_parallel_tool_calls)
        self.assertEqual(request.configuration.temperature, 0.5)
        self.assertEqual(request.configuration.top_k, 40)


class DevinStreamTests(unittest.TestCase):
    def _text_tool_sequence(self):
        state = DevinStreamState("devin/swe-1-6")
        first = devin_pb2.GetChatMessageResponse(message_id="r1", delta_text="Hi")
        first.usage.input_tokens = 20
        first.usage.cache_read_tokens = 300
        second = devin_pb2.GetChatMessageResponse()
        call = second.delta_tool_calls.add()
        call.id = "c1"
        call.name = "t"
        call.arguments_json = '{"a":'
        third = devin_pb2.GetChatMessageResponse()
        call = third.delta_tool_calls.add()
        call.arguments_json = '{"a":1}'
        third.usage.input_tokens = 10
        third.usage.output_tokens = 5
        third.usage.cache_read_tokens = 1000
        third.usage.cache_write_tokens = 200
        events = (
            state.feed(first) + state.feed(second) + state.feed(third) + state.finish()
        )
        return events

    def test_text_and_tool_sequence(self):
        events = self._text_tool_sequence()
        kinds = [event for event, _ in events]
        # Responses SSE contract: lifecycle events open the stream, and the
        # id is decided once (clients reject streams starting at
        # output_item.added or carrying non-resp_ upstream ids).
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertIn("response.output_item.added", kinds)
        self.assertIn("response.output_text.delta", kinds)
        self.assertIn("response.function_call_arguments.delta", kinds)
        self.assertEqual(kinds[-1], "response.completed")
        self.assertEqual([payload["sequence_number"] for _, payload in events], list(range(len(events))))
        created_id = events[0][1]["response"]["id"]
        self.assertTrue(created_id.startswith("resp_"))
        final = events[-1][1]["response"]
        self.assertEqual(final["id"], created_id)
        self.assertEqual(
            final["usage"],
            {"input_tokens": 1210, "output_tokens": 5, "total_tokens": 1215,
             "input_tokens_details": {"cached_tokens": 1000}},
        )
        calls = [item for item in final["output"] if item["type"] == "function_call"]
        self.assertEqual(calls[0]["arguments"], '{"a":1}')
        self.assertEqual(calls[0]["call_id"], "c1")
        deltas = "".join(payload["delta"] for kind, payload in events
                         if kind == "response.function_call_arguments.delta")
        self.assertEqual(deltas, calls[0]["arguments"])

    def test_invalid_arguments_repair_or_mark(self):
        state = DevinStreamState("devin/x")
        state.feed(devin_pb2.GetChatMessageResponse(message_id="r", delta_text="t"))
        broken = devin_pb2.GetChatMessageResponse()
        call = broken.delta_tool_calls.add()
        call.id = "c9"
        call.name = "t"
        call.arguments_json = "{oops"
        state.feed(broken)
        final = state.finish()[-1][1]["response"]
        calls = [item for item in final["output"] if item["type"] == "function_call"]
        self.assertEqual(json.loads(calls[0]["arguments"]), {"__parseError": "{oops"})

    def test_max_tokens_completes_incomplete(self):
        state = DevinStreamState("devin/x")
        state.feed(
            devin_pb2.GetChatMessageResponse(
                message_id="r", delta_text="t", stop_reason=3
            )
        )
        self.assertEqual(state.finish()[-1][0], "response.incomplete")

    def test_chunked_http_truncation_emits_failed_terminal(self):
        from http.client import HTTPResponse
        from muse_devin_wire import frame_message
        message = devin_pb2.GetChatMessageResponse(message_id="resp_chunk", delta_text="Hi", stop_reason=2)
        frame = frame_message(message.SerializeToString())
        raw = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
               + f"{len(frame):x}\r\n".encode() + frame + b"\r\n40\r\nshort")
        class Socket:
            def makefile(self, *args):
                return io.BytesIO(raw)
        response = HTTPResponse(Socket())
        response.begin()
        with patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")), patch(
            "muse_devin.open_chat_stream", return_value=response
        ):
            events = list(run_turn_events({"input": "hi"}, [], devin_model(), "k"))
        kinds = [kind for kind, _ in events]
        self.assertIn("response.output_text.delta", kinds)
        self.assertEqual(kinds[-1], "response.failed")
        self.assertNotIn("response.completed", kinds)
        self.assertTrue(response.isclosed())

    def test_nonterminal_stop_reasons_cannot_complete(self):
        from muse_devin_wire import frame_message
        for reason in (1, 9):
            for trailer in (b"", frame_message(b"{}", 2)):
                with self.subTest(reason=reason, trailer=bool(trailer)):
                    message = devin_pb2.GetChatMessageResponse(message_id="resp_partial", delta_text="part", stop_reason=reason)
                    state = DevinStreamState("devin/test")
                    state.feed(message)
                    self.assertEqual(state.finish()[-1][0], "response.failed")
                    response = io.BytesIO(frame_message(message.SerializeToString()) + trailer)
                    with patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")), patch(
                        "muse_devin.open_chat_stream", return_value=response
                    ):
                        events = list(run_turn_events({"input": "hi"}, [], devin_model(), "k"))
                    kinds = [kind for kind, _ in events]
                    self.assertEqual(kinds[-1], "response.failed")
                    self.assertNotIn("response.completed", kinds)

    def test_midstream_failure_events(self):
        state = DevinStreamState("devin/x")
        state.feed(devin_pb2.GetChatMessageResponse(message_id="r", delta_text="t"))
        events = state.fail("boom", "upstream_error")
        self.assertEqual([kind for kind, _ in events], ["response.failed"])
        self.assertEqual(events[-1][0], "response.failed")
        self.assertTrue(all(isinstance(payload["sequence_number"], int) for _, payload in events))

    def test_turn_rejects_context_overflow(self):
        from muse_devin import run_turn_events
        from muse_devin_wire import (
            CONNECT_COMPRESSED_FLAG,
            CONNECT_END_STREAM_FLAG,
            frame_message,
        )

        big_text = "x" * (600 * 1024)
        trailer = (
            '{"error":{"code":"invalid_argument","message":"internal error"}}'
        )
        stream = frame_message(trailer.encode(), CONNECT_END_STREAM_FLAG)

        class FakeStream(io.BytesIO):
            def close(self):
                pass

        body = {
            "input": [
                {"type": "message", "role": "assistant", "content": big_text},
                {"type": "message", "role": "user", "content": "hi"},
            ]
        }
        with (
            patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")),
            patch("muse_devin.open_chat_stream", return_value=FakeStream(stream)),
        ):
            with self.assertRaises(DevinChatError) as raised:
                list(run_turn_events(body, [], devin_model(), "k"))
        self.assertEqual(raised.exception.code, "devin_context_overflow")

    def test_turn_error_trailer_with_details_yields_failed(self):
        from muse_devin import run_turn_events
        from muse_devin_wire import CONNECT_END_STREAM_FLAG, frame_message

        trailer = (
            '{"error":{"code":"unavailable","message":"overloaded",'
            '"details":[{"type":"retryInfo","value":"later"}]}}'
        )
        stream = frame_message(trailer.encode(), CONNECT_END_STREAM_FLAG)

        class FakeStream(io.BytesIO):
            def close(self):
                pass

        with (
            patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")),
            patch("muse_devin.open_chat_stream", return_value=FakeStream(stream)),
        ):
            events = list(
                run_turn_events({"input": "hi"}, [], devin_model(), "k")
            )
        kinds = [event for event, _ in events]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[2], "response.failed")
        self.assertEqual(kinds[-1], "response.failed")
        self.assertIn("overloaded", events[2][1]["response"]["error"]["message"])
        self.assertIn("retryInfo: later", events[2][1]["response"]["error"]["message"])

    def test_turn_streams_text_to_completion(self):
        from muse_devin import run_turn_events
        from muse_devin_wire import CONNECT_COMPRESSED_FLAG, frame_message

        first = devin_pb2.GetChatMessageResponse(message_id="r", delta_text="Hi", stop_reason=2)
        body = frame_message(
            gzip.compress(first.SerializeToString()), CONNECT_COMPRESSED_FLAG
        )

        class FakeStream(io.BytesIO):
            def close(self):
                pass

        with (
            patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")),
            patch("muse_devin.open_chat_stream", return_value=FakeStream(body)),
        ):
            events = list(
                run_turn_events({"input": "hi"}, [], devin_model(), "k")
            )
        kinds = [event for event, _ in events]
        self.assertIn("response.output_text.delta", kinds)
        self.assertEqual(kinds[-1], "response.completed")
        self.assertEqual(
            events[-1][1]["response"]["output"][0]["content"][0]["text"], "Hi"
        )


class DevinGatewayTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("muse_devin.read_cli_token", return_value=(None, None))
        self.addCleanup(patcher.stop)
        patcher.start()

    def _handler(self, cfg):
        seen = {}

        def send_json(status, payload):
            seen["status"] = status
            seen["payload"] = payload

        handler = SimpleNamespace(
            config=cfg,
            headers={},
            server_version="muse-gateway-test",
            wfile=io.BytesIO(),
            _send_upstream_headers=lambda *args, **kwargs: None,
            send_json=send_json,
            _log_event=lambda *args, **kwargs: None,
        )
        return handler, seen

    def test_resolve_model_prefix(self):
        cfg = devin_config()
        self.assertEqual(
            cfg.resolve_model("devin/swe-1-6"), ("devin", "swe-1-6")
        )

    def test_catalog_lists_seeds_offline(self):
        cfg = devin_config()
        entries = provider_model_catalog(cfg, "devin")
        self.assertEqual(
            [entry["model_id"] for entry in entries],
            ["devin/swe-1-6", "devin/swe-1-6-fast"],
        )
        self.assertIn("swe-1-6", cfg._devin_roster)
        self.assertFalse(entries[0]["limits_verified"])

    def test_explicit_models_curate(self):
        cfg = devin_config(devin_models="swe-1-6")
        entries = provider_model_catalog(cfg, "devin")
        self.assertEqual([entry["model_id"] for entry in entries], ["devin/swe-1-6"])

    def test_provider_ready_matrix(self):
        self.assertTrue(_devin_provider_ready(devin_config()))
        missing = devin_config(devin_api_key=None)
        with tempfile.TemporaryDirectory() as directory:
            missing.devin_auth_db = os.path.join(directory, "agent.db")
            self.assertFalse(_devin_provider_ready(missing))

    def test_credential_status_reports(self):
        ok, detail = _credential_status(devin_config(), "devin")
        self.assertTrue(ok)
        self.assertIn("explicit token", detail)

    def test_forward_nonstream_returns_final(self):
        cfg = devin_config()
        handler, seen = self._handler(cfg)
        final = {"id": "r", "status": "completed", "output": []}
        with patch("muse_devin.run_turn_final", return_value=final) as mocked:
            MuseShimHandler.forward_devin(
                handler, {"model": "swe-1-6", "input": "hi"}
            )
        self.assertEqual(seen["status"], 200)
        self.assertEqual(seen["payload"], final)
        self.assertEqual(mocked.call_args[0][2].uid, "swe-1-6")

    def test_forward_stream_prestream_failure_before_headers(self):
        cfg = devin_config()
        handler, _ = self._handler(cfg)
        header_calls = []
        handler._send_upstream_headers = (
            lambda *args, **kwargs: header_calls.append(True)
        )

        def failing():
            raise DevinChatError("boom")
            yield

        with patch("muse_devin.run_turn_events", return_value=failing()):
            with self.assertRaises(ProviderError) as raised:
                MuseShimHandler.forward_devin(
                    handler, {"model": "swe-1-6", "input": "hi", "stream": True}
                )
        self.assertEqual(raised.exception.code, "devin_chat_error")
        self.assertEqual(header_calls, [])
        self.assertEqual(handler.wfile.getvalue(), b"")

    def test_forward_stream_empty_turn_errors(self):
        cfg = devin_config()
        handler, _ = self._handler(cfg)
        header_calls = []
        handler._send_upstream_headers = (
            lambda *args, **kwargs: header_calls.append(True)
        )
        with patch("muse_devin.run_turn_events", return_value=iter([])):
            with self.assertRaises(ProviderError) as raised:
                MuseShimHandler.forward_devin(
                    handler, {"model": "swe-1-6", "input": "hi", "stream": True}
                )
        self.assertEqual(raised.exception.code, "devin_chat_error")
        self.assertEqual(header_calls, [])

    def test_forward_stream_emits_primed_first_event(self):
        cfg = devin_config()
        handler, _ = self._handler(cfg)
        header_calls = []
        handler._send_upstream_headers = (
            lambda *args, **kwargs: header_calls.append(True)
        )
        events = [
            ("response.output_text.delta", {
                "type": "response.output_text.delta", "delta": "Hi",
            }),
            ("response.completed", {
                "type": "response.completed", "response": {"id": "r"},
            }),
        ]
        with patch("muse_devin.run_turn_events", return_value=iter(events)):
            MuseShimHandler.forward_devin(
                handler, {"model": "swe-1-6", "input": "hi", "stream": True}
            )
        self.assertEqual(len(header_calls), 1)
        body = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("event: response.output_text.delta", body)
        self.assertIn("event: response.completed", body)
        self.assertLess(
            body.index("event: response.output_text.delta"),
            body.index("event: response.completed"),
        )

    def test_forward_unknown_model_errors(self):
        cfg = devin_config()
        handler, _ = self._handler(cfg)
        with self.assertRaises(ProviderError) as raised:
            MuseShimHandler.forward_devin(handler, {"model": "nope", "input": "hi"})
        self.assertEqual(raised.exception.code, "devin_unknown_model")

    def test_forward_missing_credential_errors(self):
        cfg = devin_config(devin_api_key=None)
        with tempfile.TemporaryDirectory() as directory:
            cfg.devin_auth_db = os.path.join(directory, "agent.db")
            handler, _ = self._handler(cfg)
            with self.assertRaises(ProviderError) as raised:
                MuseShimHandler.forward_devin(
                    handler, {"model": "swe-1-6", "input": "hi"}
                )
        self.assertEqual(raised.exception.code, "devin_credential_unavailable")

    def test_provider_error_with_upstream_code_responds(self):
        # Regression: _upstream_diagnostic carries its own "code" (e.g.
        # invalid_value); splatting it beside our code= crashed with
        # TypeError and dropped the connection (client: "malformed
        # response data" after 10 retries).
        seen = {}

        def send_json(status, payload):
            seen["status"] = status
            seen["payload"] = payload

        logged = {}

        def log_event(event, **fields):
            logged[event] = fields

        cfg = devin_config()
        handler = SimpleNamespace(
            config=cfg,
            headers={},
            _response_started=False,
            _request_id="test-code-collision",
            _upstream_diagnostic={
                "type": "invalid_request_error",
                "code": "invalid_value",
            },
            _log_event=log_event,
            send_json=send_json,
        )
        MuseShimHandler.send_provider_error(
            handler,
            ProviderError("Codex upstream returned HTTP 400.",
                          status=400, code="codex_upstream_error"),
        )
        self.assertEqual(seen["status"], 400)
        self.assertEqual(seen["payload"]["error"]["type"],
                         "codex_upstream_error")
        self.assertEqual(
            logged["provider_error"]["upstream_code"], "invalid_value")

    def test_forward_crash_returns_500_json(self):
        # End to end over loopback: an unexpected exception inside forward
        # must answer 500 JSON, never drop the connection.
        cfg = devin_config()
        try:
            server = MuseShimServer(("127.0.0.1", 0), cfg)
        except OSError:
            self.skipTest("loopback sockets are unavailable")
        thread = threading.Thread(target=server.serve_forever, daemon=True)

        def explode(handler_self, body):
            raise RuntimeError("boom")

        with patch("muse_gateway.MuseShimHandler.forward_devin",
                   new=explode):
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/responses",
                    data=json.dumps(
                        {"model": "devin/swe-1-6", "input": "hi"}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual(raised.exception.code, 500)
        payload = json.loads(raised.exception.read().decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "internal_error")


if __name__ == "__main__":
    unittest.main()
