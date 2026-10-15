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
from unittest.mock import patch

from muse_gateway import Config, MuseShimServer

import devin_pb2
from muse_devin import DevinChatError, DevinCredentialError, DevinModel, DevinStreamState, credential_state, deref_schema_refs, fetch_auth, normalize_configs, normalize_schema_for_google, read_store_credentials, resolve_access_token, run_turn_events, select_credential
from muse_devin_wire import ConnectFrameDecoder, DevinAuthError, DevinFrameError, DevinHttpError, decode_unary, frame_message, unary_rpc


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

    def test_frames_reassemble_across_chunks(self):
        # Literal protocol bytes, not a round-trip through our own encoder.
        wire = b"\x00\x00\x00\x00\x05hello\x01\x00\x00\x00\x06world!"
        for boundary in range(len(wire) + 1):
            with self.subTest(boundary=boundary):
                decoder = ConnectFrameDecoder()
                events = decoder.decode(wire[:boundary]) + decoder.decode(wire[boundary:])
                self.assertEqual(events, [(0, b"hello"), (1, b"world!")])

    def test_oversize_length_prefix_fails_before_buffering(self):
        decoder = ConnectFrameDecoder(max_payload_bytes=4)
        with self.assertRaises(DevinFrameError):
            decoder.decode(frame_message(b"toolong", 0))

    def test_unary_decodes_bare_and_gzip_but_rejects_garbage(self):
        # Literal field bytes, independent of the protobuf serializer.
        for body in (b"\x0a\x01j", gzip.compress(b"\x0a\x01j")):
            self.assertEqual(decode_unary(devin_pb2.GetUserJwtResponse, body).user_jwt, "j")
        self.assertIsNone(decode_unary(devin_pb2.GetUserJwtResponse, b"\xff\xff\xff"))

    def test_http_error_releases_response_even_when_exception_is_retained(self):
        body = io.BytesIO(b'{"error":{"message":"busy"}}')
        error = urllib.error.HTTPError("http://127.0.0.1/rpc", 503, "busy",
                                       {"Content-Type": "application/json"}, body)
        try:
            with patch("muse_devin_wire.urllib.request.urlopen", side_effect=error):
                with self.assertRaises(DevinHttpError) as raised:
                    unary_rpc("http://127.0.0.1", "/rpc", b"request")
            self.assertEqual(raised.exception.status, 503)
            self.assertIn("busy", str(raised.exception))
            self.assertTrue(body.closed, "HTTP error response leaked until exception garbage collection")
        finally:
            error.close()


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

    def test_fetch_auth_marks_persistent_401(self):
        from muse_devin import auth_failure_mark, fetch_auth
        from muse_devin_wire import DevinHttpError

        def fail_401(*args, **kwargs):
            raise DevinHttpError("request", 401, "unauthorized")

        with patch("muse_devin.unary_rpc", side_effect=fail_401):
            with self.assertRaises(DevinAuthError):
                fetch_auth("mark-on-401-token", "https://x")
        self.assertIsNotNone(auth_failure_mark("mark-on-401-token"))

        def fail_500(*args, **kwargs):
            raise DevinHttpError("request", 500, "oops")

        with patch("muse_devin.unary_rpc", side_effect=fail_500):
            with self.assertRaises(DevinAuthError):
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
        response.default_override_model_config.model_uid = "adaptive"
        self.assertEqual({model.logical_id: model.is_default for model in normalize_configs(response, "https://x")[0]},
                         {"adaptive": True, "swe-1-6": False}, "Account override selects a catalog model independently of family defaults")

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


class DevinTranslationTests(unittest.TestCase):
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
        with self.assertRaises(DevinChatError):
            normalize_schema_for_google({"type": ["string", "number"]})


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


class DevinGatewayTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("muse_devin.read_cli_token", return_value=(None, None))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_unknown_model_and_missing_credential_are_public_json_rejections(self):
        for model, key, expected in (("nope", "dummy-key", "devin_unknown_model"),
                                     ("swe-1-6", None, "devin_credential_unavailable")):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                cfg = devin_config(devin_api_key=key, devin_auth_db=os.path.join(directory, "missing.db"),
                                   log_file=os.path.join(directory, "gateway.log"))
                server = MuseShimServer(("127.0.0.1", 0), cfg)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/responses",
                        data=json.dumps({"model": "devin/" + model, "input": "hi", "stream": True}).encode(),
                        headers={"Content-Type": "application/json"})
                    with patch("muse_devin.fetch_auth", side_effect=AssertionError("Rejected input reached inference")):
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=5)
                    with caught.exception as response:
                        self.assertEqual(response.code, 400)
                        self.assertEqual(response.headers.get_content_type(), "application/json")
                        self.assertEqual(json.load(response)["error"]["type"], expected)
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
