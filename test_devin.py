"""Devin provider tests. Fully offline.

Covers the Connect wire layer, credentials, discovery normalization, chat
translation, stream consumption, and gateway wiring. Network boundaries are
mocked at muse_devin.unary_rpc / open_chat_stream; byte vectors come from
junie's prost golden tests (independent oracle).
"""

import argparse
import gzip
import io
import json
import os
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from muse_gateway import Config, MuseShimServer, run_doctor
from tests.peers import serving

import devin_pb2
from muse_devin import DevinChatError, DevinCredentialError, DevinModel, DevinStreamState, credential_state, deref_schema_refs, fetch_auth, mark_auth_failure, normalize_configs, normalize_schema_for_google, resolve_access_token, run_turn_events
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


class UnclosableBytesIO(io.BytesIO):
    def close(self):
        pass


class DevinWireTests(unittest.TestCase):
    def test_golden_request_bytes(self):
        auth = devin_pb2.GetUserJwtRequest()
        auth.metadata.api_key = "k"
        auth.metadata.user_jwt = "j"
        self.assertEqual(
            auth.SerializeToString(),
            bytes([0x0A, 0x07, 0x1A, 0x01, 0x6B, 0xAA, 0x01, 0x01, 0x6A]),
        )
        chat = devin_pb2.GetChatMessageRequest(
            prompt="p",
            chat_model_uid="m",
            request_type=devin_pb2.ChatMessageRequestType.Value(
                "CHAT_MESSAGE_REQUEST_TYPE_CASCADE"
            ),
            cascade_id="c",
            execution_id="e",
        )
        chat.metadata.api_key = "k"
        chat.metadata.user_jwt = "j"
        chat.model_assignment_jwt = "jwt"
        self.assertEqual(
            chat.SerializeToString(),
            bytes([
                0x0A, 0x07, 0x1A, 0x01, 0x6B, 0xAA, 0x01, 0x01, 0x6A,
                0x12, 0x01, 0x70, 0x38, 0x05, 0x82, 0x01, 0x01, 0x63,
                0xAA, 0x01, 0x01, 0x6D, 0xB2, 0x01, 0x01, 0x65,
                0xD2, 0x01, 0x03, 0x6A, 0x77, 0x74,
            ]),
        )

    def test_frame_decoder_chunk_splits_and_oversize_prefixes(self):
        # Literal protocol bytes, not a round-trip through our own encoder.
        wire = b"\x00\x00\x00\x00\x05hello\x01\x00\x00\x00\x06world!"
        for boundary in range(len(wire) + 1):
            with self.subTest(boundary=boundary):
                decoder = ConnectFrameDecoder()
                events = decoder.decode(wire[:boundary]) + decoder.decode(wire[boundary:])
                self.assertEqual(events, [(0, b"hello"), (1, b"world!")])
        with self.assertRaises(DevinFrameError):
            ConnectFrameDecoder(max_payload_bytes=4).decode(frame_message(b"toolong", 0))
        # A valid frame before an oversized prefix is delivered whether the
        # prefix arrives combined or split; the oversized frame then fails.
        valid = b"\x00\x00\x00\x00\x01x"
        oversized = b"\x00" + struct.pack(">I", 5)
        combined = ConnectFrameDecoder(max_payload_bytes=4)
        self.assertEqual(combined.decode(valid + oversized), [(0, b"x")])
        self.assertEqual(combined.pending_bytes, len(oversized))
        with self.assertRaises(DevinFrameError):
            combined.decode()
        split = ConnectFrameDecoder(max_payload_bytes=4)
        self.assertEqual(split.decode(valid), [(0, b"x")])
        with self.assertRaises(DevinFrameError):
            split.decode(oversized)

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
            with patch("muse_devin_wire.build_gateway_opener", return_value=unittest.mock.Mock(**{"open.side_effect": error})):
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

    def test_explicit_cli_and_unconfigured_chain(self):
        self.assertEqual(credential_state()["state"], "unconfigured")
        with self.assertRaisesRegex(DevinCredentialError, "devin auth login"):
            resolve_access_token()
        with patch("muse_devin.read_cli_token", return_value=("cli-token", "cli.toml")) as read:
            self.assertEqual(resolve_access_token(), ("cli-token", "devin-cli:cli.toml"))
            self.assertEqual(credential_state()["source"], "devin-cli")
            read.reset_mock()
            self.assertEqual(resolve_access_token("explicit-token"), ("explicit-token", "explicit"))
            self.assertEqual(credential_state("explicit-token")["source"], "explicit")
            read.assert_not_called()

    def test_quarantine_is_source_independent_without_silent_fallback(self):
        from muse_devin import _AUTH_FAILURE_MARKS, _token_fingerprint

        token = "quarantined-token"
        mark_auth_failure(token, "probe 401")
        self.addCleanup(_AUTH_FAILURE_MARKS.pop, _token_fingerprint(token), None)
        with patch("muse_devin.read_cli_token", return_value=("healthy-cli", "cli.toml")) as read:
            self.assertEqual(credential_state(token)["state"], "auth-failed")
            with self.assertRaisesRegex(DevinCredentialError, "quarantined"):
                resolve_access_token(token)
            read.assert_not_called()
        with patch("muse_devin.read_cli_token", return_value=(token, "cli.toml")):
            self.assertEqual(credential_state()["state"], "auth-failed")
            with self.assertRaisesRegex(DevinCredentialError, "quarantined"):
                resolve_access_token()
        with patch("muse_devin.read_cli_token", return_value=("new-cli", "cli.toml")):
            self.assertEqual(resolve_access_token(), ("new-cli", "devin-cli:cli.toml"))

    def test_doctor_reports_cli_readiness_without_tokens_or_row_listing(self):
        cfg = devin_config(devin_api_key=None, upstream="http://127.0.0.1:1")
        cfg.doctor_provider = "devin"
        output = io.StringIO()
        with patch("muse_devin.read_cli_token", return_value=("doctor-secret", "cli.toml")):
            self.assertEqual(run_doctor(cfg, output=output), 0)
        rendered = output.getvalue()
        self.assertIn("credential ready (Devin CLI login)", rendered)
        self.assertNotIn("doctor-secret", rendered)
        self.assertNotIn("devin store:", rendered)
        self.assertNotIn("devin credential:", rendered)

    def test_server_directed_base_must_be_https_unless_loopback(self):
        # P2-09: the auth response cannot downgrade the session to cleartext.
        import contextlib

        def auth_reply(url):
            body = devin_pb2.GetUserJwtResponse(user_jwt="jwt", custom_api_server_url=url).SerializeToString()
            return lambda *args, **kwargs: (200, "application/proto", body)

        with patch("muse_devin.unary_rpc", side_effect=auth_reply("http://evil.example")), \
                self.assertRaises(DevinAuthError):
            fetch_auth("tok", "https://x")
        for url in ("http://127.0.0.1:9", "https://chat.example"):
            err = io.StringIO()
            with self.subTest(url=url), contextlib.redirect_stderr(err), \
                    patch("muse_devin.unary_rpc", side_effect=auth_reply(url)):
                _, jwt, effective = fetch_auth("tok", "https://x")
            self.assertEqual((jwt, effective), ("jwt", url))
            self.assertIn("directed chat traffic to " + url, err.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            devin_config(devin_upstream="http://onprem.example/devin")
        self.assertIn("uses plain http", err.getvalue())

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

    def test_cli_token_file_variants(self):
        from muse_devin import read_cli_token

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_toml(
                directory,
                'windsurf_api_key = "tok-123"\napi_server_url = "https://x"\n',
            )
            self.assertEqual(read_cli_token(path), ("tok-123", path))
            self.assertEqual(read_cli_token(os.path.join(directory, "nope.toml")), (None, None))
            for text in ('api_server_url = "https://x"\n', "windsurf_api_key = \n"):
                path = self._write_toml(directory, text)
                self.assertEqual(read_cli_token(path), (None, None))
            path = self._write_toml(directory, '# CLI credentials\nwindsurf_api_key = "  cli-token  "\n')
            with patch.dict("sys.modules", {"tomllib": None}):
                self.assertEqual(read_cli_token(path), ("cli-token", path))

    def test_cli_candidates_prefer_xdg_data_home(self):
        from muse_devin import _cli_credentials_candidates

        with tempfile.TemporaryDirectory() as home:
            with patch("os.path.expanduser", return_value=home), patch.dict(
                os.environ, {"XDG_DATA_HOME": "", "XDG_CONFIG_HOME": ""}, clear=False,
            ):
                os.environ.pop("XDG_DATA_HOME", None)
                os.environ.pop("XDG_CONFIG_HOME", None)
                candidates = _cli_credentials_candidates()
            self.assertEqual(
                candidates[0], os.path.join(home, ".local", "share", "devin", "credentials.toml"),
            )
            with patch("os.path.expanduser", return_value=home), patch.dict(
                os.environ, {"XDG_DATA_HOME": os.path.join(home, "data")}, clear=False,
            ):
                candidates = _cli_credentials_candidates()
            self.assertEqual(
                candidates[0], os.path.join(home, "data", "devin", "credentials.toml"),
            )

    def test_runtime_catalog_and_inference_follow_cli_relogin_and_explicit_token(self):
        from muse_devin import _AUTH_FAILURE_MARKS, _token_fingerprint
        from muse_gateway import _devin_provider_ready

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_toml(directory, 'windsurf_api_key = "cli-first"\n')
            cfg = devin_config(provider="devin", devin_api_key=None,
                               log_file=os.path.join(directory, "gateway.log"))
            server = MuseShimServer(("127.0.0.1", 0), cfg)
            with serving(server) as base:
                request = urllib.request.Request(
                    base + "/responses",
                    data=json.dumps({"model": "devin/swe-1-6", "input": "hi"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with patch("muse_devin._cli_credentials_candidates", return_value=[path]), patch(
                    "muse_devin.run_turn_final", return_value={"id": "resp_test"},
                ) as turn:
                    for token in ("cli-first", "cli-relogin"):
                        original = f'windsurf_api_key = "{token}"\n'
                        self._write_toml(directory, original)
                        self.assertTrue(_devin_provider_ready(cfg))
                        with urllib.request.urlopen(base + "/muse-code/models", timeout=5) as response:
                            self.assertIn("devin/swe-1-6", [row["id"] for row in json.load(response)["data"]])
                        with urllib.request.urlopen(request, timeout=5) as response:
                            self.assertEqual(json.load(response), {"id": "resp_test"})
                        self.assertEqual(turn.call_args.args[3], token)
                        with open(path, encoding="utf-8") as handle:
                            self.assertEqual(handle.read(), original)
                    cfg.devin_api_key = "explicit-runtime-token"
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                    self.assertEqual(turn.call_args.args[3], "explicit-runtime-token")
                    mark_auth_failure(cfg.devin_api_key)
                    self.addCleanup(_AUTH_FAILURE_MARKS.pop, _token_fingerprint(cfg.devin_api_key), None)
                    self.assertFalse(_devin_provider_ready(cfg))
                    turn.reset_mock()
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request, timeout=5)
                    with caught.exception as response:
                        self.assertEqual(json.load(response)["error"]["type"], "devin_credential_unavailable")
                    turn.assert_not_called()


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

    def test_google_normalize_strips_spills_and_keeps_nullability(self):
        schema = {
            "type": ["string", "null"],
            "minLength": 3,
            "description": "d",
            "properties": {
                "a": {"type": ["null", "integer"]},
                "b": {"type": "integer"},
            },
        }
        cleaned = normalize_schema_for_google(schema)
        self.assertEqual((cleaned["type"], cleaned["nullable"]), ("string", True))
        self.assertNotIn("minLength", cleaned)
        self.assertEqual(cleaned["description"], "d\n\n{minLength: 3}")
        self.assertEqual(cleaned["properties"]["a"], {"type": "integer", "nullable": True})
        self.assertNotIn("nullable", cleaned["properties"]["b"])
        with self.assertRaises(DevinChatError):
            normalize_schema_for_google({"type": ["string", "number"]})


class DevinRequestPolicyTests(unittest.TestCase):
    """P1-05 (inverts R08 Devin half): per-request policy beats capability."""

    TOOL = {"type": "function", "name": "read_file", "parameters": {"type": "object"}}

    def _request(self, **body):
        from muse_devin import build_chat_request, build_tools
        model = devin_model(supports_parallel=True)
        return build_chat_request("", {"input": "hi", **body}, model, model.uid, "k", "jwt",
                                  "cascade", [], build_tools([self.TOOL], False))

    def test_none_serial_named_required_and_stop_sequences_reach_the_wire(self):
        default = self._request()
        self.assertEqual((default.tool_choice.option_name, len(default.tools)), ("auto", 1))
        # M12 live capture: option_name "required" forced a call on swe-1-6.
        required = self._request(tool_choice="required")
        self.assertEqual(required.tool_choice.WhichOneof("choice"), "option_name")
        self.assertEqual(required.tool_choice.option_name, "required")
        self.assertFalse(default.disable_parallel_tool_calls)
        self.assertEqual(len(self._request(tool_choice="none").tools), 0)
        self.assertTrue(self._request(parallel_tool_calls=False).disable_parallel_tool_calls)
        named = self._request(tool_choice={"type": "function", "name": "read_file"})
        self.assertEqual(named.tool_choice.WhichOneof("choice"), "tool_name")
        self.assertEqual(named.tool_choice.tool_name, "read_file")
        self.assertEqual(list(self._request(stop_sequences=["END"]).configuration.stop_patterns), ["END"])

    def test_required_without_tools_is_rejected_before_inference(self):
        with patch(
            "muse_devin.fetch_auth", side_effect=AssertionError("policy check reached upstream")
        ):
            with self.assertRaises(DevinChatError) as caught:
                list(run_turn_events({"input": "hi", "tool_choice": "required"}, [], devin_model(), "k"))
        self.assertEqual(caught.exception.code, "devin_unsupported_tool_choice")

    def test_unrepresentable_controls_are_rejected_before_inference(self):
        cases = (
            ({"tool_choice": {"type": "function", "name": "missing"}}, "devin_unsupported_tool_choice"),
            ({"tool_choice": {"type": "web_search"}}, "devin_unsupported_tool_choice"),
            ({"parallel_tool_calls": "no"}, "devin_invalid_request"),
            ({"stop": 7}, "devin_invalid_request"),
            ({"stop": ["END", 7]}, "devin_invalid_request"),
            ({"stop": ""}, "devin_invalid_request"),
            ({"max_output_tokens": 0}, "invalid_output_limit"),
            ({"max_output_tokens": -5}, "invalid_output_limit"),
        )
        for extra, code in cases:
            with self.subTest(extra=extra), patch(
                "muse_devin.fetch_auth", side_effect=AssertionError("policy check reached upstream")
            ):
                with self.assertRaises(DevinChatError) as caught:
                    list(run_turn_events({"input": "hi", **extra}, [self.TOOL], devin_model(), "k"))
                self.assertEqual(caught.exception.code, code)


class DevinHistoryReferenceTests(unittest.TestCase):
    def test_item_reference_resolves_inline_or_is_rejected(self):
        # P2-04: resolvable references name inline content; others are rejected.
        from muse_devin import history_to_prompts
        inline = {"type": "message", "id": "msg_1", "role": "assistant", "content": "kept"}
        _, staged, _ = history_to_prompts({"input": [
            {"role": "user", "content": "hi"}, inline,
            {"type": "item_reference", "id": "msg_1"}, {"role": "user", "content": "go"}]}, "c")
        self.assertEqual([prompt["text"] for prompt in staged], ["hi", "kept", "go"])
        with patch("muse_devin.fetch_auth", side_effect=AssertionError("reached upstream")), \
                self.assertRaises(DevinChatError) as caught:
            list(run_turn_events({"input": [{"role": "user", "content": "hi"},
                                            {"type": "item_reference", "id": "msg_missing"}]},
                                 [], devin_model(), "k"))
        self.assertEqual(caught.exception.code, "devin_unresolved_reference")

    def test_base64_images_reach_devin_proto_and_url_images_are_rejected(self):
        from muse_devin import build_history_prompts, history_to_prompts

        _, staged, _ = history_to_prompts({"input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,AAEC"},
            ],
        }]}, "cascade")
        protos = build_history_prompts(staged, "cascade")
        self.assertEqual(protos[0].prompt, "look")
        self.assertEqual(len(protos[0].images), 1)
        self.assertEqual(protos[0].images[0].base64_data, "AAEC")
        self.assertEqual(protos[0].images[0].mime_type, "image/jpeg")
        with self.assertRaises(DevinChatError):
            history_to_prompts({"input": [{
                "role": "user",
                "content": [{"type": "input_image", "image_url": "https://example.com/image.png"}],
            }]}, "cascade")

    def test_colliding_wire_ids_are_allocated_distinctly_in_history(self):
        from muse_devin import history_to_prompts, normalize_tool_call_id

        seen = set()
        self.assertEqual(normalize_tool_call_id("call:x", seen), "call_x")
        self.assertEqual(normalize_tool_call_id("call_x", seen), "call_x-2")
        _, staged, call_ids = history_to_prompts({"input": [
            {"type": "function_call", "call_id": "call:x", "name": "echo", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_x", "name": "echo", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call:x", "output": "first"},
            {"type": "function_call_output", "call_id": "call_x", "output": "second"},
        ]}, "c")
        calls = staged[0]["tool_calls"]
        self.assertEqual([call["id"] for call in calls], ["call_x", "call_x-2"])
        self.assertEqual([prompt["tool_call_id"] for prompt in staged[1:]], ["call_x", "call_x-2"])
        self.assertEqual(call_ids, {"call:x": "call_x", "call_x": "call_x-2"})


class DevinStreamTests(unittest.TestCase):
    def _run_stream(self, stream, body=None, track=None, register_secret=None):
        with patch("muse_devin.fetch_auth", return_value=("k", "jwt", "https://x")), patch(
            "muse_devin.open_chat_stream", return_value=stream
        ):
            return list(run_turn_events(
                body or {"input": "hi"}, [], devin_model(), "k",
                track=track, register_secret=register_secret,
            ))

    def test_open_response_is_tracked_for_shutdown(self):
        from muse_devin_wire import CONNECT_END_STREAM_FLAG, frame_message

        message = devin_pb2.GetChatMessageResponse(
            message_id="r-track", delta_text="Hi", stop_reason=2)
        frames = (
            frame_message(message.SerializeToString())
            + frame_message(b"{}", CONNECT_END_STREAM_FLAG)
        )
        stream = UnclosableBytesIO(frames)
        tracked = []
        registered = []
        events = self._run_stream(stream, track=tracked.append, register_secret=registered.append)
        self.assertEqual(tracked, [stream])
        self.assertEqual(registered, ["k", "jwt"])
        self.assertEqual(events[-1][0], "response.completed")

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

    def test_fail_is_idempotent_and_preserves_started_events_after_feed_error(self):
        state = DevinStreamState("devin/swe-1-6")
        message = devin_pb2.GetChatMessageResponse(message_id="r-fail")
        call = message.delta_tool_calls.add()
        call.id, call.name = "c1", "t"
        call.arguments_json = "x" * (8 * 1024 * 1024 + 1)
        with self.assertRaises(DevinChatError):
            state.feed(message)
        events = state.fail("boom", "upstream_error")
        self.assertEqual(
            [kind for kind, _ in events],
            ["response.created", "response.in_progress", "response.failed"],
        )
        self.assertEqual([payload["sequence_number"] for _, payload in events], [0, 1, 2])
        self.assertEqual(state.fail("again", "upstream_error"), [])

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

    def test_empty_tool_arguments_and_min_log_prob_complete(self):
        # M12 live capture: a forced no-arg call arrives with
        # arguments_json="" and stop 10; claude-haiku-5-5 ends normal
        # short text turns with stop 4 (MIN_LOG_PROB).
        state = DevinStreamState("devin/swe-1-6")
        message = devin_pb2.GetChatMessageResponse(message_id="r-empty", stop_reason=10)
        call = message.delta_tool_calls.add()
        call.id, call.name, call.arguments_json = "c1", "record_color", ""
        events = state.feed(message) + state.finish()
        self.assertEqual(events[-1][0], "response.completed")
        final_call = events[-1][1]["response"]["output"][0]
        self.assertEqual(final_call["arguments"], "{}")

        state = DevinStreamState("devin/claude-haiku-5-5")
        message = devin_pb2.GetChatMessageResponse(
            message_id="r-haiku", delta_text="LIVE_OK", stop_reason=4)
        events = state.feed(message) + state.finish()
        self.assertEqual(events[-1][0], "response.completed")
        self.assertEqual(events[-1][1]["response"]["status"], "completed")

    def test_nested_prefix_fragments_survive_every_chunk_boundary(self):
        # P1-04 (inverts R07): a delta that happens to share the received
        # prefix is appended, never mistaken for a snapshot.
        target = '{"x":{"x":1}}'
        splits = [[target[:i], target[i:]] for i in range(1, len(target))]
        splits += [[target[:i], target[i:j], target[j:]]
                   for i in range(1, len(target)) for j in range(i + 1, len(target))]
        splits.append(['{"x":', '{"x":1}}'])
        for parts in splits:
            with self.subTest(parts=parts):
                state = DevinStreamState("devin/swe-1-6")
                events = []
                for index, fragment in enumerate(parts):
                    message = devin_pb2.GetChatMessageResponse(message_id="r-split")
                    call = message.delta_tool_calls.add()
                    if index == 0:
                        call.id, call.name = "c1", "t"
                    call.arguments_json = fragment
                    events += state.feed(message)
                events += state.finish()
                self.assertEqual(events[-1][0], "response.completed")
                arguments = events[-1][1]["response"]["output"][0]["arguments"]
                self.assertEqual(json.loads(arguments), {"x": {"x": 1}})
                self.assertNotIn("__parseError", arguments)

    def test_oversized_tool_arguments_end_in_one_failed_terminal(self):
        # P1-13: per-block arguments are capped at Kiro's _BLOCK_LIMIT.
        from muse_devin_wire import CONNECT_END_STREAM_FLAG, frame_message
        frames = b""
        for index in range(9):
            message = devin_pb2.GetChatMessageResponse(message_id="r-big")
            call = message.delta_tool_calls.add()
            if index == 0:
                call.id, call.name = "c1", "t"
            call.arguments_json = "x" * (1024 * 1024)
            frames += frame_message(message.SerializeToString())
        frames += frame_message(b"{}", CONNECT_END_STREAM_FLAG)
        events = self._run_stream(UnclosableBytesIO(frames))
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds.count("response.failed"), 1)
        self.assertEqual(kinds[-1], "response.failed")
        self.assertNotIn("response.completed", kinds)
        self.assertIn("safety size limit", events[-1][1]["response"]["error"]["message"])

    def test_text_thinking_and_final_response_caps_are_byte_based(self):
        import muse_devin

        with patch.object(muse_devin, "_STREAM_BLOCK_LIMIT", 4):
            state = DevinStreamState("devin/swe-1-6")
            with self.assertRaises(DevinChatError):
                state.feed(devin_pb2.GetChatMessageResponse(
                    message_id="r-text-cap", delta_text="12345"))
            state = DevinStreamState("devin/swe-1-6")
            with self.assertRaises(DevinChatError):
                state.feed(devin_pb2.GetChatMessageResponse(
                    message_id="r-think-cap", delta_thinking="12345"))
        with patch.object(muse_devin, "_STREAM_RESPONSE_LIMIT", 64):
            final = muse_devin._enforce_final_response_size({
                "id": "resp_big", "object": "response", "status": "completed",
                "model": "devin/swe-1-6", "output": [{"type": "message", "text": "x" * 100}],
            })
        self.assertEqual(final["status"], "failed")
        self.assertEqual(final["error"]["code"], "upstream_response_too_large")
        self.assertLessEqual(len(json.dumps(final).encode()), 512)

    def test_tool_before_text_keeps_output_index_order(self):
        # P2-04: a deferred tool released after streaming text never makes
        # output_index run backwards for positional consumers.
        from tests.contracts import consume
        from muse_providers import response_sse_line
        state = DevinStreamState("devin/swe-1-6")
        tool = devin_pb2.GetChatMessageResponse(message_id="r-order")
        call = tool.delta_tool_calls.add()
        call.id, call.name, call.arguments_json = "c1", "t", "{}"
        text = devin_pb2.GetChatMessageResponse(delta_text="after", stop_reason=10)
        events = state.feed(tool) + state.feed(text) + state.finish()
        added = [payload["output_index"] for kind, payload in events if kind == "response.output_item.added"]
        self.assertEqual(added, sorted(added))
        final = consume(b"".join(response_sse_line(kind, payload) for kind, payload in events))
        self.assertEqual([item["type"] for item in final["output"]], ["message", "function_call"])

    def test_item_ids_are_allocated_once_and_never_collide(self):
        # P1-03: call_x/fc_x derive the same function ID; text/tool/text and
        # repeated deltas must still publish distinct, stable item IDs.
        state = DevinStreamState("devin/swe-1-6")
        messages = [devin_pb2.GetChatMessageResponse(message_id="r-ids", delta_text="a")]
        for wire_id in ("call_x", "fc_x"):
            message = devin_pb2.GetChatMessageResponse()
            call = message.delta_tool_calls.add()
            call.id, call.name, call.arguments_json = wire_id, "t", "{}"
            messages.append(message)
        messages.append(devin_pb2.GetChatMessageResponse(delta_text="b"))
        events = [event for message in messages for event in state.feed(message)] + state.finish()
        final = events[-1][1]["response"]
        ids = [item["id"] for item in final["output"]]
        self.assertEqual([item["type"] for item in final["output"]],
                         ["message", "function_call", "function_call"])
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual([item.get("call_id") for item in final["output"][1:]], ["call_x", "fc_x"])
        text_ids = {payload["item_id"] for kind, payload in events if kind == "response.output_text.delta"}
        self.assertEqual(text_ids, {ids[0]})
        for kind, payload in events:
            if kind == "response.output_item.added":
                self.assertIn(payload["item"]["id"], ids)
            if "item_id" in payload:
                self.assertIn(payload["item_id"], ids)
        self.assertEqual(len(state._used_item_ids), len(ids))

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
        events = self._run_stream(response)
        kinds = [kind for kind, _ in events]
        self.assertIn("response.output_text.delta", kinds)
        self.assertEqual(kinds[-1], "response.failed")
        self.assertNotIn("response.completed", kinds)
        self.assertTrue(response.isclosed())

    def test_turn_rejects_context_overflow(self):
        from muse_devin_wire import CONNECT_END_STREAM_FLAG, frame_message

        stream = frame_message(
            b'{"error":{"code":"invalid_argument","message":"internal error"}}',
            CONNECT_END_STREAM_FLAG,
        )
        body = {"input": [
            {"type": "message", "role": "assistant", "content": "x" * (600 * 1024)},
            {"type": "message", "role": "user", "content": "hi"},
        ]}
        with self.assertRaises(DevinChatError) as raised:
            self._run_stream(UnclosableBytesIO(stream), body)
        self.assertEqual(raised.exception.code, "devin_context_overflow")

    def test_turn_error_trailer_with_details_yields_failed(self):
        from muse_devin_wire import CONNECT_END_STREAM_FLAG, frame_message, parse_trailer_error

        trailer = json.dumps({"error": {
            "code": "unavailable" + "x" * 5000,
            "message": "overloaded " + "m" * 5000,
            "details": [{"type": "retryInfo", "value": "later " + "d" * 5000}],
        }})
        parsed = parse_trailer_error(trailer)
        self.assertLessEqual(len(parsed.code), 128)
        self.assertLessEqual(len(parsed.message.encode()), 512)
        self.assertLessEqual(len(parsed.formatted.encode()), 512)
        self.assertLessEqual(len(parsed.raw), 2001)
        stream = frame_message(trailer.encode(), CONNECT_END_STREAM_FLAG)
        events = self._run_stream(UnclosableBytesIO(stream))
        kinds = [event for event, _ in events]
        self.assertEqual(kinds[:2], ["response.created", "response.in_progress"])
        self.assertEqual(kinds[2], "response.failed")
        self.assertEqual(kinds[-1], "response.failed")
        message = events[2][1]["response"]["error"]["message"]
        self.assertIn("overloaded", message)
        self.assertLessEqual(len(message.encode()), 512)
        self.assertNotIn("m" * 1000, message)
        self.assertNotIn("d" * 1000, message)


class DevinGatewayTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("muse_devin.read_cli_token", return_value=(None, None))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_unknown_model_and_missing_credential_are_public_json_rejections(self):
        for model, key, expected in (("nope", "dummy-key", "devin_unknown_model"),
                                     ("swe-1-6", None, "devin_credential_unavailable")):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                cfg = devin_config(devin_api_key=key, log_file=os.path.join(directory, "gateway.log"))
                server = MuseShimServer(("127.0.0.1", 0), cfg)
                with serving(server) as base:
                    req = urllib.request.Request(base + "/responses",
                        data=json.dumps({"model": "devin/" + model, "input": "hi", "stream": True}).encode(),
                        headers={"Content-Type": "application/json"})
                    with patch("muse_devin.fetch_auth", side_effect=AssertionError("Rejected input reached inference")):
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=5)
                    with caught.exception as response:
                        self.assertEqual(response.code, 400)
                        self.assertEqual(response.headers.get_content_type(), "application/json")
                        self.assertEqual(json.load(response)["error"]["type"], expected)


if __name__ == "__main__":
    unittest.main()
