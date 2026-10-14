"""Loopback protocol peers, not mocks of gateway/converter functions.

Inbound Connect bytes are authored here from the pinned schema's field tags,
not produced by the gateway's frame encoder or response serializer. Existing
Junie golden-vector tests independently check those tags against prost.
These peers simulate protocols, NOT provider accounts or model intelligence.
"""

import argparse
from contextlib import contextmanager
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.request

import devin_pb2
from muse_gateway import Config, MuseShimServer


def varint(value):
    encoded = bytearray()
    while value > 127:
        encoded.append((value & 127) | 128)
        value >>= 7
    return bytes(encoded) + bytes([value])


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    raw = value.encode() if isinstance(value, str) else value
    return varint((number << 3) | 2) + varint(len(raw)) + raw


def envelope(payload, flags=0):
    return bytes([flags]) + len(payload).to_bytes(4, "big") + payload


def chat(text="", *, stop=0, call=None):
    raw = field(1, "fixture-response")
    if text:
        raw += field(3, text)
    if stop:
        raw += field(5, stop)
    if call is not None:
        raw += field(6, b"".join(field(i, value) for i, value in enumerate(call, 1) if value))
    return envelope(raw)


TRAILER = envelope(b"{}", 2)


def text_turn(text="Hello, \u03bb \U0001f44b"):
    return chat(text[:3]) + chat(text[3:], stop=2) + TRAILER


def tool_turn(name="echo", arguments='{"text":"OK"}'):
    # Exercise cumulative arguments, omitted continuation ID and late name;
    # a single complete tool fixture misses these boundary conditions.
    prefix = arguments[:max(1, len(arguments) // 2)]
    return (chat(call=("fixture-call", "", prefix))
            + chat(call=("", name, arguments), stop=10) + TRAILER)


def responses_text(text, response_id="resp_peer"):
    """Independent Responses fixture for native byte-passthrough routes."""
    item = {"type": "message", "id": "msg_peer", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    created = {"id": response_id, "object": "response", "status": "in_progress", "output": []}
    added = {**item, "status": "in_progress", "content": []}
    payloads = [
        {"type": "response.created", "response": created},
        {"type": "response.in_progress", "response": created},
        {"type": "response.output_item.added", "output_index": 0, "item": added},
        {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_peer", "content_index": 0, "delta": text},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {**created, "status": "completed", "output": [item]}},
    ]
    return b"".join(("event: " + event["type"] + "\ndata: " + json.dumps({**event, "sequence_number": index}) + "\n\n").encode()
                    for index, event in enumerate(payloads))


def anthropic_text(text="CLAUDE_ACCEPTED"):
    events = [
        ("message_start", {"message": {"id": "msg_anthropic_peer", "model": "claude-test", "role": "assistant", "content": [],
                                         "usage": {"input_tokens": 4, "output_tokens": 0}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        ("message_stop", {}),
    ]
    return b"".join(("event: " + kind + "\ndata: " + json.dumps({"type": kind, **payload}) + "\n\n").encode()
                    for kind, payload in events)


class Peer:
    def __init__(self, turns=(), *, catalog=None, hold_after=None):
        self.turns = list(turns)
        self.catalog = catalog
        self.hold_after = hold_after
        self.release = threading.Event()
        self.requests = []
        self.errors = []

    def handler(self):
        peer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                status = 200
                if self.path.endswith("GetUserJwt"):
                    body, mime = field(1, "fixture-jwt"), "application/proto"
                elif self.path.endswith("GetCliModelConfigs"):
                    body, mime = peer.catalog or b"", "application/proto"
                else:
                    try:
                        if self.path.endswith("GetChatMessage"):
                            flags, length = raw[0], int.from_bytes(raw[1:5], "big")
                            if len(raw[5:]) != length:
                                raise ValueError("invalid outgoing Connect envelope")
                            decoded = gzip.decompress(raw[5:]) if flags & 1 else raw[5:]
                            request = devin_pb2.GetChatMessageRequest.FromString(decoded)
                            mime = "application/connect+proto"
                        else:
                            request = json.loads(raw)
                            mime = "text/event-stream"
                        peer.requests.append(request)
                        if not peer.turns:
                            raise ValueError("unexpected extra inference request")
                        turn = peer.turns.pop(0)
                        if isinstance(turn, tuple):
                            status, body = turn
                            mime = "application/json"
                        else:
                            body = turn
                    except Exception as error:
                        peer.errors.append(str(error))
                        status, body, mime = 500, b'{"error":"fixture failure"}', "application/json"
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    if peer.hold_after is not None and self.path.endswith(("GetChatMessage", "/responses")):
                        self.wfile.write(body[:peer.hold_after])
                        self.wfile.flush()
                        # Only a client observation releases the rest; timeout
                        # is a cleanup guard, not an assertion about speed.
                        peer.release.wait(8)
                        self.wfile.write(body[peer.hold_after:])
                    else:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Negative consumer tests intentionally disconnect.

        return Handler


@contextmanager
def serving(server):
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        if worker.is_alive():
            raise AssertionError("fixture server failed to stop")


@contextmanager
def gateway(directory, peer, **settings):
    with serving(ThreadingHTTPServer(("127.0.0.1", 0), peer.handler())) as upstream:
        options = dict(provider="auto", devin_upstream=upstream, meta_upstream=upstream,
                       codex_upstream=upstream, devin_api_key="fixture-devin-token",
                       meta_api_key="fixture-meta-token", model_discovery=False,
                       log_file=str(Path(directory) / "gateway.log"), timeout=4)
        options.update(settings)
        if options["provider"] in {"devin", "claude"}:
            options["upstream"] = upstream
        config = Config(argparse.Namespace(**options))
        # Config's auto route has no codex_upstream constructor override.
        config.codex_upstream = upstream
        config.claude_upstream = upstream
        try:
            with serving(MuseShimServer(("127.0.0.1", 0), config)) as base:
                yield base
        finally:
            peer.release.set()


def request(base, body):
    return urllib.request.Request(base + "/responses", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "x-tbh-session-id": "fixture-session"})


def post(base, body):
    with urllib.request.urlopen(request(base, body), timeout=5) as response:
        return response.headers.get_content_type(), response.read()
