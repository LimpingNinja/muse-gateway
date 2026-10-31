#!/usr/bin/env python3
"""Loopback OpenAI Responses proxy for Muse Code.

Modes:

* ``auto`` advertises Meta, Codex, Claude, Devin, Kiro, and configured local models to Muse
  and routes requests according to the selection in Muse's ``/model`` menu.
* ``generic`` / ``http`` (the historical mode) forwards Responses requests
  to any Responses-compatible endpoint.
* ``codex`` uses the native ChatGPT Codex Responses endpoint and Codex CLI
  OAuth credentials.
* ``claude`` uses Claude Code print mode by default and adapts its structured
  result back to Responses for Muse. Direct Messages API mode remains optional.
* ``meta`` (gateway fork) forwards to Meta's own Responses endpoint with the
  native wire format preserved: namespace tools are NOT flattened and
  reasoning/cache fields are NOT stripped. ``auto`` advertises ``meta/`` models
  alongside Codex and Claude.
* ``local`` (gateway fork) forwards to a user-configured Responses endpoint
  (LiteLLM, Ollama via an adapter, ...) with the usual flattened tools.
  ``auto`` advertises ``local/`` aliases only when explicitly configured.
* ``devin`` uses the Connect/protobuf API with a session token, Devin CLI login,
  or an explicit session token; the CLI is not used for inference.
* ``kiro`` translates Responses through a separately configured Kiro Gateway
  Messages service, using only that service's proxy key.

The implementation needs a single runtime dependency (protobuf, for the
Devin provider's generated bindings). Codex uses its ordinary
provider transport; Claude CLI mode deliberately delegates inference to the
locally installed official Claude Code client.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import secrets
import select
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
from http.client import HTTPException, IncompleteRead, RemoteDisconnected
from dataclasses import dataclass as _dataclass
from typing import Any, Iterable

from muse_providers import (
    response_socket,
    set_response_idle_timeout,
    AuthError,
    DEFAULT_CLAUDE_ENDPOINT,
    DEFAULT_CODEX_ENDPOINT,
    ProviderError,
    bounded_client_error_code,
    bounded_client_error_text,
    anthropic_json_to_responses,
    build_gateway_opener,
    claude_headers,
    convert_anthropic_stream,
    codex_headers,
    correct_codex_effort,
    credentials_usable,
    discover_claude_credentials,
    discover_codex_credentials,
    refused_symlink_credential_labels,
    demote_codex_encrypted_reasoning,
    drop_or_reject_previous_response_id,
    _flatten_tools,
    _input_items,
    endpoint_for,
    prepare_codex_request,
    resolve_claude_credentials,
    resolve_codex_credentials,
    response_sse_line,
    responses_to_anthropic,
    validate_claude_endpoint,
    validate_claude_token_url,
    validate_codex_endpoint,
    validate_codex_token_url,
)
try:
    import muse_devin
except Exception as _devin_import_error:  # protobuf absent or wrong version (P1-17, M2)
    if not isinstance(_devin_import_error, ImportError) and type(_devin_import_error).__name__ != "VersionError":
        raise

    class _MissingDevin:
        """Stand-in so non-Devin routes, --help, doctor and --models still run."""

        class DevinWireError(Exception):
            code = "devin_unavailable"

        DevinChatError = DevinAuthError = DevinHttpError = DevinCredentialError = DevinWireError
        REQUEST_POLICY_CODES: frozenset[str] = frozenset()
        reason = (
            f"Devin needs the protobuf runtime, which this interpreter ({sys.executable}) "
            f"cannot import ({_devin_import_error}); run `python -m pip install .`."
        )

        def credential_state(self, **_kwargs: Any) -> dict[str, Any]:
            return {"state": "unavailable", "note": self.reason}

        def seed_models(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return []

        def __getattr__(self, name: str) -> Any:
            def unavailable(*_args: Any, **_kwargs: Any) -> Any:
                raise ProviderError(self.reason, status=503, code="devin_unavailable")
            return unavailable

    muse_devin = _MissingDevin()  # type: ignore[assignment]
import muse_kiro
try:
    from muse_images import _IMAGE_TYPES, normalize_request_images
except ImportError:  # partial source copy without muse_images.py (same idiom as muse_devin above)
    _IMAGE_TYPES = frozenset({"input_image", "image_url", "image"})

    def normalize_request_images(  # type: ignore[misc]
        body: dict[str, Any], *, route: str, supports_images: bool | None,
        allow_files: bool, notify: Any = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Refuse image-bearing requests rather than forward unchecked images."""
        if _count_image_parts(body):
            raise ProviderError(
                "Image input needs muse_images.py beside muse_gateway.py; reinstall the gateway.",
                status=503, code="image_support_unavailable",
            )
        return body, {"count": 0, "data": 0, "url": 0, "file": 0, "bytes": 0}
from muse_models import positive_int, provider_catalog, resolve_limits
from muse_capture import CaptureSession, CapturingWriter, current_capture, encoded_secret_forms, source_fingerprint


DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
# Kiro's max_tokens is only a hint upstream, not a response-size guarantee.
KIRO_JSON_RESPONSE_LIMIT = 16 * 1024 * 1024
KIRO_REJECTED_BODY_LIMIT = 64 * 1024
KIRO_REJECTED_BODY_TIMEOUT = 0.2
DEFAULT_CODEX_MODELS = tuple(provider_catalog("codex")["fallbacks"])
DEFAULT_CLAUDE_MODELS = tuple(provider_catalog("claude")["fallbacks"])
DEFAULT_META_UPSTREAM = "https://api.meta.ai/v1"
# Devin (Codeium Cascade) Connect RPC base. Paths are service routes, not
# Responses endpoints, so provider_url("devin") returns this verbatim.
DEFAULT_DEVIN_UPSTREAM = "https://server.codeium.com"
DEFAULT_META_MODELS = tuple(provider_catalog("meta")["fallbacks"])
META_REASONING_EFFORTS = tuple(provider_catalog("meta")["reasoning_efforts"])
META_DEFAULT_REASONING_EFFORT = provider_catalog("meta")["default_reasoning_effort"]
# Internal Codex slugs: hidden from the menu. gpt-reserve stays routable as
# the silent quota-fallback; both stay explicitly addressable.
CODEX_HIDDEN_MODELS = ("gpt-reserve", "codex-auto-review")
CODEX_RESERVE_MODEL = "gpt-reserve"
CODEX_MODELS_ENDPOINT = (
    "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
)
MODEL_CATALOG_TTL_SECONDS = 3600
MODEL_CATALOG_RETRY_SECONDS = 60


class ShimError(Exception):
    """An error that should be returned to Muse as a client error."""


def env_value(name: str, default: str | None = None) -> str | None:
    """Read only the current setting, never a retired alias."""
    return os.environ.get(name, default)


def env_flag(name: str, default: bool = False) -> bool:
    value = env_value(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


MAX_TIMEOUT_SECONDS = 24 * 60 * 60


def _validated_timeout(value: Any, default: float, name: str) -> float:
    if value is None or value == "":
        return float(default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number of seconds between 0 and {MAX_TIMEOUT_SECONDS}.")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number of seconds between 0 and {MAX_TIMEOUT_SECONDS}.") from error
    if not math.isfinite(seconds) or seconds <= 0 or seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"{name} must be a finite number of seconds between 0 and {MAX_TIMEOUT_SECONDS}.")
    return seconds


def _timeout_seconds(value: str) -> float:
    try:
        return _validated_timeout(value, 600.0, "timeout")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _validated_concurrency(value: Any) -> int:
    if value is None or value == "":
        return 8
    if isinstance(value, bool):
        raise ValueError("--max-concurrent-requests must be an integer between 1 and 1024.")
    try:
        limit = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--max-concurrent-requests must be an integer between 1 and 1024.") from error
    if str(value).strip() != str(limit) or limit < 1 or limit > 1024:
        raise ValueError("--max-concurrent-requests must be an integer between 1 and 1024.")
    return limit


def _concurrency_limit(value: str) -> int:
    try:
        return _validated_concurrency(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_model_list(value: str | None, fallback: tuple[str, ...]) -> list[str]:
    models = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return list(dict.fromkeys(models or fallback))


def _menu_filter(value: str | None) -> list[str]:
    names = _parse_model_list(value, ())
    for name in names:
        provider, slash, model = name.partition("/")
        if not slash or provider not in {"meta", "codex", "claude", "devin", "kiro", "local"} or not model or model.startswith(":"):
            raise ValueError("MUSE_GATEWAY_MODELS requires exact provider/model IDs; no bare IDs or colon aliases.")
    return names


def _is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _claude_endpoint_from_upstream(upstream: str) -> str:
    value = str(upstream or "").rstrip("/")
    if value.endswith("/messages"):
        return value
    if value.endswith("/v1"):
        return value + "/messages"
    return value + "/v1/messages"


def _validate_native_config(
    *,
    provider: str,
    upstream: str,
    codex_token_url: str | None,
    claude_token_url: str | None,
    insecure_upstream: bool,
    allow_oauth_host_override: bool,
) -> None:
    if provider not in {"codex", "claude"}:
        return
    if insecure_upstream:
        raise ProviderError(
            f"--insecure-upstream is not permitted in {provider} OAuth/native mode.",
            code="insecure_native_upstream_rejected",
        )
    if provider == "codex":
        validate_codex_endpoint(
            endpoint_for(upstream, "responses"),
            allow_host_override=allow_oauth_host_override,
        )
        if codex_token_url:
            validate_codex_token_url(
                codex_token_url,
                allow_host_override=allow_oauth_host_override,
            )
    else:
        validate_claude_endpoint(
            _claude_endpoint_from_upstream(upstream),
            allow_host_override=allow_oauth_host_override,
        )
        if claude_token_url:
            validate_claude_token_url(
                claude_token_url,
                allow_host_override=allow_oauth_host_override,
            )


def flatten_namespace_tools(tools: list[Any]) -> list[Any]:
    """Flatten Muse's namespace tool groups into Responses function tools.

    One shared implementation lives in ``muse_providers._flatten_tools``
    (P2-07). The gateway transform path keeps its ``ShimError`` wire
    shape: codec errors are converted at this boundary, never leaked.
    """
    try:
        return _flatten_tools(tools)
    except ProviderError as error:
        raise ShimError(str(error)) from error


def _namespace_qualified_map(tools: list[Any]) -> dict[str, str]:
    """Map qualified history names (``ns.child``) to flattened tool names."""
    mapping: dict[str, str] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            continue
        namespace = tool.get("name")
        children = tool.get("tools")
        if not isinstance(namespace, str) or not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            name = child.get("name")
            if isinstance(name, str) and name:
                mapping.setdefault(f"{namespace}.{name}", name)
    return mapping


def transform_request(
    body: dict[str, Any], config: "Config", *, native: bool = False
) -> dict[str, Any]:
    """Rewrite one request for an upstream that is not Meta itself.

    ``native=True`` (gateway fork, ``meta/`` route) preserves the Meta wire
    format: namespace tools stay grouped and reasoning/cache fields survive,
    so Meta receives exactly what Muse sent. The ``model`` override and the
    single-provider model override still applies.
    """
    transformed = dict(body)

    if not native:
        tools = transformed.get("tools")
        if isinstance(tools, list):
            transformed["tools"] = flatten_namespace_tools(tools)
            qualified = _namespace_qualified_map(tools)
            if qualified:
                history = transformed.get("input")
                if isinstance(history, list):
                    rewritten: list[Any] = []
                    for item in history:
                        name = item.get("name") if isinstance(item, dict) else None
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                            and isinstance(name, str)
                            and name in qualified
                        ):
                            item = dict(item, name=qualified[name])
                        rewritten.append(item)
                    transformed["input"] = rewritten

    if config.model:
        transformed["model"] = config.model

    if config.portable and not native and (config.provider in {"generic", "local"} or str(body.get("model", "")).startswith(("local/", "local:"))):
        transformed.pop("include", None)
        transformed.pop("prompt_cache_key", None)
        transformed.pop("reasoning", None)

    return transformed


@_dataclass(frozen=True)
class _CatalogSnapshot:
    entries: tuple
    raw: dict
    devin_roster: Any = None
    devin_dropped: int = 0
    stamped: float = 0.0
    generation: int = 0
    outcome: str = "ok"


class Config:
    """Runtime settings with backwards-compatible historical attributes."""

    def __init__(self, args: argparse.Namespace) -> None:
        requested_provider = getattr(args, "provider", None) or getattr(args, "mode", None)
        # Namespaces built by embedders/tests historically omitted provider;
        # keep that programmatic fallback generic. The CLI explicitly supplies
        # its consumer-facing default of ``auto``.
        provider = str(requested_provider or "generic").lower()
        if provider == "http":
            provider = "generic"
        self.provider: str = provider
        self.host: str = str(getattr(args, "host", DEFAULT_HOST))
        self.port: int = int(getattr(args, "port", DEFAULT_PORT))
        raw_upstream = getattr(args, "upstream", None)
        if raw_upstream:
            upstream = str(raw_upstream)
        elif getattr(args, f"{provider}_upstream", None):
            upstream = str(getattr(args, f"{provider}_upstream"))
        else:
            upstream = _default_upstream_for(provider)
        self.upstream: str = upstream.rstrip("/")
        self.codex_upstream: str = (
            self.upstream if provider == "codex" else (getattr(args, "codex_upstream", None) or _default_upstream_for("codex"))
        ).rstrip("/")
        self.claude_upstream: str = (
            self.upstream if provider == "claude" else (getattr(args, "claude_upstream", None) or _default_upstream_for("claude"))
        ).rstrip("/")
        # Gateway fork: per-route upstreams. In single-provider mode the shared
        # --upstream wins (same convention as codex/claude); otherwise each
        # route falls back to its own flag/env/default chain.
        self.meta_upstream: str = (
            self.upstream
            if provider == "meta"
            else str(
                getattr(args, "meta_upstream", None)
                or _default_upstream_for("meta")
            )
        ).rstrip("/")
        explicit_local = getattr(args, "local_upstream", None)
        self.local_upstream: str = (
            self.upstream
            if provider == "local"
            else str(
                # Explicit "" (--local-upstream=) disables; only None defaults.
                explicit_local if explicit_local is not None
                else _default_upstream_for("local")
            )
        ).rstrip("/")
        self.devin_upstream: str = (
            self.upstream
            if provider == "devin"
            else str(
                getattr(args, "devin_upstream", None)
                or _default_upstream_for("devin")
            )
        ).rstrip("/")
        self.devin_api_key: str | None = getattr(args, "devin_api_key", None)
        _warn_cleartext_devin_upstream(self.devin_upstream)
        self.kiro_api_key: str | None = getattr(args, "kiro_api_key", None)
        self.kiro_requested = bool(
            provider == "kiro" or self.kiro_api_key
            or getattr(args, "kiro_upstream", None)
        )
        self.kiro_upstream = muse_kiro.normalize_upstream(
            self.upstream if provider == "kiro" else (
                getattr(args, "kiro_upstream", None) or _default_upstream_for("kiro")
            )
        )
        if self.kiro_api_key:
            self.kiro_api_key = muse_kiro.validate_proxy_key(self.kiro_api_key)
        self.kiro_autostart: bool = bool(getattr(args, "kiro_autostart", False))
        self.kiro_server_dir: str = os.path.abspath(os.path.expanduser(
            getattr(args, "kiro_server_dir", None) or os.path.join("~", "kiro-gateway")
        ))
        # Set by ensure_kiro_gateway during launch; None keeps doctor output unchanged.
        self._kiro_service_note: str | None = None
        # Gateway fork: model discovery runs only when the local upstream was
        # explicitly configured (flag or env), so an unconfigured install
        # advertises no local/ entries and makes no discovery calls.
        self.local_upstream_explicit = bool(self.local_upstream)
        self.api_key: str | None = getattr(args, "api_key", None)
        self.meta_api_key: str | None = getattr(args, "meta_api_key", None)
        self.local_api_key: str | None = getattr(args, "local_api_key", None)
        self.access_token: str | None = getattr(args, "access_token", None)
        self.model: str | None = getattr(args, "model", None)
        self.portable: bool = bool(getattr(args, "portable", False))
        self.insecure_upstream: bool = bool(getattr(args, "insecure_upstream", False))
        self.allow_oauth_host_override: bool = bool(
            getattr(args, "allow_oauth_host_override", False)
        )
        self.allow_non_loopback: bool = bool(getattr(args, "allow_non_loopback", False))
        self.allow_image_files: bool = bool(getattr(args, "allow_image_files", False))
        self.require_inbound_token: bool = bool(getattr(args, "require_inbound_token", False))
        self.inbound_token: str | None = (
            str(getattr(args, "inbound_token", None) or os.environ.get("MUSE_GATEWAY_INBOUND_TOKEN", "")).strip() or None
        )
        if self.require_inbound_token and not self.inbound_token:
            raise ProviderError(
                "--require-inbound-token needs MUSE_GATEWAY_INBOUND_TOKEN set in the gateway process environment.",
                code="inbound_token_missing",
            )
        self.verbose: bool = bool(getattr(args, "verbose", False))
        self.log_file: str = os.path.abspath(os.path.expanduser(
            getattr(args, "log_file", None) or "muse-gateway.log"
        ))
        _capture_dir = getattr(args, "capture_dir", None)
        self.capture_dir: str | None = (
            os.path.expanduser(str(_capture_dir)) if _capture_dir else None
        )
        self.timeout: float = _validated_timeout(getattr(args, "timeout", 600.0), 600.0, "--timeout")
        self.max_body_bytes: int = int(
            getattr(args, "max_body_bytes", DEFAULT_MAX_BODY_BYTES)
        )
        if self.max_body_bytes <= 0:
            raise ValueError("max incoming body size must be a positive integer")
        self.refresh_skew_seconds = 120
        self.codex_home: str | None = getattr(args, "codex_home", None)
        self.codex_auth_path: str | None = getattr(args, "codex_auth_path", None)
        self.codex_token_url: str | None = getattr(args, "codex_token_url", None)
        self.claude_credentials: str | None = getattr(args, "claude_credentials", None)
        self.claude_token_url: str | None = getattr(args, "claude_token_url", None)
        self.claude_transport: str = str(
            getattr(args, "claude_transport", "cli") or "cli"
        ).lower()
        if self.claude_transport not in {"cli", "http"}:
            raise ValueError("Claude transport must be 'cli' or 'http'.")
        self.claude_cli: str = str(getattr(args, "claude_cli", "claude") or "claude")
        _passthrough = getattr(args, "claude_cli_passthrough", None)
        if _passthrough is None:
            _passthrough = "oauth"
        if _passthrough not in _CLAUDE_PASSTHROUGH_SETS:
            raise ProviderError(
                f"invalid --claude-cli-passthrough value {_passthrough!r}; "
                "expected one of: api-key, all, none, oauth.",
                code="invalid_claude_passthrough",
            )
        self.claude_cli_passthrough: str = str(_passthrough)
        self.stream_idle_timeout: float = _validated_timeout(
            getattr(args, "stream_idle_timeout", 120.0), 120.0, "--stream-idle-timeout"
        )
        self.turn_timeout: float = _validated_timeout(
            getattr(args, "turn_timeout", 3600.0), 3600.0, "--turn-timeout"
        )
        self.max_concurrent_requests: int = _validated_concurrency(
            getattr(args, "max_concurrent_requests", 8)
        )
        self.doctor_provider: str | None = getattr(args, "doctor_provider", None)
        self.codex_models = list(DEFAULT_CODEX_MODELS)
        self.claude_models = list(DEFAULT_CLAUDE_MODELS)
        self.meta_models = list(DEFAULT_META_MODELS)
        self.menu_models = _menu_filter(env_value("MUSE_GATEWAY_MODELS"))
        self._raw_model_rosters: dict[str, dict[str, dict[str, Any]]] = {}
        self._catalog_diagnostics: dict[str, str] = {}
        self._reported_catalog_diagnostics: set[str] = set()
        self.model_discovery: bool = bool(getattr(args, "model_discovery", True))
        self._model_catalog_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._kiro_catalog_status = "not checked; inference not validated"
        # Devin roster stash: logical_id -> DevinModel, refreshed with the
        # catalog so requests reuse discovery without extra RPCs.
        self._devin_roster: dict[str, Any] = {}
        self._devin_rosters: dict[str, tuple[dict[str, Any], int]] = {}
        self._devin_dropped_fusion: int = 0
        # Atomic catalog snapshots (P1-02): built off-lock, published by one
        # assignment under a short lock; legacy views mirror the snapshot.
        self._catalog_snapshots: dict[str, _CatalogSnapshot] = {}
        self._catalog_lock = threading.Lock()
        self._catalog_generation: dict[str, int] = {}
        self._catalog_flight_locks: dict[str, threading.Lock] = {}
        self._catalog_flight_guard = threading.Lock()
        self._catalog_last_nocache: dict[str, float] = {}
        if not _is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ProviderError(
                f"refusing non-loopback bind address {self.host!r}; "
                "use --allow-non-loopback only when remote access is intentional.",
                code="non_loopback_bind_rejected",
            )
        if not _is_loopback_host(self.host) and not (self.require_inbound_token and self.inbound_token):
            raise ProviderError(
                f"refusing non-loopback bind address {self.host!r} without an inbound token; "
                "set MUSE_GATEWAY_INBOUND_TOKEN and pass --require-inbound-token.",
                code="non_loopback_token_required",
            )
        native_providers = (
            ("codex", "claude") if self.provider == "auto" else (self.provider,)
        )
        for native_provider in native_providers:
            native_upstream = (
                self.codex_upstream
                if native_provider == "codex"
                else self.claude_upstream
            )
            _validate_native_config(
                provider=native_provider,
                upstream=native_upstream,
                codex_token_url=self.codex_token_url,
                claude_token_url=self.claude_token_url,
                insecure_upstream=self.insecure_upstream,
                allow_oauth_host_override=self.allow_oauth_host_override,
            )

    def upstream_url(self, path: str) -> str:
        return f"{self.upstream}/{path.lstrip('/')}"

    def provider_url(self, provider: str | None = None) -> str:
        selected = provider or self.provider
        if selected == "codex":
            return endpoint_for(self.codex_upstream, "responses")
        if selected == "claude":
            return _claude_endpoint_from_upstream(self.claude_upstream)
        if selected == "meta":
            return f"{self.meta_upstream}/responses"
        if selected == "local":
            return f"{self.local_upstream}/responses"
        if selected == "devin":
            return self.devin_upstream
        if selected == "kiro":
            return self.kiro_upstream + "/messages"
        return self.upstream_url("responses")

    def resolve_model(self, model: Any) -> tuple[str, str]:
        value = str(model or "").strip()
        if not value:
            raise ProviderError(
                "request is missing a model; open Muse's /model menu and choose one.",
                code="model_missing",
            )

        # One prefix table (P2-07): uniform strip + empty -> model_missing,
        # generalizing Kiro's former behavior to every route.
        for route_name, route_label in (
            ("codex", "Codex"), ("claude", "Claude"), ("meta", "Meta"),
            ("local", "the local upstream"), ("devin", "Devin"), ("kiro", "Kiro"),
        ):
            for prefix in (f"{route_name}/", f"{route_name}:"):
                if not value.startswith(prefix):
                    continue
                if self.provider not in {"auto", route_name}:
                    raise ProviderError(
                        f"selected model belongs to {route_label}.",
                        code="model_provider_mismatch",
                    )
                model_name = value[len(prefix):].strip()
                if not model_name:
                    raise ProviderError(
                        f"{route_name.capitalize()} model name is empty.", code="model_missing"
                    )
                return route_name, model_name

        if self.provider in {"generic", "codex", "claude", "meta", "local", "devin", "kiro"}:
            return self.provider, value
        # Gateway fork: explicit membership wins over the name heuristics
        # below, so a user-configured local alias is never misrouted.
        # Discovered aliases route the same way (cached via the catalog),
        # so --model <alias> works with or without an explicit list.
        if value in self._discovered_local_names():
            return "local", value
        if value in self.meta_models or value.lower().startswith("muse-"):
            return "meta", value
        if value in self.claude_models or value.lower().startswith("claude-"):
            return "claude", value
        if value in self._devin_roster:
            return "devin", value
        if value in self.codex_models or value.lower().startswith(
            ("gpt-", "o1", "o3", "o4")
        ):
            return "codex", value
        raise ProviderError(
            f"cannot infer a provider for model {value!r}; select a prefixed model from Muse's /model menu.",
            code="model_provider_unknown",
        )

    def _discovered_local_names(self) -> set[str]:
        """Alias ids the local upstream reported, via the cached catalog."""
        try:
            entries = provider_model_catalog(self, "local")
        except Exception:
            return set()
        return set(self._snapshot_raw("local"))

    def secret_values(self) -> list[str]:
        values: list[str] = []
        for value in (
            self.api_key,
            self.access_token,
            self.meta_api_key,
            self.local_api_key,
            self.devin_api_key,
            self.kiro_api_key,
            self.inbound_token,
        ):
            if isinstance(value, str) and value:
                values.append(value)
        return values

    def model_output_limit(self, provider: str, name: str) -> int | None:
        row = self._snapshot_raw(provider).get(name, {})
        return positive_int(row.get("output_limit")) or resolve_limits(f"{provider}/{name}")["output_limit"]

    def _snapshot_raw(self, provider: str) -> dict[str, dict[str, Any]]:
        snapshot = self._catalog_snapshots.get(provider)
        if snapshot is not None:
            return snapshot.raw
        return self._raw_model_rosters.get(provider, {})

    def _flight_lock(self, key: str) -> threading.Lock:
        with self._catalog_flight_guard:
            return self._catalog_flight_locks.setdefault(key, threading.Lock())

    def _publish_catalog(self, key: str, provider: str, *, entries, raw, stamped,
                         generation: int, outcome: str = "ok",
                         devin_roster=None, devin_dropped: int = 0) -> bool:
        with self._catalog_lock:
            if generation != self._catalog_generation.get(key, 0):
                return False
            existing = self._catalog_snapshots.get(key)
            if outcome not in {"ok", "ok_nonempty", "disabled"} and existing is not None and existing.outcome in {"ok", "ok_nonempty"}:
                # Preserve last-good live entries/raw/roster on a failed
                # refresh. The caller-supplied backdated stamp still records
                # when discovery may be retried.
                snapshot = _CatalogSnapshot(
                    entries=existing.entries, raw=dict(existing.raw),
                    devin_roster=existing.devin_roster,
                    devin_dropped=existing.devin_dropped, stamped=stamped,
                    generation=generation, outcome=outcome,
                )
            else:
                snapshot = _CatalogSnapshot(
                    entries=tuple(entries), raw=dict(raw), devin_roster=devin_roster,
                    devin_dropped=devin_dropped, stamped=stamped,
                    generation=generation, outcome=outcome,
                )
            self._catalog_snapshots[key] = snapshot
            self._model_catalog_cache[key] = (stamped, list(snapshot.entries))
            self._raw_model_rosters[provider] = dict(snapshot.raw)
            if snapshot.devin_roster is not None:
                self._devin_roster = snapshot.devin_roster
                self._devin_dropped_fusion = snapshot.devin_dropped
                self._devin_rosters[key] = (snapshot.devin_roster, snapshot.devin_dropped)
                while len(self._devin_rosters) > 4:
                    evicted = next(iter(self._devin_rosters))
                    self._devin_rosters.pop(evicted)
                    if evicted != key:
                        self._catalog_snapshots.pop(evicted, None)
                        self._model_catalog_cache.pop(evicted, None)
                        self._catalog_generation.pop(evicted, None)
                        flight = self._catalog_flight_locks.get(evicted)
                        if flight is not None and not flight.locked():
                            self._catalog_flight_locks.pop(evicted, None)
            return True


# Muse requires these catalog fields; they describe this gateway's catalog
# format, not provider release dates. One source for both (P3-06).
CATALOG_RELEASE_DATE = "2026-08-06"
CATALOG_CREATED_UNIX = 1785974400  # CATALOG_RELEASE_DATE at 00:00 UTC

_PROVIDER_BLURB = {
    "codex": "OpenAI Codex subscription model",
    "claude": "Claude model via Claude Code (CLI turns arrive whole)",
    "meta": "Muse Spark model",
    "local": "Local model through muse-gateway",
    "kiro": "Kiro model through Kiro Gateway",
    "devin": "Devin model via Connect RPC",
}


def _compact_tokens(value: int) -> str:
    """Short context size for the /model menu, e.g. 262144 -> "262k"."""
    if value >= 1_000_000:
        return f"{round(value / 1_000_000)}M"
    if value >= 1_000:
        return f"{round(value / 1_000)}k"
    return str(value)


def _catalog_description(
    provider: str,
    context_limit: int,
    *,
    limits_verified: bool = True,
    limit_source: str = "live",
) -> str:
    blurb = _PROVIDER_BLURB.get(provider, "Model through muse-gateway")
    description = f"(ctx {_compact_tokens(context_limit)}) {blurb}"
    if limit_source != "live" or not limits_verified:
        description += " · fallback limits (not live-verified)"
    return description


def _catalog_entry(
    provider: str,
    model: str,
    *,
    context_limit: int,
    output_limit: int,
    is_default: bool,
    limits_verified: bool = True,
    limit_source: str = "live",
    context_source: str = "live",
    output_source: str = "live",
    supports_images: bool | None = None,
    reasoning_efforts: list[str] | None = None,
    default_reasoning_effort: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    routed_id = f"{provider}/{model}"
    provider_label = {"codex": "Codex", "claude": "Claude", "meta": "Meta", "local": "Local"}.get(
        provider, provider.title()
    )
    if supports_images is None:
        # Only advertise image input where there is a basis for it. Local
        # endpoints vary widely, so they opt in via catalog evidence instead.
        supports_images = provider in {"codex", "claude", "meta"}
    return {
        "model_id": routed_id,
        "display_label": f"{provider_label} · {label or model}",
        "release_date": CATALOG_RELEASE_DATE,
        "context_limit": context_limit,
        "output_limit": output_limit,
        "is_current": is_default,
        "is_default": is_default,
        "description": _catalog_description(
            provider,
            context_limit,
            limits_verified=limits_verified,
            limit_source=limit_source,
        ) + f" · context: {context_source}; output: {output_source}",
        "limits_verified": limits_verified,
        "limit_source": limit_source,
        "context_source": context_source,
        "output_source": output_source,
        "supports_images": supports_images,
        "reasoning_efforts": reasoning_efforts or [],
        "default_reasoning_effort": default_reasoning_effort,
    }


def _wire_catalog_model(row: dict[str, Any]) -> dict[str, Any]:
    model_id = str(row["model_id"])
    family = model_id.split("/", 1)[0] if "/" in model_id else "generic"
    supports_images = row.get(
        "supports_images", family in {"codex", "claude", "meta"}
    )
    levels = row.get("reasoning_efforts", [])
    reasoning = bool(levels)
    variants = {level: {"reasoningEffort": level} for level in levels}
    options: dict[str, Any] = {"forceReasoning": reasoning, "include": []}
    if row.get("default_reasoning_effort") in levels:
        options["reasoningEffort"] = row["default_reasoning_effort"]
    return {
        "id": model_id,
        "object": "model",
        "created": CATALOG_CREATED_UNIX,
        "owned_by": family,
        "metadata": {
            "muse-code": {
                "name": row["display_label"],
                "family": family,
                "release_date": row["release_date"],
                "is_hidden": False,
                "attachment": True,
                "reasoning": reasoning,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"] if supports_images else ["text"],
                    "output": ["text"],
                },
                "limit": {
                    "context": row["context_limit"],
                    "output": row["output_limit"],
                },
                "options": options,
                "variants": variants,
                "description": row["description"],
            }
        },
    }


def muse_catalog_response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the official network catalog schema consumed by Muse Code."""
    return {
        "object": "list",
        "data": [_wire_catalog_model(row) for row in rows],
    }


def _discover_provider_models(
    config: Config, provider: str, *, authorization: str | None = None
) -> list[dict[str, Any]]:
    """Fetch the account-visible catalog; return an empty list on any failure."""
    if not config.model_discovery:
        return []
    # Claude Code owns model selection/auth in CLI mode. Do not inspect direct
    # Anthropic credentials or make a separate HTTP catalog request here.
    if provider == "claude" and config.claude_transport == "cli":
        return []
    if provider == "meta":
        return _discover_meta_models(config, authorization=authorization)
    if provider == "local":
        return _discover_local_models(config)
    try:
        if provider == "codex":
            state = discover_codex_credentials(config)
            if state is None:
                return []
            request = urllib.request.Request(
                CODEX_MODELS_ENDPOINT,
                method="GET",
                headers=codex_headers(state.access_token),
            )
            with _credentialed_urlopen(
                request, timeout=min(config.timeout, 8.0), config=config
            ) as response:
                payload = _read_bounded_json(response, limit=CATALOG_JSON_RESPONSE_LIMIT)
            raw_models = payload.get("models", []) if isinstance(payload, dict) else []
            result = []
            for item in raw_models:
                if not isinstance(item, dict) or not isinstance(item.get("slug"), str):
                    continue
                slug = item["slug"].strip()
                limits = _discovery_model_limits(item, f"codex/{slug}")
                levels = item.get("supported_reasoning_levels")
                # 'ultra' is an orchestration mode, not a Responses wire effort.
                wire_levels = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
                efforts = [
                    level["effort"] for level in (levels if isinstance(levels, list) else [])
                    if isinstance(level, dict) and level.get("effort") in wire_levels
                ]
                result.append({
                    "model": slug,
                    **limits,
                    "reasoning_efforts": efforts,
                    "default_reasoning_effort": item.get("default_reasoning_level"),
                })
            return [item for item in result if item["model"]]

        state = discover_claude_credentials(config)
        if state is None or state.auth_mode != "api_key":
            return []
        endpoint = config.provider_url("claude").removesuffix("/messages") + "/models"
        headers = claude_headers(state)
        headers.pop("Content-Type", None)
        request = urllib.request.Request(endpoint, method="GET", headers=headers)
        with _credentialed_urlopen(
            request, timeout=min(config.timeout, 8.0), config=config
        ) as response:
            payload = _read_bounded_json(response, limit=CATALOG_JSON_RESPONSE_LIMIT)
        raw_models = payload.get("data", []) if isinstance(payload, dict) else []
        result = []
        for item in raw_models:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("id"), str)
                or not item["id"].strip()
            ):
                continue
            result.append({
                "model": item["id"].strip(),
                **_discovery_model_limits(item, "claude/" + item["id"].strip()),
            })
        return result
    except (
        AuthError,
        ProviderError,
        OSError,
        HTTPException,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return []


# Upstream /models payloads vary: OpenAI-style, LiteLLM-enriched, and
# Codex-style shapes all appear in the wild. Accept the common aliases so a
# missing key degrades to a documented fallback instead of a wrong number.
_INPUT_LIMIT_KEYS = (
    "max_input_tokens",
    "context_window",
    "context_length",
    "context_limit",
    "max_context_tokens",
)
_OUTPUT_LIMIT_KEYS = (
    "max_output_tokens", "max_completion_tokens", "output_limit", "max_tokens",
)


def _discovery_model_limits(item: dict[str, Any], model_id: str) -> dict[str, Any]:
    context = next((n for key in _INPUT_LIMIT_KEYS if (n := positive_int(item.get(key))) is not None), None)
    output = next((n for key in _OUTPUT_LIMIT_KEYS if (n := positive_int(item.get(key))) is not None), None)
    return resolve_limits(model_id, context, output)


def _discovery_supports_images(item: dict[str, Any]) -> bool:
    """Detect image-input evidence in one catalog entry, if any is present."""
    for key in ("supports_vision", "supports_image", "supports_images", "vision"):
        if item.get(key) is True:
            return True
    for key in ("capabilities", "supported_modalities"):
        value = item.get(key)
        if isinstance(value, list) and any(
            str(entry).lower() in {"vision", "image", "images"} for entry in value
        ):
            return True
    modalities = item.get("modalities")
    if isinstance(modalities, dict):
        inputs = modalities.get("input")
        if isinstance(inputs, list) and any(
            str(entry).lower() in {"vision", "image", "images"} for entry in inputs
        ):
            return True
    return False


def _get_discovery_json(
    url: str, headers: dict[str, str], timeout: float
) -> Any:
    request = urllib.request.Request(url, method="GET", headers=headers)
    with _credentialed_urlopen(request, timeout=timeout) as response:
        return _read_bounded_json(response, limit=CATALOG_JSON_RESPONSE_LIMIT)


_DISCOVERY_ERRORS = (OSError, HTTPException, ValueError, TypeError, json.JSONDecodeError, ProviderError)


def _parse_openai_style_models(payload: Any, provider: str) -> list[dict[str, Any]]:
    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(raw_models, list):
        return []
    result = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        name = item.get("id")
        if not isinstance(name, str) or not name.strip():
            continue
        result.append({
            "model": name.strip(),
            **_discovery_model_limits(item, f"{provider}/{name.strip()}"),
            "supports_images": _discovery_supports_images(item),
        })
    return result


def _discover_meta_models(
    config: Config, *, authorization: str | None = None
) -> list[dict[str, Any]]:
    """List Meta models from ``{meta_upstream}/models`` when possible.

    With a static key the gateway probes on its own. In subscription setups
    the catalog request may carry Muse's own bearer, which is the same
    credential forward_meta sends to the responses endpoint, so it is reused
    for the default Meta upstream only. Anything else (or any failure)
    degrades to [], preserving the configured fallbacks, and
    provider_model_catalog caches the result.
    """
    if not config.model_discovery:
        return []
    headers: dict[str, str] | None = None
    if config.meta_api_key:
        headers = {"Authorization": f"Bearer {config.meta_api_key}"}
    elif (
        isinstance(authorization, str)
        and authorization.lower().startswith("bearer ")
        and config.meta_upstream.rstrip("/") == DEFAULT_META_UPSTREAM
    ):
        headers = {"Authorization": authorization}
    else:
        return []
    try:
        payload = _get_discovery_json(
            f"{config.meta_upstream}/models",
            headers,
            min(config.timeout, 8.0),
        )
        models = _parse_openai_style_models(payload, "meta")
        if models:
            return models
        # Tolerate a Codex-style {"models": [{"slug": ...}]} shape as well.
        raw = payload.get("models", []) if isinstance(payload, dict) else []
        result = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("slug")
            if not isinstance(name, str) or not name.strip():
                continue
            result.append({
                "model": name.strip(),
                **_discovery_model_limits(item, f"meta/{name.strip()}"),
                "supports_images": _discovery_supports_images(item),
            })
        return result
    except _DISCOVERY_ERRORS:
        return []


def _discover_ollama_tags(upstream: str, timeout: float) -> list[dict[str, Any]]:
    """Fall back to Ollama's native /api/tags listing when applicable.

    Only attempted when the configured upstream has the Ollama OpenAI-compat
    shape (``.../v1``), so non-Ollama servers never receive a stray request.
    Ollama tags carry no token limits, so entries are marked unverified.
    """
    base = upstream.rstrip("/")
    if not base.endswith("/v1"):
        return []
    base = base[: -len("/v1")].rstrip("/")
    try:
        payload = _get_discovery_json(f"{base}/api/tags", {}, timeout)
    except _DISCOVERY_ERRORS:
        return []
    raw_models = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(raw_models, list):
        return []
    result = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        result.append(
            {
                "model": name.strip(),
                **resolve_limits(f"local/{name.strip()}"),
                "supports_images": _discovery_supports_images(item),
            }
        )
    return result


def _discover_local_models(config: Config) -> list[dict[str, Any]]:
    """Discover the raw local roster; missing capacities stay missing."""
    if not config.local_upstream_explicit or not config.model_discovery:
        return []
    headers: dict[str, str] = {}
    key = config.local_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timeout = min(config.timeout, 8.0)
    try:
        payload = _get_discovery_json(
            f"{config.local_upstream}/models", headers, timeout
        )
    except _DISCOVERY_ERRORS:
        payload = None
    if payload is not None:
        models = _parse_openai_style_models(payload, "local")
        if models:
            return models
    return _discover_ollama_tags(config.local_upstream, timeout)


def _provider_is_available(config: Config, provider: str) -> bool:
    """True when this system can actually serve the provider's route.

    Codex needs usable stored credentials; Claude needs usable credentials
    too, plus its CLI executable when using the default print transport
    (the CLI owns inference, but login state is still statically visible).
    Meta is always listed: subscription forwarding cannot be verified
    statically and launch mode provides it. Devin needs a ready credential.
    Local stays opt-in through its own configured-list and discovery rules
    below. Any lookup failure means unavailable, never an exception: the
    menu must degrade, not crash.
    """
    try:
        if provider == "codex":
            return credentials_usable(discover_codex_credentials(config))[0]
        if provider == "claude":
            if config.claude_transport == "cli":
                if shutil.which(config.claude_cli) is None:
                    return False
            return credentials_usable(discover_claude_credentials(config))[0]
        if provider == "devin":
            return _devin_provider_ready(config)
        if provider == "kiro":
            return bool(config.kiro_api_key)
    except Exception:
        return False
    return True


def _devin_provider_ready(config: Config) -> bool:
    """Devin rows list only when a credential is ready."""
    try:
        state = muse_devin.credential_state(api_key=config.devin_api_key)
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("state") == "ready"


def _devin_cache_key(config: Config, token: str) -> str:
    return "devin:" + hashlib.sha256(token.encode()).hexdigest()


def _devin_catalog_entries(config: Config, access_token=None, note_error_headers=None) -> list[dict[str, Any]]:
    """Devin rows with their own roster stash (uid/routing the generic item
    shape cannot carry). Explicit devin_models curate; fusion pairings are
    dropped inside discovery (option B) with the count stashed for doctor."""
    try:
        token = access_token
        if token is None:
            token, _identity = muse_devin.resolve_access_token(api_key=config.devin_api_key)
    except (muse_devin.DevinCredentialError, muse_devin.DevinWireError, OSError, ValueError):
        config._devin_roster = {}
        config._devin_dropped_fusion = 0
        return []
    cache_key = _devin_cache_key(config, token)
    cached = config._model_catalog_cache.get(cache_key)
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        snapshot = config._catalog_snapshots.get(cache_key)
        if snapshot is not None and snapshot.devin_roster is not None:
            config._devin_roster = snapshot.devin_roster
            config._devin_dropped_fusion = snapshot.devin_dropped
            config._raw_model_rosters["devin"] = dict(snapshot.raw)
        return cached[1]
    flight = config._flight_lock(cache_key)
    if not flight.acquire(blocking=False):
        if cached:
            return cached[1]  # serve stale while a refresh is in flight
        flight.acquire()
        try:
            cached = config._model_catalog_cache.get(cache_key)
            if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
                return cached[1]
        finally:
            flight.release()
        flight.acquire()
    try:
        generation = config._catalog_generation.get(cache_key, 0)
        dropped = 0
        live_ok = False
        if config.model_discovery:
            try:
                models, dropped, _warnings = muse_devin.discover_models(
                    token, config.devin_upstream, timeout=min(config.timeout, 8.0)
                )
                live_ok = bool(models)
            except muse_devin.DevinHttpError as error:
                if callable(note_error_headers):
                    note_error_headers(getattr(error, "headers", None))
                models = []
            except (muse_devin.DevinWireError, OSError, ValueError):
                models = []
            if not models:
                models = muse_devin.seed_models(config.devin_upstream)
        else:
            models = muse_devin.seed_models(config.devin_upstream)
        roster = {}
        for model in models:
            roster[model.logical_id] = model
            roster.setdefault(model.uid, model)
        ordered = sorted(models, key=lambda model: not model.is_default)
        items = [
            {"model": model.logical_id, "context_limit": model.context_limit,
             "output_limit": model.output_limit, "context_source": model.context_source,
             "output_source": model.output_source,
             "limits_verified": not model.unverified,
             "limit_source": model.context_source if model.context_source == model.output_source else "mixed",
             "supports_images": model.supports_images, "reasoning_efforts": model.efforts,
             "default_reasoning_effort": model.default_effort, "label": model.label}
            for model in ordered
        ]
        entries = _catalog_for_items(config, "devin", items)
        outcome = "ok_nonempty" if live_ok else ("disabled" if not config.model_discovery else "failed")
        if outcome in ("ok_nonempty", "disabled"):
            stamped = time.time()
        else:
            stamped = time.time() - (MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS)
        raw = {item["model"]: item for item in items}
        if not config._publish_catalog(
            cache_key, "devin", entries=entries, raw=raw, stamped=stamped,
            generation=generation, outcome=outcome,
            devin_roster=roster, devin_dropped=dropped,
        ):
            stale = config._model_catalog_cache.get(cache_key)
            if stale:
                return stale[1]
        snapshot = config._catalog_snapshots.get(cache_key)
        return list(snapshot.entries) if snapshot is not None else entries
    finally:
        flight.release()


def _catalog_for_items(config: Config, provider: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve raw discovery, but publish only complete numeric capacities.

    Pure: callers publish (entries, raw) atomically via _publish_catalog.
    """
    entries = []
    for item in items:
        if not positive_int(item.get("context_limit")) or not positive_int(item.get("output_limit")):
            continue
        entries.append(_catalog_entry(
            provider, item["model"], context_limit=item["context_limit"],
            output_limit=item["output_limit"], is_default=not entries,
            limits_verified=item.get("limits_verified", False),
            limit_source=item.get("limit_source", "missing"),
            context_source=item.get("context_source", "missing"),
            output_source=item.get("output_source", "missing"),
            supports_images=False if provider == "claude" and config.claude_transport == "cli" else item.get(
                "supports_images", provider in {"codex", "claude", "meta"}),
            reasoning_efforts=item.get("reasoning_efforts") or (list(META_REASONING_EFFORTS) if provider == "meta" else None),
            default_reasoning_effort=item.get("default_reasoning_effort") or (META_DEFAULT_REASONING_EFFORT if provider == "meta" else None),
            label=item.get("label"),
        ))
    return entries


def _menu_entries(config: Config, provider: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roster = config._snapshot_raw(provider)
    incomplete = [f"{provider}/{name}" for name, item in roster.items()
                  if not item.get("context_limit") or not item.get("output_limit")]
    available = {entry["model_id"] for entry in entries}
    unresolved = [name for name in config.menu_models if name.startswith(provider + "/") and name not in available]
    notes = []
    if incomplete:
        notes.append("omitted: missing capacity metadata for " + ", ".join(incomplete))
    if unresolved:
        notes.append("unresolved MUSE_GATEWAY_MODELS IDs (not entitlement): " + ", ".join(unresolved))
    detail = bounded_client_error_text("; ".join(notes))
    config._catalog_diagnostics[provider] = detail
    if detail and detail not in config._reported_catalog_diagnostics:
        print("muse-gateway catalog: " + detail, file=sys.stderr)
        config._reported_catalog_diagnostics.add(detail)
    wanted = set(config.menu_models)
    return [entry for entry in entries if (not wanted or entry["model_id"] in wanted)
            and not (provider == "codex" and entry["model_id"].split("/", 1)[1] in CODEX_HIDDEN_MODELS)]


def _kiro_catalog_entries(config: Config) -> list[dict[str, Any]]:
    """Names from Kiro Gateway, never account entitlement or capacity evidence."""
    if not config.kiro_api_key:
        return []
    cached = config._model_catalog_cache.get("kiro")
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        return cached[1]
    flight = config._flight_lock("kiro")
    if not flight.acquire(blocking=False):
        if cached:
            return cached[1]
        flight.acquire()
        try:
            cached = config._model_catalog_cache.get("kiro")
            if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
                return cached[1]
        finally:
            flight.release()
        flight.acquire()
    try:
        return _kiro_catalog_entries_locked(config)
    finally:
        flight.release()


def _kiro_catalog_entries_locked(config: Config) -> list[dict[str, Any]]:
    generation = config._catalog_generation.get("kiro", 0)
    names: list[str] = []
    discovery_ok = False
    if config.model_discovery:
        try:
            names = muse_kiro.discover_models(
                config.kiro_upstream, config.kiro_api_key,
                timeout=min(config.timeout, 8.0),
            )
            discovery_ok = True
            config._kiro_catalog_status = (
                f"authenticated catalog reachable ({len(names)} names); inference not validated"
            )
        except (ProviderError, OSError, HTTPException, ValueError, TypeError, RecursionError) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            # Kiro is optional: its unavailable catalog cannot hide other routes.
            config._kiro_catalog_status = "authenticated catalog unavailable; inference not validated"
    else:
        config._kiro_catalog_status = "catalog discovery disabled; inference not validated"
    items = [
        {"model": name, **resolve_limits(f"kiro/{name}"), "supports_images": False}
        for name in names
    ]
    entries = _catalog_for_items(config, "kiro", items)
    outcome = "ok_nonempty" if (discovery_ok and names) else ("disabled" if not config.model_discovery else "failed")
    stamped = time.time()
    if outcome not in ("ok_nonempty", "disabled"):
        stamped -= MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS
    raw = {item["model"]: item for item in items}
    if not config._publish_catalog("kiro", "kiro", entries=entries, raw=raw,
                                   stamped=stamped, generation=generation, outcome=outcome):
        stale = config._model_catalog_cache.get("kiro")
        if stale:
            return stale[1]
    snapshot = config._catalog_snapshots.get("kiro")
    return list(snapshot.entries) if snapshot is not None else entries


def provider_model_catalog(
    config: Config, provider: str, *, authorization: str | None = None
) -> list[dict[str, Any]]:
    if provider == "kiro":
        return _menu_entries(config, provider, _kiro_catalog_entries(config))
    if provider == "devin":
        entries = _devin_catalog_entries(config) if _provider_is_available(config, provider) else []
        return _menu_entries(config, provider, entries)
    cached = config._model_catalog_cache.get(provider)
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        return _menu_entries(config, provider, cached[1])
    flight = config._flight_lock(provider)
    if not flight.acquire(blocking=False):
        if cached:
            return _menu_entries(config, provider, cached[1])
        flight.acquire()
        try:
            cached = config._model_catalog_cache.get(provider)
            if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
                return _menu_entries(config, provider, cached[1])
        finally:
            flight.release()
        flight.acquire()
    try:
        return _provider_model_catalog_locked(config, provider, authorization=authorization)
    finally:
        flight.release()


def _provider_model_catalog_locked(
    config: Config, provider: str, *, authorization: str | None = None
) -> list[dict[str, Any]]:
    generation = config._catalog_generation.get(provider, 0)
    if provider in {"codex", "claude", "devin"} and not _provider_is_available(
        config, provider
    ):
        return _menu_entries(config, provider, [])

    discovered = _discover_provider_models(
        config, provider, authorization=authorization
    )
    configured = {"codex": config.codex_models, "claude": config.claude_models,
                  "meta": config.meta_models, "local": []}.get(provider)
    if configured is None:
        raise ProviderError("unknown catalog provider", code="unknown_catalog_provider")
    items = discovered or [{"model": model, **resolve_limits(f"{provider}/{model}")}
                           for model in configured]
    entries = _catalog_for_items(config, provider, items)
    outcome = "ok_nonempty" if discovered else ("disabled" if not config.model_discovery else "failed")
    if outcome not in ("ok_nonempty", "disabled"):
        # Discovery was attempted and came back empty: backdate the entry so
        # it expires after the short retry interval instead of serving stale
        # fallbacks for the full hour. A fresh login or recovered upstream is
        # picked up on the next fetch without a gateway restart.
        stamped = time.time() - (
            MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS
        )
    else:
        stamped = time.time()
    raw = {item["model"]: item for item in items}
    if not config._publish_catalog(provider, provider, entries=entries, raw=raw,
                                   stamped=stamped, generation=generation, outcome=outcome):
        stale = config._model_catalog_cache.get(provider)
        if stale:
            return _menu_entries(config, provider, stale[1])
    snapshot = config._catalog_snapshots.get(provider)
    return _menu_entries(config, provider, list(snapshot.entries) if snapshot is not None else entries)


def _codex_model_efforts(config: Config, model: Any) -> list[str]:
    """Model-specific Codex wire efforts from the catalog, else [] (unknown).

    An empty result means "no evidence" (bundled fallbacks carry none), in
    which case callers must pass the effort through untouched. Never fails:
    discovery or lookup problems degrade to unknown rather than breaking
    forwarding.
    """
    try:
        provider_model_catalog(config, "codex")
        entries = [{"model_id": "codex/" + name, **row}
                   for name, row in config._snapshot_raw("codex").items()]
    except (ProviderError, ShimError, ValueError, OSError, KeyError, AttributeError):
        return []
    wanted = str(model or "").split("/")[-1].strip().lower()
    if not wanted:
        return []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = str(entry.get("model_id", "")).split("/")[-1].strip().lower()
        if candidate == wanted:
            efforts = entry.get("reasoning_efforts") or []
            return [level for level in efforts if isinstance(level, str)]
    return []


def _model_supports_images(
    config: Config, provider: str, model: Any, *, authorization: str | None
) -> bool | None:
    """Image-input capability for one routed model; None means unknown.

    Mirrors _codex_model_efforts: catalog problems degrade to the catalog's
    own defaults instead of breaking forwarding.
    """
    if provider == "claude" and config.claude_transport == "cli":
        return False
    if provider == "kiro":
        return False
    if provider not in {"codex", "claude", "meta", "local"}:
        return None
    try:
        provider_model_catalog(config, provider, authorization=authorization)
        row = config._snapshot_raw(provider).get(str(model or ""))
    except (ProviderError, ShimError, ValueError, OSError, KeyError, AttributeError):
        row = None
    if isinstance(row, dict) and "supports_images" in row:
        return bool(row["supports_images"])
    return provider in {"codex", "claude", "meta"}


def _claude_cli_output_schema(
    tool_names: list[str], *, min_calls: int = 0, max_calls: int = 8
) -> dict[str, Any]:
    name_schema: dict[str, Any] = {"type": "string"}
    if tool_names:
        name_schema["enum"] = tool_names
    calls_schema: dict[str, Any] = {"type": "array", "maxItems": max_calls if tool_names else 0}
    if min_calls and tool_names:
        calls_schema["minItems"] = min_calls
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tool_calls": {
                **calls_schema,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": name_schema,
                        "arguments_json": {"type": "string"},
                    },
                    "required": ["name", "arguments_json"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["text", "tool_calls"],
        "additionalProperties": False,
    }


def _claude_cli_prompt(payload: dict[str, Any]) -> str:
    request_document = {
        "system": payload.get("system", []),
        "messages": payload.get("messages", []),
        "tools": payload.get("tools", []),
        "tool_choice": payload.get("tool_choice", {"type": "auto"}),
    }
    return (
        "Continue the conversation represented by the JSON document below. "
        "Honor its system instructions and message order. If a listed tool is "
        "needed, return it in tool_calls without pretending it was executed; "
        "arguments_json must be a JSON object encoded as a string and must match "
        "that tool's input_schema. If no tool is needed, return the complete "
        "assistant answer in text and an empty tool_calls array. Do not mention "
        "this transport envelope.\n\nREQUEST_JSON\n"
        + json.dumps(request_document, ensure_ascii=False, separators=(",", ":"))
    )


def _usage_int(value: Any) -> int:
    """Bad CLI usage numbers count as 0, never a 500 (P3-03 #10)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def run_claude_cli(
    body: dict[str, Any], config: Config, *, track: Any = None
) -> tuple[dict[str, Any], Any]:
    """Execute one Muse turn through Claude Code print mode.

    ``track``, when given, is called with the owned ``Popen`` handle so
    server shutdown can terminate it (P1-10). Callers that pass
    ``handler._track_upstream`` get cleanup on request end and shutdown.
    """
    model = str(body.get("model") or "").strip()
    if not model:
        raise ProviderError("Claude CLI request is missing a model.", code="model_missing")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model) or model.startswith("-"):
        raise ProviderError(
            "Claude CLI model name contains characters that cannot be passed as an argument.",
            code="claude_cli_invalid_model",
        )
    model_limit = config.model_output_limit("claude", model)
    payload, tool_names = responses_to_anthropic(
        body,
        oauth=False,
        max_output_tokens=model_limit,
    )
    if payload.get("stop_sequences"):
        raise ProviderError(
            "Claude CLI transport cannot enforce stop sequences; omit stop or use --claude-transport http.",
            code="claude_cli_stop_unsupported",
        )
    # P2-02: `claude -p` has no token-limit flag (checked against the
    # installed CLI's --help), so a cap below the model limit cannot be
    # honored; refuse it rather than pretend.
    requested = body.get("max_output_tokens", body.get("max_tokens"))
    if requested is not None and model_limit and int(payload.get("max_tokens") or 0) < model_limit:
        raise ProviderError(
            "Claude CLI transport cannot enforce max_output_tokens below the model limit; "
            "omit it or use --claude-transport http.",
            code="claude_cli_output_limit_unsupported",
        )
    for message in payload.get("messages", []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image" for part in content
        ):
            raise ProviderError(
                "Claude CLI transport supports text input only; use --claude-transport http for image input.",
                code="claude_cli_images_unsupported",
            )
    available_tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    available_names = [
        str(tool.get("name"))
        for tool in available_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    # P1-05: request policy constrains the schema and the parsed result.
    forced = payload.get("tool_choice") if isinstance(payload.get("tool_choice"), dict) else {}
    min_calls = 1 if available_names and forced.get("type") in {"any", "tool"} else 0
    max_calls = 1 if forced.get("disable_parallel_tool_use") else 8
    if forced.get("type") == "tool" and forced.get("name") in available_names:
        available_names = [str(forced["name"])]
    schema = _claude_cli_output_schema(available_names, min_calls=min_calls, max_calls=max_calls)
    executable = config.claude_cli
    if os.name == "nt":
        # POSIX exec already searches PATH like doctor's shutil.which; Windows
        # CreateProcess does not apply PATHEXT, so resolve .cmd/.exe here (P2-02/P2-14).
        executable = (shutil.which(config.claude_cli) or shutil.which(f"{config.claude_cli}.cmd")
                      or shutil.which(f"{config.claude_cli}.exe") or config.claude_cli)
    command = [
        executable,
        "-p",
        "--safe-mode",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--model",
        model,
        "--tools",
        "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        "--system-prompt",
        "You are a model backend for Muse Code. Return only the requested structured result.",
    ]
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if isinstance(effort, str) and "haiku" not in model.lower():
        normalized_effort = {"minimal": "low", "ultra": "max"}.get(
            effort.lower(), effort.lower()
        )
        if normalized_effort in {"low", "medium", "high", "xhigh", "max"}:
            command.extend(["--effort", normalized_effort])

    environment = _child_environment(
        keep=_CLAUDE_PASSTHROUGH_SETS[config.claude_cli_passthrough] | _CLAUDE_HARMLESS_VARS
    )
    environment.pop("CLAUDECODE", None)
    environment.pop("CLAUDE_CODE_ENTRYPOINT", None)
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
    except FileNotFoundError as error:
        raise ProviderError(
            f"Claude CLI executable {config.claude_cli!r} was not found. Install Claude Code or pass --claude-cli.",
            status=502,
            code="claude_cli_missing",
        ) from error
    if callable(track):
        track(child)
    try:
        stdout, stderr = child.communicate(
            input=_claude_cli_prompt(payload), timeout=config.timeout
        )
    except subprocess.TimeoutExpired as error:
        _stop_child(child)
        raise ProviderError(
            f"Claude CLI exceeded the {config.timeout:g}s timeout.",
            status=504,
            code="claude_cli_timeout",
        ) from error
    completed = types.SimpleNamespace(
        returncode=child.returncode, stdout=stdout, stderr=stderr
    )

    result: dict[str, Any] | None = None
    try:
        decoded = json.loads(str(completed.stdout or ""))
        if isinstance(decoded, dict):
            result = decoded
    except json.JSONDecodeError:
        pass
    if result is not None and result.get("is_error") is True:
        message = str(result.get("result") or "Claude CLI reported an error.")[:500]
        raw_status = result.get("api_error_status")
        status = int(raw_status) if isinstance(raw_status, int) else 502
        raise ProviderError(
            message,
            status=status,
            code="claude_cli_rate_limited" if status == 429 else "claude_cli_failed",
        )
    if completed.returncode != 0:
        detail = " ".join(str(completed.stderr or "").strip().splitlines())[-500:]
        suffix = f": {detail}" if detail else ""
        raise ProviderError(
            f"Claude CLI exited with status {completed.returncode}{suffix}",
            status=502,
            code="claude_cli_failed",
        )
    if result is None:
        raise ProviderError(
            "Claude CLI returned invalid JSON.",
            status=502,
            code="claude_cli_invalid_response",
        )
    structured = result.get("structured_output")
    if not isinstance(structured, dict):
        try:
            structured = json.loads(str(result.get("result") or ""))
        except json.JSONDecodeError as error:
            raise ProviderError(
                "Claude CLI returned no structured output.",
                status=502,
                code="claude_cli_invalid_response",
            ) from error
    text = structured.get("text")
    calls = structured.get("tool_calls")
    if not isinstance(text, str) or not isinstance(calls, list):
        raise ProviderError(
            "Claude CLI structured output has an invalid shape.",
            status=502,
            code="claude_cli_invalid_response",
        )
    if len(calls) < min_calls or len(calls) > (max_calls if available_names else 0):
        raise ProviderError(
            "Claude CLI result violates the request's tool policy.",
            status=502,
            code="claude_cli_invalid_response",
        )

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in available_names:
            raise ProviderError(
                "Claude CLI returned an unknown tool call.",
                status=502,
                code="claude_cli_invalid_tool",
            )
        try:
            arguments = json.loads(str(call.get("arguments_json") or "{}"))
        except json.JSONDecodeError as error:
            raise ProviderError(
                "Claude CLI returned invalid tool arguments.",
                status=502,
                code="claude_cli_invalid_tool",
            ) from error
        if not isinstance(arguments, dict):
            raise ProviderError(
                "Claude CLI tool arguments must be a JSON object.",
                status=502,
                code="claude_cli_invalid_tool",
            )
        content.append(
            {
                "type": "tool_use",
                "id": "toolu_" + secrets.token_hex(12),
                "name": str(call["name"]),
                "input": arguments,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    return (
        {
            "id": "msg_" + secrets.token_hex(12),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "usage": {
                "input_tokens": _usage_int(usage.get("input_tokens")),
                "output_tokens": _usage_int(usage.get("output_tokens")),
            },
        },
        tool_names,
    )


def _anthropic_message_sse(payload: dict[str, Any]) -> io.BytesIO:
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": payload})
    ]
    for index, block in enumerate(payload.get("content", [])):
        if block.get("type") == "text":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block.get("text", "")},
                    },
                )
            )
        else:
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(
                                block.get("input", {}),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    },
                )
            )
        events.append(
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        )
    events.extend(
        [
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": payload.get("stop_reason")},
                    "usage": payload.get("usage", {}),
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    raw = b"".join(
        (
            f"event: {event_name}\n"
            f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
        ).encode("utf-8")
        for event_name, event in events
    )
    return io.BytesIO(raw)


_LOG_TOKEN_PATTERN = re.compile(
    rb"(?:devin-session-token\$[^\s\"'<>]+|"
    rb"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|"
    rb"sk-[A-Za-z0-9_-]{16,})"
)


def _redact_bytes(payload: bytes, secrets: Iterable[str]) -> bytes:
    result = payload
    values = set()
    for secret in secrets:
        if secret:
            values.add(secret)
            if secret.lower().startswith("bearer "):
                values.add(secret[7:].strip())
    forms = {form for secret in values for form in encoded_secret_forms(secret)}
    for form in sorted(forms, key=len, reverse=True):
        result = result.replace(form, b"[REDACTED]")
    return _LOG_TOKEN_PATTERN.sub(b"[REDACTED]", result)


_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}


def _upstream_diagnostic(payload: bytes) -> dict[str, Any]:
    """Extract known error categories, never arbitrary upstream text."""
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeError):
        return {"reason": "unstructured_upstream_error"}
    if not isinstance(value, dict):
        return {"reason": "unstructured_upstream_error"}
    if isinstance(value.get("response"), dict):
        value = value["response"]
    error = value.get("error", value)
    if not isinstance(error, dict):
        error = {}
    # Upstream uses both "detail" and "Detail" across models/endpoints.
    message = error.get("message", value.get("detail", value.get("Detail", "")))
    result: dict[str, Any] = {}
    for field in ("type", "code", "param"):
        marker = error.get(field)
        if isinstance(marker, str) and re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", marker):
            result[field] = marker
    if message == "System messages are not allowed":
        result["reason"] = "system_messages_not_allowed"
    elif message == "Input must be a list":
        result["reason"] = "input_must_be_list"
    elif message == "Stream must be set to true":
        result["reason"] = "stream_required"
    if error.get("param") == "reasoning.effort":
        result["param"] = "reasoning.effort"
        if error.get("code") == "unsupported_value":
            result["reason"] = "unsupported_reasoning_effort"
            if isinstance(message, str) and "Supported values are:" in message:
                tail = message.split("Supported values are:", 1)[1]
                result["supported_efforts"] = [
                    word for word in re.findall(r"'([a-z]+)'", tail)
                    if word in _REASONING_EFFORTS
                ]
    return result or {"reason": "upstream_rejected_request"}


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-JSON numeric constant")


def _capture_rejected_kiro_body(upstream: Any) -> None:
    _read_kiro_error_prefix(upstream, capture_only=True)


def _read_kiro_error_prefix(
    upstream: Any, *, label: str = "upstream", capture_only: bool = False,
) -> bytes:
    """One bounded, short read; body evidence cannot replace a known error.

    Chunked read1 can loop over framing/trailers despite a socket timeout, so
    omit it. For other bodies, short reads are not EOF: require a completed
    Content-Length or observed close-delimited EOF to claim completeness.
    """
    capture = current_capture()
    if capture is None and capture_only:
        return b""
    incomplete_marker = "kiro_http_error_body_incomplete" if label == "upstream.error" else "kiro_rejected_body_incomplete"

    def mark(marker):
        if capture is not None and marker not in capture.errors:
            capture.errors.append(marker)

    def record(method, *args, **kwargs):
        if capture is not None:
            try:
                getattr(capture, method)(*args, **kwargs)
            except Exception:
                mark("kiro_error_capture_failed")
                mark(incomplete_marker)
                return False
        return True

    sock = None
    previous_timeout = None
    restore_timeout = False
    prefix = b""
    body_complete = False
    body_label = label + ".body"
    try:
        if label == "upstream.error":
            record("json", label + ".headers", {
                "status": getattr(upstream, "status", None),
                "headers": dict(getattr(upstream, "headers", None) or {}),
            })
        if getattr(upstream, "chunked", False):
            record("event", body_label + ".omitted",
                   reason="chunked_framing_has_no_absolute_read_bound", complete=False)
            mark(incomplete_marker)
            return prefix
        reader = getattr(upstream, "read1", None)
        sock = getattr(getattr(getattr(upstream, "fp", None), "raw", None), "_sock", None)
        if not callable(reader) or sock is None:
            record("event", body_label + ".omitted", reason="bounded_reader_unavailable", complete=False)
            mark(incomplete_marker)
            return prefix
        previous_timeout = sock.gettimeout()
        sock.settimeout(min(previous_timeout, KIRO_REJECTED_BODY_TIMEOUT)
                        if previous_timeout is not None else KIRO_REJECTED_BODY_TIMEOUT)
        restore_timeout = True
        prefix = reader(KIRO_REJECTED_BODY_LIMIT + 1)
        stored = record("bytes", body_label, prefix)
        remaining = getattr(upstream, "length", None)
        framed_end = remaining == 0 if remaining is not None else upstream.isclosed()
        body_complete = len(prefix) <= KIRO_REJECTED_BODY_LIMIT and framed_end
        capture_complete = stored and body_complete
        record("event", body_label + ".rejected", bytes=len(prefix), limit=KIRO_REJECTED_BODY_LIMIT,
               complete=capture_complete, representation="credentials_scrubbed_prefix")
        if not capture_complete:
            mark(incomplete_marker)
    except Exception as error:
        partial = getattr(error, "partial", b"")
        if isinstance(partial, bytes) and partial:
            prefix = partial[:KIRO_REJECTED_BODY_LIMIT + 1]
            record("bytes", body_label, prefix)
        record("event", body_label + ".omitted", reason=type(error).__name__, bytes=len(prefix), complete=False)
        mark(incomplete_marker)
    finally:
        if restore_timeout:
            try:
                sock.settimeout(previous_timeout)
            except OSError:
                pass
    # A parseable prefix need not be the complete HTTP entity. Keep it in the
    # private evidence capture, but never classify an incomplete representation.
    return prefix if body_complete else b""


def _capture_http_error(error) -> bytes:
    """Keep normal error projection unchanged; drain evidence only on opt-in."""
    try:
        first = error.read(65536)
    except IncompleteRead as partial:
        first = partial.partial[:65536] if isinstance(partial.partial, bytes) else b""
        capture = current_capture()
        if capture is not None:
            capture.event("upstream_error_capture_incomplete", got=len(first))
    except (OSError, HTTPException):
        first = b""
    capture = current_capture()
    if capture is not None:
        label = "upstream.error" if "upstream.error.body" not in capture.streams else capture.label("upstream.error")
        capture.json(label + ".headers", {"status": error.code, "headers": dict(error.headers or {})})
        try:
            error_value = json.loads(first)
        except (ValueError, UnicodeError, RecursionError):
            capture.event(label + ".json_unavailable", reason="non_json_or_incomplete_prefix")
        else:
            # Register structured credentials before saving the raw body too.
            capture.json(label, error_value)
        capture.bytes(label + ".body", first)
        observed = len(first)
        while observed < 64 * 1024 * 1024:
            try:
                chunk = error.read(min(65536, 64 * 1024 * 1024 - observed))
            except (OSError, HTTPException) as capture_error:
                capture.event(label + ".incomplete", kind=type(capture_error).__name__, complete=False)
                capture.errors.append("upstream_error_capture_incomplete")
                break
            if not chunk:
                capture.event(label + ".eof", bytes=observed)
                break
            observed += len(chunk)
            capture.bytes(label + ".body", chunk)
        else:
            capture.event(label + ".limit", bytes=observed, complete=False)
    return first


def _request_diagnostic(body: dict[str, Any]) -> dict[str, Any]:
    """Record shape and recognized settings, never prompts or tool arguments."""
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    history = body.get("input")
    items = history if isinstance(history, list) else []
    model = body.get("model")
    safe_model = (
        model if isinstance(model, str)
        and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", model)
        and not re.search(r"sk-|eyJ|token|secret|password|credential", model, re.I)
        else "[omitted]"
    )
    return {
        "model": safe_model,
        "effort": effort if isinstance(effort, str) and effort in _REASONING_EFFORTS else "unspecified_or_other",
        "previous_response_id_present": bool(body.get("previous_response_id")),
        "input_items": len(items),
        "system_messages": sum(isinstance(i, dict) and i.get("role") == "system" for i in items),
        "tool_calls": sum(isinstance(i, dict) and i.get("type") == "function_call" for i in items),
        "tool_results": sum(isinstance(i, dict) and i.get("type") == "function_call_output" for i in items),
        "image_parts": _count_image_parts(body),
    }


def _count_image_parts(body: dict[str, Any]) -> int:
    """Shape only: a superset of the parts the image normalizer inspects."""
    def count(parts: Any) -> int:
        if not isinstance(parts, list):
            return 0
        return sum(isinstance(part, dict) and str(part.get("type") or "").strip().lower() in _IMAGE_TYPES
                   for part in parts)

    total = count(body.get("instructions")) + count(body.get("system"))
    for item in _input_items(body):
        if isinstance(item, dict):
            total += count(item.get("content")) + count(item.get("output"))
    return total


def _ssl_context(config: Config) -> ssl.SSLContext | None:
    if config.insecure_upstream:
        return ssl._create_unverified_context()  # noqa: SLF001 - explicit opt-in
    return None


def _credentialed_urlopen(request: Any, *, timeout: float, config: Config | None = None) -> Any:
    """Open a credentialed request through the scoped gateway opener (P0-04)."""
    url = getattr(request, "full_url", None) or (request if isinstance(request, str) else None)
    opener = build_gateway_opener(
        url, ssl_context=_ssl_context(config) if config is not None else None
    )
    return opener.open(request, timeout=timeout)


_IDLE_TIMEOUT_FALLBACK_LOGGED = False


def _set_stream_idle_timeout(upstream: Any, config: Config | None) -> None:
    """Best-effort: switch an opened upstream socket to the idle budget (P1-09).

    If the socket is unreachable on this Python, the connect timeout stays in
    force and the fallback is reported once.
    """
    global _IDLE_TIMEOUT_FALLBACK_LOGGED
    if config is None:
        return
    idle = getattr(config, "stream_idle_timeout", None)
    if not idle:
        return
    if (not set_response_idle_timeout(upstream, float(idle))
            and isinstance(upstream, http.client.HTTPResponse)
            and not _IDLE_TIMEOUT_FALLBACK_LOGGED):
        _IDLE_TIMEOUT_FALLBACK_LOGGED = True
        print("muse-gateway: upstream socket not reachable; --timeout stays the stream read budget",
              file=sys.stderr, flush=True)


class _CaptureReadlineStream:
    """Wrap an SSE stream so capture observes the raw upstream bytes."""

    def __init__(self, upstream: Any) -> None:
        self._upstream = upstream

    def readline(self, limit: int = -1):
        chunk = self._upstream.readline(limit)
        if chunk:
            capture = current_capture()
            if capture is not None:
                data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                capture.event("upstream.read", bytes=len(data))
                capture.bytes("upstream.body", data)
                capture.observe_sse("upstream", data)
        return chunk

    def __getattr__(self, name: str):
        return getattr(self._upstream, name)


def _iter_upstream_chunks(upstream: Any, *, content_type: str) -> Iterable[bytes]:
    """Yield promptly from an upstream stream without a large read buffer."""
    def observed(chunk):
        data = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
        capture = current_capture()
        if capture is not None:
            capture.event("upstream.read", bytes=len(data))
            capture.bytes("upstream.body", data)
            if "text/event-stream" in content_type.lower():
                capture.observe_sse("upstream", data)
        return data

    if "text/event-stream" in content_type.lower():
        reader = getattr(upstream, "readline", None)
        if callable(reader):
            while True:
                chunk = reader(UPSTREAM_SSE_LINE_LIMIT + 1)
                if chunk and len(chunk) > UPSTREAM_SSE_LINE_LIMIT and not chunk.endswith(b"\n" if isinstance(chunk, bytes) else "\n"):
                    raise ProviderError(
                        "Upstream SSE line exceeded the safety size limit.",
                        status=502, code="upstream_protocol_error",
                    )
                if not chunk:
                    if current_capture() is not None:
                        current_capture().event("upstream.eof")
                    return
                yield observed(chunk)

    reader = getattr(upstream, "read1", None)
    if not callable(reader):
        reader = getattr(upstream, "read", None)
    if not callable(reader):
        return
    while True:
        chunk = reader(8192)
        if not chunk:
            if current_capture() is not None:
                current_capture().event("upstream.eof")
            return
        yield observed(chunk)


UPSTREAM_JSON_RESPONSE_LIMIT = 64 * 1024 * 1024


def _read_bounded_upstream_body(upstream: Any, *, limit: int = UPSTREAM_JSON_RESPONSE_LIMIT) -> bytes:
    """Bounded non-stream upstream read (P1-13).

    Streaming passthrough totals stay uncapped by design (bytes flow
    through; only lines/accumulations are capped). Truncation and oversize
    are distinct operator signals with distinct codes.
    """
    try:
        raw = upstream.read(limit + 1)
    except (IncompleteRead, HTTPException, OSError) as error:
        partial = getattr(error, "partial", b"")
        if current_capture() is not None and isinstance(partial, bytes) and partial:
            current_capture().bytes("upstream.body", partial[:limit + 1])
            current_capture().event("upstream.body.incomplete", complete=False)
        raise ProviderError(
            "Upstream response ended before its declared HTTP body length.",
            status=502,
            code="upstream_truncated",
        ) from error
    data = raw.encode("utf-8") if isinstance(raw, str) else raw
    capture = current_capture()
    if capture is not None:
        capture.bytes("upstream.body", data)
    if len(data) > limit:
        raise ProviderError(
            "Upstream response exceeded the safety size limit.",
            status=502,
            code="upstream_response_too_large",
        )
    if getattr(upstream, "length", None) not in {None, 0}:
        raise ProviderError(
            "Upstream response ended before its declared HTTP body length.",
            status=502,
            code="upstream_truncated",
        )
    if capture is not None:
        capture.event("upstream.eof")
    return data


CATALOG_JSON_RESPONSE_LIMIT = 4 * 1024 * 1024
HEALTH_JSON_RESPONSE_LIMIT = 64 * 1024


def _read_bounded_json(response: Any, *, limit: int) -> Any:
    raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ProviderError(
            "JSON response exceeded the safety size limit.",
            status=502, code="upstream_response_too_large",
        )
    return json.loads(raw.decode("utf-8"))


class _ClientCancelWatch:
    """Shut the upstream socket on client hangup or turn deadline.

    A watcher thread polls the client socket (no async runtime). Closing
    the upstream unblocks a relay waiting on a silent upstream; existing
    BrokenPipe handling stays as the backstop for writes.
    """

    POLL_SECONDS = 1.0

    def __init__(self, handler: Any, upstream: Any, *, deadline: float | None = None) -> None:
        self.handler, self.upstream = handler, upstream
        self.deadline = deadline
        self.fired = False
        self.deadline_fired = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_ClientCancelWatch":
        if getattr(self.handler, "connection", None) is not None or self.deadline is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _interrupt_upstream(self) -> None:
        sock = response_socket(self.upstream)
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
            else:
                self.upstream.close()
        except (OSError, AttributeError):
            pass

    def _run(self) -> None:
        while True:
            wait = self.POLL_SECONDS
            if self.deadline is not None:
                wait = min(wait, max(0.0, self.deadline - time.monotonic()))
            if self._stop.wait(wait):
                return
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.deadline_fired = True
                self._interrupt_upstream()
                return
            if _client_hung_up(self.handler):
                self.fired = True
                self._interrupt_upstream()
                return

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.POLL_SECONDS + 1)


UPSTREAM_SSE_LINE_LIMIT = 4 * 1024 * 1024
UPSTREAM_SSE_EVENT_LIMIT = 8 * 1024 * 1024
UPSTREAM_STREAM_RESPONSE_LIMIT = 64 * 1024 * 1024
_SSE_FRAME_BOUNDARY = re.compile(rb"\r\n\r\n|\n\n|\r\r")


def _parse_sse_frame(frame: bytes) -> tuple[str, dict[str, Any]] | None:
    """Parse one complete SSE frame into a Responses event payload."""
    event_names: list[str] = []
    data_lines: list[bytes] = []
    for line in frame.splitlines():
        if not line or line.startswith(b":"):
            continue
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"event":
            event_names.append(value.decode("utf-8", errors="strict").strip())
        elif field == b"data":
            data_lines.append(value)
    if not data_lines:
        return None
    raw = b"\n".join(data_lines)
    if raw == b"[DONE]":
        return ("[DONE]", {})
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SSE data payload is not an object")
    event_type = payload.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("SSE data payload has no event type")
    if any(name != event_type for name in event_names):
        raise ValueError("SSE event name does not match payload type")
    return event_type, payload


def _relay_parsed_sse(handler: Any, iterator: Any, watch: _ClientCancelWatch, observe: Any) -> bool:
    """Relay validated Responses SSE; False when the client cancelled."""
    buffer = bytearray()
    total = 0
    next_sequence = 0
    resp_id = "resp_" + secrets.token_hex(8)
    saw_created = False
    saw_terminal = False
    missing_sequence = 0
    failure_code = "upstream_truncated"
    failure_message = "Upstream stream ended without a terminal event."
    failure_log = "upstream_stream_truncated"

    def write_event(event_type: str, payload: dict[str, Any]) -> None:
        nonlocal next_sequence
        payload = dict(payload)
        payload["sequence_number"] = next_sequence
        next_sequence += 1
        handler.wfile.write(response_sse_line(event_type, payload))
        handler.wfile.flush()

    def ensure_created(first_payload: dict[str, Any] | None = None) -> None:
        nonlocal resp_id, saw_created
        if saw_created:
            return
        if isinstance(first_payload, dict):
            response_obj = first_payload.get("response")
            if isinstance(response_obj, dict) and isinstance(response_obj.get("id"), str) and response_obj["id"]:
                resp_id = response_obj["id"]
        snapshot = {"id": resp_id, "object": "response", "status": "in_progress", "output": []}
        write_event("response.created", {"type": "response.created", "response": dict(snapshot)})
        if not (isinstance(first_payload, dict) and first_payload.get("type") == "response.in_progress"):
            write_event("response.in_progress", {"type": "response.in_progress", "response": dict(snapshot)})
        saw_created = True

    def write_failed(code: str, message: str) -> None:
        ensure_created()
        handler._stream_failed = True
        write_event("response.failed", {
            "type": "response.failed",
            "response": {
                "id": resp_id, "object": "response", "status": "failed", "output": [],
                "error": {
                    "type": bounded_client_error_code(code, "upstream_truncated"),
                    "code": bounded_client_error_code(code, "upstream_truncated"),
                    "message": bounded_client_error_text(message),
                },
            },
        })

    while not saw_terminal:
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        except (OSError, HTTPException, TimeoutError, ProviderError) as error:
            if watch.fired:
                handler.close_connection = True
                if callable(observe):
                    observe("client_disconnected", disconnect_origin="client")
                return False
            if isinstance(error, ProviderError):
                failure_code = error.code
                failure_message = str(error)
                failure_log = failure_code
            else:
                failure_log = "upstream_stream_interrupted"
                if callable(observe):
                    observe(failure_log, kind=type(error).__name__)
            break
        if watch.fired:
            handler.close_connection = True
            if callable(observe):
                observe("client_disconnected", disconnect_origin="client")
            return False
        total += len(chunk)
        if total > UPSTREAM_STREAM_RESPONSE_LIMIT:
            failure_code = "upstream_response_too_large"
            failure_message = "Upstream stream exceeded the safety size limit."
            failure_log = failure_code
            buffer.clear()
            break
        buffer.extend(chunk)
        while True:
            match = _SSE_FRAME_BOUNDARY.search(buffer)
            if match is None:
                if len(buffer) > UPSTREAM_SSE_EVENT_LIMIT:
                    failure_code = "upstream_protocol_error"
                    failure_message = "Upstream SSE event exceeded the safety size limit."
                    failure_log = failure_code
                    buffer.clear()
                break
            frame = bytes(buffer[:match.start()])
            del buffer[:match.end()]
            if len(frame) > UPSTREAM_SSE_EVENT_LIMIT:
                failure_code = "upstream_protocol_error"
                failure_message = "Upstream SSE event exceeded the safety size limit."
                failure_log = failure_code
                buffer.clear()
                break
            try:
                parsed = _parse_sse_frame(frame)
            except (ValueError, UnicodeError):
                failure_code = "upstream_protocol_error"
                failure_message = "Upstream stream sent a malformed SSE frame."
                failure_log = failure_code
                buffer.clear()
                break
            if parsed is None:
                continue
            event_type, payload = parsed
            if event_type == "[DONE]":
                continue
            if not isinstance(payload.get("sequence_number"), int) or isinstance(payload.get("sequence_number"), bool):
                missing_sequence += 1
            if event_type == "error":
                error_obj = payload.get("error") if isinstance(payload.get("error"), dict) else {}
                code = error_obj.get("code") or error_obj.get("type") or payload.get("code") or "upstream_error"
                message = error_obj.get("message") or payload.get("message") or "Upstream stream returned an error."
                if callable(observe):
                    observe("upstream_stream_error", **_upstream_diagnostic(json.dumps(payload).encode()))
                write_failed(str(code), str(message))
                saw_terminal = True
                break
            response_obj = payload.get("response")
            if isinstance(response_obj, dict):
                response_obj = dict(response_obj)
                payload = dict(payload)
                payload["response"] = response_obj
                if not saw_created and event_type == "response.created":
                    if isinstance(response_obj.get("id"), str) and response_obj["id"]:
                        resp_id = response_obj["id"]
                elif not saw_created:
                    ensure_created(payload)
                    response_obj = dict(payload["response"])
                    payload = dict(payload)
                    payload["response"] = response_obj
                response_obj["id"] = resp_id
            elif not saw_created and event_type != "response.created":
                ensure_created(payload)
            if event_type == "response.created":
                saw_created = True
            if event_type == "response.failed" and callable(observe):
                observe("upstream_stream_error", **_upstream_diagnostic(json.dumps(payload).encode()))
            if event_type in {"response.failed", "response.incomplete"}:
                handler._stream_failed = True
            write_event(event_type, payload)
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                saw_terminal = True
                break
            if watch.fired:
                handler.close_connection = True
                if callable(observe):
                    observe("client_disconnected", disconnect_origin="client")
                return False
        if failure_code != "upstream_truncated" and not buffer and total > 0 and not saw_terminal:
            # A protocol/size failure above has consumed/discarded the frame;
            # terminal synthesis below is the next and only output.
            if failure_log in {"upstream_protocol_error", "upstream_response_too_large"}:
                break
    if watch.fired and not saw_terminal:
        handler.close_connection = True
        if callable(observe):
            observe("client_disconnected", disconnect_origin="client")
        return False
    if missing_sequence and callable(observe):
        observe("upstream_missing_sequence_number", count=missing_sequence)
    if not saw_terminal:
        if buffer.strip() and failure_code == "upstream_truncated":
            failure_message = "Upstream stream ended with an incomplete SSE frame."
        if getattr(watch, "deadline_fired", False):
            failure_code = "turn_timeout"
            failure_message = "Turn exceeded the configured deadline."
            failure_log = failure_code
        if callable(observe):
            observe(failure_log)
        write_failed(failure_code, failure_message)
    return True


def _client_hung_up(handler: Any) -> bool:
    """True when the inbound client has closed its socket (EOF is readable)."""
    connection = getattr(handler, "connection", None)
    if connection is None or not hasattr(connection, "fileno"):
        return False
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def _write_upstream_body(handler: BaseHTTPRequestHandler, upstream: Any, *, stream: bool, content_type: str) -> None:
    if stream:
        observe = getattr(handler, "_log_event", None)
        iterator = iter(_iter_upstream_chunks(upstream, content_type=content_type))
        watch_factory = getattr(handler, "_stream_watch", None)
        watch = watch_factory(upstream) if callable(watch_factory) else _ClientCancelWatch(handler, upstream)
        with watch:
            _relay_parsed_sse(handler, iterator, watch, observe)
        return
    payload = _read_bounded_upstream_body(upstream)
    if payload:
        handler.wfile.write(payload)
        handler.wfile.flush()


def _buffered_codex_response(upstream: Any, content_type: str) -> dict[str, Any]:
    """Collapse one streamed Codex turn into its terminal response object.

    The Codex backend requires ``stream: true`` ("Stream must be set to
    true"), so non-streaming inbound turns are streamed upstream and
    collapsed here back to the single Responses object the caller asked
    for. Failure terminals raise; a stream that ends without one is a
    truncation error, never silent success.

    The upstream ``Content-Type`` is only a hint for which reader to try
    first: a proxy may label a streamed SSE body ``application/json``
    (or omit the header entirely), so a non-SSE label whose body parses
    as SSE is collapsed like any other stream. A body that is neither a
    JSON object nor an SSE stream with a terminal still truncates.
    """
    def consider(name: str, raw: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        terminal = name or str(data.get("type") or "")
        if terminal in {"response.completed", "response.incomplete"}:
            response = data.get("response")
            return response if isinstance(response, dict) else None
        if terminal == "response.failed":
            response = data.get("response")
            error = response.get("error") if isinstance(response, dict) else None
            detail: dict[str, Any] = {"reason": "upstream_response_failed"}
            code = error.get("code") if isinstance(error, dict) else None
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", code):
                detail["code"] = code
            raise ProviderError(f"Codex turn failed. detail={json.dumps(detail)}",
                                status=502, code="codex_upstream_error")
        if terminal == "error":
            detail = _upstream_diagnostic(raw.encode("utf-8"))
            raise ProviderError(f"Codex turn failed. detail={json.dumps(detail)}",
                                status=502, code="codex_upstream_error")
        return None

    def collapse_sse_lines(lines: Iterable[str]) -> dict[str, Any] | None:
        event_name = ""
        data_lines: list[str] = []
        for line in lines:
            if not line.strip():
                if data_lines:
                    name, raw = event_name, "\n".join(data_lines)
                    event_name, data_lines = "", []
                    final = consider(name, raw)
                    if final is not None:
                        return final
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                if sum(len(part) for part in data_lines) > 4 * 1024 * 1024:
                    raise ProviderError("Codex SSE event exceeded the safety size limit.",
                                        status=502, code="upstream_protocol_error")
        if data_lines:
            return consider(event_name, "\n".join(data_lines))
        return None

    if "text/event-stream" not in content_type.lower():
        try:
            payload = _read_bounded_upstream_body(upstream)
        except ProviderError:
            raise
        except OSError as error:
            raise ProviderError("Codex turn ended without a response body.",
                                status=502, code="codex_truncated_stream") from error
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        parse_error: ValueError | None = None
        try:
            data = json.loads(payload)
        except ValueError as error:
            parse_error, data = error, None
        if isinstance(data, dict):
            return data
        lines = payload.splitlines()
        if any(line.startswith(("event:", "data:")) for line in lines):
            final = collapse_sse_lines(lines)
            if final is not None:
                return final
        raise ProviderError("Codex turn ended without a response body.",
                            status=502, code="codex_truncated_stream") from parse_error

    def streamed_lines() -> Iterable[str]:
        total = 0
        for chunk in _iter_upstream_chunks(upstream, content_type=content_type):
            total += len(chunk)
            if total > UPSTREAM_STREAM_RESPONSE_LIMIT:
                raise ProviderError("Codex stream exceeded the safety size limit.",
                                    status=502, code="upstream_response_too_large")
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            yield from text.splitlines()

    final = collapse_sse_lines(streamed_lines())
    if final is not None:
        return final
    raise ProviderError("Codex stream ended without a terminal response event.",
                        status=502, code="codex_truncated_stream")


def _select_bearer(
    static_key: str | None, incoming_auth: str | None, *, missing: str
) -> str:
    """Pick the upstream Authorization header for one Responses forward.

    A statically configured key wins; otherwise the bearer Muse itself sent
    is forwarded. Gateway fork: shared by the generic, meta, and local routes.
    """
    if static_key:
        return f"Bearer {static_key}"
    if incoming_auth:
        return incoming_auth
    raise ShimError(missing)


_forwarding_warnings: set[str] = set()


_cleartext_devin_warned: set[str] = set()


def _warn_cleartext_devin_upstream(url: str) -> None:
    """User-supplied http:// Devin upstream off loopback: warn, never refuse (P2-09)."""
    parsed = urllib.parse.urlsplit(url or "")
    if parsed.scheme != "http" or _is_loopback_host(parsed.hostname or "") or url in _cleartext_devin_warned:
        return
    _cleartext_devin_warned.add(url)
    print(f"WARNING: --devin-upstream {parsed.hostname} uses plain http; the Devin session token "
          "and conversation travel unencrypted.", file=sys.stderr, flush=True)


def _warn_credential_forwarding_once(route: str, url: str, *, custom: bool) -> None:
    """Warn once when a Muse bearer is forwarded to a non-default upstream.

    Static route keys take precedence, so reaching this point means the Muse
    client credential itself is being sent to the upstream. That is the
    documented subscription mechanism for Meta's own endpoint, but it deserves
    a loud note when the destination is anything else.
    """
    if not custom or route in _forwarding_warnings:
        return
    _forwarding_warnings.add(route)
    destination = urllib.parse.urlsplit(url)
    host = destination.hostname or "[invalid host]"
    print(
        f"WARNING: muse-gateway {route}/ has no static API key, so the Muse "
        f"authorization header is forwarded to the custom upstream host {host}. Only "
        "use upstreams you trust; prefer a dedicated route API key.",
        file=sys.stderr,
        flush=True,
    )


def _gateway_version() -> str:
    try:
        from importlib.metadata import version

        return version("muse-gateway")
    except Exception:
        pass
    try:
        text = Path(__file__).with_name("pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "0.0.0+unknown"


class MuseShimHandler(BaseHTTPRequestHandler):
    server_version = f"muse-gateway/{_gateway_version()}"
    protocol_version = "HTTP/1.1"
    # Idle keep-alive budget only; upstream sockets use their own timeouts (P1-09).
    timeout = 30
    body_timeout_floor = 60.0

    def setup(self) -> None:
        super().setup()
        self._capture_request = None
        self._inference_slot_acquired = False
        self._request_started_at: float | None = None
        self._pending_capture_incoming: tuple[dict[str, Any], dict[str, str], bytes] | None = None
        self._pending_capture_transformed: tuple[dict[str, Any], str, str] | None = None
        if getattr(self.server, "capture_session", None) is not None:
            self.wfile = CapturingWriter(self.wfile)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        finally:
            self._release_tracked()
            capture = getattr(self, "_capture_request", None)
            if capture is not None:
                if getattr(self, "_pending_capture_incoming", None) is not None or getattr(self, "_pending_capture_transformed", None) is not None:
                    capture.event(
                        "capture.request_artifacts_omitted",
                        reason="credential_registration_incomplete",
                    )
                    self._pending_capture_incoming = None
                    self._pending_capture_transformed = None
                capture.finish(status=getattr(self, "_response_status", None),
                               stream_failed=getattr(self, "_stream_failed", False),
                               close_connection=self.close_connection)
                self._capture_request = None

    @property
    def config(self) -> Config:
        return self.server.config  # type: ignore[attr-defined]

    def _register_secret(self, value: Any) -> None:
        """Register a resolved credential for capture + log redaction (P1-07)."""
        if not isinstance(value, str) or not value:
            return
        secrets = getattr(self, "_request_secrets", None)
        if secrets is None:
            secrets = self._request_secrets = []
        if value not in secrets:
            secrets.append(value)
        capture = current_capture()
        if capture is not None:
            capture.add_secret(value)

    def _flush_pending_capture_artifacts(self) -> None:
        """Write request artifacts only after route credentials are known."""
        capture = current_capture()
        if capture is None:
            self._pending_capture_incoming = None
            self._pending_capture_transformed = None
            return
        incoming = getattr(self, "_pending_capture_incoming", None)
        if incoming is not None:
            body, headers, raw = incoming
            capture.json("incoming", body)
            capture.json("incoming.headers", headers)
            capture.bytes("incoming.body", raw)
            self._pending_capture_incoming = None
        transformed = getattr(self, "_pending_capture_transformed", None)
        if transformed is not None:
            body, provider, model = transformed
            capture.json("transformed", body)
            capture.event("route.selected", provider=provider, model=model)
            self._pending_capture_transformed = None

    def _turn_deadline(self) -> float | None:
        started = getattr(self, "_request_started_at", None)
        timeout = getattr(self.config, "turn_timeout", None)
        if started is None or not timeout:
            return None
        return started + float(timeout)

    def _stream_watch(self, upstream: Any) -> _ClientCancelWatch:
        return _ClientCancelWatch(self, upstream, deadline=self._turn_deadline())

    def _acquire_inference_slot(self) -> bool:
        semaphore = getattr(getattr(self, "server", None), "inference_semaphore", None)
        if semaphore is None:
            return True
        if not semaphore.acquire(blocking=False):
            return False
        self._inference_slot_acquired = True
        return True

    def _track_upstream(self, resource: Any) -> None:
        """Let server shutdown close this request's upstream (P1-10)."""
        track = getattr(getattr(self, "server", None), "track", None)
        if callable(track):
            track(resource)
            tracked = getattr(self, "_tracked_resources", None)
            if tracked is None:
                tracked = self._tracked_resources = []
            tracked.append(resource)

    def _release_tracked(self) -> None:
        untrack = getattr(getattr(self, "server", None), "untrack", None)
        for resource in getattr(self, "_tracked_resources", None) or []:
            if callable(untrack):
                untrack(resource)
        self._tracked_resources = []
        if getattr(self, "_inference_slot_acquired", False):
            semaphore = getattr(getattr(self, "server", None), "inference_semaphore", None)
            if semaphore is not None:
                semaphore.release()
            self._inference_slot_acquired = False

    def _host_header_allowed(self, *, require_present: bool = False) -> bool:
        headers = getattr(self, "headers", None)
        host = headers.get("Host") if headers is not None else None
        if not host:
            return not require_present
        hostname = urllib.parse.urlsplit("//" + str(host).strip()).hostname
        if not hostname:
            return False
        hostname = hostname.lower().rstrip(".")
        if getattr(self.config, "allow_non_loopback", False):
            return True
        allowed = {"localhost", "127.0.0.1", "::1"}
        configured = str(getattr(self.config, "host", "") or "").lower().rstrip(".")
        if configured and configured not in {"0.0.0.0", "::", "*"}:
            allowed.add(configured)
        return hostname in allowed or _is_loopback_host(hostname)

    def _inbound_token_authorized(self) -> bool:
        if not getattr(self.config, "require_inbound_token", False):
            return True
        expected = getattr(self.config, "inbound_token", None)
        if not expected:
            return False
        candidates = []
        custom = self.headers.get("x-muse-gateway-token")
        if custom:
            candidates.append(str(custom).strip())
        authorization = self.headers.get("Authorization") or ""
        if authorization.lower().startswith("bearer "):
            candidates.append(authorization[7:].strip())
        return any(secrets.compare_digest(candidate, expected) for candidate in candidates if candidate)

    def _reject_untrusted_inbound(self, *, post: bool = False) -> bool:
        if not self._host_header_allowed(require_present=post):
            self.send_provider_error(ProviderError(
                "Request Host is not allowed for this loopback gateway.",
                status=403, code="bad_host",
            ))
            return True
        if post and self.headers.get("Origin"):
            self.send_provider_error(ProviderError(
                "Browser-origin inference requests are not allowed.",
                status=403, code="browser_origin_rejected",
            ))
            return True
        if post and not self._inbound_token_authorized():
            self.send_provider_error(ProviderError(
                "A valid gateway inbound token is required.",
                status=401, code="inbound_token_required",
            ))
            return True
        return False

    def _note_upstream_error_headers(self, headers: Any) -> str | None:
        """Keep a strict Retry-After and one bounded request ID (P1-08)."""
        if not headers:
            return None
        secrets = tuple(self._all_secrets())
        retry = muse_kiro._retry_after(headers, secrets)
        if muse_kiro._single_header(headers, "Retry-After") is not None:
            self._retry_after_observation = (
                "clamped" if retry and retry.get("clamped") else "accepted" if retry else "rejected"
            )
        if retry:
            self._retry_after = retry["value"]
        for header_name in ("request-id", "x-request-id", "anthropic-request-id"):
            candidate = muse_kiro._single_header(headers, header_name)
            if (
                candidate
                and len(candidate) <= 128
                and all(33 <= ord(c) <= 126 for c in candidate)
                and muse_kiro._secret_free(candidate, secrets)
            ):
                self._upstream_request_id = candidate
                return candidate
        return None

    def _all_secrets(self) -> list[str]:
        return [
            *self.config.secret_values(),
            *getattr(self, "_request_secrets", []),
            self.headers.get("Authorization", "") if getattr(self, "headers", None) else "",
        ]

    def _log_event(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "request_id": getattr(self, "_request_id", None),
            **getattr(self, "_diagnostic", {}),
            **fields,
        }
        capture = current_capture()
        if capture is not None:
            capture.event("gateway." + event, **{key: value for key, value in record.items() if key != "event"})
        known_secrets = self._all_secrets()
        rendered = _redact_bytes(
            json.dumps(record, ensure_ascii=True).encode("utf-8"), known_secrets
        ).decode("utf-8")
        self.server.diagnostics.info(rendered)  # type: ignore[attr-defined]

    def send_response(self, code: int, message: str | None = None) -> None:
        self._response_started = True
        self._response_status = code
        if code >= 400:
            self._stream_failed = True
        super().send_response(code, message)

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.config.verbose:
            # BaseHTTPRequestHandler includes raw request paths in its log.
            self._log_event("http_access", status=getattr(self, "_response_status", None))

    def send_json(self, status: int, value: Any, *, headers: dict[str, str] | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        for name, field in (headers or {}).items():
            self.send_header(name, field)
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def send_provider_error(self, error: ProviderError | ShimError, *, status: int | None = None) -> None:
        # Rejections before forwarding are failures too. In particular, a
        # transform error must not leave a successful-looking capture outcome.
        self._stream_failed = True
        error_status = status or getattr(error, "status", 400)
        if getattr(error, "headers", None) is not None:
            self._note_upstream_error_headers(getattr(error, "headers"))
        code = bounded_client_error_code(getattr(error, "code", "shim_error"), "shim_error")
        kiro_fields = muse_kiro.error_fields(error) if isinstance(error, ProviderError) else {}
        if "code" in kiro_fields:
            kiro_fields["code"] = bounded_client_error_code(kiro_fields["code"], code)
        extra = dict(getattr(self, "_upstream_diagnostic", {}))
        # Upstream diagnostics travel as one sub-dict so no upstream key can
        # collide with our own log fields (the old `code` splat TypeError
        # killed the error response entirely).
        self._log_event("provider_error", status=error_status, code=code,
                        upstream=extra,
                        retry_after_observation=getattr(self, "_retry_after_observation", None),
                        **({"metadata": kiro_fields["metadata"]} if kiro_fields else {}))
        if getattr(self, "_response_started", False):
            # A partial response cannot be replaced by a second HTTP response.
            self.close_connection = True
            return
        message = bounded_client_error_text(_redact_bytes(
            str(error).encode("utf-8"),
            self._all_secrets(),
        ).decode("utf-8"))
        try:
            retry = kiro_fields.get("metadata", {}).get("retry_after")
            retry_value = retry["value"] if retry else getattr(self, "_retry_after", None)
            error_body = {"message": message, "type": code,
                          "request_id": getattr(self, "_request_id", None), **kiro_fields}
            upstream_request_id = getattr(self, "_upstream_request_id", None)
            if upstream_request_id:
                error_body["upstream_request_id"] = upstream_request_id
            self.send_json(
                int(error_status),
                {"error": error_body},
                headers={"Retry-After": retry_value} if retry_value else None,
            )
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Muse went away (ESC/cancel) while we answered. Stay quiet.
            return

    def do_GET(self) -> None:
        """Answer, never drop: a failing GET is a JSON error (P2-13)."""
        self._request_id = secrets.token_hex(6)
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.close_connection = True
        except (ProviderError, ShimError) as error:
            self.send_provider_error(error)
        except Exception as error:  # noqa: BLE001 - last-resort answer, not a crash
            self._log_event("internal_error", kind=type(error).__name__)
            self.send_provider_error(ProviderError(
                "The gateway could not build this response; see the gateway log.",
                status=500, code="internal_error"))

    def _handle_get(self) -> None:
        if self._reject_untrusted_inbound(post=False):
            return
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "muse-gateway",
                    "version": _gateway_version(),
                    "pid": os.getpid(),
                    "fingerprint": getattr(self.server, "source_fingerprint", None),
                    "provider": self.config.provider,
                    "claude_transport": self.config.claude_transport,
                    "image_files": self.config.allow_image_files,
                    "supported_routes": {
                        "version": 1,
                        "providers": ["generic", "codex", "claude", "meta", "devin", "kiro", "local"],
                    },
                    **(self.server.capture_session.health() if self.server.capture_session is not None else {}),
                },
            )
            return

        if path in {"/muse-code/models", "/v1/muse-code/models"}:
            cache_control = self.headers.get("Cache-Control", "")
            if "no-cache" in cache_control.lower():
                # Model switches re-read the menu: an explicit no-cache fetch
                # recalls discovery (rate-limited to once per 10 s; the last
                # snapshot keeps serving meanwhile, never dropped).
                now = time.time()
                last = self.config._catalog_last_nocache.get("all", 0.0)
                if now - last >= 10.0:
                    self.config._catalog_last_nocache["all"] = now
                    with self.config._catalog_lock:
                        for key in list(self.config._model_catalog_cache):
                            self.config._catalog_generation[key] = (
                                self.config._catalog_generation.get(key, 0) + 1
                            )
                        self.config._model_catalog_cache.clear()
            inbound_auth = self.headers.get("Authorization")
            if self.config.provider == "auto":
                models: list[dict[str, Any]] = []
                for route in ("codex", "claude", "meta", "devin", "kiro", "local"):
                    models.extend(
                        dict(item)
                        for item in provider_model_catalog(
                            self.config, route, authorization=inbound_auth
                        )
                    )
                self.send_json(200, muse_catalog_response(models))
                return
            entries = []
            if self.config.provider in {"kiro", "codex", "claude", "meta", "devin", "local"}:
                entries = provider_model_catalog(self.config, self.config.provider, authorization=inbound_auth)
            self.send_json(200, muse_catalog_response(entries))
            return

        self.send_json(404, {"error": {"message": "not found", "type": "shim_error"}})

    def _read_request_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ShimError("Content-Length must be an integer") from exc
        if content_length <= 0:
            raise ShimError("request body is empty")
        if content_length > self.config.max_body_bytes:
            raise ShimError(
                f"request body exceeds the configured limit of {self.config.max_body_bytes} bytes"
            )
        # Inbound body deadline (P1-09/M4): one absolute monotonic deadline,
        # not a per-read timeout that a dribbling client can keep resetting.
        deadline = time.monotonic() + max(self.body_timeout_floor, float(self.config.timeout))
        chunks: list[bytes] = []
        remaining = content_length
        try:
            while remaining:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise TimeoutError("request body deadline exceeded")
                self.connection.settimeout(budget)
                reader = getattr(self.rfile, "read1", self.rfile.read)
                chunk = reader(min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        except (TimeoutError, socket.timeout) as exc:
            self.close_connection = True
            raise ProviderError("request body was not received in time.",
                                status=408, code="request_timeout") from exc
        finally:
            self.connection.settimeout(self.timeout)
        if len(raw) < content_length:
            self.close_connection = True
            raise ShimError("request body ended before Content-Length")
        try:
            body = json.loads(raw)
        except RecursionError as exc:
            raise ShimError("request body is too deeply nested") from exc
        except json.JSONDecodeError as exc:
            raise ShimError("request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise ShimError("request body must be a JSON object")
        if current_capture() is not None:
            # Defer disk writes until the selected route credential has been
            # registered; the incoming body may echo that credential.
            self._pending_capture_incoming = (body, dict(self.headers), raw)
        return body

    def do_POST(self) -> None:
        self._request_id = secrets.token_hex(6)
        self._request_started_at = time.monotonic()
        self._response_started = False
        self._stream_failed = False
        self._request_secrets: list[str] = []
        self._retry_after: str | None = None
        self._retry_after_observation: str | None = None
        self._upstream_request_id: str | None = None
        self._diagnostic: dict[str, Any] = {}
        self._upstream_diagnostic: dict[str, Any] = {}
        session = getattr(self.server, "capture_session", None)
        if session is not None:
            try:
                self._capture_request = session.request(
                    self._request_id,
                    [*self.config.secret_values(), self.headers.get("Authorization", "")],
                )
            except OSError as capture_error:
                self._log_event("capture_unavailable", kind=type(capture_error).__name__)
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/responses", "/v1/responses"}:
            self.send_json(404, {"error": {"message": "not found", "type": "shim_error"}})
            return
        if self._reject_untrusted_inbound(post=True):
            return
        if not self._acquire_inference_slot():
            self._retry_after = "1"
            self.send_provider_error(ProviderError(
                "Gateway is at its concurrent inference limit; retry shortly.",
                status=503, code="gateway_busy",
            ))
            return

        try:
            body = self._read_request_body()
            self._diagnostic = _request_diagnostic(body)
            self._log_event("request_received")
            # Gateway fork: resolve before transforming so the meta/ route can
            # keep the native wire format. Resolving on the pinned model first
            # preserves the old order's observable behavior for other routes.
            provider, upstream_model = self.config.resolve_model(
                self.config.model or body.get("model")
            )
            transformed = transform_request(
                body, self.config, native=(provider == "meta")
            )
            transformed["model"] = upstream_model
            if provider != "devin":
                # Images become checked bytes before the request is forwarded.
                # The catalog is consulted only when the request carries images.
                image_route = (
                    "claude-cli"
                    if provider == "claude" and self.config.claude_transport == "cli"
                    else provider
                )
                supports_images = (
                    _model_supports_images(
                        self.config, provider, upstream_model,
                        authorization=self.headers.get("Authorization"),
                    )
                    if self._diagnostic.get("image_parts")
                    else None
                )
                transformed, _image_stats = normalize_request_images(
                    transformed, route=image_route, supports_images=supports_images,
                    allow_files=self.config.allow_image_files, notify=self._log_event,
                )
            limit = self.config.model_output_limit(provider, upstream_model)
            requested = transformed.get("max_output_tokens")
            # Devin resolves the credential-bound model later in forward_devin;
            # its request builder applies that model's output bound.
            if provider not in {"codex", "devin"} and limit and positive_int(requested):
                transformed["max_output_tokens"] = min(int(requested), limit)
            self._diagnostic["provider"] = provider
            if current_capture() is not None:
                self._pending_capture_transformed = (transformed, provider, upstream_model)
        except (ProviderError, ShimError, ValueError) as error:
            local_error = (
                error
                if isinstance(error, (ProviderError, ShimError))
                else ShimError(str(error))
            )
            self.send_provider_error(local_error)
            return

        try:
            if provider == "codex":
                self.forward_codex(transformed)
            elif provider == "claude":
                if self.config.claude_transport == "cli":
                    self.forward_claude_cli(transformed)
                else:
                    self.forward_claude(transformed)
            elif provider == "meta":
                self.forward_meta(transformed)
            elif provider == "local":
                self.forward_local(transformed)
            elif provider == "devin":
                self.forward_devin(transformed)
            elif provider == "kiro":
                self.forward_kiro(transformed)
            else:
                self.forward_generic(transformed)
        except RemoteDisconnected:
            # Upstream closed before/while answering. This is ALSO a
            # ConnectionResetError — it must be caught before the client-disconnect
            # trio or it is misdiagnosed as the client going away (de-glm C1).
            self.send_provider_error(ProviderError(
                "Upstream closed the connection without a complete response.",
                status=502, code="upstream_reset",
            ))
        except HTTPException:
            # IncompleteRead, BadStatusLine, ... — response-phase upstream
            # failure, never a client disconnect.
            self.send_provider_error(ProviderError(
                "Upstream connection failed while responding.",
                status=502, code="upstream_reset"))
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            if getattr(self, "_response_started", False):
                self._log_event("client_disconnected", disconnect_origin="client")
                self.close_connection = True
            else:  # no client byte written: the reset came from upstream
                self._log_event("upstream_disconnected", disconnect_origin="upstream")
                self.send_provider_error(ProviderError(
                    "Upstream connection reset.", status=502, code="upstream_reset"))
        except (ProviderError, ShimError) as error:
            self.send_provider_error(error)
        except urllib.error.HTTPError as error:
            try:
                detail = _upstream_diagnostic(_capture_http_error(error))
            finally:
                error.close()
            candidate = self._note_upstream_error_headers(error.headers)
            if candidate:
                detail = {**detail, "upstream_request_id": candidate}
            self._log_event("upstream_error", status=error.code, upstream=detail)
            self.send_provider_error(ProviderError(
                f"Upstream returned HTTP {error.code}: {json.dumps(detail)}",
                status=error.code, code="upstream_http_error",
            ))
        except (urllib.error.URLError, TimeoutError, OSError):
            self.send_provider_error(ProviderError(
                "Upstream connection failed; check the gateway log and upstream availability.",
                status=502, code="upstream_unreachable",
            ))
        except Exception as error:
            # Last resort: a dropped connection surfaces client-side as
            # "malformed response data" after 10 blind retries. Always
            # answer instead, and leave the traceback in the diagnostics
            # log (frames only, no prompt or response content).
            self._log_event(
                "internal_error", kind=type(error).__name__,
                traceback=traceback.format_exc(limit=8)[-2000:],
            )
            if getattr(self, "_response_started", False):
                self.close_connection = True
                return
            self.send_provider_error(ProviderError(
                "gateway internal error; see the gateway log",
                status=500, code="internal_error",
            ))
        else:
            self._log_event("request_finished", status=getattr(self, "_response_status", None),
                            stream_failed=self._stream_failed)

    def _send_upstream_headers(self, upstream: Any, *, content_type: str, stream: bool) -> None:
        capture = current_capture()
        if capture is not None:
            capture.json("upstream.headers", {"status": int(getattr(upstream, "status", 200)),
                                               "headers": dict(upstream.headers)})
        self.send_response(int(getattr(upstream, "status", 200)))
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache" if stream else "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def forward_generic(self, body: dict[str, Any]) -> None:
        authorization = _select_bearer(
            self.config.api_key,
            self.headers.get("Authorization"),
            missing=(
                "no upstream API key; pass --api-key in generic mode or supply client authorization"
            ),
        )
        url = self.config.upstream_url("responses")
        if not self.config.api_key:
            _warn_credential_forwarding_once("generic", url, custom=True)
        self._post_responses(url, body, authorization)

    def forward_meta(self, body: dict[str, Any]) -> None:
        """Forward to Meta with the native wire format (see transform_request)."""
        authorization = _select_bearer(
            self.config.meta_api_key,
            self.headers.get("Authorization"),
            missing=(
                "no Meta API key; set META_API_KEY for API billing, or launch Muse "
                "in two-terminal mode with a real META_API_KEY "
                "(`run` mode always sends a placeholder bearer)"
            ),
        )
        meta_url = self.config.provider_url("meta")
        if not self.config.meta_api_key and self.headers.get("Authorization"):
            _warn_credential_forwarding_once(
                "meta",
                meta_url,
                custom=meta_url.rstrip("/") != DEFAULT_META_UPSTREAM + "/responses",
            )
        self._post_responses(meta_url, body, authorization)

    def forward_local(self, body: dict[str, Any]) -> None:
        """Forward only the local route's optional key, never Muse's bearer."""
        if not self.config.local_upstream:
            raise ProviderError("local route disabled; set MUSE_GATEWAY_LOCAL_UPSTREAM.", code="local_not_configured")
        authorization = f"Bearer {self.config.local_api_key}" if self.config.local_api_key else None
        self._post_responses(self.config.provider_url("local"), body, authorization)

    def forward_kiro(self, body: dict[str, Any]) -> None:
        """Translate through the external Messages service, never native OAuth."""
        self._stream_failed = True
        key = muse_kiro.validate_proxy_key(self.config.kiro_api_key)
        model_name = str(body.get("model") or "")
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]
        budget = self.config.model_output_limit("kiro", model_name.strip())
        payload, tool_names = muse_kiro.prepare_request(
            body, max_output_tokens=budget,
        )
        if body.get("previous_response_id") and "previous_response_id" not in payload:
            self._log_event("previous_response_id_dropped")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        endpoint = self.config.provider_url("kiro")
        self._register_secret(key)
        self._flush_pending_capture_artifacts()
        capture = current_capture()
        if capture is not None:
            capture.json("outgoing", payload)
            # Body/open are recorded once inside muse_kiro.open_request as
            # kiro.outgoing.body / kiro.upstream.open (no duplicate writes).
        stream = body.get("stream") is True
        upstream_status = None
        upstream_headers = None
        secrets = tuple(self._all_secrets())
        try:
            with muse_kiro.open_request(
                endpoint, key, body=encoded, timeout=self.config.timeout,
            ) as upstream:
                _set_stream_idle_timeout(upstream, self.config)
                self._track_upstream(upstream)
                upstream_status = getattr(upstream, "status", None)
                upstream_headers = upstream.headers
                content_type = upstream.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                expected = "text/event-stream" if stream else "application/json"
                if capture is not None:
                    capture.json("upstream.headers", {
                        "status": int(getattr(upstream, "status", 200)),
                        "headers": dict(upstream.headers),
                    })
                if content_type != expected:
                    _capture_rejected_kiro_body(upstream)
                    raise ProviderError(
                        f"Kiro Gateway returned an unexpected media type; expected {expected}.",
                        status=502, code="kiro_invalid_response",
                    )
                if not stream:
                    try:
                        raw = upstream.read(KIRO_JSON_RESPONSE_LIMIT + 1)
                    except IncompleteRead as error:
                        if capture is not None:
                            capture.bytes("upstream.body", error.partial[:KIRO_JSON_RESPONSE_LIMIT + 1])
                            capture.event("upstream.body.incomplete", complete=False)
                            capture.errors.append("kiro_response_body_incomplete")
                        raise ProviderError(
                            "Kiro Gateway JSON response was interrupted before its HTTP body ended.",
                            status=502, code="kiro_invalid_response",
                        ) from error
                    if capture is not None:
                        capture.bytes("upstream.body", raw)
                    if len(raw) > KIRO_JSON_RESPONSE_LIMIT:
                        if capture is not None:
                            capture.event("upstream.body.limit", bytes=len(raw), limit=KIRO_JSON_RESPONSE_LIMIT, complete=False)
                            capture.errors.append("kiro_response_body_limit")
                        raise ProviderError(
                            "Kiro Gateway JSON response exceeded the safety size limit.",
                            status=502, code="kiro_invalid_response",
                        )
                    if getattr(upstream, "length", None) not in {None, 0}:
                        if capture is not None:
                            capture.event("upstream.body.incomplete", complete=False)
                            capture.errors.append("kiro_response_body_incomplete")
                        raise ProviderError(
                            "Kiro Gateway JSON response ended before its declared HTTP body length.",
                            status=502, code="kiro_invalid_response",
                        )
                    if capture is not None:
                        capture.event("upstream.eof")
                    try:
                        result = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
                    except (UnicodeError, ValueError, RecursionError) as error:
                        raise ProviderError(
                            "Kiro Gateway returned invalid JSON.",
                            status=502, code="kiro_invalid_response",
                        ) from error
                    if isinstance(result, dict) and result.get("type") == "error":
                        raise muse_kiro.upstream_error(
                            source="json", upstream_status=upstream_status, headers=upstream_headers,
                            payload=muse_kiro.load_error_payload(raw), secrets=secrets,
                        )
                    converted = muse_kiro.convert_json(
                        result, model=str(payload["model"]), tool_names=tool_names,
                        headers=upstream_headers, upstream_status=upstream_status, secrets=secrets,
                    )
                    self.send_json(200, converted)
                    self._stream_failed = converted.get("status") != "completed"
                    return
                pending = muse_kiro.stream_events(
                    upstream, model=str(payload["model"]), tool_names=tool_names,
                    secrets=secrets,
                )
                terminal = False
                try:
                    with self._stream_watch(upstream) as watch:
                        try:
                            first = next(pending)
                        except StopIteration as error:
                            raise ProviderError(
                                "Kiro Gateway returned an empty stream.",
                                status=502, code="kiro_invalid_response",
                            ) from error
                        if first[0] in {"response.failed", "response.incomplete", "error"}:
                            raise ProviderError(
                                "Kiro Gateway could not complete the response.",
                                status=502, code="kiro_failed_response",
                            )
                        self._send_upstream_headers(
                            upstream, content_type="text/event-stream", stream=True,
                        )
                        event_type, value = first
                        while True:
                            self.wfile.write(response_sse_line(event_type, value))
                            self.wfile.flush()
                            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                                terminal = True
                                self._stream_failed = event_type != "response.completed"
                                if event_type == "response.failed":
                                    failure = (value.get("response") or {}).get("error") or {}
                                    self._log_event("kiro_upstream_error", metadata=failure.get("metadata"))  # P3-05 n1
                                break
                            if watch.fired:
                                self._log_event("client_disconnected", disconnect_origin="client")
                                self.close_connection = True
                                return
                            try:
                                event_type, value = next(pending)
                            except StopIteration:
                                break
                        if not terminal:
                            raise ProviderError(
                                "Kiro Gateway stream ended without a terminal response.",
                                status=502, code="turn_timeout" if watch.deadline_fired else "kiro_stream_truncated",
                            )
                finally:
                    pending.close()
        except urllib.error.HTTPError as error:
            self._stream_failed = True
            try:
                prefix = _read_kiro_error_prefix(error.fp, label="upstream.error")
                failure = muse_kiro.upstream_error(
                    source="http", upstream_status=error.code, headers=error.headers,
                    payload=muse_kiro.load_error_payload(prefix), secrets=secrets,
                )
                muse_kiro._capture("event", "kiro.upstream.failed", **muse_kiro.error_fields(failure))
            finally:
                error.close()
            raise failure from error
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as error:
            self._stream_failed = True
            raise ProviderError(
                "Kiro Gateway connection failed or its response was interrupted.",
                status=502, code="kiro_upstream_unreachable",
                metadata=muse_kiro.error_metadata(source="transport", upstream_status=upstream_status,
                                                   headers=upstream_headers, secrets=secrets),
            ) from error
        except ProviderError as error:
            if error.metadata is None:
                error.metadata = muse_kiro.error_metadata(source="protocol", upstream_status=upstream_status,
                                                         headers=upstream_headers, secrets=secrets)
            raise
        except BaseException:
            self._stream_failed = True
            raise
        finally:
            self._log_event("kiro_response_finished", stream=stream, stream_failed=self._stream_failed)

    def forward_devin(self, body: dict[str, Any]) -> None:
        """Translate one Responses request onto Devin's Connect RPCs."""
        if "previous_response_id" in body:
            had_id = bool(body.get("previous_response_id"))
            body = drop_or_reject_previous_response_id(
                body, code="devin_previous_response_unsupported"
            )
            if had_id:
                notify = getattr(self, "_log_event", None)
                if callable(notify):
                    notify("previous_response_id_dropped")
        config = self.config
        try:
            token, _identity = muse_devin.resolve_access_token(api_key=config.devin_api_key)
            self._register_secret(token)
            self._flush_pending_capture_artifacts()
        except muse_devin.DevinCredentialError as error:
            raise ProviderError(str(error), code="devin_credential_unavailable") from error
        try:
            _ = _devin_catalog_entries(
                config, access_token=token, note_error_headers=self._note_upstream_error_headers
            )
        except muse_devin.DevinHttpError as error:
            self._note_upstream_error_headers(getattr(error, "headers", None))
            raise ProviderError(
                "Devin upstream error.", status=502, code="devin_upstream_error"
            ) from error
        except muse_devin.DevinWireError as error:
            raise ProviderError(
                "Devin upstream error.", status=502, code="devin_wire_error"
            ) from error
        logical_id = str(body.get("model") or "").strip()
        roster, _dropped = config._devin_rosters[_devin_cache_key(config, token)]
        model = roster.get(logical_id)
        if model is None:
            raise ProviderError(
                f"unknown Devin model {logical_id!r}; see --models for the roster.",
                code="devin_unknown_model",
            )
        body, _image_stats = normalize_request_images(
            body, route="devin", supports_images=model.supports_images,
            allow_files=config.allow_image_files, notify=getattr(self, "_log_event", None),
        )
        reasoning = body.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        if not isinstance(effort, str) or not effort:
            effort = None
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        stream = body.get("stream") is True
        conversation_id = self.headers.get("x-tbh-session-id") or body.get("conversation_id") or body.get("session_id")
        self._log_event(
            "devin_turn",
            model=f"devin/{model.logical_id}",
            effort=effort or model.default_effort or "default",
            router=bool(model.is_router),
        )
        watch_box: dict[str, _ClientCancelWatch] = {}

        def track_devin(resource: Any) -> None:
            self._track_upstream(resource)
            watch = self._stream_watch(resource)
            watch.__enter__()
            watch_box["watch"] = watch

        try:
            if not stream:
                try:
                    final = muse_devin.run_turn_final(
                        body, tools, model, token,
                        base_url=config.devin_upstream, effort=effort, timeout=config.timeout,
                        conversation_id=conversation_id,
                        idle_timeout=config.stream_idle_timeout,
                        track=track_devin,
                        register_secret=self._register_secret,
                    )
                finally:
                    if "watch" in watch_box:
                        watch_box.pop("watch").__exit__()
                self._stream_failed = final.get("status") != "completed"
                self.send_json(200, final)
                return
            pending = muse_devin.run_turn_events(
                body, tools, model, token,
                base_url=config.devin_upstream, effort=effort, timeout=config.timeout,
                conversation_id=conversation_id,
                idle_timeout=config.stream_idle_timeout,
                track=track_devin,
                register_secret=self._register_secret,
            )
            try:
                try:
                    first_event = next(pending)
                except StopIteration:
                    raise muse_devin.DevinChatError(
                        "Devin turn ended without a terminal event"
                    )
                watch = watch_box.get("watch")
                if watch is not None and watch.fired:
                    self._log_event("client_disconnected", disconnect_origin="client")
                    self.close_connection = True
                    return
                self._send_upstream_headers(
                    types.SimpleNamespace(status=200, headers={}),
                    content_type="text/event-stream",
                    stream=True,
                )
                event_type, payload = first_event
                terminal = False
                last_sequence = -1
                response_id = "resp_" + secrets.token_hex(8)
                while True:
                    if event_type in {"response.failed", "response.incomplete"}:
                        self._stream_failed = True
                    if isinstance(payload, dict):
                        sequence = payload.get("sequence_number")
                        if isinstance(sequence, int) and not isinstance(sequence, bool):
                            last_sequence = max(last_sequence, sequence)
                        response_obj = payload.get("response")
                        if isinstance(response_obj, dict) and isinstance(response_obj.get("id"), str):
                            response_id = response_obj["id"]
                    self.wfile.write(response_sse_line(event_type, payload))
                    self.wfile.flush()
                    if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                        terminal = True
                        break
                    if watch is not None and watch.fired:
                        self._log_event("client_disconnected", disconnect_origin="client")
                        self.close_connection = True
                        return
                    try:
                        event_type, payload = next(pending)
                    except StopIteration:
                        break
                if not terminal:
                    self._stream_failed = True
                    deadline_hit = bool(watch is not None and watch.deadline_fired)
                    code = "turn_timeout" if deadline_hit else "upstream_truncated"
                    self._log_event(code)
                    self.wfile.write(response_sse_line("response.failed", {
                        "type": "response.failed", "sequence_number": last_sequence + 1,
                        "response": {"id": response_id, "object": "response", "status": "failed",
                                     "output": [], "error": {"type": code, "code": code,
                                                             "message": "Turn exceeded the configured deadline." if deadline_hit else "Devin stream ended without a terminal event."}},
                    }))
                    self.wfile.flush()
            finally:
                if "watch" in watch_box:
                    watch_box.pop("watch").__exit__()
                pending.close()
        except muse_devin.DevinAuthError as error:
            status = getattr(error.__cause__, "status", None)
            self._note_upstream_error_headers(getattr(error.__cause__, "headers", None))
            if status == 429:
                raise ProviderError("Devin auth endpoint is rate limited.", status=429,
                                    code="devin_upstream_error") from error
            raise AuthError(
                "Devin authentication failed"
                + (f" (HTTP {status})." if status else "."),
                code="devin_upstream_unauthorized",
            ) from error
        except muse_devin.DevinChatError as error:
            status = (
                400
                if error.code == "devin_context_overflow"
                or error.code in muse_devin.REQUEST_POLICY_CODES
                else 502
            )
            raise ProviderError(
                str(error), status=status, code=error.code or "devin_chat_error"
            ) from error
        except muse_devin.DevinHttpError as error:
            self._note_upstream_error_headers(getattr(error, "headers", None))
            if error.status == 401:
                muse_devin.mark_auth_failure(token)
                raise AuthError("Devin upstream rejected the credentials.",
                                code="devin_upstream_unauthorized") from error
            raise ProviderError("Devin upstream error.",
                                status=429 if error.status == 429 else 502,
                                code="devin_upstream_error") from error
        except muse_devin.DevinWireError as error:
            raise ProviderError(
                "Devin upstream error.", status=502, code="devin_wire_error"
            ) from error

    def _post_responses(
        self, url: str, body: dict[str, Any], authorization: str | None
    ) -> None:
        """POST one Responses body and stream the upstream reply back to Muse."""
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                **({"Authorization": authorization} if authorization else {}),
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "User-Agent": self.server_version,
            },
        )
        if authorization:
            self._register_secret(authorization)
        self._flush_pending_capture_artifacts()
        capture = current_capture()
        if capture is not None:
            capture.bytes("outgoing.body", request.data)
            capture.json("outgoing", body)
            capture.json("outgoing.headers", dict(request.header_items()))
            capture.event("upstream.open", url=url)
        with _credentialed_urlopen(
            request, timeout=self.config.timeout, config=self.config
        ) as upstream:
            _set_stream_idle_timeout(upstream, self.config)
            self._track_upstream(upstream)
            content_type = upstream.headers.get("Content-Type", "application/octet-stream")
            stream = "text/event-stream" in content_type.lower() or body.get("stream") is True
            if not stream:
                # Read before headers: truncation is a clean 502, never a
                # short body under committed headers (P0-08/P1-13).
                if capture is not None:
                    capture.json("upstream.headers", {"status": int(getattr(upstream, "status", 200)),
                                                      "headers": dict(upstream.headers)})
                data = _read_bounded_upstream_body(upstream)
                self.send_response(int(getattr(upstream, "status", 200)))
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                if data:
                    self.wfile.write(data)
                    self.wfile.flush()
                return
            self._send_upstream_headers(upstream, content_type=content_type, stream=stream)
            _write_upstream_body(
                self,
                upstream,
                stream=stream,
                content_type=content_type,
            )

    def forward_codex(self, body: dict[str, Any]) -> None:
        state = resolve_codex_credentials(self.config)
        self._register_secret(state.access_token)
        self._register_secret(state.refresh_token)
        self._flush_pending_capture_artifacts()
        if not body.get("model"):
            raise ProviderError(
                "Codex request is missing model; send model or pin one with --model in a provider-specific server mode.",
                code="model_missing",
            )
        endpoint = self.config.provider_url("codex")
        validate_codex_endpoint(
            endpoint,
            allow_host_override=self.config.allow_oauth_host_override,
        )
        had_chained_id = bool(body.get("previous_response_id"))
        codex_body = prepare_codex_request(body)
        inbound_stream = body.get("stream") is True
        # The Codex backend requires streaming ("Stream must be set to
        # true"). Non-streaming turns are streamed upstream and collapsed
        # back to a single response object below.
        codex_body["stream"] = True
        if had_chained_id and "previous_response_id" not in codex_body:
            # The unresolvable chain reference was dropped in favor of the
            # complete inline turn. Logged so the session keeps working but
            # the rewrite stays visible in diagnostics.
            notify = getattr(self, "_log_event", None)
            if callable(notify):
                notify("previous_response_id_dropped")
        reasoning = codex_body.get("reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        if isinstance(effort, str):
            # Reconcile with model-specific evidence so a stale Muse variant
            # (e.g. "minimal" for gpt-6-astra) never reaches Codex as an
            # opaque 400. Unknown models pass through untouched.
            supported = _codex_model_efforts(self.config, codex_body.get("model"))
            corrected, was_corrected = correct_codex_effort(effort, supported)
            if was_corrected:
                codex_body["reasoning"] = dict(reasoning, effort=corrected)
                notify = getattr(self, "_log_event", None)
                if callable(notify):
                    notify(
                        "effort_corrected",
                        from_effort=effort,
                        to_effort=corrected,
                        supported_efforts=supported,
                    )
        encoded_body = json.dumps(
            codex_body,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        capture = current_capture()
        if capture is not None:
            capture.bytes("outgoing.body", encoded_body)
            capture.event("upstream.request_encoded", bytes=len(encoded_body))
        auth_refreshed = False
        reasoning_recovered = False
        reserve_recovered = False
        # Initial attempt plus at most one retry each for auth refresh,
        # reasoning demotion, and reserve fallback.
        for attempt in range(4):
            request = urllib.request.Request(
                endpoint,
                data=encoded_body,
                method="POST",
                headers=codex_headers(state.access_token),
            )
            if capture is not None:
                capture.add_secret(state.access_token)
                capture.json("outgoing.headers", dict(request.header_items()))
                capture.event("upstream.open", attempt=attempt, url=endpoint)
            try:
                with _credentialed_urlopen(
                    request, timeout=self.config.timeout, config=self.config
                ) as upstream:
                    _set_stream_idle_timeout(upstream, self.config)
                    self._track_upstream(upstream)
                    content_type = upstream.headers.get("Content-Type", "application/json")
                    if not inbound_stream:
                        self.send_json(200, _buffered_codex_response(upstream, content_type))
                        return
                    stream = "text/event-stream" in content_type.lower() or codex_body.get("stream") is True
                    self._send_upstream_headers(upstream, content_type=content_type, stream=stream)
                    _write_upstream_body(
                        self,
                        upstream,
                        stream=stream,
                        content_type=content_type,
                    )
                    return
            except urllib.error.HTTPError as error:
                if error.code in {401, 403} and not auth_refreshed:
                    error.close()
                    try:
                        state = resolve_codex_credentials(self.config, force_refresh=True)
                    except ProviderError as refresh_error:
                        raise AuthError(
                            "Codex upstream rejected the OAuth credentials and refresh failed. "
                            "Run `codex login` and try again.",
                            code="codex_upstream_unauthorized",
                        ) from refresh_error
                    self._register_secret(state.access_token)
                    self._register_secret(state.refresh_token)
                    auth_refreshed = True
                    continue
                if error.code in {401, 403}:
                    error.close()
                    raise AuthError(
                        "Codex upstream rejected the OAuth credentials. Run `codex login` and try again.",
                        code="codex_upstream_unauthorized",
                    ) from error
                try:
                    raw_detail = _capture_http_error(error)
                except (OSError, HTTPException):
                    raw_detail = b""
                finally:
                    error.close()
                detail = _upstream_diagnostic(raw_detail)
                self._note_upstream_error_headers(error.headers)
                self._upstream_diagnostic = detail
                if error.code == 400 and detail.get("code") == "invalid_encrypted_content" and not reasoning_recovered:
                    recovered, count = demote_codex_encrypted_reasoning(codex_body)
                    if count:
                        codex_body = recovered
                        encoded_body = json.dumps(codex_body, ensure_ascii=False, separators=(",", ":")).encode()
                        reasoning_recovered = True
                        self._log_event("codex_reasoning_demoted", count=count,
                                        reason="upstream_rejected_encrypted_content")
                        if capture is not None:
                            capture.bytes("outgoing.recovered.body", encoded_body)
                        continue
                if (
                    not reserve_recovered
                    and codex_body.get("model") != CODEX_RESERVE_MODEL
                    and (
                        error.code == 429
                        or detail.get("code") in {"insufficient_quota", "quota_exceeded"}
                    )
                ):
                    # Silent quota fallback to the Luna-class reserve alias.
                    # The menu hides it; the request still visibly logs it.
                    from_model = codex_body.get("model")
                    codex_body = dict(codex_body, model=CODEX_RESERVE_MODEL)
                    encoded_body = json.dumps(codex_body, ensure_ascii=False, separators=(",", ":")).encode()
                    reserve_recovered = True
                    self._log_event("codex_reserve_fallback", from_model=from_model,
                                    reason="upstream_quota_exhausted")
                    if capture is not None:
                        capture.bytes("outgoing.reserve.body", encoded_body)
                        capture.event("upstream.reserve_fallback", from_model=from_model)
                    continue
                capture_note = ""
                if capture is not None:
                    capture_note = f"; private capture: {capture.directory}"
                else:
                    capture_note = (
                        "; enable --capture-dir on a new "
                        "gateway to retain the upstream error message"
                    )
                raise ProviderError(
                    f"Codex upstream returned HTTP {error.code}. "
                    f"detail={json.dumps(detail)}{capture_note}",
                    status=error.code,
                    code="codex_upstream_error",
                ) from error

    def forward_claude_cli(self, body: dict[str, Any]) -> None:
        for name in _CLAUDE_PASSTHROUGH_SETS[self.config.claude_cli_passthrough]:
            self._register_secret(os.environ.get(name, ""))
        self._flush_pending_capture_artifacts()
        payload, tool_names = run_claude_cli(
            body, self.config, track=self._track_upstream
        )
        if body.get("previous_response_id"):
            self._log_event("previous_response_id_dropped")
        model = str(payload.get("model") or body.get("model") or "")
        if body.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for chunk in convert_anthropic_stream(
                _anthropic_message_sse(payload),
                model=model,
                tool_names=tool_names,
                coalesce_text=True,
                native_usage=True,
            ):
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        self.send_json(
            200,
            anthropic_json_to_responses(
                payload,
                model=model,
                tool_names=tool_names,
                coalesce_text=True,
                native_usage=True,
            ),
        )

    def forward_claude(self, body: dict[str, Any]) -> None:
        state = resolve_claude_credentials(self.config)
        self._register_secret(state.access_token)
        self._register_secret(state.refresh_token)
        self._flush_pending_capture_artifacts()
        payload, tool_names = responses_to_anthropic(
            body,
            oauth=state.auth_mode != "api_key",
            model_override=(
                self.config.model if self.config.provider == "claude" else None
            ),
            max_output_tokens=self.config.model_output_limit("claude", str(body.get("model") or "")),
        )
        if body.get("previous_response_id"):
            self._log_event("previous_response_id_dropped")
        endpoint = self.config.provider_url("claude")
        validate_claude_endpoint(
            endpoint,
            allow_host_override=self.config.allow_oauth_host_override,
        )
        encoded_body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        capture = current_capture()
        if capture is not None:
            capture.bytes("outgoing.body", encoded_body)
            capture.json("outgoing", payload)
        for attempt in range(2):
            request = urllib.request.Request(
                endpoint,
                data=encoded_body,
                method="POST",
                headers=claude_headers(state, stream=bool(body.get("stream"))),
            )
            if capture is not None:
                capture.json("outgoing.headers", dict(request.header_items()))
                capture.event("upstream.open", attempt=attempt, url=endpoint)
            try:
                with _credentialed_urlopen(
                    request, timeout=self.config.timeout, config=self.config
                ) as upstream:
                    _set_stream_idle_timeout(upstream, self.config)
                    self._track_upstream(upstream)
                    content_type = upstream.headers.get("Content-Type", "application/json")
                    stream = "text/event-stream" in content_type.lower() or body.get("stream") is True
                    if stream:
                        # Prefetch the first converted event before headers:
                        # a first-read failure can still be answered 502;
                        # later failures stay 200 + response.failed (P0-08).
                        converted = convert_anthropic_stream(
                            _CaptureReadlineStream(upstream) if capture is not None else upstream,
                            model=str(payload.get("model") or ""),
                            tool_names=tool_names,
                            coalesce_text=True,
                            native_usage=True,
                        )
                        try:
                            with self._stream_watch(upstream) as watch:
                                first_chunk = next(converted, None)
                                if watch.fired:
                                    self._log_event("client_disconnected", disconnect_origin="client")
                                    self.close_connection = True
                                    return
                                self._send_upstream_headers(
                                    upstream,
                                    content_type="text/event-stream",
                                    stream=True,
                                )
                                if first_chunk is not None:
                                    self.wfile.write(first_chunk)
                                    self.wfile.flush()
                                for chunk in converted:
                                    if watch.fired:
                                        self._log_event("client_disconnected", disconnect_origin="client")
                                        self.close_connection = True
                                        return
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                                if watch.deadline_fired:
                                    self._log_event("turn_timeout")
                                return
                        finally:
                            converted.close()
                    try:
                        response_payload = json.loads(_read_bounded_upstream_body(upstream).decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise ProviderError(
                            "Claude returned invalid JSON.",
                            status=502,
                            code="claude_invalid_response",
                        ) from exc
                    if not isinstance(response_payload, dict):
                        raise ProviderError(
                            "Claude returned an invalid response.",
                            status=502,
                            code="claude_invalid_response",
                        )
                    converted = anthropic_json_to_responses(
                        response_payload,
                        model=str(payload.get("model") or ""),
                        tool_names=tool_names,
                        coalesce_text=True,
                        native_usage=True,
                    )
                    encoded = json.dumps(converted, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
            except urllib.error.HTTPError as error:
                # Evidence first (capture only), status-only projection to
                # the client; dispose before retry/refresh.
                try:
                    _capture_http_error(error)
                    self._note_upstream_error_headers(error.headers)
                finally:
                    error.close()
                if error.code in {401, 403} and attempt == 0 and state.auth_mode != "api_key":
                    try:
                        state = resolve_claude_credentials(self.config, force_refresh=True)
                        self._register_secret(state.access_token)
                        self._register_secret(state.refresh_token)
                    except ProviderError as refresh_error:
                        raise AuthError(
                            "Claude upstream rejected the OAuth credentials and refresh failed. "
                            "Run `claude setup-token` or `claude login` and try again.",
                            code="claude_upstream_unauthorized",
                        ) from refresh_error
                    continue
                if error.code in {401, 403}:
                    raise AuthError(
                        "Claude upstream rejected the credentials. Run `claude setup-token` or `claude login` and try again.",
                        code="claude_upstream_unauthorized",
                    ) from error
                raise ProviderError(
                    f"Claude upstream returned HTTP {error.code}.",
                    status=error.code,
                    code="claude_upstream_error",
                ) from error


def _shutdown_tracked_response(resource: Any) -> None:
    """Interrupt a tracked response without closing a running generator."""
    if isinstance(resource, types.GeneratorType):
        return
    sock = resource if isinstance(resource, socket.socket) else response_socket(resource)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
            return
        except OSError:
            pass
    close = getattr(resource, "close", None)
    if callable(close):
        close()


class _PrivateRotatingFileHandler(RotatingFileHandler):
    """Rotating diagnostics whose files stay owner-only where supported."""

    def _open(self):
        try:
            os.close(os.open(self.baseFilename, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
        except OSError:
            pass  # super()._open() raises the real error (P1-15 presents it)
        stream = super()._open()
        try:
            os.chmod(self.baseFilename, 0o600)
        except OSError:
            pass
        return stream


class MuseShimServer(ThreadingHTTPServer):
    daemon_threads = True
    # Shutdown closes tracked in-flight work instead of joining handlers (P1-10).
    block_on_close = False

    def server_bind(self) -> None:
        if self.config.kiro_requested and os.name == "nt":
            # Unlike a connection probe, this excludes simultaneous binds and
            # sockets whose owners have bound but have not started listening.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, address: tuple[str, int], config: Config) -> None:
        if config.kiro_requested and _port_is_busy(*address):
            raise OSError("Kiro settings require a new gateway port; refusing occupied-port reuse")
        self.config = config
        self.capture_session = None
        self._inflight: set[Any] = set()
        self._inflight_lock = threading.Lock()
        self._closed_inflight_total = 0
        self.inference_semaphore = threading.BoundedSemaphore(config.max_concurrent_requests)
        # Computed once, never per /health request (P1-15).
        self.source_fingerprint = source_fingerprint()
        if config.kiro_requested:
            # These are instance settings: native/non-Kiro reuse is unchanged.
            # Never combine SO_EXCLUSIVEADDRUSE with the base's SO_REUSEADDR.
            self.allow_reuse_address = False
            self.allow_reuse_port = False
        super().__init__(address, MuseShimHandler)
        try:
            self.capture_session = CaptureSession(config.capture_dir) if config.capture_dir else None
            self._log_handler = _PrivateRotatingFileHandler(
                config.log_file, maxBytes=5 * 1024 * 1024, backupCount=2,
                encoding="utf-8",
            )
            self._log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.diagnostics = logging.Logger(f"muse-gateway.{self.server_port}", logging.INFO)
            self.diagnostics.propagate = False
            self.diagnostics.addHandler(self._log_handler)
            self.diagnostics.info(json.dumps({"event": "gateway_started", "port": self.server_port}))
            if self.capture_session is not None:
                self.diagnostics.info(json.dumps({"event": "private_capture_enabled",
                                                 **self.capture_session.health(),
                                                 "directory": str(self.capture_session.directory)}))
        except BaseException:
            self.server_close()
            raise

    def track(self, resource: Any) -> None:
        with self._inflight_lock:
            self._inflight.add(resource)

    def untrack(self, resource: Any) -> None:
        with self._inflight_lock:
            self._inflight.discard(resource)

    @property
    def closed_inflight_total(self) -> int:
        with self._inflight_lock:
            return self._closed_inflight_total

    def close_inflight(self) -> int:
        """Close in-flight upstream responses and terminate tracked children."""
        with self._inflight_lock:
            resources, self._inflight = list(self._inflight), set()
            self._closed_inflight_total += len(resources)
        children = [resource for resource in resources if isinstance(resource, subprocess.Popen)]
        responses = [resource for resource in resources if not isinstance(resource, subprocess.Popen)]
        for resource in responses:
            try:
                _shutdown_tracked_response(resource)
            except Exception:
                pass
        if children:
            try:
                _stop_children(children)
            except Exception:
                pass
        return len(resources)

    def shutdown(self) -> None:
        super().shutdown()
        self.close_inflight()

    def server_close(self) -> None:
        super().server_close()
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            diagnostics = getattr(self, "diagnostics", None)
            if diagnostics is not None:
                diagnostics.removeHandler(handler)
            handler.close()
            self._log_handler = None


def _default_upstream_for(provider: str) -> str:
    return {
        "kiro": env_value("MUSE_GATEWAY_KIRO_UPSTREAM") or muse_kiro.DEFAULT_UPSTREAM,
        "codex": DEFAULT_CODEX_ENDPOINT, "claude": DEFAULT_CLAUDE_ENDPOINT,
        "meta": DEFAULT_META_UPSTREAM, "devin": DEFAULT_DEVIN_UPSTREAM,
        "local": env_value("MUSE_GATEWAY_LOCAL_UPSTREAM") or "",
        "generic": DEFAULT_UPSTREAM, "http": DEFAULT_UPSTREAM,
    }.get(provider, "")


def _reject_removed_devin_config(argv: list[str]) -> None:
    """Reject retired selectors without reading files or echoing values."""
    removed_flags = {"--devin-auth-db", "--devin-credential-id", "--devin-default-file"}
    removed = {value.partition("=")[0] for value in argv} & removed_flags
    for suffix in ("AUTH_DB", "CREDENTIAL_ID", "DEFAULT_FILE"):
        for prefix in ("MUSE_GATEWAY_", "MUSE_SHIM_"):
            name = prefix + "DEVIN_" + suffix
            if name in os.environ:
                removed.add(name)
    if removed:
        raise ProviderError(
            "removed Devin credential selection configuration: " + ", ".join(sorted(removed))
            + "; remove it and use `devin auth login` or an explicit Devin token. "
            "Credential databases and saved preferences are no longer read or written.",
            code="removed_devin_credential_config",
        )


_CLI_ONLY_SETTINGS = {
    "HOST": "--host", "PROVIDER": "--provider", "MODEL": "--model",
    "NO_MODEL_DISCOVERY": "--no-model-discovery", "PORTABLE": "--portable",
    "MAX_BODY_BYTES": "--max-body-bytes", "INSECURE_UPSTREAM": "--insecure-upstream",
    "VERBOSE": "--verbose", "CODEX_AUTH": "--codex-auth-path",
    "CLAUDE_CREDENTIALS": "--claude-credentials", "CLAUDE_CLI": "--claude-cli",
    "DEVIN_UPSTREAM": "--devin-upstream", "META_UPSTREAM": "--meta-upstream",
    "CODEX_UPSTREAM": "--codex-upstream", "CLAUDE_UPSTREAM": "--claude-upstream",
    "CODEX_TOKEN_URL": "--codex-token-url", "CLAUDE_TOKEN_URL": "--claude-token-url",
    "ALLOW_OAUTH_HOST_OVERRIDE": "--allow-oauth-host-override",
    "ALLOW_NON_LOOPBACK": "--allow-non-loopback",
    "ALLOW_IMAGE_FILES": "--allow-image-files",
    "REQUIRE_INBOUND_TOKEN": "--require-inbound-token",
    "KIRO_AUTOSTART": "--kiro-autostart", "KIRO_SERVER_DIR": "--kiro-server-dir",
    "CAPTURE_DIR": "--capture-dir",
}
_RETIRED_SETTINGS = {
    **_CLI_ONLY_SETTINGS,
    **{p + "_MODELS": "MUSE_GATEWAY_MODELS (exact provider/model IDs; filter only)"
       for p in ("META", "CODEX", "CLAUDE", "DEVIN", "KIRO", "LOCAL")},
    **{p + "_BASE_URL": ("MUSE_GATEWAY_" + p + "_UPSTREAM" if p in {"KIRO", "LOCAL"}
                           else "--" + p.lower() + "-upstream")
       for p in ("META", "CODEX", "CLAUDE", "DEVIN", "KIRO", "LOCAL")},
    "UPSTREAM": "explicit --upstream in generic mode",
    "API_KEY": "explicit --api-key in generic mode; route-specific keys otherwise",
    "META_API_KEY": "META_API_KEY (API billing only; not subscription launch)",
    "DEVIN_API_KEY": "DEVIN_API_KEY",
    "CODEX_ACCESS_TOKEN": "CODEX_HOME/auth.json or --access-token",
    "CODEX_REFRESH_TOKEN": "CODEX_HOME/auth.json",
    "CLAUDE_ACCESS_TOKEN": "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_REFRESH_TOKEN": "CLAUDE_CODE_REFRESH_TOKEN",
    "KIRO_MODEL_BUDGETS": "exact model metadata; budget overrides were removed",
    "KIRO_CONTEXT_LIMIT": "exact model metadata; budget overrides were removed",
    "KIRO_OUTPUT_LIMIT": "exact model metadata; budget overrides were removed",
    "MAX_OUTPUT_TOKENS": "per-request max_output_tokens (no global cap)",
    "REFRESH_SKEW": "internal refresh timing (no public setting)",
}
_RETIRED_FLAGS = {
    **{"--" + p + "-models": "MUSE_GATEWAY_MODELS (exact provider/model IDs; filter only)"
       for p in ("meta", "codex", "claude", "devin", "kiro", "local")},
    **{"--" + p + "-base-url": "--" + p + "-upstream"
       for p in ("meta", "codex", "claude", "devin", "kiro", "local")},
    "--kiro-context-limit": "exact model metadata; remove budget overrides",
    "--kiro-output-limit": "exact model metadata; remove budget overrides",
    "--kiro-model-budgets": "exact model metadata; remove budget overrides",
    "--max-output-tokens": "per-request max_output_tokens (no global cap)",
    "--refresh-skew": "internal refresh timing; remove this flag",
}


def _reject_retired_flags(argv: list[str]) -> None:
    for token in argv:
        name = token.partition("=")[0]
        if name in _RETIRED_FLAGS:
            raise ProviderError(f"removed gateway flag {name}; use {_RETIRED_FLAGS[name]}.", code="retired_configuration")


def _warn_retired_environment() -> None:
    external = {"CODEX_ACCESS_TOKEN": "CODEX_HOME/auth.json or --access-token",
                "OPENAI_CODEX_ACCESS_TOKEN": "CODEX_HOME/auth.json or --access-token",
                "CODEX_REFRESH_TOKEN": "CODEX_HOME/auth.json", "ANTHROPIC_TOKEN": "CLAUDE_CODE_OAUTH_TOKEN"}
    for name in sorted(os.environ):
        replacement = external.get(name)
        current = None
        if name.startswith(("MUSE_SHIM_", "MUSE_GATEWAY_")):
            prefix = "MUSE_SHIM_" if name.startswith("MUSE_SHIM_") else "MUSE_GATEWAY_"
            suffix = name[len(prefix):]
            replacement = _RETIRED_SETTINGS.get(suffix)
            if prefix == "MUSE_SHIM_":
                current = "MUSE_GATEWAY_" + suffix
                replacement = replacement or current
        if not replacement:
            continue
        target = replacement.split(" ", 1)[0]
        conflict = (current and current in os.environ) or (target in os.environ and target != name)
        if conflict:
            raise ProviderError(f"conflicting retired setting {name}; remove it and use {replacement}.", code="retired_configuration_conflict")
        print(f"warning: {name} is removed and ignored; use {replacement}.", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "muse-gateway: loopback Muse proxy with a unified Meta, Codex, "
            "Claude, Devin, Kiro, and local model catalog (forked from muse-shim)"
        ),
        epilog=(
            "For Muse subscription auth use `muse-gateway launch [MUSE_ARGS...]` "
            "(or the muse-gateway.ps1 / Bash wrappers, which forward to it). "
            "`python -m muse_gateway run [MUSE_ARGS...]` uses a placeholder Muse "
            "key instead; server options go before `run` or `launch`. "
            "Check credentials with `muse-gateway doctor [--doctor-provider NAME]`."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("auto", "generic", "http", "codex", "claude", "meta", "devin", "kiro", "local", "doctor"),
        default="auto",
        help="provider mode or doctor",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "generic", "http", "codex", "claude", "meta", "devin", "kiro", "local"),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=env_value("MUSE_GATEWAY_PORT", str(DEFAULT_PORT)))
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--model", default=None, help="single-provider model override only")
    parser.add_argument("--codex-upstream", default=None, help="guarded native/development endpoint override")
    parser.add_argument("--claude-upstream", default=None, help="guarded native/development endpoint override")
    parser.add_argument(
        "--meta-upstream",
        default=None,
        help="Meta Responses base URL (default: https://api.meta.ai/v1; custom hosts receive native Meta payloads and authorization)",
    )
    parser.add_argument(
        "--meta-api-key",
        default=env_value("META_API_KEY"),
        help="explicit Meta API billing key (META_API_KEY); conflicts with subscription launch",
    )
    parser.add_argument(
        "--local-upstream",
        default=env_value("MUSE_GATEWAY_LOCAL_UPSTREAM"),
        help="local Responses base URL (unset disables local routing)",
    )
    parser.add_argument(
        "--local-api-key",
        default=env_value("MUSE_GATEWAY_LOCAL_API_KEY"),
        help="optional local key (unset sends no Authorization header)",
    )
    parser.add_argument(
        "--devin-upstream",
        default=None,
        help="Devin Connect RPC base URL (default: https://server.codeium.com)",
    )
    parser.add_argument(
        "--devin-api-key",
        default=env_value("DEVIN_API_KEY"),
        help="explicit Devin session token (DEVIN_API_KEY; else CLI login)",
    )
    parser.add_argument(
        "--kiro-upstream", default=env_value("MUSE_GATEWAY_KIRO_UPSTREAM"),
        help="Kiro Gateway Messages base URL (default: http://127.0.0.1:8000/v1; loopback HTTP or verified HTTPS)",
    )
    parser.add_argument(
        "--kiro-api-key", default=env_value("MUSE_GATEWAY_KIRO_API_KEY"),
        help="Kiro Gateway proxy key only (prefer MUSE_GATEWAY_KIRO_API_KEY; never native Kiro or Claude credentials)",
    )
    parser.add_argument(
        "--kiro-autostart",
        action="store_true",
        default=False,
        help="launch starts the local Kiro Gateway checkout when its loopback service is down (default: off; CLI-only, never from env or file)",
    )
    parser.add_argument(
        "--kiro-server-dir", default=None,
        help="Kiro Gateway checkout to autostart (default: ~/kiro-gateway; CLI-only)",
    )
    parser.add_argument(
        "--no-model-discovery",
        action="store_false",
        dest="model_discovery",
        default=True,
        help="do not query provider model catalogs; advertise configured fallbacks only",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        default=False,
        help="strip reasoning and cache fields for local/generic Responses servers only",
    )
    parser.add_argument(
        "--insecure-upstream",
        action="store_true",
        default=False,
        help="disable upstream TLS verification (development only)",
    )
    parser.add_argument(
        "--allow-oauth-host-override",
        action="store_true",
        default=False,
        help="DANGEROUS development-only opt-in for non-native OAuth hosts; never enable from an environment variable",
    )
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        default=False,
        help="allow binding beyond loopback (requires --require-inbound-token)",
    )
    parser.add_argument(
        "--require-inbound-token",
        action="store_true",
        default=False,
        help="require MUSE_GATEWAY_INBOUND_TOKEN on inference POSTs (x-muse-gateway-token or Bearer)",
    )
    parser.add_argument(
        "--allow-image-files",
        action="store_true",
        default=False,
        help="accept file:// image references and send those files' bytes upstream "
             "(CLI-only; off by default; absolute paths, no symlinks, images only)",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=DEFAULT_MAX_BODY_BYTES,
        help=f"maximum incoming JSON body size (default: {DEFAULT_MAX_BODY_BYTES})",
    )
    parser.add_argument("--timeout", type=_timeout_seconds, default=env_value("MUSE_GATEWAY_TIMEOUT", "600"),
                        help="connect/headers budget in seconds (also the Claude CLI subprocess deadline)")
    parser.add_argument("--stream-idle-timeout", type=_timeout_seconds,
                        default=env_value("MUSE_GATEWAY_STREAM_IDLE_TIMEOUT", "120"),
                        help="upstream stream idle budget in seconds after headers (default: 120)")
    parser.add_argument("--turn-timeout", type=_timeout_seconds,
                        default=env_value("MUSE_GATEWAY_TURN_TIMEOUT", "3600"),
                        help="absolute streaming turn deadline in seconds (default: 3600)")
    parser.add_argument("--max-concurrent-requests", type=_concurrency_limit,
                        default=env_value("MUSE_GATEWAY_MAX_CONCURRENT_REQUESTS", "8"),
                        help="maximum concurrent inference POSTs (default: 8)")
    parser.add_argument(
        "--capture-dir", default=None,
        help="opt-in private request/response captures (CLI-only; contains conversation content; "
             "adds per-chunk disk writes, so streams can arrive slower while enabled)",
    )
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME"))
    parser.add_argument("--codex-auth-path", default=None)
    parser.add_argument("--codex-token-url", default=None, help="guarded native/development OAuth endpoint")
    parser.add_argument("--claude-credentials", default=None)
    parser.add_argument("--claude-token-url", default=None, help="guarded native/development OAuth endpoint")
    parser.add_argument(
        "--claude-transport",
        choices=("cli", "http"),
        default=env_value("MUSE_GATEWAY_CLAUDE_TRANSPORT", "cli"),
        help="Claude inference transport (default: cli, using claude -p)",
    )
    parser.add_argument(
        "--claude-cli",
        default="claude",
        help="Claude Code executable used by CLI transport",
    )
    parser.add_argument(
        "--claude-cli-passthrough",
        choices=sorted(_CLAUDE_PASSTHROUGH_SETS),
        default=env_value("MUSE_GATEWAY_CLAUDE_CLI_PASSTHROUGH", "oauth"),
        help=(
            "which Claude credential env vars the `claude -p` child may inherit: "
            "oauth (default), api-key, all, or none. Server-side: applies to the "
            "serving gateway's Config, so a launcher attaching to a running "
            "gateway inherits that server's policy"
        ),
    )
    parser.add_argument("--version", action="version", version=f"muse-gateway {_gateway_version()}")
    parser.add_argument(
        "--doctor-provider",
        choices=("generic", "codex", "claude", "meta", "devin", "kiro", "local"),
        default=None,
    )
    parser.add_argument(
        "--models",
        action="store_true",
        default=False,
        help="print the unified model catalog (one <provider>/<name> id per "
        "line) and exit without starting the server",
    )
    parser.add_argument(
        "--models-provider",
        choices=("codex", "claude", "meta", "devin", "kiro", "local"),
        default=None,
        help="with --models, list only this provider's catalog",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="config file for server, doctor, models or launch (default: beside muse_gateway.py only)",
    )
    parser.add_argument(
        "--log-file", default=env_value("MUSE_GATEWAY_LOG_FILE"),
        help="rotating diagnostic log (default: ./muse-gateway.log; no request/response content)",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def _gateway_flag_options() -> set[str]:
    """Zero-argument gateway options, derived from argparse (P1-11)."""
    return {
        option
        for action in _build_parser()._actions
        if action.nargs == 0
        for option in action.option_strings
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    try:
        raw = list(sys.argv[1:] if argv is None else argv)
        _reject_removed_devin_config(raw)
        _reject_retired_flags(raw)
        _warn_retired_environment()
    except ProviderError as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    provider = args.provider or args.mode
    if provider == "http":
        provider = "generic"
    args.provider = provider
    if not args.upstream:
        args.upstream = getattr(args, f"{provider}_upstream", None) or _default_upstream_for(provider)
    try:
        _menu_filter(env_value("MUSE_GATEWAY_MODELS"))
        if provider == "kiro" or args.kiro_upstream or args.kiro_api_key:
            muse_kiro.normalize_upstream(args.upstream if provider == "kiro" else (args.kiro_upstream or muse_kiro.DEFAULT_UPSTREAM))
            if args.kiro_api_key:
                muse_kiro.validate_proxy_key(args.kiro_api_key)
        if provider == "auto" and args.model:
            raise ProviderError(
                "--model cannot pin auto mode; choose a model "
                "in Muse with /model, or start a provider-specific mode.",
                code="auto_model_override_rejected",
            )
        if args.max_body_bytes <= 0:
            raise ProviderError(
                "--max-body-bytes must be a positive integer.",
                code="invalid_body_limit",
            )
        if not _is_loopback_host(args.host) and not args.allow_non_loopback:
            raise ProviderError(
                f"refusing non-loopback bind address {args.host!r}; use --allow-non-loopback when intentional.",
                code="non_loopback_bind_rejected",
            )
        inbound_token = os.environ.get("MUSE_GATEWAY_INBOUND_TOKEN", "").strip()
        if args.require_inbound_token and not inbound_token:
            raise ProviderError(
                "--require-inbound-token needs MUSE_GATEWAY_INBOUND_TOKEN set in the gateway process environment.",
                code="inbound_token_missing",
            )
        if not _is_loopback_host(args.host) and not (args.require_inbound_token and inbound_token):
            raise ProviderError(
                f"refusing non-loopback bind address {args.host!r} without an inbound token; "
                "set MUSE_GATEWAY_INBOUND_TOKEN and pass --require-inbound-token.",
                code="non_loopback_token_required",
            )
        native_providers = ("codex", "claude") if provider == "auto" else (provider,)
        for native_provider in native_providers:
            native_upstream = (
                args.upstream
                if provider == native_provider
                else (getattr(args, f"{native_provider}_upstream", None) or _default_upstream_for(native_provider))
            )
            _validate_native_config(
                provider=native_provider,
                upstream=native_upstream,
                codex_token_url=args.codex_token_url,
                claude_token_url=args.claude_token_url,
                insecure_upstream=args.insecure_upstream,
                allow_oauth_host_override=args.allow_oauth_host_override,
            )
    except (ProviderError, ValueError) as error:
        parser.error(str(error))
    return args


def _credential_status(config: Config, provider: str) -> tuple[bool, str]:
    if provider == "kiro":
        if not config.kiro_api_key:
            return False, "proxy key not configured; set MUSE_GATEWAY_KIRO_API_KEY (no native credential fallback)"
        entries = provider_model_catalog(config, "kiro")
        ok = not (config.model_discovery and "unavailable" in config._kiro_catalog_status)
        detail = f"proxy key configured; {config._kiro_catalog_status}; {len(entries)} menu entries; capacities use exact-ID field provenance"
        if config._catalog_diagnostics.get("kiro"):
            detail += "; " + config._catalog_diagnostics["kiro"]
        if config._kiro_service_note:
            detail = f"{detail}; {config._kiro_service_note}"
        return ok, detail
    if provider == "generic":
        if config.api_key:
            return True, "API key configured"
        return False, "API key not configured"
    if provider == "meta":
        if config.meta_api_key:
            return True, "API key configured"
        return (
            False,
            "no static API key configured; subscription forwarding is checked "
            "only when Muse sends a request",
        )
    if provider == "local":
        if not config.local_upstream_explicit:
            return False, "local disabled; set MUSE_GATEWAY_LOCAL_UPSTREAM"
        provider_model_catalog(config, "local")
        available = [row for row in config._snapshot_raw("local").values()
                     if positive_int(row.get("context_limit")) and positive_int(row.get("output_limit"))]
        auth = "local key configured" if config.local_api_key else "no Authorization header (key optional)"
        if available:
            return True, f"{auth}; {len(available)} models with capacity metadata; inference not validated"
        detail = config._catalog_diagnostics.get("local") or "catalog unavailable, empty, disabled, or missing required capacities"
        return False, f"{auth}; {detail}; inference not validated"
    if provider == "claude" and config.claude_transport == "cli":
        executable = shutil.which(config.claude_cli)
        if not executable:
            return False, f"Claude CLI executable {config.claude_cli!r} not found"
        try:
            cli_state = discover_claude_credentials(config)
        except Exception:
            cli_state = None
        cli_usable, cli_reason = credentials_usable(cli_state)
        if not cli_usable:
            return (
                False,
                f"Claude CLI found but no usable login ({cli_reason}); "
                "run `claude login` or set an API key",
            )
        return (
            True,
            f"Claude CLI print transport (non-streaming: turns arrive whole); executable={executable}; {cli_reason} login",
        )
    if provider == "devin":
        state = muse_devin.credential_state(api_key=config.devin_api_key)
        status = state.get("state")
        if status == "ready":
            identity = "Devin CLI login" if state.get("source") == "devin-cli" else "explicit token"
            return True, f"credential ready ({identity})"
        return False, state.get("note") or f"credential {status}"
    try:
        state = discover_codex_credentials(config) if provider == "codex" else discover_claude_credentials(config)
    except Exception:
        state = None
    if state is None:
        symlinks = refused_symlink_credential_labels(config, provider)
        if symlinks:
            return False, f"symlinked credential refused at {symlinks[0]}; replace the symlink with the real file"
        if provider == "codex":
            return False, "no OAuth credentials found; run `codex login`"
        return False, "no OAuth login or API key found; run `claude login`"
    usable, reason = credentials_usable(state)
    if not usable:
        hint = "codex login" if provider == "codex" else "claude login"
        return False, f"credentials unusable ({reason}); run `{hint}` and try again"
    expiry = "expiry unknown"
    if state.expires_at is not None:
        expiry = "expired" if state.expires_at <= time.time() else "not expired"
    refresh = "refresh available" if state.refresh_token else "no refresh token"
    mode = "API key" if state.auth_mode == "api_key" else "OAuth"
    return True, f"{mode}; {refresh}; {expiry}; source={state.source}"


def run_doctor(config: Config, *, output: Any = None) -> int:
    output = output or sys.stdout
    print("muse-gateway doctor", file=output)
    print(f"listen: {config.host}:{config.port}", file=output)
    if config.provider != "auto":
        print(f"generic endpoint configured: {'yes' if bool(config.upstream) else 'no'}", file=output)
    selected = getattr(config, "doctor_provider", None)
    if selected is None and config.provider in {"codex", "claude", "meta", "local", "devin", "kiro"}:
        selected = config.provider
    if selected:
        providers = (selected,)
    elif config.provider in {"auto", "doctor"}:
        # Include local routes configured through discovery or explicit IDs.
        providers = ("codex", "claude", "meta", "devin")
        if config.kiro_requested:
            providers = (*providers, "kiro")
        if config.local_upstream_explicit:
            providers = (*providers, "local")
    else:
        providers = ("codex", "claude")
    ready = True
    for provider in providers:
        ok, detail = _credential_status(config, provider)
        ready = ready and ok
        print(f"{provider}: {'ready' if ok else 'not ready'} ({detail})", file=output)
        if provider == "claude":
            if config.claude_transport == "cli":
                print(
                    "claude capability: inference uses Claude Code print mode; "
                    "Muse tools are bridged through structured output",
                    file=output,
                )
                print(
                    f"claude cli passthrough: {config.claude_cli_passthrough} "
                    "(--claude-cli-passthrough; CLI turns arrive whole, not token-streamed)",
                    file=output,
                )
            else:
                print(
                    "claude capability: direct Messages API mode; Muse reasoning "
                    "is not sent as Anthropic thinking",
                    file=output,
                )
        if provider == "kiro":
            print(
                "kiro capability: separate Kiro Gateway Messages service; text/tools, "
                "thinking disabled; no hard output-cap or forced-tool guarantee; inference not validated by doctor",
                file=output,
            )
    print("secrets: not displayed", file=output)
    return 0 if ready else 1


def run_models(config: Config, provider: str | None = None, *, output: Any = None) -> int:
    """Print the unified model catalog, one <provider>/<name> id per line.

    Uses the same discovery and availability gating as the running server,
    so this lists exactly what Muse /model would offer. A provider that
    fails discovery is skipped with a stderr warning; the exit code stays 0
    so scripts can parse stdout unconditionally.
    """
    output = output or sys.stdout
    if provider:
        providers = (provider,)
    elif config.provider in {"codex", "claude", "meta", "local", "devin", "kiro"}:
        providers = (config.provider,)
    else:
        providers = ("codex", "claude", "meta", "devin", "kiro")
        if config.local_upstream_explicit:
            providers = (*providers, "local")
    for name in providers:
        try:
            entries = provider_model_catalog(config, name)
        except Exception:
            print(f"muse-gateway models: {name} catalog unavailable; inference not validated", file=sys.stderr)
            continue
        if not entries and not config._catalog_diagnostics.get(name):
            print(f"muse-gateway models: {name} has no selectable models (unavailable, empty, disabled, or filtered); inference not validated", file=sys.stderr)
        for entry in entries:
            model_id = entry.get("model_id") if isinstance(entry, dict) else None
            if model_id:
                print(model_id, file=output)
    return 0


def _server_destination(config: Config) -> str:
    if config.provider == "auto":
        return "Codex + Claude + Meta + Devin + Kiro + Local (select with Muse /model)"
    if config.provider == "claude" and config.claude_transport == "cli":
        return f"Claude CLI ({config.claude_cli} -p)"
    return config.provider_url()


def _existing_shim_is_compatible(
    base_url: str, provider: str, claude_transport: str = "cli"
) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as response:
            payload = _read_bounded_json(response, limit=HEALTH_JSON_RESPONSE_LIMIT)
        compatible = (
            response.status == 200
            and isinstance(payload, dict)
            and payload.get("ok") is True
            and payload.get("service") in {"muse-gateway", "muse-shim"}
            and payload.get("provider") == provider
            and payload.get("claude_transport") == claude_transport
        )
    except (OSError, ValueError, json.JSONDecodeError, ProviderError):
        return False
    if compatible:
        _warn_version_mismatch(base_url, payload)
    return compatible


def _warn_version_mismatch(base_url: str, health: dict[str, Any]) -> None:
    """Reuse warns (never refuses) when the running gateway differs (P1-15)."""
    running = health.get("version")
    mine = _gateway_version()
    if running != mine:
        print(
            f"warning: reusing the gateway on {base_url} reporting version "
            f"{running or 'unknown'} (pid {health.get('pid', '?')}); this launcher is {mine}. "
            "Stop it and relaunch to use this version.",
            file=sys.stderr, flush=True,
        )


class _LauncherSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def _signal_owned_child(process: subprocess.Popen[Any], *, kill: bool = False) -> None:
    """Signal one gateway-owned child, never an unrelated process (P1-10)."""
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            group = os.getpgid(process.pid)
        except OSError:
            group = None
        if group not in (None, 0, 1) and group == process.pid:
            try:
                os.killpg(group, signal.SIGKILL if kill else signal.SIGTERM)
                return
            except (OSError, ProcessLookupError):
                pass  # Fall through to the direct handle below.
    try:
        if kill:
            process.kill()
        else:
            process.terminate()
    except OSError:
        pass


def _wait_for_children(processes: list[subprocess.Popen[Any]], timeout: float) -> list[subprocess.Popen[Any]]:
    deadline = time.monotonic() + timeout
    alive = []
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            alive.append(process)
    return alive


def _stop_children(processes: Iterable[subprocess.Popen[Any]]) -> None:
    """Signal every owned child first, then wait against shared deadlines."""
    live = [process for process in processes if process.poll() is None]
    for process in live:
        _signal_owned_child(process)
    live = _wait_for_children(live, 3.0)
    for process in live:
        _signal_owned_child(process, kill=True)
    _wait_for_children(live, 2.0)


def _stop_child(process: subprocess.Popen[Any]) -> None:
    """Stop a gateway-owned child, never an unrelated process (P1-10).

    Only the given ``Popen`` handle is ever signalled. On POSIX, when the
    owned child leads its own process group (Kiro autostart uses
    ``start_new_session=True``), the whole group is signalled so detached
    grandchildren cannot outlive the child. Anything else gets a direct
    terminate/kill. Every wait is bounded; proceed regardless.
    """
    _stop_children([process])


def _subcommand_index(argv: list[str], word: str) -> int | None:
    """Find a subcommand without mistaking an option value for it."""
    flag_options = _gateway_flag_options()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"run", "launch"}:
            return index if token == word else None
        if token == "--":
            return None
        if token in flag_options or not token.startswith("-") or "=" in token:
            index += 1
        else:
            # Every non-flag gateway option consumes exactly one value.
            index += 2
    return None


def _run_command_index(argv: list[str]) -> int | None:
    """Find the run subcommand without mistaking an option value for it."""
    return _subcommand_index(argv, "run")


def _launch_command_index(argv: list[str]) -> int | None:
    """Find the launch subcommand without mistaking an option value for it."""
    return _subcommand_index(argv, "launch")


def run_muse(config: Config, muse_args: list[str], *, output: Any = None) -> int:
    """Run the local gateway and the Muse client as one foreground command."""
    output = output or sys.stderr
    server: MuseShimServer | None = None
    server_thread: threading.Thread | None = None
    child: subprocess.Popen[Any] | None = None
    previous_signal_handlers: dict[int, Any] = {}
    base_url = f"http://{config.host}:{config.port}"

    if config.kiro_requested and _port_is_busy(config.host, config.port):
        print(
            f"muse-gateway Kiro settings require a new gateway on {base_url}; "
            "the port is occupied. Choose --port; existing servers are not reused or stopped.",
            file=output, flush=True,
        )
        return 1
    try:
        server = MuseShimServer((config.host, config.port), config)
    except OSError as error:
        if config.kiro_requested or not _existing_shim_is_compatible(
            base_url, config.provider, config.claude_transport
        ):
            print(
                f"muse-gateway could not listen on {base_url}: {error}",
                file=output,
                flush=True,
            )
            return 1
        print(f"muse-gateway reusing {base_url}", file=output, flush=True)
    else:
        base_url = f"http://{config.host}:{server.server_port}"
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(
            f"muse-gateway {config.provider} listening on {base_url} "
            f"-> {_server_destination(config)}",
            file=output,
            flush=True,
        )

    environment = _child_environment()
    environment["META_API_KEY"] = "local-gateway-placeholder"
    muse_bin = shutil.which("muse") or shutil.which("muse.cmd") or "muse"
    command = [
        muse_bin,
        "--provider",
        "meta",
        "--base-url",
        base_url,
        *muse_args,
    ]

    if threading.current_thread() is threading.main_thread():
        def handle_launcher_signal(signum: int, _frame: Any) -> None:
            raise _LauncherSignal(signum)

        # SIGHUP does not exist on Windows; handle whichever signals do.
        launcher_signals = tuple(
            signum
            for signum in (getattr(signal, "SIGHUP", None), signal.SIGTERM)
            if signum is not None
        )
        for signum in launcher_signals:
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_launcher_signal)

    try:
        child = subprocess.Popen(command, env=environment)
        return int(child.wait())
    except FileNotFoundError:
        print("muse-gateway could not find `muse` on PATH.", file=output, flush=True)
        return 127
    except KeyboardInterrupt:
        return 130
    except _LauncherSignal as received:
        return 128 + received.signum
    finally:
        for signum in previous_signal_handlers:
            signal.signal(signum, signal.SIG_IGN)
        if child is not None:
            _stop_child(child)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)


LAUNCH_USAGE = """\
muse-gateway launch: start the gateway, then Muse on top of it.

Meta models ride your `muse login` subscription via header forwarding, so no
Meta API key is configured. Codex uses your existing `codex login`.

Usage:
  muse-gateway launch [LAUNCH_OPTIONS] [MUSE_ARGS ...]

Launcher options must come first: the first subcommand, unknown flag, or `--`
ends option parsing and everything from there goes to `muse`.

  -p, --port PORT            listen port (default 8787)
  --local-upstream URL       local Responses base URL
  --local-api-key KEY        local upstream API key
  --devin-upstream URL       Devin Connect RPC base URL
  --devin-api-key KEY        Devin session token (else CLI login)
  --kiro-upstream URL        separate Kiro Gateway Messages base URL
  --kiro-api-key KEY         Kiro Gateway proxy key (prefer environment)
  --kiro-server-dir DIR      Kiro Gateway checkout to autostart (~/kiro-gateway)
  --kiro-autostart            start the checkout when its loopback service is down
  --check-only               preflight + catalog check, then exit (no Muse)
  --env-file PATH            config file (default: beside muse_gateway.py only)
  --muse-bin PATH            Muse executable (default: auto-detect)
  --log-file PATH            rotating diagnostics (default: ./muse-gateway.log)
  -h                         this help; --help passes through to Muse

Config precedence: flags, then current environment, then the env file.
Local is disabled unless MUSE_GATEWAY_LOCAL_UPSTREAM is set; its key is optional.
MUSE_GATEWAY_MODELS optionally filters exact provider/model IDs, never entitlement.
Advanced server options go before `launch`. Removed gateway flags are rejected.
"""

_LAUNCH_FLAG_OPTIONS = {"--no-model-discovery", "--portable", "--verbose", "--insecure-upstream",
                        "--allow-oauth-host-override", "--allow-non-loopback", "--require-inbound-token",
                        "--allow-image-files"}
_LAUNCH_VALUE_OPTIONS = {
    "--host": "host", "--timeout": "timeout", "--max-body-bytes": "max_body_bytes",
    "--capture-dir": "capture_dir", "--codex-auth-path": "codex_auth_path",
    "--codex-home": "codex_home", "--access-token": "access_token",
    "--codex-upstream": "codex_upstream", "--codex-token-url": "codex_token_url",
    "--claude-upstream": "claude_upstream", "--claude-token-url": "claude_token_url",
    "--claude-credentials": "claude_credentials", "--claude-cli": "claude_cli",
    "--claude-cli-passthrough": "claude_cli_passthrough",
    "--stream-idle-timeout": "stream_idle_timeout", "--turn-timeout": "turn_timeout",
    "--max-concurrent-requests": "max_concurrent_requests",
    "--claude-transport": "claude_transport", "--meta-upstream": "meta_upstream",
    "--meta-api-key": "meta_api_key",
    "--port": "port",
    "--local-upstream": "local_upstream",
    "--local-api-key": "local_api_key",
    "--devin-upstream": "devin_upstream",
    "--devin-api-key": "devin_api_key",
    "--kiro-upstream": "kiro_upstream",
    "--kiro-api-key": "kiro_api_key",
    "--kiro-server-dir": "kiro_server_dir",
    "--env-file": "env_file",
    "--muse-bin": "muse_bin",
    "--log-file": "log_file",
}


def _split_launch_args(
    argv: list[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Split `launch` args into (launcher options, Muse args).

    Mirrors the source wrapper contract: leading launcher options are
    consumed; the first subcommand, unknown flag, or `--` ends option parsing.
    `--help` is intentionally not a launcher option so it reaches Muse; `-h`
    prints launcher help instead.
    """
    _reject_removed_devin_config(argv)
    _reject_retired_flags(argv)
    opts = argparse.Namespace(
        port=None,
        check_only=False,
        server_flags=[],
        kiro_autostart=False,
        local_upstream=None,
        local_api_key=None,
        env_file=None,
        muse_bin=None,
        log_file=None,
        help=False,
    )
    muse_args: list[str] = []
    index = 0
    total = len(argv)
    while index < total:
        token = argv[index]
        if token == "--":
            muse_args.extend(argv[index + 1 :])
            break
        if token == "-h":
            opts.help = True
            index += 1
            continue
        if token in ("-p", "--port"):
            index += 1
            if index >= total:
                raise ProviderError(
                    "launch option --port needs a value.",
                    code="launch_option_value_missing",
                )
            try:
                opts.port = int(argv[index])
            except ValueError as error:
                raise ProviderError(
                    f"launch option --port needs an integer, got {argv[index]!r}.",
                    code="launch_option_value_invalid",
                ) from error
        elif token == "--check-only":
            opts.check_only = True
        elif token in _LAUNCH_FLAG_OPTIONS:
            opts.server_flags.append(token)
        elif token == "--kiro-autostart":
            opts.kiro_autostart = True
        elif token in _LAUNCH_VALUE_OPTIONS:
            index += 1
            if index >= total:
                raise ProviderError(
                    f"launch option {token} needs a value.",
                    code="launch_option_value_missing",
                )
            value = argv[index]
            setattr(opts, _LAUNCH_VALUE_OPTIONS[token], value)
        elif token.startswith("--") and "=" in token:
            name, _, value = token.partition("=")
            if name == "--port":
                try:
                    opts.port = int(value)
                except ValueError as error:
                    raise ProviderError(
                        f"launch option --port needs an integer, got {value!r}.",
                        code="launch_option_value_invalid",
                    ) from error
            elif name in _LAUNCH_VALUE_OPTIONS:
                setattr(opts, _LAUNCH_VALUE_OPTIONS[name], value)
            else:
                muse_args.extend(argv[index:])
                break
        else:
            muse_args.extend(argv[index:])
            break
        index += 1
    return opts, muse_args


_FILE_ENV_NAMES: set[str] = set()  # names load_env_file set in this process

# Mirrors tests/check_muse.py:68-78 — the project's own isolation policy.
_CHILD_ENV_DROP_PREFIXES = (
    "MUSE_GATEWAY_", "MUSE_SHIM_", "META_", "DEVIN_", "CODEX_",
    "OPENAI_", "ANTHROPIC_", "CLAUDE_", "KIRO_",
    "AWS_", "AZURE_", "GCP_", "GOOGLE_CLOUD_",
)
_SECRET_ENV_NAME = re.compile(
    r"(?:^|_)(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|ACCESS_KEY|CREDENTIALS?)(?:$|_)"
)

# Claude CLI credential passthrough policy (--claude-cli-passthrough).
_CLAUDE_OAUTH_VARS = frozenset({"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_REFRESH_TOKEN"})
_CLAUDE_API_KEY_VARS = frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"})
_CLAUDE_HARMLESS_VARS = frozenset({"CLAUDE_CONFIG_DIR"})  # location, not a credential
_CLAUDE_PASSTHROUGH_SETS = {
    "none": frozenset(),
    "oauth": _CLAUDE_OAUTH_VARS,
    "api-key": _CLAUDE_API_KEY_VARS,
    "all": _CLAUDE_OAUTH_VARS | _CLAUDE_API_KEY_VARS,
}


def _child_environment(*, keep: frozenset[str] = frozenset()) -> dict[str, str]:
    """Environment for a spawned child: everything we loaded or broker is gone.

    `keep` names exact variables to retain despite matching a dropped prefix;
    callers pass the resolved Claude passthrough set for `claude -p` only."""
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in keep:
            env[name] = value
        elif (
            name in _FILE_ENV_NAMES
            or name.startswith(_CHILD_ENV_DROP_PREFIXES)
            or _SECRET_ENV_NAME.search(name)
        ):
            continue
        else:
            env[name] = value
    return env


def load_env_file(path: str) -> int:
    """Load KEY=VALUE lines from path into the environment without overwriting.

    Returns the number of variables set. A missing file sets nothing.
    """
    try:
        with open(path, encoding="utf-8-sig") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return 0
    loaded = 0
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, _, value = text.partition("=")
        name = name.strip()
        name = re.sub(r"^export\s+", "", name)
        if (
            not name
            or not (name[0].isalpha() or name[0] == "_")
            or not all(char.isalnum() or char == "_" for char in name)
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'":
            quote = value[0]
            closing = -1
            scan = 1
            while True:
                candidate = value.find(quote, scan)
                if candidate < 0:
                    break
                rest = value[candidate + 1:].strip()
                if not rest or rest.startswith("#"):
                    closing = candidate
                    break
                scan = candidate + 1
            if closing > 0:
                value = value[1:closing]
            elif len(value) >= 2 and value[-1] == quote:
                value = value[1:-1]
        else:
            # Unquoted values: a whitespace-delimited ' #' starts a comment.
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if name in os.environ:
            continue
        os.environ[name] = value
        _FILE_ENV_NAMES.add(name)
        loaded += 1
    return loaded


def _default_env_file() -> str | None:
    """Locate the env file beside muse_gateway.py only.

    The working directory is the project under edit, i.e. untrusted input:
    never adopt its .env implicitly. Operators opt in with --env-file. A
    symlinked beside-module file is deliberately followed; the resolved
    target is returned so startup reporting names the file actually read."""
    script_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    return os.path.realpath(script_env) if os.path.isfile(script_env) else None


def _load_config_file(argv: list[str], *, explicit: str | None = None) -> tuple[str | None, int]:
    """Load the env file and return (path, settings_loaded).

    Lookup is beside-module only (P0-02). A named --env-file that does not
    exist is an error, not a silent no-op (de-wonder F6)."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file")
    opts, _ = bootstrap.parse_known_args(argv)
    requested = explicit if explicit is not None else opts.env_file
    if requested is not None:
        path = os.path.expanduser(requested)
        if not os.path.isfile(path):
            raise ProviderError(
                f"--env-file {requested!r} does not exist or is not a file.",
                code="env_file_missing",
            )
    else:
        path = _default_env_file()
    if path is None:
        return None, 0
    return path, load_env_file(path)


def _wait_for_health(base_url: str, timeout: float = 15.0) -> dict[str, Any] | None:
    """Poll /health until it reports ok; return its payload or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                payload = _read_bounded_json(response, limit=HEALTH_JSON_RESPONSE_LIMIT)
            if isinstance(payload, dict) and payload.get("ok") is True:
                return payload
        except Exception:
            pass
        time.sleep(0.2)
    return None


def _color_enabled() -> bool:
    """ANSI color only on a live terminal that did not opt out."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _paint(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


_KIRO_AUTOSTART_TIMEOUT = 30.0
_KIRO_AUTOSTART_POLL = 0.5


def _kiro_service_healthy(upstream: str, key: str, *, timeout: float = 3.0) -> bool:
    try:
        with muse_kiro.open_request(
            muse_kiro.normalize_upstream(upstream) + "/models", key, timeout=timeout
        ) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _kiro_server_python(server_dir: str) -> str:
    """Prefer the checkout's venv python; fall back to this interpreter."""
    if os.name == "nt":
        candidate = os.path.join(server_dir, ".venv", "Scripts", "python.exe")
    else:
        candidate = os.path.join(server_dir, ".venv", "bin", "python")
    if os.path.isfile(candidate):
        return candidate
    return sys.executable


def ensure_kiro_gateway(
    config: Config, *, timeout: float = _KIRO_AUTOSTART_TIMEOUT
) -> subprocess.Popen[Any] | None:
    """Start the local Kiro Gateway checkout when enabled and needed.

    Returns the child this call started (the caller owns its lifetime), or
    None when the service was already up, autostart is off, or the start
    failed. The outcome lands in ``config._kiro_service_note`` for the
    startup status line. Only loopback upstreams are ever autostarted.
    """
    if not config.kiro_api_key:
        return None
    if _kiro_service_healthy(config.kiro_upstream, config.kiro_api_key):
        config._kiro_service_note = "service already running"
        return None
    if not config.kiro_autostart:
        config._kiro_service_note = (
            "service not reachable; start it or pass --kiro-autostart"
        )
        return None
    try:
        split = urllib.parse.urlsplit(config.kiro_upstream)
        host = split.hostname or ""
        port = split.port or 8000
    except ValueError:
        config._kiro_service_note = "autostart refused: unparsable Kiro upstream URL"
        return None
    if not _is_loopback_host(host):
        config._kiro_service_note = (
            f"autostart refused: upstream {host or config.kiro_upstream} is not loopback"
        )
        return None
    server_dir = config.kiro_server_dir
    entry = os.path.join(server_dir, "main.py")
    if not os.path.isfile(entry):
        config._kiro_service_note = (
            f"autostart failed: main.py not found in {server_dir}"
        )
        return None
    print(f"Kiro Gateway not reachable; autostarting from {server_dir}...")
    log_path = os.path.join(server_dir, "autostart.log")
    try:
        log_handle = open(log_path, "ab")
    except OSError:
        log_handle = None
    child: subprocess.Popen[Any] | None = None
    try:
        popen_kwargs: dict[str, Any] = {
            "cwd": server_dir,
            "stdout": log_handle or subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "env": _child_environment(),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kwargs["start_new_session"] = True
        child = subprocess.Popen(
            [_kiro_server_python(server_dir), "main.py", "--port", str(port)],
            **popen_kwargs,
        )
    except OSError as error:
        config._kiro_service_note = f"autostart failed: {error}"
        return None
    finally:
        if log_handle is not None:
            log_handle.close()
    assert child is not None
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if child.poll() is not None:
                child.wait()
                config._kiro_service_note = (
                    f"autostart failed: service exited during startup; see {log_path}"
                )
                return None
            if _kiro_service_healthy(config.kiro_upstream, config.kiro_api_key):
                config._kiro_service_note = f"service autostarted (pid {child.pid})"
                return child
            time.sleep(_KIRO_AUTOSTART_POLL)
    except BaseException:
        _stop_child(child)
        raise
    _stop_child(child)
    config._kiro_service_note = (
        f"autostart failed: service did not answer within {timeout:g}s; see {log_path}"
    )
    return None


def _print_provider_status(config: Config, provider: str) -> None:
    """One startup line per provider: Checking <Name>... <status>.

    Green when auth is usable, red with the reason when the route will
    degrade to no models. Plain text when piped or NO_COLOR is set.
    """
    ok, detail = _credential_status(config, provider)
    if ok:
        status = _paint(f"Found Auth! ({detail})", "32")
    else:
        status = _paint(f"No auth ({detail})", "31")
    print(f"Checking {provider.capitalize()}... {status}")


def _port_is_busy(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def launch_main(
    pre_argv: list[str], post_argv: list[str] | None = None
) -> int:
    """Run the subscription launcher: gateway plus Muse as one command.

    Server options go before the `launch` token, launcher options
    (--port, --check-only, --local-*, --env-file, --muse-bin) right after it,
    and everything else passes through to `muse`. Unlike `run`, no placeholder
    Muse key is set, so Muse subscription authentication is forwarded.
    """
    try:
        launch_opts, muse_args = _split_launch_args(list(post_argv or []))
    except ProviderError as error:
        print(f"muse-gateway launch: {error}", file=sys.stderr)
        return 2
    if launch_opts.help:
        print(LAUNCH_USAGE)
        return 0
    try:
        _env_path, _env_count = _load_config_file(pre_argv, explicit=launch_opts.env_file)
    except ProviderError as error:
        print(f"muse-gateway launch: {error}", file=sys.stderr)
        return 2
    print(f"config: loaded {_env_count} setting(s) from {_env_path}" if _env_path else "config: no env file",
          file=sys.stderr, flush=True)
    # Post-launch flags are explicit CLI values, not post-validation patches.
    parser_argv = list(pre_argv) if pre_argv else ["auto"]
    parser_argv.extend(launch_opts.server_flags)
    for flag, attribute in _LAUNCH_VALUE_OPTIONS.items():
        if flag in {"--env-file", "--muse-bin"}:
            continue
        value = getattr(launch_opts, attribute, None)
        if value is not None:
            parser_argv.append(f"{flag}={value}")
    if getattr(launch_opts, "kiro_autostart", False):
        parser_argv.append("--kiro-autostart")
    try:
        server_args = parse_args(parser_argv)
    except SystemExit as exited:
        return int(exited.code or 2)
    # Launch always serves the unified catalog, like the source wrappers. A
    # provider-specific --model pin cannot survive that, so reject it (P1-11).
    if server_args.model:
        print("muse-gateway launch: --model pins a provider-specific server mode, but launch "
              "always serves the unified catalog; choose the model in Muse with /model.",
              file=sys.stderr)
        return 2
    server_args.provider = "auto"
    server_args.mode = "auto"
    if server_args.meta_api_key:
        print("muse-gateway launch: META_API_KEY/--meta-api-key conflicts with subscription launch; unset the static key or use the manual API-billing server workflow.", file=sys.stderr)
        return 2
    try:
        config = Config(server_args)
    except (ProviderError, ValueError) as error:
        print(f"muse-gateway launch: {error}", file=sys.stderr)
        return 2

    if config.kiro_requested and _port_is_busy(config.host, config.port):
        print(
            f"error: port {config.port} is occupied. Kiro settings require a new "
            "gateway; choose --port. Existing servers are not reused or stopped.",
            file=sys.stderr,
        )
        return 1

    local_enabled = config.local_upstream_explicit
    if not local_enabled:
        print("local disabled: no MUSE_GATEWAY_LOCAL_UPSTREAM; skipping local-backend preflight.", file=sys.stderr)

    if not launch_opts.check_only:
        muse_bin = (
            launch_opts.muse_bin
            or shutil.which("muse")
            or shutil.which("muse.cmd")
        )
        if not muse_bin:
            print("error: muse not found on PATH", file=sys.stderr)
            return 1
    else:
        muse_bin = launch_opts.muse_bin or "muse"
    print("config: local " + ("configured (optional route key only)" if local_enabled else "disabled"))
    if local_enabled:
        ok, detail = _credential_status(config, "local")
        if not ok:
            print(f"local: not ready ({detail})", file=sys.stderr)
            print(
                "error: local route not ready (see doctor output above)",
                file=sys.stderr,
            )
            return 1
    _print_provider_status(config, "codex")
    _print_provider_status(config, "claude")
    print(
        "Checking Meta... "
        + _paint("Subscription login (OAuth forwarding)", "32")
    )
    _print_provider_status(config, "devin")
    # Signal handling and cleanup scope start before autostart, so a signal
    # during the startup window cannot orphan the detached child (P1-10).
    previous_signal_handlers: dict[int, Any] = {}
    child: subprocess.Popen[Any] | None = None
    kiro_child: subprocess.Popen[Any] | None = None
    server: MuseShimServer | None = None
    server_thread: threading.Thread | None = None
    if threading.current_thread() is threading.main_thread():
        def handle_launcher_signal(signum: int, _frame: Any) -> None:
            raise _LauncherSignal(signum)

        launcher_signals = tuple(
            signum
            for signum in (getattr(signal, "SIGHUP", None), signal.SIGTERM)
            if signum is not None
        )
        for signum in launcher_signals:
            previous_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_launcher_signal)
    box: dict[str, Any] = {}
    try:
        return _launch_lifecycle(config, launch_opts, muse_args, muse_bin, box)
    except _LauncherSignal as received:
        return 128 + received.signum
    finally:
        for signum in previous_signal_handlers:
            signal.signal(signum, signal.SIG_IGN)
        kiro_child = box.get("kiro_child")
        child = box.get("child")
        server = box.get("server")
        server_thread = box.get("server_thread")
        owned_children = [owned for owned in (child, kiro_child) if owned is not None]
        if owned_children:
            _stop_children(owned_children)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)


def _launch_lifecycle(
    config: Config, launch_opts: argparse.Namespace, muse_args: list[str],
    muse_bin: str, box: dict[str, Any],
) -> int:
    """Body of launch_main; every owned resource is published to ``box``
    as soon as it exists so the caller's cleanup sees it on any exit."""
    box["kiro_child"] = ensure_kiro_gateway(config)
    _print_provider_status(config, "kiro")

    base_url = f"http://{config.host}:{config.port}"
    server: MuseShimServer | None = None
    if _port_is_busy(config.host, config.port):
        # Another terminal got here first: attach to its gateway when it is
        # one of ours instead of failing. (The explicit probe matters: the
        # server sets SO_REUSEADDR, which lets a second bind succeed on
        # Windows, so try-bind alone cannot detect the collision there.)
        if config.kiro_requested or not _existing_shim_is_compatible(
            base_url, "auto", config.claude_transport
        ):
            print(
                f"error: port {config.port} is already in use (another gateway?). "
                "Stop it or use --port.",
                file=sys.stderr,
            )
            return 1
        print(f"muse-gateway reusing existing gateway on {base_url}")
        if (
            launch_opts.local_upstream is not None
            or launch_opts.local_api_key is not None
        ):
            print(
                "warning: reusing the running gateway; this terminal's "
                "--local-* settings are ignored while it runs.",
                file=sys.stderr,
            )
        if any(
            getattr(launch_opts, option, None) is not None
            for option in (
                "devin_upstream",
                "devin_api_key",
            )
        ):
            print(
                "warning: reusing the running gateway; this terminal's "
                "--devin-* settings are ignored while it runs.",
                file=sys.stderr,
            )
    else:
        try:
            server = MuseShimServer((config.host, config.port), config)
        except OSError as error:
            print(
                f"muse-gateway could not listen on {base_url}: {error}",
                file=sys.stderr,
            )
            return 1
        box["server"] = server
        base_url = f"http://{config.host}:{server.server_port}"
        server_thread = box["server_thread"] = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
    health = _wait_for_health(base_url)
    if health is None:
        print(f"error: gateway did not come up on {base_url}", file=sys.stderr)
        return 1
    print(f"gateway up: provider {health.get('provider', '?')} on {base_url}")
    if server is None:
        print("diagnostics: logged by the running gateway (owned by the running gateway; "
              "this terminal opened no log)")
    else:
        print(f"diagnostics: {config.log_file} (rotating; no prompt or response content)")
    if launch_opts.check_only:
        try:
            with urllib.request.urlopen(
                base_url + "/muse-code/models", timeout=10
            ) as response:
                catalog = _read_bounded_json(response, limit=CATALOG_JSON_RESPONSE_LIMIT)
            ids = [item.get("id", "?") for item in catalog.get("data", [])]
        except Exception as error:
            print(f"error: catalog check failed: {error}", file=sys.stderr)
            return 1
        print(f"catalog ({len(ids)} entries):")
        for model_id in ids:
            print(f"  {model_id}")
        return 0
    extra: list[str] = []
    if "--provider" not in muse_args and not any(
        item.startswith("--provider=") for item in muse_args
    ):
        extra = ["--provider", "meta", "--base-url", base_url]
    environment = _child_environment()
    environment.pop("META_API_KEY", None)

    try:
        child = box["child"] = subprocess.Popen(
            [muse_bin, *extra, *muse_args], env=environment
        )
        return int(child.wait())
    except FileNotFoundError:
        print("muse-gateway could not find `muse` on PATH.", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        return 130


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_removed_devin_config(raw_argv)
        _reject_retired_flags(raw_argv)
    except ProviderError as error:
        print(f"muse-gateway: {error}", file=sys.stderr)
        return 2
    launch_index = _launch_command_index(raw_argv)
    if launch_index is not None:
        return launch_main(raw_argv[:launch_index], raw_argv[launch_index + 1:])
    run_index = _run_command_index(raw_argv)
    gateway_argv = raw_argv[:run_index] if run_index is not None else raw_argv
    if not any(token in {"-h", "--help", "--version"} for token in gateway_argv):
        try:
            _env_path, _env_count = _load_config_file(gateway_argv)
        except ProviderError as error:
            print(f"muse-gateway: {error}", file=sys.stderr)
            return 2
        print(f"config: loaded {_env_count} setting(s) from {_env_path}" if _env_path else "config: no env file",
              file=sys.stderr, flush=True)
    if run_index is not None:
        args = parse_args(gateway_argv)
        config = Config(args)
        return run_muse(config, raw_argv[run_index + 1:])
    args = parse_args(raw_argv)
    if args.models:
        config = Config(args)
        return run_models(config, provider=args.models_provider)
    if args.mode == "doctor":
        config = Config(args)
        config.doctor_provider = args.doctor_provider
        return run_doctor(config)
    config = Config(args)
    if not _is_loopback_host(config.host):
        print(
            "WARNING: muse-gateway is listening on a non-loopback address; "
            "remote clients can reach the proxy.",
            file=sys.stderr,
            flush=True,
        )
    base_url = f"http://{config.host}:{config.port}"
    stopping: list[str] = []
    server: Any = None

    def stop_on_signal(signum: int, _frame: Any) -> None:
        # serve_forever runs on this thread: shut down from a helper thread.
        stopping.append(signal.Signals(signum).name)
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()

    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        # Install before binding so a ready port can never precede the
        # stop-signal handlers. SIGHUP does not exist on Windows.
        for signum in (getattr(signal, "SIGHUP", None), signal.SIGTERM):
            if signum is not None:
                previous_handlers[signum] = signal.signal(signum, stop_on_signal)
    try:
        server = MuseShimServer((config.host, config.port), config)
    except OSError as error:
        # Same plain-language taxonomy as launch (P1-15); exit 1, no traceback.
        log_dir = os.path.dirname(config.log_file)
        if getattr(error, "filename", None) and os.path.abspath(str(error.filename)) in {
            config.log_file, log_dir,
        }:
            print(f"muse-gateway: cannot open log file {config.log_file}: {error.strerror or error}. "
                  "Pass a writable --log-file.", file=sys.stderr)
        elif _port_is_busy(config.host, config.port) and _existing_shim_is_compatible(
            base_url, config.provider, config.claude_transport
        ):
            print(f"muse-gateway: a compatible gateway is already serving {base_url}; "
                  "use it, or stop it before starting another.", file=sys.stderr)
        else:
            print(f"muse-gateway: could not listen on {base_url}: {error.strerror or error}. "
                  "Stop the other process or use --port.", file=sys.stderr)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        return 1
    print(
        f"muse-gateway {config.provider} listening on http://{config.host}:{server.server_port} "
        f"-> {_server_destination(config)}",
        file=sys.stderr,
        flush=True,
    )
    try:
        if stopping:
            threading.Thread(target=server.shutdown, daemon=True).start()
        server.serve_forever()
    except KeyboardInterrupt:
        stopping.append("SIGINT")
    finally:
        server.close_inflight()
        closed = server.closed_inflight_total
        server.server_close()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    if stopping:
        print(f"muse-gateway: stopped on {stopping[0]} (closed {closed} in-flight request(s)).",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
