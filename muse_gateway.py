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
  or OMP credential store; the CLI is not used for inference.
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
import os
import re
import secrets
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
from http.client import HTTPException, IncompleteRead
from typing import Any, Iterable

from muse_providers import (
    AuthError,
    CLAUDE_OAUTH_BETAS,
    CLAUDE_OAUTH_CLIENT_ID,
    DEFAULT_CLAUDE_BASE_URL,
    DEFAULT_CODEX_BASE_URL,
    CODEX_OAUTH_CLIENT_ID,
    DEFAULT_CLAUDE_ENDPOINT,
    DEFAULT_CODEX_ENDPOINT,
    ProviderError,
    anthropic_json_to_responses,
    claude_headers,
    convert_anthropic_sse,
    convert_anthropic_stream,
    codex_headers,
    correct_codex_effort,
    credentials_usable,
    discover_claude_credentials,
    discover_codex_credentials,
    demote_codex_encrypted_reasoning,
    endpoint_for,
    prepare_codex_request,
    refresh_claude_oauth,
    refresh_codex_oauth,
    resolve_claude_credentials,
    resolve_codex_credentials,
    response_sse_line,
    responses_to_anthropic,
    translate_anthropic_sse,
    validate_claude_endpoint,
    validate_claude_token_url,
    validate_codex_endpoint,
    validate_codex_token_url,
)
import muse_devin
import muse_kiro
from muse_capture import CaptureSession, CapturingWriter, current_capture, encoded_secret_forms


DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024
# Kiro's max_tokens is only a hint upstream, not a response-size guarantee.
KIRO_JSON_RESPONSE_LIMIT = 16 * 1024 * 1024
KIRO_REJECTED_BODY_LIMIT = 64 * 1024
KIRO_REJECTED_BODY_TIMEOUT = 0.2
DEFAULT_CODEX_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
)
DEFAULT_CLAUDE_MODELS = (
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5-20251001",
)
# Bundled Meta catalog limits; output limits are unverified fallback values.
# See README.md for discovery and capability limitations.
DEFAULT_META_UPSTREAM = "https://api.meta.ai/v1"
# Devin (Codeium Cascade) Connect RPC base. Paths are service routes, not
# Responses endpoints, so provider_url("devin") returns this verbatim.
DEFAULT_DEVIN_UPSTREAM = "https://server.codeium.com"
DEFAULT_META_MODELS = (
    "muse-spark-1.3",
    "muse-spark-1.2",
)
# Muse documents these as the Meta reasoning-effort levels (`muse exec
# --help`: "Meta reasoning effort: none|minimal|low|medium|high|xhigh|max|ultra
# (default: high)"). Meta rows carry no per-model discovery evidence for
# efforts, so the documented set is advertised for every meta/ row.
META_REASONING_EFFORTS = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)
META_DEFAULT_REASONING_EFFORT = "high"
META_MODEL_LIMITS = {
    "muse-spark-1.3": (1_048_576, 32_768),
    "muse-spark-1.3-contributor": (1_048_576, 32_768),
    "muse-spark-1.2": (1_048_576, 32_768),
    "muse-spark-1.2-contributor": (1_048_576, 32_768),
    "muse-spark-1.1": (1_048_576, 32_768),
}
META_DEFAULT_LIMITS = (1_048_576, 32_768)
# Fallbacks when the local catalog does not supply per-model token limits.
LOCAL_DEFAULT_LIMITS = (128_000, 32_768)
CLAUDE_MODEL_LIMITS = {
    "claude-fable-5": (1_000_000, 128_000),
    "claude-opus-5": (1_000_000, 128_000),
    "claude-sonnet-5": (1_000_000, 128_000),
    "claude-haiku-4-5-20251001": (200_000, 64_000),
}
CODEX_MODELS_ENDPOINT = (
    "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0"
)
MODEL_CATALOG_TTL_SECONDS = 3600
MODEL_CATALOG_RETRY_SECONDS = 60


class ShimError(Exception):
    """An error that should be returned to Muse as a client error."""


def _legacy_env_name(name: str) -> str | None:
    """Map a MUSE_GATEWAY_* name to its MUSE_SHIM_* predecessor, if any."""
    if name.startswith("MUSE_GATEWAY_"):
        return "MUSE_SHIM_" + name[len("MUSE_GATEWAY_"):]
    return None


def env_value(name: str, default: str | None = None) -> str | None:
    """Read a MUSE_GATEWAY_* variable, falling back to MUSE_SHIM_*.

    The MUSE_SHIM_* spelling predates the rename and keeps working so
    existing setups do not break; when both spellings are set, the
    MUSE_GATEWAY_* value wins. Presence counts, even for empty strings,
    exactly like os.environ.get.
    """
    if name in os.environ:
        return os.environ[name]
    legacy = _legacy_env_name(name)
    if legacy is not None and legacy in os.environ:
        return os.environ[legacy]
    return default


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        legacy = _legacy_env_name(name)
        if legacy is not None:
            value = os.environ.get(legacy)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_model_list(value: str | None, fallback: tuple[str, ...]) -> list[str]:
    models = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return list(dict.fromkeys(models or fallback))


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
    """Flatten Muse's namespace tool groups into Responses function tools."""
    flattened: list[Any] = []
    names: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            flattened.append(tool)
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                if tool["name"] in names:
                    raise ShimError(f"duplicate tool name {tool['name']!r}")
                names.add(tool["name"])
            continue

        namespace = tool.get("name", "<unnamed>")
        children = tool.get("tools")
        if not isinstance(children, list):
            raise ShimError(f"Muse namespace {namespace!r} has no tools list")

        for child in children:
            if not isinstance(child, dict):
                raise ShimError(f"Muse namespace {namespace!r} contains a non-object tool")
            name = child.get("name")
            if not isinstance(name, str) or not name:
                raise ShimError(f"Muse namespace {namespace!r} contains an unnamed tool")
            if name in names:
                raise ShimError(
                    f"duplicate tool name {name!r} after flattening namespace {namespace!r}"
                )
            names.add(name)
            flattened.append(child)

    return flattened


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
    ``max_output_tokens`` cap still apply - both are explicit user choices.
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

    if config.max_output_tokens is not None:
        requested = transformed.get("max_output_tokens")
        if isinstance(requested, (int, float)) and not isinstance(requested, bool):
            transformed["max_output_tokens"] = min(int(requested), config.max_output_tokens)
        else:
            transformed["max_output_tokens"] = config.max_output_tokens

    if config.portable and not native:
        transformed.pop("include", None)
        transformed.pop("prompt_cache_key", None)
        transformed.pop("reasoning", None)

    return transformed


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
        elif provider == "kiro" and getattr(args, "kiro_upstream", None):
            upstream = str(args.kiro_upstream)
        else:
            upstream = _default_upstream_for(provider)
        self.upstream: str = upstream.rstrip("/")
        self.codex_upstream: str = (
            self.upstream if provider == "codex" else _default_upstream_for("codex")
        ).rstrip("/")
        self.claude_upstream: str = (
            self.upstream if provider == "claude" else _default_upstream_for("claude")
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
        self.local_upstream: str = (
            self.upstream
            if provider == "local"
            else str(
                getattr(args, "local_upstream", None)
                or _default_upstream_for("local")
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
        self.devin_auth_db: str | None = getattr(args, "devin_auth_db", None)
        self.devin_credential_id: str | None = getattr(
            args, "devin_credential_id", None
        )
        self.kiro_api_key: str | None = getattr(args, "kiro_api_key", None)
        self.kiro_models: list[str] = _parse_model_list(
            getattr(args, "kiro_models", None), ()
        )
        self.kiro_context_limit = int(getattr(
            args, "kiro_context_limit", muse_kiro.DEFAULT_CONTEXT_LIMIT
        ))
        self.kiro_output_limit = int(getattr(
            args, "kiro_output_limit", muse_kiro.DEFAULT_OUTPUT_LIMIT
        ))
        if self.kiro_context_limit <= 0 or self.kiro_output_limit <= 0:
            raise ValueError("Kiro context and output budgets must be positive integers")
        self.kiro_requested = bool(
            provider == "kiro" or self.kiro_api_key
            or getattr(args, "kiro_upstream", None) or self.kiro_models
            or self.kiro_context_limit != muse_kiro.DEFAULT_CONTEXT_LIMIT
            or self.kiro_output_limit != muse_kiro.DEFAULT_OUTPUT_LIMIT
        )
        if self.kiro_requested and getattr(args, "max_output_tokens", None) is not None:
            if int(args.max_output_tokens) <= 0:
                raise ValueError("Kiro max output budget must be a positive integer")
        self.kiro_upstream = muse_kiro.normalize_upstream(
            self.upstream if provider == "kiro" else (
                getattr(args, "kiro_upstream", None) or _default_upstream_for("kiro")
            )
        )
        if self.kiro_api_key:
            self.kiro_api_key = muse_kiro.validate_proxy_key(self.kiro_api_key)
        # Gateway fork: model discovery runs only when the local upstream was
        # explicitly configured (flag or env), so an unconfigured install
        # advertises no local/ entries and makes no discovery calls.
        self.local_upstream_explicit: bool = bool(
            getattr(args, "local_upstream", None)
        )
        self.api_key: str | None = getattr(args, "api_key", None)
        self.meta_api_key: str | None = getattr(args, "meta_api_key", None)
        self.local_api_key: str | None = getattr(args, "local_api_key", None)
        self.access_token: str | None = getattr(args, "access_token", None)
        self.model: str | None = getattr(args, "model", None)
        self.portable: bool = bool(getattr(args, "portable", False))
        self.max_output_tokens: int | None = getattr(args, "max_output_tokens", None)
        self.insecure_upstream: bool = bool(getattr(args, "insecure_upstream", False))
        self.allow_oauth_host_override: bool = bool(
            getattr(args, "allow_oauth_host_override", False)
        )
        self.allow_non_loopback: bool = bool(getattr(args, "allow_non_loopback", False))
        self.verbose: bool = bool(getattr(args, "verbose", False))
        self.log_file: str = os.path.abspath(
            getattr(args, "log_file", None) or "muse-gateway.log"
        )
        self.capture_dir: str | None = getattr(args, "capture_dir", None)
        self.timeout: float = float(getattr(args, "timeout", 600.0) or 600.0)
        self.max_body_bytes: int = int(
            getattr(args, "max_body_bytes", DEFAULT_MAX_BODY_BYTES)
        )
        if self.max_body_bytes <= 0:
            raise ValueError("max incoming body size must be a positive integer")
        self.refresh_skew_seconds: int = int(getattr(args, "refresh_skew", 120) or 120)
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
        self.doctor_provider: str | None = getattr(args, "doctor_provider", None)
        self.codex_models: list[str] = _parse_model_list(
            getattr(args, "codex_models", None), DEFAULT_CODEX_MODELS
        )
        self.claude_models: list[str] = _parse_model_list(
            getattr(args, "claude_models", None), DEFAULT_CLAUDE_MODELS
        )
        self.codex_models_explicit: bool = bool(getattr(args, "codex_models", None))
        self.claude_models_explicit: bool = bool(getattr(args, "claude_models", None))
        # Gateway fork: meta ships with defaults (always advertised); local
        # advertises the explicit list when set, else whatever the local
        # upstream reports from its /models endpoint (see discovery below).
        self.meta_models: list[str] = _parse_model_list(
            getattr(args, "meta_models", None), DEFAULT_META_MODELS
        )
        self.meta_models_explicit: bool = bool(getattr(args, "meta_models", None))
        self.local_models: list[str] = _parse_model_list(
            getattr(args, "local_models", None), ()
        )
        # Devin: an explicit list curates discovered logical ids/uids (never
        # fabricated: discovery owns routing metadata).
        self.devin_models: list[str] = _parse_model_list(
            getattr(args, "devin_models", None), ()
        )
        self.devin_models_explicit: bool = bool(getattr(args, "devin_models", None))
        self.model_discovery: bool = bool(getattr(args, "model_discovery", True))
        self._model_catalog_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._kiro_catalog_status = "not checked; inference not validated"
        # Devin roster stash: logical_id -> DevinModel, refreshed with the
        # catalog so requests reuse discovery without extra RPCs.
        self._devin_roster: dict[str, Any] = {}
        self._devin_rosters: dict[str, tuple[dict[str, Any], int]] = {}
        self._devin_dropped_fusion: int = 0
        if not _is_loopback_host(self.host) and not self.allow_non_loopback:
            raise ProviderError(
                f"refusing non-loopback bind address {self.host!r}; "
                "use --allow-non-loopback only when remote access is intentional.",
                code="non_loopback_bind_rejected",
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

        for prefix in ("codex/", "codex:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "codex"}:
                    raise ProviderError(
                        "selected model belongs to Codex.",
                        code="model_provider_mismatch",
                    )
                return "codex", value[len(prefix) :]
        for prefix in ("claude/", "claude:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "claude"}:
                    raise ProviderError(
                        "selected model belongs to Claude.",
                        code="model_provider_mismatch",
                    )
                return "claude", value[len(prefix) :]
        for prefix in ("meta/", "meta:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "meta"}:
                    raise ProviderError(
                        "selected model belongs to Meta.",
                        code="model_provider_mismatch",
                    )
                return "meta", value[len(prefix) :]
        for prefix in ("local/", "local:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "local"}:
                    raise ProviderError(
                        "selected model belongs to the local upstream.",
                        code="model_provider_mismatch",
                    )
                return "local", value[len(prefix) :]
        for prefix in ("devin/", "devin:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "devin"}:
                    raise ProviderError(
                        "selected model belongs to Devin.",
                        code="model_provider_mismatch",
                    )
                return "devin", value[len(prefix) :]

        for prefix in ("kiro/", "kiro:"):
            if value.startswith(prefix):
                if self.provider not in {"auto", "kiro"}:
                    raise ProviderError(
                        "selected model belongs to Kiro.",
                        code="model_provider_mismatch",
                    )
                model = value[len(prefix) :].strip()
                if not model:
                    raise ProviderError("Kiro model name is empty.", code="model_missing")
                return "kiro", model

        if self.provider in {"generic", "codex", "claude", "meta", "local", "devin", "kiro"}:
            return self.provider, value
        # Gateway fork: explicit membership wins over the name heuristics
        # below, so a user-configured local alias is never misrouted.
        # Discovered aliases route the same way (cached via the catalog),
        # so --model <alias> works with or without an explicit list.
        if value in self.local_models or value in self._discovered_local_names():
            return "local", value
        if value in self.meta_models or value.lower().startswith("muse-"):
            return "meta", value
        if value in self.claude_models or value.lower().startswith("claude-"):
            return "claude", value
        if value in self.devin_models:
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
        return {
            model_id[len("local/"):]
            for entry in entries
            if (model_id := str(entry.get("model_id", ""))).startswith("local/")
        }

    def secret_values(self) -> list[str]:
        values: list[str] = []
        for value in (
            self.api_key,
            self.access_token,
            self.meta_api_key,
            self.local_api_key,
            self.devin_api_key,
            self.kiro_api_key,
        ):
            if isinstance(value, str) and value:
                values.append(value)
        return values

    def kiro_effective_output_limit(self) -> int:
        """The same unverified operational budget for catalog and requests."""
        return min(self.kiro_output_limit, self.max_output_tokens) if self.max_output_tokens is not None else self.kiro_output_limit


_PROVIDER_BLURB = {
    "codex": "OpenAI Codex subscription model",
    "claude": "Claude model via Claude Code",
    "meta": "Muse Spark model",
    "local": "Local model through muse-gateway",
    "kiro": "Kiro model through Kiro Gateway",
}


def _compact_tokens(value: int) -> str:
    """Short context size for the /model menu, e.g. 262144 -> "262k"."""
    if value >= 1_000_000:
        return f"{round(value / 1_000_000)}M"
    if value >= 1_000:
        return f"{round(value / 1_000)}k"
    return str(value)


def _catalog_description(
    provider: str, context_limit: int, *, limits_verified: bool = True
) -> str:
    blurb = _PROVIDER_BLURB.get(provider, "Model through muse-gateway")
    description = f"(ctx {_compact_tokens(context_limit)}) {blurb}"
    if not limits_verified:
        description += " · limits unverified"
    return description


def _catalog_entry(
    provider: str,
    model: str,
    *,
    context_limit: int,
    output_limit: int,
    is_default: bool,
    limits_verified: bool = True,
    supports_images: bool | None = None,
    reasoning_efforts: list[str] | None = None,
    default_reasoning_effort: str | None = None,
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
        "display_label": f"{provider_label} · {model}",
        "release_date": "2026-08-06",
        "context_limit": context_limit,
        "output_limit": output_limit,
        "is_current": is_default,
        "is_default": is_default,
        "description": _catalog_description(
            provider, context_limit, limits_verified=limits_verified
        ),
        "limits_verified": limits_verified,
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
        "created": 1785974400,
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
            with urllib.request.urlopen(
                request, timeout=min(config.timeout, 8.0)
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_models = payload.get("models", []) if isinstance(payload, dict) else []
            result = []
            for item in raw_models:
                if not isinstance(item, dict) or not isinstance(item.get("slug"), str):
                    continue
                context, output, verified = _discovery_model_limits(item, 272000, 32768)
                levels = item.get("supported_reasoning_levels")
                # 'ultra' is an orchestration mode, not a Responses wire effort.
                wire_levels = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
                efforts = [
                    level["effort"] for level in (levels if isinstance(levels, list) else [])
                    if isinstance(level, dict) and level.get("effort") in wire_levels
                ]
                result.append({
                    "model": item["slug"].strip(),
                    "context_limit": context, "output_limit": output,
                    "limits_verified": verified,
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
        with urllib.request.urlopen(
            request, timeout=min(config.timeout, 8.0)
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        raw_models = payload.get("data", []) if isinstance(payload, dict) else []
        return [
            {
                "model": item["id"].strip(),
                "context_limit": int(item.get("max_input_tokens") or 200000),
                "output_limit": int(item.get("max_tokens") or 32768),
            }
            for item in raw_models
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item["id"].strip()
        ]
    except (
        AuthError,
        ProviderError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        return []


def _positive_int(value: Any, fallback: int) -> int:
    """Coerce an upstream limit to a positive int, else the fallback."""
    if isinstance(value, bool):
        return fallback
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback
    return number if number > 0 else fallback


def _positive_int_or_none(value: Any) -> int | None:
    """Coerce an upstream limit to a positive int, else None."""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


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
    "max_output_tokens",
    "max_completion_tokens",
    "output_limit",
)


def _discovery_model_limits(
    item: dict[str, Any], context_default: int, output_default: int
) -> tuple[int, int, bool]:
    """Extract (context, output, verified) limits from one catalog entry."""
    context_limit: int | None = None
    for key in _INPUT_LIMIT_KEYS:
        context_limit = _positive_int_or_none(item.get(key))
        if context_limit is not None:
            break
    output_limit: int | None = None
    for key in _OUTPUT_LIMIT_KEYS:
        output_limit = _positive_int_or_none(item.get(key))
        if output_limit is not None:
            break
    verified = context_limit is not None and output_limit is not None
    return (
        context_limit or context_default,
        output_limit or output_default,
        verified,
    )


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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


_DISCOVERY_ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError)


def _parse_openai_style_models(
    payload: Any, context_default: int, output_default: int
) -> list[dict[str, Any]]:
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
        context_limit, output_limit, verified = _discovery_model_limits(
            item, context_default, output_default
        )
        result.append(
            {
                "model": name.strip(),
                "context_limit": context_limit,
                "output_limit": output_limit,
                "limits_verified": verified,
                "supports_images": _discovery_supports_images(item),
            }
        )
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
        models = _parse_openai_style_models(payload, *META_DEFAULT_LIMITS)
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
            context_limit, output_limit, verified = _discovery_model_limits(
                item, *META_DEFAULT_LIMITS
            )
            result.append(
                {
                    "model": name.strip(),
                    "context_limit": context_limit,
                    "output_limit": output_limit,
                    "limits_verified": verified,
                    "supports_images": _discovery_supports_images(item),
                }
            )
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
    context_default, output_default = LOCAL_DEFAULT_LIMITS
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
                "context_limit": context_default,
                "output_limit": output_default,
                "limits_verified": False,
                "supports_images": _discovery_supports_images(item),
            }
        )
    return result


def _discover_local_models(config: Config) -> list[dict[str, Any]]:
    """List models from the local upstream's OpenAI-style /models endpoint.

    Runs for explicitly configured upstreams; an explicit
    MUSE_GATEWAY_LOCAL_MODELS list still curates which names are advertised,
    but discovered limits enrich the entries that match. Per-model limits are
    read from common upstream keys when present, else LOCAL_DEFAULT_LIMITS
    (marked unverified). A bare Ollama server answers via /api/tags as a
    fallback. Any failure degrades to [], and provider_model_catalog caches
    the result, so a down endpoint cannot slow every catalog fetch.
    """
    if not config.local_upstream_explicit or not config.model_discovery:
        return []
    headers: dict[str, str] = {}
    key = config.local_api_key or config.api_key
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
        models = _parse_openai_style_models(payload, *LOCAL_DEFAULT_LIMITS)
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
        state = muse_devin.credential_state(
            auth_db_path=config.devin_auth_db,
            credential_id=config.devin_credential_id,
            api_key=config.devin_api_key,
        )
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("state") == "ready"


def _devin_cache_key(config: Config, token: str) -> str:
    return "devin:" + hashlib.sha256(token.encode()).hexdigest()


def _devin_catalog_entries(config: Config, access_token=None) -> list[dict[str, Any]]:
    """Devin rows with their own roster stash (uid/routing the generic item
    shape cannot carry). Explicit devin_models curate; fusion pairings are
    dropped inside discovery (option B) with the count stashed for doctor."""
    try:
        token = access_token
        if token is None:
            token, _identity = muse_devin.resolve_access_token(
                auth_db_path=config.devin_auth_db,
                credential_id=config.devin_credential_id,
                api_key=config.devin_api_key,
            )
    except (muse_devin.DevinCredentialError, muse_devin.DevinWireError, OSError, ValueError):
        config._devin_roster = {}
        config._devin_dropped_fusion = 0
        return []
    cache_key = _devin_cache_key(config, token)
    cached = config._model_catalog_cache.get(cache_key)
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        config._devin_roster, config._devin_dropped_fusion = config._devin_rosters[cache_key]
        return cached[1]
    dropped = 0
    if config.model_discovery:
        try:
            models, dropped, _warnings = muse_devin.discover_models(
                token, config.devin_upstream
            )
        except (muse_devin.DevinWireError, OSError, ValueError):
            models = []
        if not models:
            models = muse_devin.seed_models(config.devin_upstream)
    else:
        models = muse_devin.seed_models(config.devin_upstream)
    if config.devin_models_explicit:
        wanted = set(config.devin_models)
        models = [
            model
            for model in models
            if model.logical_id in wanted or model.uid in wanted
        ]
    roster = {}
    for model in models:
        roster[model.logical_id] = model
        roster.setdefault(model.uid, model)
    config._devin_roster = roster
    config._devin_dropped_fusion = dropped
    ordered = sorted(models, key=lambda model: not model.is_default)
    entries = [
        _catalog_entry(
            "devin",
            model.logical_id,
            context_limit=model.context_limit,
            output_limit=min(model.output_limit, config.max_output_tokens)
            if config.max_output_tokens
            else model.output_limit,
            is_default=index == 0,
            limits_verified=not model.unverified,
            supports_images=model.supports_images,
            reasoning_efforts=model.efforts or None,
            default_reasoning_effort=model.default_effort,
        )
        for index, model in enumerate(ordered)
    ]
    if not entries and config.model_discovery:
        stamped = time.time() - (
            MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS
        )
    else:
        stamped = time.time()
    config._devin_rosters[cache_key] = (roster, dropped)
    config._model_catalog_cache[cache_key] = (stamped, entries)
    return entries


def _kiro_catalog_entries(config: Config) -> list[dict[str, Any]]:
    """Names from Kiro Gateway, never account entitlement or capacity evidence."""
    if not config.kiro_api_key:
        return []
    cached = config._model_catalog_cache.get("kiro")
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        return cached[1]
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
    names = config.kiro_models or names
    entries = [
        _catalog_entry(
            "kiro", name,
            context_limit=config.kiro_context_limit,
            output_limit=config.kiro_effective_output_limit(),
            is_default=index == 0,
            limits_verified=False,
            supports_images=False,
        )
        for index, name in enumerate(names)
    ]
    stamped = time.time()
    if config.model_discovery and not discovery_ok:
        stamped -= MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS
    config._model_catalog_cache["kiro"] = (stamped, entries)
    return entries


def provider_model_catalog(
    config: Config, provider: str, *, authorization: str | None = None
) -> list[dict[str, Any]]:
    if provider == "kiro":
        return _kiro_catalog_entries(config)
    if provider == "devin":
        if not _provider_is_available(config, provider):
            return []
        return _devin_catalog_entries(config)
    cached = config._model_catalog_cache.get(provider)
    if cached and time.time() - cached[0] < MODEL_CATALOG_TTL_SECONDS:
        return cached[1]
    if provider in {"codex", "claude", "devin"} and not _provider_is_available(
        config, provider
    ):
        # Never advertise rows that could only fail at request time. Not
        # cached, so a fresh login/CLI install appears on the next fetch.
        return []

    discovered = _discover_provider_models(
        config, provider, authorization=authorization
    )
    if provider == "codex":
        configured = config.codex_models
        configured_explicit = config.codex_models_explicit
    elif provider == "claude":
        configured = config.claude_models
        configured_explicit = config.claude_models_explicit
    elif provider == "meta":
        configured = config.meta_models
        configured_explicit = config.meta_models_explicit
    elif provider == "local":
        configured = config.local_models
        configured_explicit = False
    else:
        raise ProviderError(
            f"cannot build a model catalog for unknown provider {provider!r}.",
            code="unknown_catalog_provider",
        )
    # Retain the legacy native fallback labels; they are not live verification.
    trusted_fallback = provider in {"codex", "claude"}
    fallbacks = {}
    for model in configured:
        if provider == "claude":
            context_limit, output_limit = CLAUDE_MODEL_LIMITS.get(
                model, (200_000, 32_768)
            )
        elif provider == "meta":
            context_limit, output_limit = META_MODEL_LIMITS.get(
                model, META_DEFAULT_LIMITS
            )
        elif provider == "local":
            context_limit, output_limit = LOCAL_DEFAULT_LIMITS
        else:
            context_limit, output_limit = 272_000, 32_768
        fallbacks[model] = {
            "model": model,
            "context_limit": context_limit,
            "output_limit": output_limit,
            "limits_verified": trusted_fallback,
            "supports_images": provider in {"codex", "claude", "meta"},
        }
    if (provider == "local" and configured) or (
        provider == "meta" and configured_explicit
    ):
        # An explicit list curates names; discovery only enriches limits.
        # (Local configured models are always explicit; Meta ships bundled
        # defaults, so only an explicit MUSE_GATEWAY_META_MODELS curates.)
        defaults = dict(fallbacks)
        for item in discovered:
            if item["model"] in defaults:
                defaults[item["model"]] = item
    else:
        defaults = fallbacks if not discovered or configured_explicit else {}
        for item in discovered:
            defaults[item["model"]] = item

    entries = [
        _catalog_entry(
            provider,
            item["model"],
            context_limit=item["context_limit"],
            output_limit=min(item["output_limit"], config.max_output_tokens)
            if config.max_output_tokens
            else item["output_limit"],
            is_default=index == 0,
            limits_verified=item.get("limits_verified", trusted_fallback),
            supports_images=False if provider == "claude" and config.claude_transport == "cli" else item.get(
                "supports_images", provider in {"codex", "claude", "meta"}
            ),
            reasoning_efforts=item.get("reasoning_efforts")
            or (list(META_REASONING_EFFORTS) if provider == "meta" else None),
            default_reasoning_effort=item.get("default_reasoning_effort")
            or (META_DEFAULT_REASONING_EFFORT if provider == "meta" else None),
        )
        for index, item in enumerate(defaults.values())
    ]
    if not discovered and config.model_discovery:
        # Discovery was attempted and came back empty: backdate the entry so
        # it expires after the short retry interval instead of serving stale
        # fallbacks for the full hour. A fresh login or recovered upstream is
        # picked up on the next fetch without a gateway restart.
        stamped = time.time() - (
            MODEL_CATALOG_TTL_SECONDS - MODEL_CATALOG_RETRY_SECONDS
        )
    else:
        stamped = time.time()
    config._model_catalog_cache[provider] = (stamped, entries)
    return entries


def _codex_model_efforts(config: Config, model: Any) -> list[str]:
    """Model-specific Codex wire efforts from the catalog, else [] (unknown).

    An empty result means "no evidence" (bundled fallbacks carry none), in
    which case callers must pass the effort through untouched. Never fails:
    discovery or lookup problems degrade to unknown rather than breaking
    forwarding.
    """
    try:
        entries = provider_model_catalog(config, "codex")
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


def _claude_cli_output_schema(tool_names: list[str]) -> dict[str, Any]:
    name_schema: dict[str, Any] = {"type": "string"}
    if tool_names:
        name_schema["enum"] = tool_names
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "maxItems": 8 if tool_names else 0,
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


def run_claude_cli(
    body: dict[str, Any], config: Config
) -> tuple[dict[str, Any], Any]:
    """Execute one Muse turn through Claude Code print mode."""
    model = str(body.get("model") or "").strip()
    if not model:
        raise ProviderError("Claude CLI request is missing a model.", code="model_missing")
    payload, tool_names = responses_to_anthropic(
        body,
        oauth=False,
        max_output_tokens=config.max_output_tokens,
    )
    available_tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
    available_names = [
        str(tool.get("name"))
        for tool in available_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]
    schema = _claude_cli_output_schema(available_names)
    command = [
        config.claude_cli,
        "-p",
        "--safe-mode",
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

    environment = os.environ.copy()
    # Claude Code refuses accidental recursive launches. The gateway is an
    # intentional transport boundary, so do not inherit parent-agent markers.
    environment.pop("CLAUDECODE", None)
    environment.pop("CLAUDE_CODE_ENTRYPOINT", None)
    try:
        completed = subprocess.run(
            command,
            input=_claude_cli_prompt(payload),
            text=True,
            capture_output=True,
            timeout=config.timeout,
            env=environment,
            check=False,
        )
    except FileNotFoundError as error:
        raise ProviderError(
            f"Claude CLI executable {config.claude_cli!r} was not found. Install Claude Code or set MUSE_GATEWAY_CLAUDE_CLI.",
            status=502,
            code="claude_cli_missing",
        ) from error
    except subprocess.TimeoutExpired as error:
        raise ProviderError(
            f"Claude CLI exceeded the {config.timeout:g}s timeout.",
            status=504,
            code="claude_cli_timeout",
        ) from error

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
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
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
    return result


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
    first = error.read(65536)
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
    }


def _ssl_context(config: Config) -> ssl.SSLContext | None:
    if config.insecure_upstream:
        return ssl._create_unverified_context()  # noqa: SLF001 - explicit opt-in
    return None


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
                chunk = reader()
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


def _write_upstream_body(handler: BaseHTTPRequestHandler, upstream: Any, *, stream: bool, content_type: str) -> None:
    if stream:
        frame = bytearray()
        event_name = b""
        observe = getattr(handler, "_log_event", None)

        def observe_frame() -> None:
            try:
                event = json.loads(frame)
            except (ValueError, UnicodeError):
                event = None
            failed = event_name in (b"error", b"response.failed") or (
                isinstance(event, dict) and event.get("type") in ("error", "response.failed")
            )
            if failed and callable(observe):
                handler._stream_failed = True  # type: ignore[attr-defined]
                observe("upstream_stream_error", **_upstream_diagnostic(bytes(frame)))
            frame.clear()

        for chunk in _iter_upstream_chunks(upstream, content_type=content_type):
            if callable(observe) and "text/event-stream" in content_type.lower():
                # Observe bounded SSE frames without changing forwarded bytes.
                if chunk.strip() == b"":
                    observe_frame()
                    event_name = b""
                elif chunk.startswith(b"event:"):
                    event_name = chunk[6:].strip()[:64]
                elif chunk.startswith(b"data:") and len(frame) < 65536:
                    frame.extend((chunk[5:].strip() + b"\n")[:65536 - len(frame)])
            handler.wfile.write(chunk)
            handler.wfile.flush()
        if frame or event_name:
            observe_frame()
        return
    payload = upstream.read()
    capture = current_capture()
    if capture is not None:
        capture.bytes("upstream.body", payload.encode() if isinstance(payload, str) else payload)
        capture.event("upstream.eof")
    if payload:
        handler.wfile.write(payload.encode("utf-8") if isinstance(payload, str) else payload)
        handler.wfile.flush()


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


class MuseShimHandler(BaseHTTPRequestHandler):
    server_version = "muse-gateway/0.5.0"
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self._capture_request = None
        if getattr(self.server, "capture_session", None) is not None:
            self.wfile = CapturingWriter(self.wfile)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        finally:
            capture = getattr(self, "_capture_request", None)
            if capture is not None:
                capture.finish(status=getattr(self, "_response_status", None),
                               stream_failed=getattr(self, "_stream_failed", False),
                               close_connection=self.close_connection)
                self._capture_request = None

    @property
    def config(self) -> Config:
        return self.server.config  # type: ignore[attr-defined]

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
        known_secrets = self.config.secret_values()
        known_secrets.append(self.headers.get("Authorization", ""))
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
        code = getattr(error, "code", "shim_error")
        kiro_fields = muse_kiro.error_fields(error) if isinstance(error, ProviderError) else {}
        extra = dict(getattr(self, "_upstream_diagnostic", {}))
        if "code" in extra:
            # The upstream error code must not collide with our own `code=`
            # (that TypeError killed the error response and the client saw
            # a dropped connection as "malformed response data").
            extra["upstream_code"] = extra.pop("code")
        self._log_event("provider_error", status=error_status, code=code,
                        **extra, **({"metadata": kiro_fields["metadata"]} if kiro_fields else {}))
        if getattr(self, "_response_started", False):
            # A partial response cannot be replaced by a second HTTP response.
            self.close_connection = True
            return
        message = _redact_bytes(
            str(error).encode("utf-8"),
            [*self.config.secret_values(), self.headers.get("Authorization", "")],
        ).decode("utf-8")
        try:
            retry = kiro_fields.get("metadata", {}).get("retry_after")
            self.send_json(
                int(error_status),
                {"error": {"message": message, "type": code,
                           "request_id": getattr(self, "_request_id", None), **kiro_fields}},
                headers={"Retry-After": retry["value"]} if retry else None,
            )
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Muse went away (ESC/cancel) while we answered. Stay quiet.
            return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "muse-gateway",
                    "provider": self.config.provider,
                    "claude_transport": self.config.claude_transport,
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
                # Model switches re-read the menu: let an explicit no-cache
                # catalog fetch recall discovery instead of serving the TTL cache.
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
            if self.config.provider == "kiro":
                entries = provider_model_catalog(self.config, "kiro")
                if self.config.model and self.config.kiro_api_key:
                    _, model = self.config.resolve_model(self.config.model)
                    entries = [_catalog_entry(
                        "kiro", model,
                        context_limit=self.config.kiro_context_limit,
                        output_limit=self.config.kiro_effective_output_limit(),
                        is_default=True, limits_verified=False, supports_images=False,
                    )]
                self.send_json(200, muse_catalog_response(entries))
                return
            if self.config.provider in {"codex", "claude", "meta", "devin", "local"} and not self.config.model:
                entries = provider_model_catalog(
                    self.config, self.config.provider, authorization=inbound_auth
                )
                if entries:
                    self.send_json(200, muse_catalog_response(entries))
                    return
                # else fall through: a route with no configured models (only
                # possible for local/) advertises a single upstream entry.
            model = self.config.model or {
                "codex": "codex-model-from-request",
                "claude": "claude-model-from-request",
                "meta": "meta-model-from-request",
                "devin": "devin-model-from-request",
                "local": "local-model-from-request",
            }.get(self.config.provider, "upstream-model")
            provider = (
                self.config.provider
                if self.config.provider in {"codex", "claude", "meta", "devin", "local"}
                else "generic"
            )
            self.send_json(
                200,
                muse_catalog_response(
                    [
                        {
                            "model_id": model,
                            "display_label": model,
                            "release_date": "2026-08-06",
                            "context_limit": 200000,
                            "output_limit": self.config.max_output_tokens or 32768,
                            "is_current": True,
                            "is_default": True,
                            "description": _catalog_description(provider, 200000),
                        }
                    ]
                ),
            )
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
        raw = self.rfile.read(content_length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ShimError("request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise ShimError("request body must be a JSON object")
        capture = current_capture()
        if capture is not None:
            capture.json("incoming", body)
            capture.json("incoming.headers", dict(self.headers))
            capture.bytes("incoming.body", raw)
        return body

    def do_POST(self) -> None:
        self._request_id = secrets.token_hex(6)
        self._response_started = False
        self._stream_failed = False
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
            self._diagnostic["provider"] = provider
            if current_capture() is not None:
                current_capture().json("transformed", transformed)
                current_capture().event("route.selected", provider=provider, model=upstream_model)
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
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self._log_event("client_disconnected")
            self.close_connection = True
        except (ProviderError, ShimError) as error:
            self.send_provider_error(error)
        except urllib.error.HTTPError as error:
            try:
                detail = _upstream_diagnostic(_capture_http_error(error))
            finally:
                error.close()
            self._log_event("upstream_error", status=error.code, **detail)
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
        content_length = upstream.headers.get("Content-Length")
        if not stream and content_length:
            self.send_header("Content-Length", content_length)
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def forward_generic(self, body: dict[str, Any]) -> None:
        authorization = _select_bearer(
            self.config.api_key,
            self.headers.get("Authorization"),
            missing=(
                "no upstream API key; set MUSE_GATEWAY_API_KEY or launch Muse with META_API_KEY"
            ),
        )
        self._post_responses(
            self.config.upstream_url("responses"), body, authorization
        )

    def forward_meta(self, body: dict[str, Any]) -> None:
        """Forward to Meta with the native wire format (see transform_request)."""
        authorization = _select_bearer(
            self.config.meta_api_key,
            self.headers.get("Authorization"),
            missing=(
                "no Meta API key; set MUSE_GATEWAY_META_API_KEY, or launch Muse "
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
        """Forward one request to the user-configured local Responses endpoint."""
        authorization = _select_bearer(
            self.config.local_api_key or self.config.api_key,
            self.headers.get("Authorization"),
            missing=(
                "no local API key; set MUSE_GATEWAY_LOCAL_API_KEY (or MUSE_GATEWAY_API_KEY "
                "when the local upstream is shared), or launch Muse with META_API_KEY"
            ),
        )
        local_url = self.config.provider_url("local")
        if (
            not (self.config.local_api_key or self.config.api_key)
            and self.headers.get("Authorization")
        ):
            host = urllib.parse.urlsplit(local_url).hostname or ""
            _warn_credential_forwarding_once(
                "local", local_url, custom=not _is_loopback_host(host)
            )
        self._post_responses(local_url, body, authorization)

    def forward_kiro(self, body: dict[str, Any]) -> None:
        """Translate through the external Messages service, never native OAuth."""
        self._stream_failed = True
        key = muse_kiro.validate_proxy_key(self.config.kiro_api_key)
        budget = self.config.kiro_effective_output_limit()
        payload, tool_names = muse_kiro.prepare_request(
            body, max_output_tokens=budget,
        )
        if body.get("previous_response_id") and "previous_response_id" not in payload:
            self._log_event("previous_response_id_dropped")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        endpoint = self.config.provider_url("kiro")
        capture = current_capture()
        if capture is not None:
            capture.add_secret(key)
            capture.json("outgoing", payload)
            capture.bytes("outgoing.body", encoded)
            capture.event("upstream.open", url=endpoint)
        stream = body.get("stream") is True
        upstream_status = None
        upstream_headers = None
        secrets = tuple([*self.config.secret_values(), self.headers.get("Authorization", "")])
        try:
            with muse_kiro.open_request(
                endpoint, key, body=encoded, timeout=self.config.timeout,
            ) as upstream:
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
                                self._log_event("kiro_upstream_error", metadata=value["response"]["error"]["metadata"])
                            break
                        try:
                            event_type, value = next(pending)
                        except StopIteration:
                            break
                    if not terminal:
                        raise ProviderError(
                            "Kiro Gateway stream ended without a terminal response.",
                            status=502, code="kiro_stream_truncated",
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
            if body.get("previous_response_id") and not (body.get("input") or body.get("messages")):
                raise ProviderError(
                    "Devin response chaining requires complete inline conversation input.",
                    code="devin_previous_response_unsupported",
                )
            # Devin has no server-side chaining; the complete inline history
            # stands in for the unresolvable reference. Same event the Codex
            # path logs so diagnostics stay uniform.
            body = {key: value for key, value in body.items() if key != "previous_response_id"}
            notify = getattr(self, "_log_event", None)
            if callable(notify):
                notify("previous_response_id_dropped")
        config = self.config
        try:
            token, _identity = muse_devin.resolve_access_token(
                auth_db_path=config.devin_auth_db,
                credential_id=config.devin_credential_id,
                api_key=config.devin_api_key,
            )
            if current_capture() is not None:
                current_capture().add_secret(token)
        except muse_devin.DevinCredentialError as error:
            raise ProviderError(str(error), code="devin_credential_unavailable") from error
        _ = _devin_catalog_entries(config, access_token=token)
        logical_id = str(body.get("model") or "").strip()
        roster, _dropped = config._devin_rosters[_devin_cache_key(config, token)]
        model = roster.get(logical_id)
        if model is None:
            raise ProviderError(
                f"unknown Devin model {logical_id!r}; see --models for the roster.",
                code="devin_unknown_model",
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
        try:
            if not stream:
                final = muse_devin.run_turn_final(
                    body, tools, model, token,
                    base_url=config.devin_upstream, effort=effort, timeout=config.timeout,
                    conversation_id=conversation_id,
                )
                self.send_json(200, final)
                return
            pending = muse_devin.run_turn_events(
                body, tools, model, token,
                base_url=config.devin_upstream, effort=effort, timeout=config.timeout,
                conversation_id=conversation_id,
            )
            try:
                first_event = next(pending)
            except StopIteration:
                raise muse_devin.DevinChatError(
                    "Devin turn ended without a terminal event"
                )
            self._send_upstream_headers(
                types.SimpleNamespace(status=200, headers={}),
                content_type="text/event-stream",
                stream=True,
            )
            event_type, payload = first_event
            self.wfile.write(response_sse_line(event_type, payload))
            self.wfile.flush()
            for event_type, payload in pending:
                if event_type == "response.failed":
                    self._stream_failed = True
                chunk = response_sse_line(event_type, payload)
                self.wfile.write(chunk)
                self.wfile.flush()
        except muse_devin.DevinAuthError as error:
            raise AuthError(str(error), code="devin_upstream_unauthorized") from error
        except muse_devin.DevinChatError as error:
            status = 400 if error.code == "devin_context_overflow" else 502
            raise ProviderError(
                str(error), status=status, code=error.code or "devin_chat_error"
            ) from error
        except muse_devin.DevinHttpError as error:
            raise ProviderError(
                str(error), status=502, code="devin_upstream_error"
            ) from error
        except muse_devin.DevinWireError as error:
            raise ProviderError(
                str(error), status=502, code="devin_wire_error"
            ) from error

    def _post_responses(
        self, url: str, body: dict[str, Any], authorization: str
    ) -> None:
        """POST one Responses body and stream the upstream reply back to Muse."""
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "User-Agent": self.server_version,
            },
        )
        capture = current_capture()
        if capture is not None:
            capture.add_secret(authorization)
            capture.bytes("outgoing.body", request.data)
            capture.json("outgoing", body)
            capture.json("outgoing.headers", dict(request.header_items()))
            capture.event("upstream.open", url=url)
        with urllib.request.urlopen(
            request,
            timeout=self.config.timeout,
            context=_ssl_context(self.config),
        ) as upstream:
            content_type = upstream.headers.get("Content-Type", "application/octet-stream")
            stream = "text/event-stream" in content_type.lower() or body.get("stream") is True
            self._send_upstream_headers(upstream, content_type=content_type, stream=stream)
            _write_upstream_body(
                self,
                upstream,
                stream=stream,
                content_type=content_type,
            )

    def forward_codex(self, body: dict[str, Any]) -> None:
        state = resolve_codex_credentials(self.config)
        if current_capture() is not None:
            current_capture().add_secret(state.access_token)
        if not body.get("model"):
            raise ProviderError(
                "Codex request is missing model; set MUSE_GATEWAY_MODEL or send model.",
                code="model_missing",
            )
        endpoint = self.config.provider_url("codex")
        validate_codex_endpoint(
            endpoint,
            allow_host_override=self.config.allow_oauth_host_override,
        )
        had_chained_id = bool(body.get("previous_response_id"))
        codex_body = prepare_codex_request(body)
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
        for attempt in range(3):
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
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout,
                    context=_ssl_context(self.config),
                ) as upstream:
                    content_type = upstream.headers.get("Content-Type", "application/json")
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
                except OSError:
                    raw_detail = b""
                finally:
                    error.close()
                detail = _upstream_diagnostic(raw_detail)
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
                capture_note = ""
                if capture is not None:
                    capture_note = f"; private capture: {capture.directory}"
                else:
                    capture_note = (
                        "; enable --capture-dir (MUSE_GATEWAY_CAPTURE_DIR) on a new "
                        "gateway to retain the upstream error message"
                    )
                raise ProviderError(
                    f"Codex upstream returned HTTP {error.code}. "
                    f"detail={json.dumps(detail)}{capture_note}",
                    status=error.code,
                    code="codex_upstream_error",
                ) from error

    def forward_claude_cli(self, body: dict[str, Any]) -> None:
        payload, tool_names = run_claude_cli(body, self.config)
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
            ),
        )

    def forward_claude(self, body: dict[str, Any]) -> None:
        if "previous_response_id" in body:
            raise ProviderError(
                "Claude mode does not support previous_response_id because the gateway has no durable response-state store.",
                code="claude_previous_response_unsupported",
            )
        state = resolve_claude_credentials(self.config)
        payload, tool_names = responses_to_anthropic(
            body,
            oauth=state.auth_mode != "api_key",
            model_override=(
                self.config.model if self.config.provider == "claude" else None
            ),
            max_output_tokens=self.config.max_output_tokens,
        )
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
        for attempt in range(2):
            request = urllib.request.Request(
                endpoint,
                data=encoded_body,
                method="POST",
                headers=claude_headers(state, stream=bool(body.get("stream"))),
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout,
                    context=_ssl_context(self.config),
                ) as upstream:
                    content_type = upstream.headers.get("Content-Type", "application/json")
                    stream = "text/event-stream" in content_type.lower() or body.get("stream") is True
                    if stream:
                        self._send_upstream_headers(
                            upstream,
                            content_type="text/event-stream",
                            stream=True,
                        )
                        for chunk in convert_anthropic_stream(
                            upstream,
                            model=str(payload.get("model") or ""),
                            tool_names=tool_names,
                        ):
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        return
                    try:
                        response_payload = json.loads(upstream.read().decode("utf-8"))
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
                # This adapter projects only the status, never the body.
                # Dispose it before retry/refresh or retaining an exception.
                error.close()
                if error.code in {401, 403} and attempt == 0 and state.auth_mode != "api_key":
                    try:
                        state = resolve_claude_credentials(self.config, force_refresh=True)
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


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse cross-origin redirects: stdlib re-sends Authorization on
    301/302/303 (as a GET), which would leak the caller's bearer to the
    redirect target. Same-origin redirects keep working."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source = urllib.parse.urlsplit(req.full_url).hostname or ""
        target = urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl)).hostname or ""
        if target.lower() != source.lower():
            raise urllib.error.HTTPError(
                newurl, code,
                f"refusing cross-origin redirect to {target or '[invalid host]'}",
                headers, fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _PrivateRotatingFileHandler(RotatingFileHandler):
    """Rotating diagnostics whose files stay owner-only where supported."""

    def _open(self):
        stream = super()._open()
        try:
            os.chmod(self.baseFilename, 0o600)
        except OSError:
            pass
        return stream


class MuseShimServer(ThreadingHTTPServer):
    daemon_threads = True

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
        if config.kiro_requested:
            # These are instance settings: native/non-Kiro reuse is unchanged.
            # Never combine SO_EXCLUSIVEADDRUSE with the base's SO_REUSEADDR.
            self.allow_reuse_address = False
            self.allow_reuse_port = False
        super().__init__(address, MuseShimHandler)
        # Credentialed upstreams must never follow a redirect onto another
        # origin. Installed here so every server path is covered; tests that
        # patch urllib.request.urlopen are unaffected (they bypass the opener).
        try:
            self.capture_session = CaptureSession(config.capture_dir) if config.capture_dir else None
            urllib.request.install_opener(
                urllib.request.build_opener(_SameOriginRedirectHandler)
            )
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
    if provider == "kiro":
        return env_value("MUSE_GATEWAY_KIRO_UPSTREAM") or muse_kiro.DEFAULT_UPSTREAM
    if provider == "codex":
        return env_value(
            "MUSE_GATEWAY_CODEX_UPSTREAM",
            env_value("MUSE_GATEWAY_CODEX_BASE_URL", DEFAULT_CODEX_ENDPOINT),
        )
    if provider == "claude":
        return env_value(
            "MUSE_GATEWAY_CLAUDE_UPSTREAM",
            env_value("MUSE_GATEWAY_CLAUDE_BASE_URL", DEFAULT_CLAUDE_ENDPOINT),
        )
    if provider == "meta":
        return env_value(
            "MUSE_GATEWAY_META_UPSTREAM",
            env_value("MUSE_GATEWAY_META_BASE_URL", DEFAULT_META_UPSTREAM),
        )
    if provider == "devin":
        return env_value(
            "MUSE_GATEWAY_DEVIN_UPSTREAM",
            env_value("MUSE_GATEWAY_DEVIN_BASE_URL", DEFAULT_DEVIN_UPSTREAM),
        )
    if provider == "local":
        return env_value(
            "MUSE_GATEWAY_LOCAL_UPSTREAM",
            env_value(
                "MUSE_GATEWAY_LOCAL_BASE_URL",
                env_value("MUSE_GATEWAY_UPSTREAM", DEFAULT_UPSTREAM),
            ),
        )
    return env_value("MUSE_GATEWAY_UPSTREAM", DEFAULT_UPSTREAM)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "muse-gateway: loopback Muse proxy with a unified Meta, Codex, "
            "Claude, Devin, Kiro, and local model catalog (forked from muse-shim)"
        ),
        epilog=(
            "For Muse subscription auth use `muse-gateway launch [MUSE_ARGS...]` "
            "(or the muse-gateway.ps1 / Bash wrappers, which forward to it). "
            "`python -m muse_gateway run [MUSE_ARGS...]` uses a placeholder Muse "
            "key instead; server options go before `run` or `launch`."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("auto", "generic", "http", "codex", "claude", "meta", "devin", "kiro", "local", "doctor"),
        default=env_value("MUSE_GATEWAY_PROVIDER", "auto"),
        help="provider mode or doctor",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "generic", "http", "codex", "claude", "meta", "devin", "kiro", "local"),
    )
    parser.add_argument("--host", default=env_value("MUSE_GATEWAY_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(env_value("MUSE_GATEWAY_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--api-key", default=env_value("MUSE_GATEWAY_API_KEY"))
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--model", default=env_value("MUSE_GATEWAY_MODEL"))
    parser.add_argument(
        "--codex-models",
        default=env_value("MUSE_GATEWAY_CODEX_MODELS"),
        help="comma-separated Codex fallback models shown in Muse's /model menu",
    )
    parser.add_argument(
        "--claude-models",
        default=env_value("MUSE_GATEWAY_CLAUDE_MODELS"),
        help="comma-separated Claude fallback models shown in Muse's /model menu",
    )
    parser.add_argument(
        "--meta-models",
        default=env_value("MUSE_GATEWAY_META_MODELS"),
        help="comma-separated Meta models shown in Muse's /model menu (default: muse-spark-1.3,muse-spark-1.2)",
    )
    parser.add_argument(
        "--meta-upstream",
        default=env_value("MUSE_GATEWAY_META_UPSTREAM"),
        help="Meta Responses base URL (default: https://api.meta.ai/v1; custom hosts receive native Meta payloads and authorization)",
    )
    parser.add_argument(
        "--meta-api-key",
        default=env_value("MUSE_GATEWAY_META_API_KEY"),
        help="Meta API key (prefer the MUSE_GATEWAY_META_API_KEY env var; required for meta/ models under `run`)",
    )
    parser.add_argument(
        "--local-models",
        default=env_value("MUSE_GATEWAY_LOCAL_MODELS"),
        help="comma-separated local model aliases shown as local/ entries (default: none; opt-in)",
    )
    parser.add_argument(
        "--local-upstream",
        default=env_value("MUSE_GATEWAY_LOCAL_UPSTREAM"),
        help="local Responses base URL, e.g. a LiteLLM proxy (default: MUSE_GATEWAY_UPSTREAM)",
    )
    parser.add_argument(
        "--local-api-key",
        default=env_value("MUSE_GATEWAY_LOCAL_API_KEY"),
        help="local upstream API key (prefer the MUSE_GATEWAY_LOCAL_API_KEY env var; falls back to MUSE_GATEWAY_API_KEY)",
    )
    parser.add_argument(
        "--devin-upstream",
        default=env_value("MUSE_GATEWAY_DEVIN_UPSTREAM"),
        help="Devin Connect RPC base URL (default: https://server.codeium.com)",
    )
    parser.add_argument(
        "--devin-api-key",
        default=env_value(
            "MUSE_GATEWAY_DEVIN_API_KEY", env_value("DEVIN_API_KEY")
        ),
        help="Devin session token (prefer env vars; else Devin CLI login, then OMP credential store)",
    )
    parser.add_argument(
        "--devin-auth-db",
        default=env_value("MUSE_GATEWAY_DEVIN_AUTH_DB"),
        help="OMP agent.db path for Devin credentials (default: ~/.omp/agent/agent.db)",
    )
    parser.add_argument(
        "--devin-credential-id",
        default=env_value("MUSE_GATEWAY_DEVIN_CREDENTIAL_ID"),
        help="Devin credential row id/identity when several are active",
    )
    parser.add_argument(
        "--devin-models",
        default=env_value("MUSE_GATEWAY_DEVIN_MODELS"),
        help="comma-separated devin/ ids to list (curates discovery, default: all)",
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
        "--kiro-models", default=env_value("MUSE_GATEWAY_KIRO_MODELS"),
        help="comma-separated Kiro Gateway model IDs to curate kiro/ entries; discovery is not account entitlement evidence",
    )
    parser.add_argument(
        "--kiro-context-limit", type=int,
        default=env_value("MUSE_GATEWAY_KIRO_CONTEXT_LIMIT", str(muse_kiro.DEFAULT_CONTEXT_LIMIT)),
        help="positive unverified Kiro context budget (default: 65536)",
    )
    parser.add_argument(
        "--kiro-output-limit", type=int,
        default=env_value("MUSE_GATEWAY_KIRO_OUTPUT_LIMIT", str(muse_kiro.DEFAULT_OUTPUT_LIMIT)),
        help="positive unverified Kiro output budget, not a guaranteed upstream cap (default: 8192)",
    )
    parser.add_argument(
        "--no-model-discovery",
        action="store_false",
        dest="model_discovery",
        default=not env_flag("MUSE_GATEWAY_NO_MODEL_DISCOVERY"),
        help="do not query provider model catalogs; advertise configured fallbacks only",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        default=env_flag("MUSE_GATEWAY_PORTABLE"),
        help="strip reasoning and cache fields for stricter Responses servers",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=(
            int(env_value("MUSE_GATEWAY_MAX_OUTPUT_TOKENS"))
            if env_value("MUSE_GATEWAY_MAX_OUTPUT_TOKENS")
            else None
        ),
    )
    parser.add_argument(
        "--insecure-upstream",
        action="store_true",
        default=env_flag("MUSE_GATEWAY_INSECURE_UPSTREAM"),
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
        help="allow binding beyond loopback (remote clients can reach this proxy)",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=int(env_value("MUSE_GATEWAY_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))),
        help=f"maximum incoming JSON body size (default: {DEFAULT_MAX_BODY_BYTES})",
    )
    parser.add_argument("--timeout", type=float, default=float(env_value("MUSE_GATEWAY_TIMEOUT", "600")))
    parser.add_argument(
        "--capture-dir", default=env_value("MUSE_GATEWAY_CAPTURE_DIR"),
        help="opt-in private request/response captures (contains conversation content; credentials scrubbed)",
    )
    parser.add_argument("--refresh-skew", type=int, default=int(env_value("MUSE_GATEWAY_REFRESH_SKEW", "120")))
    parser.add_argument("--codex-home", default=os.environ.get("CODEX_HOME"))
    parser.add_argument("--codex-auth-path", default=env_value("MUSE_GATEWAY_CODEX_AUTH"))
    parser.add_argument("--codex-token-url", default=env_value("MUSE_GATEWAY_CODEX_TOKEN_URL"))
    parser.add_argument("--claude-credentials", default=env_value("MUSE_GATEWAY_CLAUDE_CREDENTIALS"))
    parser.add_argument("--claude-token-url", default=env_value("MUSE_GATEWAY_CLAUDE_TOKEN_URL"))
    parser.add_argument(
        "--claude-transport",
        choices=("cli", "http"),
        default=env_value("MUSE_GATEWAY_CLAUDE_TRANSPORT", "cli"),
        help="Claude inference transport (default: cli, using claude -p)",
    )
    parser.add_argument(
        "--claude-cli",
        default=env_value("MUSE_GATEWAY_CLAUDE_CLI", "claude"),
        help="Claude Code executable used by CLI transport",
    )
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
        help="config file loaded for --models (default: beside "
        "muse_gateway.py, else ./.env; launch takes its own --env-file "
        "after the launch token)",
    )
    parser.add_argument(
        "--log-file", default=env_value("MUSE_GATEWAY_LOG_FILE"),
        help="rotating diagnostic log (default: ./muse-gateway.log; no request/response content)",
    )
    parser.add_argument("--verbose", action="store_true", default=env_flag("MUSE_GATEWAY_VERBOSE"))
    args = parser.parse_args(argv)
    provider = args.provider or args.mode
    if provider == "http":
        provider = "generic"
    args.provider = provider
    if not args.upstream:
        args.upstream = (args.kiro_upstream if provider == "kiro" else None) or _default_upstream_for(provider)
    try:
        if args.kiro_context_limit <= 0 or args.kiro_output_limit <= 0:
            raise ProviderError("Kiro context and output budgets must be positive integers.", code="invalid_kiro_budget")
        if provider == "kiro" or args.kiro_upstream or args.kiro_api_key or args.kiro_models:
            muse_kiro.normalize_upstream(args.upstream if provider == "kiro" else (args.kiro_upstream or muse_kiro.DEFAULT_UPSTREAM))
            if args.kiro_api_key:
                muse_kiro.validate_proxy_key(args.kiro_api_key)
        if provider == "auto" and args.model:
            raise ProviderError(
                "--model / MUSE_GATEWAY_MODEL cannot pin auto mode; choose a model "
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
        native_providers = ("codex", "claude") if provider == "auto" else (provider,)
        for native_provider in native_providers:
            native_upstream = (
                args.upstream
                if provider == native_provider
                else _default_upstream_for(native_provider)
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
        return ok, f"proxy key configured; {config._kiro_catalog_status}; {len(entries)} menu entries, budgets unverified"
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
            "only when Muse sends a request (use the source gateway launchers)",
        )
    if provider == "local":
        if not (config.local_api_key or config.api_key):
            return (
                False,
                "API key not configured (set MUSE_GATEWAY_LOCAL_API_KEY or "
                "MUSE_GATEWAY_API_KEY for a shared upstream)",
            )
        if config.local_models:
            count = len(config.local_models)
            return True, f"API key configured; {count} explicit local models"
        entries = provider_model_catalog(config, "local")
        if entries:
            return (
                True,
                f"API key configured; {len(entries)} local models discovered",
            )
        if not config.local_upstream_explicit:
            return True, "API key configured; no local upstream set, no local/ entries"
        return (
            False,
            "API key configured; local discovery returned no usable models "
            "(unavailable, empty, invalid, or disabled); check the upstream "
            "or configure MUSE_GATEWAY_LOCAL_MODELS",
        )
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
            f"Claude CLI print transport; executable={executable}; {cli_reason} login",
        )
    if provider == "devin":
        state = muse_devin.credential_state(
            auth_db_path=config.devin_auth_db,
            credential_id=config.devin_credential_id,
            api_key=config.devin_api_key,
        )
        status = state.get("state")
        if status == "ready":
            identity = state.get("identity")
            if not identity:
                if state.get("source") == "devin-cli":
                    identity = "Devin CLI login"
                elif state.get("source") == "store":
                    identity = (
                        f"stored credential id {state.get('credential_id')}"
                    )
                else:
                    identity = "explicit token"
            detail = f"credential ready ({identity})"
            if state.get("source") == "store":
                try:
                    _store_path, store_rows = muse_devin.list_store_credentials(
                        config.devin_auth_db
                    )
                except Exception:
                    store_rows = []
                if len(store_rows) > 1:
                    detail += (
                        f"; using id {state.get('credential_id')}"
                        f" of {len(store_rows)} active"
                    )
            return True, detail
        return False, state.get("note") or f"credential {status}"
    try:
        state = discover_codex_credentials(config) if provider == "codex" else discover_claude_credentials(config)
    except Exception:
        state = None
    if state is None:
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


def _print_devin_rows(config: Config, output: Any) -> None:
    """List stored Devin credentials (ids and expiry, never secrets).

    Gives the ``--devin-credential-id`` menu: one line per active row with
    the selected marker, so a multi-login store stays operable.
    """
    try:
        store_path, rows = muse_devin.list_store_credentials(config.devin_auth_db)
    except Exception:
        return
    if not rows:
        return
    try:
        selected = muse_devin.credential_state(
            auth_db_path=config.devin_auth_db,
            credential_id=config.devin_credential_id,
            api_key=config.devin_api_key,
        ).get("credential_id")
    except Exception:
        selected = None
    print(f"devin store: {store_path} ({len(rows)} active)", file=output)
    for row in rows:
        expires_at = row.get("expires_at")
        if isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool):
            expires = time.strftime("%Y-%m-%d", time.gmtime(expires_at / 1000))
        else:
            expires = "unknown"
        identity = row.get("email") or row.get("identity_key") or "?"
        flags = []
        if row.get("id") == selected:
            flags.append("selected")
        if row.get("expired"):
            flags.append("expired")
        if row.get("auth_failed"):
            flags.append("auth-failed")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        print(
            f"devin credential: id={row.get('id')} identity={identity}"
            f" expires={expires}{suffix}",
            file=output,
        )


def run_doctor(config: Config, *, output: Any = None) -> int:
    output = output or sys.stdout
    print("muse-gateway doctor", file=output)
    print(f"listen: {config.host}:{config.port}", file=output)
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
        if config.local_models or config.local_upstream_explicit:
            providers = (*providers, "local")
    else:
        providers = ("codex", "claude")
    ready = bool(config.upstream)
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
        if provider == "devin":
            _print_devin_rows(config, output)
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
        if config.local_models or config.local_upstream_explicit:
            providers = (*providers, "local")
    for name in providers:
        try:
            entries = provider_model_catalog(config, name)
        except Exception as error:
            print(f"muse-gateway models: {name} discovery failed ({error})", file=sys.stderr)
            continue
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
            payload = json.loads(response.read().decode("utf-8"))
        return (
            response.status == 200
            and isinstance(payload, dict)
            and payload.get("ok") is True
            and payload.get("service") in {"muse-gateway", "muse-shim"}
            and payload.get("provider") == provider
            and payload.get("claude_transport") == claude_transport
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


class _LauncherSignal(BaseException):
    def __init__(self, signum: int) -> None:
        self.signum = signum


def _stop_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _subcommand_index(argv: list[str], word: str) -> int | None:
    """Find a subcommand without mistaking an option value for it."""
    flag_options = {
        "-h",
        "--help",
        "--no-model-discovery",
        "--portable",
        "--insecure-upstream",
        "--allow-oauth-host-override",
        "--allow-non-loopback",
        "--verbose",
    }
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

    environment = os.environ.copy()
    environment["META_API_KEY"] = "local-gateway-placeholder"
    command = [
        "muse",
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
        if child is not None:
            _stop_child(child)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)


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
  --local-models A,B         restrict local/ entries to these upstream IDs
  --devin-upstream URL       Devin Connect RPC base URL
  --devin-api-key KEY        Devin session token (else CLI login, then OMP)
  --devin-auth-db PATH       OMP agent.db path for Devin credentials
  --devin-credential-id ID   Devin credential row when several are active
  --devin-models A,B         restrict devin/ entries to these IDs
  --kiro-upstream URL        separate Kiro Gateway Messages base URL
  --kiro-api-key KEY         Kiro Gateway proxy key (prefer environment)
  --kiro-models A,B          restrict kiro/ entries to these upstream IDs
  --kiro-context-limit N     unverified positive operational context budget
  --kiro-output-limit N      unverified positive operational output budget
  --check-only               preflight + catalog check, then exit (no Muse)
  --env-file PATH            config file (default: muse_gateway.py dir, else ./)
  --muse-bin PATH            Muse executable (default: auto-detect)
  --log-file PATH            rotating diagnostics (default: ./muse-gateway.log)
  -h                         this help; --help passes through to Muse

Config precedence: flags, then environment, then the env file. Without a local
key the launcher skips local preflight; explicit local settings remain active.
"""

_LAUNCH_VALUE_OPTIONS = {
    "--port": "port",
    "--local-upstream": "local_upstream",
    "--local-api-key": "local_api_key",
    "--local-models": "local_models",
    "--devin-upstream": "devin_upstream",
    "--devin-api-key": "devin_api_key",
    "--devin-auth-db": "devin_auth_db",
    "--devin-credential-id": "devin_credential_id",
    "--devin-models": "devin_models",
    "--kiro-upstream": "kiro_upstream",
    "--kiro-api-key": "kiro_api_key",
    "--kiro-models": "kiro_models",
    "--kiro-context-limit": "kiro_context_limit",
    "--kiro-output-limit": "kiro_output_limit",
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
    opts = argparse.Namespace(
        port=None,
        check_only=False,
        local_upstream=None,
        local_api_key=None,
        local_models=None,
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
        elif token in _LAUNCH_VALUE_OPTIONS:
            index += 1
            if index >= total:
                raise ProviderError(
                    f"launch option {token} needs a value.",
                    code="launch_option_value_missing",
                )
            value = argv[index]
            if token in {"--kiro-context-limit", "--kiro-output-limit"}:
                value = _launch_positive_budget(token, value)
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
                if name in {"--kiro-context-limit", "--kiro-output-limit"}:
                    value = _launch_positive_budget(name, value)
                setattr(opts, _LAUNCH_VALUE_OPTIONS[name], value)
            else:
                muse_args.extend(argv[index:])
                break
        else:
            muse_args.extend(argv[index:])
            break
        index += 1
    return opts, muse_args


def _launch_positive_budget(option: str, value: str) -> int:
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except ValueError:
        pass
    raise ProviderError(
        f"launch option {option} needs a positive integer.",
        code="launch_option_value_invalid",
    )


def load_env_file(path: str) -> int:
    """Load KEY=VALUE lines from path into the environment without overwriting.

    Returns the number of variables set. A missing file sets nothing.
    """
    try:
        with open(path, encoding="utf-8") as handle:
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
        if name.startswith("export"):
            name = name[len("export") :].strip()
        if (
            not name
            or not (name[0].isalpha() or name[0] == "_")
            or not all(char.isalnum() or char == "_" for char in name)
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value or os.environ.get(name):
            continue
        os.environ[name] = value
        loaded += 1
    return loaded


def _default_env_file() -> str | None:
    """Locate the env file: beside muse_gateway.py, else ./.env, else None."""
    script_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(script_env):
        return script_env
    cwd_env = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(cwd_env):
        return cwd_env
    return None


def _wait_for_health(base_url: str, timeout: float = 15.0) -> dict[str, Any] | None:
    """Poll /health until it reports ok; return its payload or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
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
    env_file = launch_opts.env_file or _default_env_file()
    if env_file is not None:
        load_env_file(env_file)
    # Post-launch flags are explicit CLI values, not post-validation patches.
    # Supplying them to argparse avoids validating an overridden env default
    # (e.g. KIRO_OUTPUT_LIMIT=bad with --kiro-output-limit 4096).
    parser_argv = list(pre_argv) if pre_argv else ["auto"]
    for flag, attribute in _LAUNCH_VALUE_OPTIONS.items():
        if flag in {"--env-file", "--muse-bin"}:
            continue
        value = getattr(launch_opts, attribute, None)
        if value is not None:
            parser_argv.append(f"{flag}={value}")
    try:
        server_args = parse_args(parser_argv)
    except SystemExit as exited:
        return int(exited.code or 2)
    # Launch always serves the unified catalog, like the source wrappers.
    server_args.provider = "auto"
    server_args.mode = "auto"
    # Subscription launch must not retain a static key captured by argparse.
    server_args.meta_api_key = None
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

    default_local_upstream = "http://127.0.0.1:4000/v1"
    gateway_upstream = (
        launch_opts.local_upstream
        or getattr(server_args, "local_upstream", None)
        or default_local_upstream
    )
    gateway_key = launch_opts.local_api_key or config.local_api_key or ""
    if launch_opts.local_models is not None:
        gateway_models = _parse_model_list(launch_opts.local_models, ())
    else:
        gateway_models = config.local_models
    local_enabled = bool(gateway_key)
    os.environ["MUSE_GATEWAY_PORT"] = str(config.port)
    # Keep both Meta key slots UNSET so Muse sends its subscription login and
    # the gateway forwards it. (`run` mode can't do this: it forces a
    # placeholder key, which is why launch exists.)
    os.environ.pop("META_API_KEY", None)
    os.environ.pop("MUSE_GATEWAY_META_API_KEY", None)
    os.environ.pop("MUSE_GATEWAY_META_API_KEY", None)
    if local_enabled:
        os.environ["MUSE_GATEWAY_LOCAL_UPSTREAM"] = gateway_upstream
        os.environ["MUSE_GATEWAY_LOCAL_API_KEY"] = gateway_key
        if gateway_models:
            os.environ["MUSE_GATEWAY_LOCAL_MODELS"] = ",".join(gateway_models)
        config.local_upstream = gateway_upstream.rstrip("/")
        config.local_api_key = gateway_key
        config.local_models = list(gateway_models)
        config.local_upstream_explicit = True
    else:
        print(
            "warning: no local API key: skipping local-backend preflight. "
            "Set MUSE_GATEWAY_LOCAL_API_KEY in the environment or .env file to "
            "configure it. Continuing; explicit local upstream/model settings "
            "remain active.",
            file=sys.stderr,
        )

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
    if local_enabled:
        model_count = len(gateway_models)
        model_note = f"{model_count} explicit" if model_count else "auto-discovered"
    else:
        model_note = "preflight skipped (no local API key)"
    print(f"config: upstream={gateway_upstream} models={model_note}")
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
    _print_provider_status(config, "kiro")

    base_url = f"http://{config.host}:{config.port}"
    server: MuseShimServer | None = None
    server_thread: threading.Thread | None = None
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
            or launch_opts.local_models is not None
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
                "devin_auth_db",
                "devin_credential_id",
                "devin_models",
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
        base_url = f"http://{config.host}:{server.server_port}"
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
    previous_signal_handlers: dict[int, Any] = {}
    child: subprocess.Popen[Any] | None = None
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
    try:
        health = _wait_for_health(base_url)
        if health is None:
            print(f"error: gateway did not come up on {base_url}", file=sys.stderr)
            return 1
        print(f"gateway up: provider {health.get('provider', '?')} on {base_url}")
        print(f"diagnostics: {config.log_file} (rotating; no prompt or response content)")
        if launch_opts.check_only:
            try:
                with urllib.request.urlopen(
                    base_url + "/muse-code/models", timeout=10
                ) as response:
                    catalog = json.loads(response.read().decode("utf-8"))
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
        environment = os.environ.copy()
        environment.pop("META_API_KEY", None)
        environment.pop("MUSE_GATEWAY_META_API_KEY", None)
        environment.pop("MUSE_GATEWAY_META_API_KEY", None)
        try:
            child = subprocess.Popen(
                [muse_bin, *muse_args, *extra], env=environment
            )
            return int(child.wait())
        except FileNotFoundError:
            print("muse-gateway could not find `muse` on PATH.", file=sys.stderr)
            return 127
        except KeyboardInterrupt:
            return 130
        except _LauncherSignal as received:
            return 128 + received.signum
    finally:
        if child is not None:
            _stop_child(child)
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    run_index = _run_command_index(raw_argv)
    if run_index is not None:
        args = parse_args(raw_argv[:run_index])
        return run_muse(Config(args), raw_argv[run_index + 1:])
    launch_index = _launch_command_index(raw_argv)
    if launch_index is not None:
        return launch_main(raw_argv[:launch_index], raw_argv[launch_index + 1:])
    args = parse_args(raw_argv)
    if args.models:
        env_file = args.env_file or _default_env_file()
        if env_file is not None:
            load_env_file(env_file)
            # Parse again after loading defaults. Explicit flags still win;
            # otherwise --models would ignore its own --env-file settings.
            args = parse_args(raw_argv)
        return run_models(Config(args), provider=args.models_provider)
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
    server = MuseShimServer((config.host, config.port), config)
    print(
        f"muse-gateway {config.provider} listening on http://{config.host}:{server.server_port} "
        f"-> {_server_destination(config)}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
