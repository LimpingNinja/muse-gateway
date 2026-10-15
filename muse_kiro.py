"""Kiro Gateway's Messages boundary, without native Claude credentials.

Kiro login, account refresh and inference retries belong to the separate
service. This module deliberately owns none of them. The existing Anthropic
codecs own Responses translation; the checks here keep their permissive
defaults from turning missing or malformed upstream data into success.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from http.client import HTTPException
import ipaddress
import json
import math
import re
import ssl
from typing import Any, Iterator, Mapping
import urllib.error
import urllib.parse
import urllib.request

from muse_capture import current_capture, encoded_secret_forms
from muse_providers import (
    ProviderError, StreamState, ToolNameMap, anthropic_json_to_responses,
    convert_anthropic_event, responses_to_anthropic,
)


DEFAULT_UPSTREAM = "http://127.0.0.1:8000/v1"
DEFAULT_CONTEXT_LIMIT = 65536
DEFAULT_OUTPUT_LIMIT = 8192
_CATALOG_LIMIT = 2 * 1024 * 1024
_EVENT_LIMIT = 4 * 1024 * 1024
_BLOCK_LIMIT = 8 * 1024 * 1024
_RESPONSE_LIMIT = 64 * 1024 * 1024
_STOP_REASONS = {"end_turn", "stop_sequence", "tool_use", "max_tokens", "refusal"}

# These are explicit machine identifiers, never matches against error prose.
# A rate_limit_error can also denote a spend cap, so it is not temporary
# evidence without a valid Retry-After. These observations authorize no replay.
_ERROR_CATEGORIES = {
    "authentication_error": "authentication", "permission_error": "permission",
    "billing_error": "quota", "insufficient_quota": "quota", "quota_exceeded": "quota",
    "rate_limit_error": "rate_limit", "rate_limit_exceeded": "rate_limit",
    "overloaded_error": "overload", "invalid_request_error": "request",
    "not_found_error": "request", "request_too_large": "request",
    "context_length_exceeded": "context", "context_window_exceeded": "context",
}
_BLOCKING_CATEGORIES = {"authentication", "permission", "quota", "request", "context"}
_STATUS_CATEGORIES = {401: "authentication", 402: "quota", 403: "permission", 413: "request"}
_HTTP_DATE = (
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT",
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), [0-9]{2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT",
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?: [0-9]|[0-9]{2}) [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}",
)


def _safe_error_value(value: Any, secrets: tuple[str, ...], *, correlation: bool = False) -> str | None:
    pattern = r"[A-Za-z0-9_-]{1,128}" if correlation else r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        return None
    return value if _secret_free(value, secrets) else None


def _secret_free(value: str, secrets: tuple[str, ...]) -> bool:
    raw = value.encode("utf-8")
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        variants = (secret, secret[7:].strip()) if secret.lower().startswith("bearer ") else (secret,)
        if any(form in raw for variant in variants for form in encoded_secret_forms(variant)):
            return False
    capture = current_capture()
    known = getattr(capture, "secrets", ()) if capture is not None else ()
    if isinstance(known, (set, tuple, list)) and any(isinstance(secret, bytes) and secret and secret in raw for secret in known):
        return False
    # Reject recognizable credential forms even when they were not configured.
    return re.search(r"(?:sk-[A-Za-z0-9_-]{16,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)", value) is None


def _single_header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    # Message.get_all preserves duplicate fields, unlike dict(headers).
    values = headers.get_all(name, []) if hasattr(headers, "get_all") else [value for key, value in headers.items() if key.lower() == name.lower()]
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def _retry_after(headers: Any, secrets: tuple[str, ...]) -> dict[str, str] | None:
    value = _single_header(headers, "Retry-After")
    if value is None or len(value) > 128 or any(ord(char) < 32 or ord(char) > 126 for char in value) or not _secret_free(value, secrets):
        return None
    value = value.strip(" ")
    if re.fullmatch(r"[0-9]{1,10}", value) and int(value) <= 2147483647:
        return {"value": value, "kind": "delay-seconds"}
    matched = next((index for index, pattern in enumerate(_HTTP_DATE) if re.fullmatch(pattern, value)), None)
    if matched is None:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)  # asctime is GMT in HTTP.
        if matched == 1:
            # RFC 9110: a two-digit year more than 50 years in the future
            # represents the most recent past year with those last digits.
            year = datetime.now(timezone.utc).year
            candidate = year // 100 * 100 + parsed.year % 100
            if candidate > year + 50:
                candidate -= 100
            parsed = parsed.replace(year=candidate)
        if parsed.strftime("%a") != value[:3]:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return {"value": value, "kind": "http-date"}


def load_error_payload(raw: str | bytes) -> dict[str, Any]:
    """Only complete, bounded error JSON is evidence; reject duplicate keys."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("ambiguous error JSON")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("non-JSON number")
    if len(raw) > 65536:
        return {}
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
        return value if isinstance(value, dict) else {}
    except (ValueError, UnicodeError, RecursionError):
        return {}


def error_metadata(*, source: str, upstream_status: int | None = None,
                   payload: Any = None, headers: Any = None, secrets: tuple[str, ...] = (),
                   output_exposure: str = "none") -> dict[str, Any]:
    """Bounded evidence at our Kiro Gateway boundary, not native provenance."""
    upstream = {}
    if isinstance(payload, dict):
        # FastAPI HTTPException uses this one concrete wrapper. String detail
        # is prose, not a substitute for a typed error envelope.
        outer_id = _safe_error_value(payload.get("request_id"), secrets, correlation=True)
        if isinstance(payload.get("detail"), dict) and payload["detail"].get("type") == "error" and "error" not in payload:
            payload = payload["detail"]
            nested_id = _safe_error_value(payload.get("request_id"), secrets, correlation=True)
            if outer_id is not None and nested_id is not None:
                upstream["detail_request_id"] = nested_id
        details = payload.get("error")
        if isinstance(details, dict):
            for field in ("type", "code", "param"):
                safe = _safe_error_value(details.get(field), secrets)
                if safe is not None:
                    upstream[field] = safe
        safe = outer_id or _safe_error_value(payload.get("request_id"), secrets, correlation=True)
        if safe is not None:
            upstream["body_request_id"] = safe
    safe = _safe_error_value(_single_header(headers, "request-id"), secrets, correlation=True)
    if safe is not None:
        upstream["header_request_id"] = safe
    correlations = {upstream[field] for field in ("body_request_id", "header_request_id", "detail_request_id") if field in upstream}
    if len(correlations) > 1:
        upstream["request_id_conflict"] = True
    categories = [_ERROR_CATEGORIES[value] for field in ("code", "type") if (value := upstream.get(field)) in _ERROR_CATEGORIES]
    expected = {"authentication": {401}, "permission": {403}, "quota": {400, 402, 403, 429},
                "rate_limit": {429}, "overload": {503, 529}, "request": set(range(400, 500)), "context": {400, 413}}
    # Status 200 is the transport for SSE/JSON errors, not a subtype conflict.
    status_category = _STATUS_CATEGORIES.get(upstream_status) if source == "http" else None
    if categories and all(upstream_status in expected[item] for item in categories):
        # A compatible coarse status must not contradict more specific typed
        # evidence, e.g. quota on 403 or invalid_request_error on other 4xx.
        status_category = None
    all_categories = categories + ([status_category] if status_category else [])
    blocker = next((category for category in all_categories if category in _BLOCKING_CATEGORIES), None)
    category = blocker or next(iter(categories), source if source in {"transport", "protocol"} else "unknown")
    retry = _retry_after(headers, secrets)
    distinct = set(all_categories)
    # Specific codes refine these umbrella types without contradicting them.
    compatible = ({"request", "context"}, {"rate_limit", "quota"})
    conflict = len(distinct) > 1 and distinct not in compatible
    if source == "http" and categories:
        # Only explicitly compatible status/type evidence can be temporary.
        conflict = conflict or any(upstream_status not in expected[item] for item in categories)
    transient = False if blocker else None
    hint_supports_temporary = retry is not None and (
        category == "rate_limit" or source == "http" and upstream_status in {429, 503, 529}
    )
    if not blocker and not conflict and (category == "overload" or hint_supports_temporary):
        transient = True
    result = {"provider": "kiro", "layer": "kiro_gateway", "source": source,
              "phase": "before_response" if output_exposure == "none" else "stream",
              "upstream_status": upstream_status, "category": category, "transient": transient,
              "output_exposure": output_exposure, "upstream": upstream}
    if conflict:
        result["evidence_conflict"] = True
    if retry is not None:
        result["retry_after"] = retry
    return result


def upstream_error(*, source: str, upstream_status: int | None = None, payload: Any = None,
                   headers: Any = None, secrets: tuple[str, ...] = ()) -> ProviderError:
    metadata = error_metadata(source=source, upstream_status=upstream_status, payload=payload,
                              headers=headers, secrets=secrets)
    return ProviderError("Kiro Gateway returned a provider error; check its configuration and service logs.",
                         status=upstream_status if source == "http" and upstream_status is not None and 400 <= upstream_status <= 599 else 502,
                         code="kiro_invalid_response" if source == "json" else "kiro_upstream_error", metadata=metadata)


def error_fields(error: ProviderError) -> dict[str, Any]:
    """Kiro opts into added fields; other providers retain their envelope."""
    if error.metadata is None:
        return {}
    upstream = error.metadata.get("upstream", {})
    return {"code": upstream.get("code") or upstream.get("type") or error.code, "metadata": error.metadata}


@dataclass
class _KiroToolNames(ToolNameMap):
    tools_allowed: bool = True
    offered_wire_names: frozenset[str] = frozenset()


def _semantic_json(value: Any) -> None:
    """Require finite JSON values and strings that can actually reach UTF-8."""
    pending = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeError as error:
                raise ValueError("invalid Unicode scalar value") from error
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("non-finite JSON number")
        elif isinstance(current, (dict, list)):
            if id(current) in visited:
                continue
            visited.add(id(current))
            if isinstance(current, dict):
                if not all(isinstance(key, str) for key in current):
                    raise ValueError("non-string JSON object key")
                pending.extend(current.keys())
                pending.extend(current.values())
            else:
                pending.extend(current)
        elif current is not None and not isinstance(current, (int, bool)):
            raise ValueError("unsupported JSON value")


def _wire_identifiers(values: list[str], prefix: str, *, reserved: set[str] | None = None) -> dict[str, str]:
    """Preserve safe IDs, reserving every original before generating aliases."""
    reserved = reserved or set()
    used = {value for value in values if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)} | reserved
    result = {}
    for value in values:
        if value in result:
            continue
        if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) and value not in reserved:
            result[value] = value
            continue
        suffix = 0
        while True:
            source = value if suffix == 0 else value + ":" + str(suffix)
            candidate = prefix + hashlib.sha256(source.encode("utf-8")).hexdigest()[:48]
            if candidate not in used:
                break
            suffix += 1
        result[value] = candidate
        used.add(candidate)
    return result


def _invalid_response(message: str = "Kiro Gateway returned invalid Messages data.") -> ProviderError:
    return ProviderError(message, status=502, code="kiro_invalid_response")


def _capture(method: str, *args: Any, **kwargs: Any) -> None:
    # Evidence collection must never determine whether a turn succeeds.
    capture = current_capture()
    if capture is not None:
        try:
            getattr(capture, method)(*args, **kwargs)
        except Exception:
            # Never copy exception text: a recorder can be handling secrets.
            # A bounded manifest marker distinguishes lost evidence from a
            # complete capture without making diagnostics an inference gate.
            try:
                marker = "kiro_capture_" + method + "_failed"
                if marker not in capture.errors and len(capture.errors) < 32:
                    capture.errors.append(marker)
            except Exception:
                pass  # The recorder itself may be unavailable.


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validated_url(value: str) -> urllib.parse.SplitResult:
    message = "Kiro upstream must be an HTTP loopback or verified HTTPS URL without credentials, query or fragment."
    if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise ProviderError(message, code="kiro_invalid_upstream")
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
        host = parsed.hostname
        decoded = urllib.parse.unquote(value)
        if (parsed.scheme not in {"http", "https"} or not host or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or "?" in value or "#" in value or port == 0
                or any(ord(c) <= 32 or ord(c) == 127 for c in decoded)
                or (parsed.scheme == "http" and not _loopback(host))):
            raise ValueError("unsafe URL")
        host.encode("idna")
    except (ValueError, UnicodeError) as error:
        raise ProviderError(message, code="kiro_invalid_upstream") from error
    return parsed


def normalize_upstream(value: str) -> str:
    """Accept a service root, a /v1 base, or its /messages endpoint."""
    parsed = _validated_url(value)
    path = parsed.path.rstrip("/")
    if path.endswith("/messages"):
        path = path[:-len("/messages")]
    if not path:
        path = "/v1"
    if not path.endswith("/v1") or any(part in {".", ".."} for part in path.split("/")):
        raise ProviderError("Kiro upstream must be the service root, /v1 base, or /v1/messages endpoint.", code="kiro_invalid_upstream")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def validate_proxy_key(value: Any) -> str:
    if value is None or value == "":
        raise ProviderError("Kiro Gateway needs its own proxy key; set MUSE_GATEWAY_KIRO_API_KEY.", code="kiro_not_configured")
    if (not isinstance(value, str) or len(value) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in value)):
        raise ProviderError("Kiro proxy key must be a nonempty, printable token without whitespace.", code="kiro_invalid_key")
    return value


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Kiro redirects are disabled.", headers, fp)


def open_request(url: str, key: str, *, body: bytes | None = None, timeout: float = 8.0):
    """Open once, with a private verified opener and no redirect credentials."""
    parsed = _validated_url(url)
    key = validate_proxy_key(key)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout < float("inf"):
        raise ProviderError("Kiro timeout must be positive and finite.", code="kiro_invalid_timeout")
    if body is not None and not isinstance(body, bytes):
        raise TypeError("Kiro request body must be bytes")
    handlers = [_NoRedirects(), urllib.request.HTTPSHandler(context=ssl.create_default_context())]
    if _loopback(parsed.hostname or ""):
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    headers = {"Authorization": "Bearer " + key, "Accept": "text/event-stream, application/json" if body is not None else "application/json",
               "User-Agent": "muse-gateway", "anthropic-version": "2023-06-01"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST" if body is not None else "GET")
    _capture("add_secret", key)
    _capture("json", "kiro.outgoing.headers", headers)
    if body is not None:
        _capture("bytes", "kiro.outgoing.body", body)
    _capture("event", "kiro.upstream.open", url=url)
    return opener.open(request, timeout=timeout)


def _loads(raw: str | bytes) -> Any:
    def invalid_constant(value):
        raise ValueError("non-JSON numeric constant")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed
    try:
        return json.loads(raw, parse_constant=invalid_constant, parse_float=finite_float)
    except RecursionError as error:
        raise ValueError("JSON nesting limit exceeded") from error


def _is_json_mime(value: str) -> bool:
    mime = value.split(";", 1)[0].strip().lower()
    return mime == "application/json" or mime.startswith("application/") and mime.endswith("+json")


def discover_models(upstream: str, key: str, *, timeout: float = 8.0) -> list[str]:
    """Read names only: this catalog is not entitlement or capacity evidence."""
    with open_request(normalize_upstream(upstream) + "/models", key, timeout=timeout) as response:
        if not _is_json_mime(response.headers.get("Content-Type", "")):
            raise _invalid_response("Kiro Gateway model catalog was not JSON.")
        raw = response.read(_CATALOG_LIMIT + 1)
        _capture("bytes", "kiro.catalog.body", raw)
    if len(raw) > _CATALOG_LIMIT:
        raise _invalid_response("Kiro Gateway model catalog exceeded the read limit.")
    try:
        payload = _loads(raw)
        _semantic_json(payload)
    except (ValueError, UnicodeError) as error:
        raise _invalid_response("Kiro Gateway model catalog was invalid JSON.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise _invalid_response("Kiro Gateway model catalog must contain a data list.")
    names: list[str] = []
    for row in payload["data"]:
        name = row.get("id") if isinstance(row, dict) else None
        if not isinstance(name, str) or not name or len(name) > 256 or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in name):
            raise _invalid_response("Kiro Gateway model catalog contained an invalid model ID.")
        if name not in names:
            names.append(name)
    return names


def _text_content(content: Any, *, summary: bool = False) -> None:
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise ProviderError("Kiro requires text message content.", code="kiro_unsupported_input")
    for part in content:
        if isinstance(part, str):
            continue
        allowed = {"text", "input_text", "output_text"} | ({"summary_text"} if summary else set())
        if (not isinstance(part, dict) or not isinstance(part.get("type"), str) or part["type"] not in allowed
                or not isinstance(part.get("text"), str)):
            raise ProviderError("Kiro currently supports text input only; media and unsupported content blocks cannot be replayed.", code="kiro_unsupported_input")


def _input_items(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = body.get("input")
    if value is None:
        value = body.get("messages")
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                result.append(item)
            else:
                raise ProviderError("Kiro input must contain conversation items.", code="kiro_invalid_request")
        return result
    return []


def _validate_tools(tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        raise ProviderError("Kiro tools must be a list of function definitions.", code="kiro_invalid_tools")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ProviderError("Kiro tools must be function definitions.", code="kiro_invalid_tools")
        if tool.get("type") == "namespace":
            _validate_tools(tool.get("tools"))
        elif tool.get("type") == "function":
            function = tool.get("function", tool)
            if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip() or "parameters" in function and not isinstance(function["parameters"], dict):
                raise ProviderError("Kiro function definitions need a name and an object schema.", code="kiro_invalid_tools")
        else:
            raise ProviderError("Kiro supports function tools only; built-in server tools are not available.", code="kiro_unsupported_tools")


def prepare_request(body: Mapping[str, Any], *, max_output_tokens: int | None = None) -> tuple[dict[str, Any], ToolNameMap]:
    """Prepare full inline replay without importing foreign provider state.

    Inline replay is the caller's contract; we cannot establish completeness
    just because some history was supplied. A new prompt plus an ID alone is
    deliberately refused rather than silently starting a fresh conversation.
    """
    if not isinstance(body, Mapping):
        raise ProviderError("Kiro request must be an object.", code="kiro_invalid_request")
    try:
        prepared = copy.deepcopy(dict(body))
    except RecursionError as error:
        raise ProviderError("Kiro request nesting is too deep.", code="kiro_invalid_request") from error
    try:
        _semantic_json(prepared)
    except ValueError as error:
        raise ProviderError("Kiro request contains invalid JSON values or Unicode.", code="kiro_invalid_request") from error
    _validate_tools(prepared.get("tools"))
    for name in ("temperature", "top_p", "top_k", "stop", "stop_sequences"):
        if prepared.get(name) is not None:
            raise ProviderError("Kiro Gateway does not enforce sampling or stop controls; omit those options.", code="kiro_unsupported_option")
    parallel = prepared.get("parallel_tool_calls")
    if parallel is not None and not isinstance(parallel, bool):
        raise ProviderError("Kiro parallel_tool_calls must be a boolean when supplied.", code="kiro_invalid_request")
    if parallel is False:
        raise ProviderError("Kiro Gateway cannot enforce parallel_tool_calls=false; use another route if single-call turns are required.", code="kiro_unsupported_option")
    choice = prepared.get("tool_choice", "auto")
    if isinstance(choice, dict) and isinstance(choice.get("type"), str) and choice["type"] in {"auto", "none"}:
        choice = choice["type"]
    if choice not in (None, "auto", "none"):
        raise ProviderError("Kiro Gateway does not enforce required or named tool choices; use auto or none.", code="kiro_unsupported_tool_choice")
    prepared["tool_choice"] = choice
    for name in ("instructions", "system"):
        if prepared.get(name) is not None:
            _text_content(prepared[name])
    items = _input_items(prepared)
    if not items:
        raise ProviderError("Kiro requires inline conversation input.", code="kiro_invalid_request")
    for item in items:
        if any(item.get(field) is not None and not isinstance(item[field], str) for field in ("type", "role")):
            raise ProviderError("Kiro conversation item type and role must be strings.", code="kiro_invalid_request")
    actual = [item for item in items if item.get("type") != "item_reference"]
    by_id: dict[str, dict[str, Any]] = {}
    for item in actual:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            if item_id in by_id and by_id[item_id] != item:
                raise ProviderError("Kiro inline history has conflicting item IDs.", code="kiro_invalid_history")
            by_id[item_id] = item
    if "previous_response_id" in prepared:
        previous_id = prepared.pop("previous_response_id")
        if previous_id is not None and previous_id != "":
            has_user = any(item.get("role") == "user" or item.get("type") == "function_call_output" for item in actual)
            has_assistant = any(item.get("role") == "assistant" or item.get("type") == "function_call"
                                or item.get("type") == "message" and item.get("role") is None for item in actual)
            if not isinstance(previous_id, str) or not has_user or not has_assistant:
                raise ProviderError("Kiro has no durable response store; resend the full inline conversation instead of ID-only chaining.", code="kiro_previous_response_unsupported")
            _capture("event", "kiro.redundant_previous_response_id_removed")
    cleaned = []
    calls: dict[str, bool] = {}
    pending_calls: set[str] = set()
    results_started = False

    def assistant_item():
        nonlocal results_started
        if results_started and pending_calls:
            raise ProviderError("Kiro cannot start another assistant turn before all preceding tool results arrive.", code="kiro_invalid_history")
        results_started = False

    for item in items:
        kind = item.get("type")
        if kind == "item_reference":
            if not isinstance(item.get("id"), str) or item["id"] not in by_id:
                raise ProviderError("Kiro cannot resolve a required item reference; resend its inline content.", code="kiro_unresolved_reference")
            continue
        if kind == "reasoning":
            summary = item.get("summary", "")
            if isinstance(summary, list):
                _text_content(summary, summary=True)
                summary = "\n".join(part if isinstance(part, str) else part["text"] for part in summary)
            if not isinstance(summary, str):
                raise ProviderError("Kiro reasoning summaries must be visible text.", code="kiro_invalid_history")
            if summary:
                assistant_item()
            cleaned.append({"type": "reasoning", "summary": summary})
            continue
        if kind == "function_call":
            assistant_item()
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(call_id, str) or not call_id or call_id in calls or not isinstance(item.get("name"), str) or not item["name"].strip():
                raise ProviderError("Kiro tool history requires unique call IDs and tool names.", code="kiro_invalid_history")
            arguments = item.get("arguments", {})
            try:
                arguments = _loads(arguments) if isinstance(arguments, str) else arguments
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments are not an object")
                _semantic_json(arguments)
                json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError, UnicodeError, RecursionError) as error:
                raise ProviderError("Kiro tool history contains invalid JSON arguments.", code="kiro_invalid_history") from error
            item["arguments"] = arguments
            calls[call_id] = False
            pending_calls.add(call_id)
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or call_id not in calls or calls[call_id]:
                raise ProviderError("Kiro tool results must match one preceding inline tool call.", code="kiro_invalid_history")
            calls[call_id] = True
            pending_calls.remove(call_id)
            results_started = True
            if isinstance(item.get("output"), list):
                _text_content(item["output"])
        elif (kind in {None, "message", "text", "input_text", "output_text"}
              and item.get("role") in {"user", "assistant", "system", "developer"}) or kind == "message":
            if item.get("role") not in {None, "user", "assistant", "system", "developer"}:
                raise ProviderError("Kiro message history contains an unsupported role.", code="kiro_invalid_history")
            role = item.get("role") or ("assistant" if kind == "message" else "user")
            if role == "assistant":
                assistant_item()
            elif role == "user" and pending_calls:
                raise ProviderError("Kiro tool results must precede the next conversation turn.", code="kiro_invalid_history")
            _text_content(item.get("content", ""))
        elif kind in {"text", "input_text", "output_text"}:
            if pending_calls:
                raise ProviderError("Kiro tool results must precede the next user turn.", code="kiro_invalid_history")
            _text_content([item])
        else:
            raise ProviderError("Kiro cannot replay unsupported input items; provide text messages and complete tool history.", code="kiro_unsupported_input")
        cleaned.append(item)
    if not all(calls.values()):
        raise ProviderError("Kiro inline history is missing a tool result.", code="kiro_invalid_history")
    # The shared codec's fallback for unsafe call IDs can collide with a
    # client's already-safe ID. Normalize the copied call/result pairs first.
    call_ids = _wire_identifiers(list(calls), "kiro_call_")
    for item in cleaned:
        if item.get("type") == "function_call":
            item["call_id"] = call_ids[item.get("call_id") or item["id"]]
        elif item.get("type") == "function_call_output":
            item["call_id"] = call_ids[item["call_id"]]
    prepared["input"] = cleaned
    prepared.pop("messages", None)
    try:
        payload, tool_names = responses_to_anthropic(prepared, oauth=False, max_output_tokens=max_output_tokens)
    except RecursionError as error:
        raise ProviderError("Kiro request nesting is too deep.", code="kiro_invalid_request") from error
    if calls and not payload.get("tools"):
        raise ProviderError("Kiro Gateway cannot preserve structured tool history when current tools are disabled or empty; use another route for a no-tools continuation.",
                            code="kiro_unsupported_history_tool_policy")
    # KG intercepts the literal web_search name even when its global search
    # switch is off. Alias it (and names exceeding the wire grammar) so the
    # client, not the service, continues to own function execution.
    aliases = _wire_identifiers(list(tool_names.original_to_wire), "kiro_tool_", reserved={"web_search"})
    for tool in payload.get("tools", []):
        tool["name"] = aliases[tool["name"]]
    for message in payload["messages"]:
        for part in message["content"] if isinstance(message["content"], list) else []:
            if part["type"] == "tool_use":
                part["name"] = aliases[part["name"]]
    tool_names = _KiroToolNames(oauth=False, original_to_wire=aliases,
                               wire_to_original={wire: original for original, wire in aliases.items()},
                               tools_allowed=choice != "none",
                               offered_wire_names=frozenset(tool["name"] for tool in payload.get("tools", [])))
    payload["thinking"] = {"type": "disabled"}
    try:
        _semantic_json(payload)
        json.dumps(payload, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ProviderError("Kiro request contains invalid JSON values.", code="kiro_invalid_request") from error
    return payload, tool_names


def _valid_block(block: Any, *, ids: set[str]) -> str:
    if not isinstance(block, dict):
        raise _invalid_response()
    kind = block.get("type")
    if kind == "text":
        if not isinstance(block.get("text"), str):
            raise _invalid_response("Kiro Gateway returned invalid text content.")
    elif kind == "tool_use":
        call_id, name = block.get("id"), block.get("name")
        if not isinstance(call_id, str) or not call_id or call_id in ids or not isinstance(name, str) or not name.strip():
            raise _invalid_response("Kiro Gateway returned missing or duplicate tool identities.")
        ids.add(call_id)
        if not isinstance(block.get("input"), dict):
            raise _invalid_response("Kiro Gateway tool input must be a JSON object.")
        try:
            json.dumps(block["input"], allow_nan=False)
        except (ValueError, TypeError, RecursionError) as error:
            raise _invalid_response("Kiro Gateway returned invalid tool JSON.") from error
    else:
        raise _invalid_response("Kiro Gateway returned an unsupported content block; signed thinking and media replay are not supported.")
    return kind


def _validate_tool_policy(tool_names: ToolNameMap, name: str) -> None:
    # Replay aliases include historical functions, not just current offers.
    # A bare codec name map likewise carries no execution authority.
    offered = getattr(tool_names, "offered_wire_names", frozenset())
    if not getattr(tool_names, "tools_allowed", True) or name not in offered or name == "web_search":
        raise ProviderError("Kiro Gateway returned a tool outside the client's execution policy.", status=502, code="kiro_unexpected_tool")


def _validate_stop(reason: Any, *, has_tools: bool) -> None:
    if not isinstance(reason, str) or reason not in _STOP_REASONS:
        raise _invalid_response("Kiro Gateway did not supply a supported stop reason.")
    if reason == "tool_use" and not has_tools or has_tools and reason not in {"tool_use", "max_tokens"}:
        raise _invalid_response("Kiro Gateway stop reason disagrees with its tool output.")


def convert_json(payload: Mapping[str, Any], *, model: str, tool_names: ToolNameMap,
                 headers: Any = None, upstream_status: int | None = None,
                 secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validate a Messages result against the prepared request's tool policy."""
    if isinstance(payload, dict) and payload.get("type") == "error" and isinstance(payload.get("error"), dict):
        raise upstream_error(source="json", upstream_status=upstream_status, payload=payload,
                             headers=headers, secrets=secrets)
    try:
        _semantic_json(payload)
    except ValueError as error:
        raise _invalid_response("Kiro Gateway returned invalid JSON values or Unicode.") from error
    if not isinstance(payload, dict) or payload.get("type") != "message" or payload.get("role") != "assistant" or not isinstance(payload.get("id"), str) or not payload["id"]:
        raise _invalid_response()
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise _invalid_response("Kiro Gateway returned an empty response.")
    ids: set[str] = set()
    kinds = [_valid_block(block, ids=ids) for block in content]
    for block in content:
        if block["type"] == "tool_use":
            _validate_tool_policy(tool_names, block["name"])
    if not ids and not any(block.get("text") for block in content):
        raise _invalid_response("Kiro Gateway returned an empty response.")
    _validate_stop(payload.get("stop_reason"), has_tools="tool_use" in kinds)
    _validate_usage(payload.get("usage"))
    converted = anthropic_json_to_responses(payload, model=model, tool_names=tool_names, coalesce_text=True)
    # Distinct tool call IDs can collide in the shared synthetic item-ID
    # convention. Keep each inline replay item unambiguous.
    seen: set[str] = set()
    for index, item in enumerate(converted["output"]):
        while item["id"] in seen:
            item["id"] += "_" + str(index)
        seen.add(item["id"])
    return converted


def _validate_usage(usage: Any) -> None:
    if usage is None:
        return
    if not isinstance(usage, dict):
        raise _invalid_response("Kiro Gateway returned invalid usage data.")
    for name in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
                 "cache_read_input_tokens", "cache_creation_input_tokens"):
        if name in usage and (type(usage[name]) is not int or usage[name] < 0):
            raise _invalid_response("Kiro Gateway returned invalid usage data.")


def _sse_events(upstream: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    event_name = ""
    data_lines: list[str] = []
    frame_size = 0
    observed = 0
    first = True
    while True:
        raw = upstream.readline(_EVENT_LIMIT + 1)
        if not raw:
            # SSE events require their blank-line delimiter. An undelimited
            # message_stop at EOF is not evidence of a complete frame.
            if data_lines or event_name:
                raise _invalid_response("Kiro Gateway ended in an incomplete SSE frame.")
            return
        if not isinstance(raw, bytes):
            raise _invalid_response("Kiro Gateway stream was not a byte stream.")
        _capture("event", "kiro.upstream.read", bytes=len(raw))
        _capture("bytes", "kiro.upstream.body", raw)
        _capture("observe_sse", "kiro.upstream", raw)
        frame_size += len(raw)
        observed += len(raw)
        if observed > _RESPONSE_LIMIT:
            raise _invalid_response("Kiro Gateway response exceeded the total read limit.")
        if frame_size > _EVENT_LIMIT:
            raise _invalid_response("Kiro Gateway SSE frame exceeded the read limit.")
        try:
            line = raw.decode("utf-8-sig" if first else "utf-8").rstrip("\r\n")
        except UnicodeError as error:
            raise _invalid_response("Kiro Gateway returned invalid UTF-8.") from error
        first = False
        if not line:
            if data_lines:
                try:
                    payload = _loads("\n".join(data_lines))
                except (ValueError, UnicodeError) as error:
                    raise _invalid_response("Kiro Gateway returned malformed SSE JSON.") from error
                if not isinstance(payload, dict) or not isinstance(payload.get("type"), str) or not payload["type"] or event_name and event_name != payload["type"]:
                    raise _invalid_response("Kiro Gateway SSE event and payload types disagree.")
                try:
                    _semantic_json(payload)
                except ValueError as error:
                    raise _invalid_response("Kiro Gateway returned invalid JSON values or Unicode.") from error
                if payload["type"] == "error":
                    # The general codec remains unchanged; error metadata
                    # alone requires unambiguous keys and a smaller budget.
                    payload = load_error_payload("\n".join(data_lines)) or {"type": "error"}
                yield event_name or payload["type"], payload
            event_name, data_lines, frame_size = "", [], 0
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)


def stream_events(upstream: Any, *, model: str, tool_names: ToolNameMap,
                  secrets: tuple[str, ...] = ()) -> Iterator[tuple[str, dict[str, Any]]]:
    """Validate Messages lifecycle, then reuse the numbered Responses codec.

    A prestream failure raises before headers can be sent. Once any output
    has escaped, a failure is one numbered response.failed, never a second
    HTTP response or a misleading success. Returning at message_stop avoids
    waiting for HTTP-body EOF; the transport owner closes the response.
    Tool responses require the current policy returned by prepare_request;
    an unscoped name map does not authorize function calls.
    """
    state = StreamState(model=model, tool_names=tool_names, coalesce_text=True)
    sequence = 0
    blocks: dict[int, dict[str, Any]] = {}
    tool_ids: set[str] = set()
    precontent_restarts = 0
    output_exposure = "none"
    upstream_status = getattr(upstream, "status", None)

    def discard_tool(index):
        state_block = state.blocks.pop(index, None)
        if state_block is not None:
            state.output[:] = [item for item in state.output if item is not state_block["item"]]
        blocks[index]["events"].clear()
        blocks[index]["arguments"] = ""
        blocks[index]["initial"] = {}
        blocks[index]["initial_arguments"] = ""

    def add_block_bytes(block, fragment):
        try:
            size = len(fragment.encode("utf-8"))
        except UnicodeError as error:
            raise _invalid_response("Kiro Gateway returned invalid block Unicode.") from error
        total = block["utf8_bytes"] + size
        if total > _BLOCK_LIMIT:
            kind = "text" if block["kind"] == "text" else "tool"
            raise _invalid_response(f"Kiro Gateway {kind} block exceeded the read limit.")
        block["utf8_bytes"] = total

    def convert(kind, payload):
        nonlocal sequence, output_exposure
        events = convert_anthropic_event(kind, payload, state)
        for name, data in events:
            # Distinct wire call IDs such as call_x and fc_x can collide in
            # the shared Responses item-ID convention. Keep call_id intact,
            # and update only this output item's synthetic identity.
            if name == "response.output_item.added":
                index = data["output_index"]
                item = state.output[index]
                used = {previous["id"] for previous in state.output[:index]}
                while item["id"] in used:
                    item["id"] += "_" + str(index)
                data["item"]["id"] = item["id"]
            if "item_id" in data:
                data["item_id"] = state.output[data["output_index"]]["id"]
        block = blocks.get(payload.get("index")) if kind.startswith("content_block_") else None
        if block is not None and block["kind"] == "tool_use":
            # A syntactically truncated call is not executable. Hold its
            # events until validation, without delaying ordinary text.
            block["events"].extend(events)
            if kind != "content_block_stop":
                return
            events = block["events"]
            block["events"] = []
        for name, data in events:
            try:
                _semantic_json(data)
            except ValueError as error:
                raise _invalid_response("Kiro Gateway returned invalid outbound Unicode or JSON values.") from error
            if name not in {"response.created", "response.in_progress"}:
                output_exposure = "content"
            elif output_exposure == "none":
                output_exposure = "lifecycle"
            yield name, {**data, "sequence_number": sequence}
            sequence += 1

    try:
        headers = getattr(upstream, "headers", None)
        if headers is not None and headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
            raise _invalid_response("Kiro Gateway streaming response was not event-stream data.")
        for kind, payload in _sse_events(upstream):
            if kind == "ping":
                continue
            if kind == "error":
                raise upstream_error(source="sse", upstream_status=upstream_status, payload=payload,
                                     headers=headers, secrets=secrets)
            if kind == "message_start":
                message = payload.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("id"), str) or not message["id"] or message.get("role") != "assistant" or message.get("content") != []:
                    raise _invalid_response("Kiro Gateway returned an invalid message_start.")
                _validate_usage(message.get("usage"))
                if state.started:
                    if blocks or state.stop_reason is not None or precontent_restarts >= 8:
                        raise _invalid_response("Kiro Gateway restarted a message after content began.")
                    precontent_restarts += 1
                    _capture("event", "kiro.upstream.precontent_restart", count=precontent_restarts)
                    continue  # The service retried; keep one Responses ID.
                yield from convert(kind, payload)
                continue
            if kind not in {"content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}:
                continue  # New informational event types do not invent output.
            if not state.started:
                raise _invalid_response("Kiro Gateway sent content before message_start.")
            if kind.startswith("content_block_"):
                index = payload.get("index")
                if type(index) is not int or index < 0:
                    raise _invalid_response("Kiro Gateway returned an invalid block index.")
                if kind == "content_block_start":
                    if index in blocks or len(blocks) >= 4096 or state.stop_reason is not None or any(not block["closed"] for block in blocks.values()):
                        raise _invalid_response("Kiro Gateway duplicated or misplaced a content block.")
                    block = payload.get("content_block")
                    block_kind = _valid_block(block, ids=tool_ids)
                    if block_kind == "tool_use":
                        _validate_tool_policy(tool_names, block["name"])
                    initial_arguments = (json.dumps(block["input"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                                         if block_kind == "tool_use" else "")
                    blocks[index] = {"kind": block_kind, "closed": False, "initial": block.get("input", {}),
                                     "initial_arguments": initial_arguments, "arguments": "", "text": "", "events": [],
                                     "invalid_arguments": False, "utf8_bytes": 0, "has_argument_delta": False}
                    add_block_bytes(blocks[index], block["text"] if block_kind == "text" else initial_arguments)
                    yield from convert(kind, payload)
                    if block_kind == "text" and block["text"]:
                        blocks[index]["text"] = block["text"]
                        yield from convert("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": block["text"]}})
                    continue
                block = blocks.get(index)
                if block is None or block["closed"]:
                    raise _invalid_response("Kiro Gateway updated a missing or closed content block.")
                if kind == "content_block_delta":
                    delta = payload.get("delta")
                    if not isinstance(delta, dict):
                        raise _invalid_response("Kiro Gateway returned an invalid content delta.")
                    if block["kind"] == "text" and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        add_block_bytes(block, delta["text"])
                        block["text"] += delta["text"]
                    elif block["kind"] == "tool_use" and delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                        if block["initial"]:
                            raise _invalid_response("Kiro Gateway supplied ambiguous initial and streamed tool arguments.")
                        if not block["has_argument_delta"]:
                            # Initial {} is a stream placeholder, not a prefix
                            # of the actual JSON argument fragments.
                            block["utf8_bytes"] = 0
                            block["has_argument_delta"] = True
                        add_block_bytes(block, delta["partial_json"])
                        block["arguments"] += delta["partial_json"]
                    else:
                        raise _invalid_response("Kiro Gateway content delta disagrees with its block.")
                    yield from convert(kind, payload)
                    continue
                if block["kind"] == "tool_use":
                    arguments = block["arguments"]
                    if not arguments:
                        arguments = block["initial_arguments"]
                        block["utf8_bytes"] = 0
                        add_block_bytes(block, arguments)
                        yield from convert("content_block_delta", {"index": index, "delta": {"type": "input_json_delta", "partial_json": arguments}})
                    try:
                        parsed = _loads(arguments)
                        if not isinstance(parsed, dict):
                            raise ValueError("tool input is not an object")
                        _semantic_json(parsed)
                    except json.JSONDecodeError:
                        # The stop reason arrives later. Only max_tokens may
                        # classify this as incomplete; never publish this call.
                        block["closed"] = True
                        block["invalid_arguments"] = True
                        discard_tool(index)
                        continue
                    except (ValueError, UnicodeError) as error:
                        raise _invalid_response("Kiro Gateway ended a tool block with invalid JSON arguments.") from error
                block["closed"] = True
                yield from convert(kind, payload)
                continue
            if kind == "message_delta":
                delta = payload.get("delta")
                if not isinstance(delta, dict) or any(not block["closed"] for block in blocks.values()):
                    raise _invalid_response("Kiro Gateway ended a message with unfinished content.")
                if delta.get("stop_reason") is not None:
                    _validate_stop(delta["stop_reason"], has_tools=bool(tool_ids))
                    if any(block["invalid_arguments"] for block in blocks.values()) and delta["stop_reason"] != "max_tokens":
                        raise _invalid_response("Kiro Gateway ended a tool block with invalid JSON arguments.")
                    if state.stop_reason is not None and state.stop_reason != delta["stop_reason"]:
                        raise _invalid_response("Kiro Gateway changed its stop reason.")
                _validate_usage(payload.get("usage"))
                yield from convert(kind, payload)
                continue
            if not blocks or any(not block["closed"] for block in blocks.values()) or not tool_ids and not any(block["text"] for block in blocks.values()):
                raise _invalid_response("Kiro Gateway ended an empty or unfinished message.")
            _validate_stop(state.stop_reason, has_tools=bool(tool_ids))
            if any(block["invalid_arguments"] for block in blocks.values()) and state.stop_reason != "max_tokens":
                raise _invalid_response("Kiro Gateway ended a tool block with invalid JSON arguments.")
            _capture("event", "kiro.upstream.semantic_stop", status="incomplete" if state.stop_reason == "max_tokens" else "completed")
            yield from convert("message_stop", payload)
            return
        raise _invalid_response("Kiro Gateway stream ended before message_stop.")
    except (ProviderError, OSError, HTTPException) as error:
        if not isinstance(error, ProviderError):
            error = ProviderError("Kiro Gateway stream could not be read completely.", status=502, code="kiro_upstream_error",
                                  metadata=error_metadata(source="transport", upstream_status=upstream_status,
                                                          headers=headers, secrets=secrets))
        if error.metadata is None:
            error.metadata = error_metadata(source="protocol", upstream_status=upstream_status,
                                            headers=headers, secrets=secrets)
        error.metadata["output_exposure"] = output_exposure
        error.metadata["phase"] = "before_response" if sequence == 0 else "stream"
        _capture("event", "kiro.upstream.failed", adapter_code=error.code, **error_fields(error))
        if sequence == 0:
            raise error
        for index, block in blocks.items():
            if block["kind"] == "tool_use" and block["events"]:
                discard_tool(index)
        state.error_type = error.code
        state.error_message = str(error)
        state.finished = True
        response = state.response(status="failed")
        response["error"].update(error_fields(error))
        yield "response.failed", {"type": "response.failed", "sequence_number": sequence, "response": response}
