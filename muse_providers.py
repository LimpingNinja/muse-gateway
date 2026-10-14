"""Native provider transports used by :mod:`muse_gateway`.

The module deliberately uses only the Python standard library.  Muse speaks
the OpenAI Responses wire format; Codex already accepts that format while
Claude needs a small, explicit Responses <-> Anthropic Messages adapter.

No function in this module logs credential values or request bodies.
"""

from __future__ import annotations

import base64
import copy
import datetime as _datetime
import hashlib
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

try:  # POSIX-only coordination; the transport remains usable elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]


DEFAULT_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_USER_AGENT = "codex_cli_rs/0.0.0 (muse-gateway)"

DEFAULT_CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_TOKEN_URLS = (
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
)
CLAUDE_CODE_VERSION = "2.1.74"
CLAUDE_TOKEN_USER_AGENT = "axios/1.7.9"
CLAUDE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
CLAUDE_COMMON_BETAS = (
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
)
CLAUDE_OAUTH_BETAS = (
    "claude-code-20250219",
    "oauth-2025-04-20",
)

_TOKEN_KEY_ALIASES = ("access_token", "accessToken", "access")
_REFRESH_KEY_ALIASES = ("refresh_token", "refreshToken", "refresh")
_EXPIRY_KEY_ALIASES = (
    "expires_at",
    "expiresAt",
    "expiry",
    "expiration",
    "expires_on",
)
_TOKEN_CONTAINER_KEYS = (
    "tokens",
    "oauth",
    "oauth_tokens",
    "credentials",
    "auth",
)


def _canonical_url(value: str) -> str:
    """Normalize only harmless URL spelling differences for allow-list checks."""
    parsed = urllib.parse.urlsplit(str(value or "").strip())
    if not parsed.scheme or not parsed.netloc or parsed.username or parsed.password:
        return ""
    if parsed.query or parsed.fragment:
        return ""
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
    )


def _reject_unless_allowed(
    value: str,
    allowed: Iterable[str],
    *,
    provider: str,
    allow_host_override: bool = False,
) -> str:
    provider_code = provider.lower()
    canonical = _canonical_url(value)
    allowed_canonical = {_canonical_url(item) for item in allowed}
    if allow_host_override:
        if not canonical:
            raise ProviderError(
                f"{provider} OAuth URL is malformed.",
                code=f"{provider_code}_oauth_url_invalid",
            )
        return canonical
    if canonical not in allowed_canonical:
        raise ProviderError(
            f"{provider} OAuth endpoint is not an approved native URL; "
            "use the explicit --allow-oauth-host-override development flag "
            "only with a local fake endpoint.",
            code=f"{provider_code}_oauth_host_not_allowed",
        )
    return canonical


def validate_codex_endpoint(url: str, *, allow_host_override: bool = False) -> str:
    return _reject_unless_allowed(
        url,
        (DEFAULT_CODEX_ENDPOINT, DEFAULT_CODEX_BASE_URL),
        provider="Codex",
        allow_host_override=allow_host_override,
    )


def validate_claude_endpoint(url: str, *, allow_host_override: bool = False) -> str:
    return _reject_unless_allowed(
        url,
        (
            DEFAULT_CLAUDE_ENDPOINT,
            DEFAULT_CLAUDE_BASE_URL,
            DEFAULT_CLAUDE_BASE_URL + "/v1",
        ),
        provider="Claude",
        allow_host_override=allow_host_override,
    )


def validate_codex_token_url(url: str, *, allow_host_override: bool = False) -> str:
    return _reject_unless_allowed(
        url,
        (DEFAULT_CODEX_TOKEN_URL,),
        provider="Codex",
        allow_host_override=allow_host_override,
    )


def validate_claude_token_url(url: str, *, allow_host_override: bool = False) -> str:
    return _reject_unless_allowed(
        url,
        CLAUDE_TOKEN_URLS,
        provider="Claude",
        allow_host_override=allow_host_override,
    )


class ProviderError(Exception):
    """An error safe to expose to the local Muse client."""

    def __init__(self, message: str, *, status: int = 400, code: str = "provider_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class AuthError(ProviderError):
    """A local credential problem with an actionable next step."""

    def __init__(self, message: str, *, code: str = "auth_error"):
        super().__init__(message, status=401, code=code)


@dataclass
class CredentialState:
    provider: str
    access_token: str
    refresh_token: str | None = None
    expires_at: float | None = None
    source: str = "environment"
    path: Path | None = None
    container_path: tuple[str, ...] = ()
    access_key: str = "access_token"
    refresh_key: str | None = "refresh_token"
    expiry_key: str | None = None
    auth_mode: str = "oauth"
    scopes: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _now() -> float:
    return time.time()


def _parse_expiry(value: Any) -> float | None:
    """Parse common OAuth expiry representations into epoch seconds."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        numeric = float(value)
        if numeric > 1_000_000_000_000:
            numeric /= 1000.0
        return numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except ValueError:
            numeric = None
        if numeric is not None and numeric > 0:
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        try:
            return _datetime.datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            return None
    return None


def jwt_claims(token: str) -> dict[str, Any]:
    """Decode an unsigned JWT payload for non-secret routing claims only.

    Signature verification is intentionally not attempted here.  The OAuth
    provider remains the authority; this helper is used only for expiry and
    the account-routing header.  Malformed or opaque tokens simply return an
    empty mapping.
    """
    if not isinstance(token, str):
        return {}
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    try:
        encoded = parts[1] + ("=" * (-len(parts[1]) % 4))
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
        return payload if isinstance(payload, dict) else {}
    except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return {}


def token_account_id(token: str) -> str | None:
    claims = jwt_claims(token)
    candidates = [
        claims.get("chatgpt_account_id"),
        claims.get("account_id"),
    ]
    auth_claim = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict):
        candidates.insert(0, auth_claim.get("chatgpt_account_id"))
    for value in candidates:
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]+", value.strip()):
            return value.strip()
    return None


def token_expiry(token: str, explicit_expiry: float | None = None) -> float | None:
    if explicit_expiry is not None:
        return explicit_expiry
    claim = jwt_claims(token).get("exp")
    return _parse_expiry(claim)


def token_is_expiring(
    token: str,
    explicit_expiry: float | None = None,
    *,
    skew_seconds: int = 120,
) -> bool:
    expiry = token_expiry(token, explicit_expiry)
    return expiry is not None and expiry <= _now() + max(0, int(skew_seconds))


def _credential_file_is_readable(path: Path) -> bool:
    try:
        # Credential targets are deliberately not followed through symlinks.
        # This keeps a later atomic replacement from silently changing an
        # unrelated file and makes credential discovery match write safety.
        return not path.is_symlink() and path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _iter_token_containers(data: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    """Yield likely OAuth containers without recursively harvesting secrets."""
    seen: set[int] = set()

    def emit(value: Any, path: tuple[str, ...]) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
        if not isinstance(value, dict) or id(value) in seen:
            return
        seen.add(id(value))
        yield value, path
        for key in _TOKEN_CONTAINER_KEYS:
            child = value.get(key)
            if isinstance(child, dict):
                yield from emit(child, path + (key,))

    for key in _TOKEN_CONTAINER_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            yield from emit(value, (key,))
    yield from emit(dict(data), ())


def _find_key(container: Mapping[str, Any], aliases: Iterable[str]) -> str | None:
    for key in aliases:
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            return key
    return None


def _find_token_record(
    data: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], str, str | None, str | None] | None:
    for container, path in _iter_token_containers(data):
        access_key = _find_key(container, _TOKEN_KEY_ALIASES)
        if access_key is None:
            continue
        refresh_key = _find_key(container, _REFRESH_KEY_ALIASES)
        expiry_key = next(
            (key for key in _EXPIRY_KEY_ALIASES if key in container), None
        )
        return container, path, access_key, refresh_key, expiry_key
    return None


def _walk_path(data: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or not isinstance(current.get(key), dict):
            return None
        current = current[key]
    return current if isinstance(current, dict) else None


def _file_source_label(provider: str, path: Path, configured: bool = False) -> str:
    if provider == "codex":
        return "CODEX_HOME/auth.json" if configured else "~/.codex/auth.json"
    return "CLAUDE_CODE_CREDENTIALS" if configured else "~/.claude/.credentials.json"


def _candidate_codex_paths(config: Any) -> list[tuple[Path, bool]]:
    configured_home = str(getattr(config, "codex_home", "") or os.environ.get("CODEX_HOME", "")).strip()
    explicit = str(getattr(config, "codex_auth_path", "") or "").strip()
    candidates: list[tuple[Path, bool]] = []
    if explicit:
        candidates.append((Path(explicit).expanduser(), True))
    if configured_home:
        candidates.append((Path(configured_home).expanduser() / "auth.json", True))
    candidates.append((Path.home() / ".codex" / "auth.json", False))
    unique: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for path, configured in candidates:
        resolved = path.absolute()
        if resolved not in seen:
            seen.add(resolved)
            unique.append((path, configured))
    return unique


def _candidate_claude_paths(config: Any) -> list[tuple[Path, bool]]:
    explicit = str(
        getattr(config, "claude_credentials", "")
        or os.environ.get("MUSE_GATEWAY_CLAUDE_CREDENTIALS", "")
        or os.environ.get("MUSE_SHIM_CLAUDE_CREDENTIALS", "")
        or os.environ.get("CLAUDE_CODE_CREDENTIALS_PATH", "")
    ).strip()
    config_dir = str(os.environ.get("CLAUDE_CONFIG_DIR", "")).strip()
    candidates: list[tuple[Path, bool]] = []
    if explicit:
        candidates.append((Path(explicit).expanduser(), True))
    if config_dir:
        candidates.append((Path(config_dir).expanduser() / ".credentials.json", True))
    candidates.append((Path.home() / ".claude" / ".credentials.json", False))
    unique: list[tuple[Path, bool]] = []
    seen: set[Path] = set()
    for path, configured in candidates:
        resolved = path.absolute()
        if resolved not in seen:
            seen.add(resolved)
            unique.append((path, configured))
    return unique


def _state_from_file(
    provider: str,
    path: Path,
    data: dict[str, Any],
    *,
    source_label: str,
) -> CredentialState | None:
    found = _find_token_record(data)
    if found is None:
        return None
    container, path_parts, access_key, refresh_key, expiry_key = found
    access = container.get(access_key)
    if not isinstance(access, str) or not access.strip():
        return None
    refresh = container.get(refresh_key) if refresh_key else None
    if not isinstance(refresh, str) or not refresh.strip():
        refresh = None
    expiry = _parse_expiry(container.get(expiry_key)) if expiry_key else None
    scopes = container.get("scopes")
    return CredentialState(
        provider=provider,
        access_token=access.strip(),
        refresh_token=refresh.strip() if refresh else None,
        expires_at=expiry,
        source=source_label,
        path=path,
        container_path=path_parts,
        access_key=access_key,
        refresh_key=refresh_key,
        expiry_key=expiry_key,
        scopes=list(scopes) if isinstance(scopes, list) else None,
    )


def _env_first(names: Iterable[str]) -> tuple[str, str] | tuple[None, None]:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return name, value
    return None, None


def discover_codex_credentials(config: Any) -> CredentialState | None:
    env_name, access = _env_first(
        (
            "MUSE_GATEWAY_CODEX_ACCESS_TOKEN",
            "MUSE_SHIM_CODEX_ACCESS_TOKEN",
            "CODEX_ACCESS_TOKEN",
            "OPENAI_CODEX_ACCESS_TOKEN",
        )
    )
    if not access:
        access = str(getattr(config, "access_token", "") or "").strip()
        env_name = "command line" if access else None
    if access:
        _, refresh = _env_first(
            (
                "MUSE_GATEWAY_CODEX_REFRESH_TOKEN",
                "MUSE_SHIM_CODEX_REFRESH_TOKEN",
                "CODEX_REFRESH_TOKEN",
            )
        )
        return CredentialState(
            provider="codex",
            access_token=access,
            refresh_token=refresh or None,
            expires_at=token_expiry(access),
            source=env_name or "environment",
            path=None,
        )

    for path, configured in _candidate_codex_paths(config):
        if not _credential_file_is_readable(path):
            continue
        data = _read_json_object(path)
        if data is None:
            continue
        state = _state_from_file(
            "codex", path, data, source_label=_file_source_label("codex", path, configured)
        )
        if state:
            return state
    return None


def _is_oauth_token(value: str) -> bool:
    if not value:
        return False
    if value.startswith("sk-ant-api"):
        return False
    return value.startswith(("sk-ant-", "cc-", "eyJ", "oauth-"))


def _read_claude_keychain() -> CredentialState | None:
    """Best-effort macOS Keychain support; inference never uses a subprocess."""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True,
            text=True,
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    access = oauth.get("accessToken") or oauth.get("access_token")
    if not isinstance(access, str) or not access.strip():
        return None
    refresh = oauth.get("refreshToken") or oauth.get("refresh_token")
    return CredentialState(
        provider="claude",
        access_token=access.strip(),
        refresh_token=refresh.strip() if isinstance(refresh, str) and refresh.strip() else None,
        expires_at=_parse_expiry(oauth.get("expiresAt") or oauth.get("expires_at")),
        source="macOS Keychain",
        path=None,
        scopes=list(oauth.get("scopes")) if isinstance(oauth.get("scopes"), list) else None,
    )


def discover_claude_credentials(config: Any) -> CredentialState | None:
    env_name, token = _env_first(
        (
            "MUSE_GATEWAY_CLAUDE_ACCESS_TOKEN",
            "MUSE_SHIM_CLAUDE_ACCESS_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_TOKEN",
        )
    )
    if not token:
        token = str(getattr(config, "access_token", "") or "").strip()
        env_name = "command line" if token else None
    if token:
        refresh_name, refresh = _env_first(
            (
                "MUSE_GATEWAY_CLAUDE_REFRESH_TOKEN",
                "MUSE_SHIM_CLAUDE_REFRESH_TOKEN",
                "CLAUDE_CODE_REFRESH_TOKEN",
            )
        )
        if not _is_oauth_token(token):
            return CredentialState(
                provider="claude",
                access_token=token,
                source=env_name or "environment",
                auth_mode="api_key",
            )
        return CredentialState(
            provider="claude",
            access_token=token,
            refresh_token=refresh or None,
            expires_at=token_expiry(token),
            source=env_name or "environment",
        )

    api_key = str(
        getattr(config, "api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    ).strip()
    if api_key:
        if _is_oauth_token(api_key):
            return CredentialState(
                provider="claude",
                access_token=api_key,
                refresh_token=None,
                expires_at=token_expiry(api_key),
                source="ANTHROPIC_API_KEY",
            )
        return CredentialState(
            provider="claude",
            access_token=api_key,
            source="ANTHROPIC_API_KEY",
            auth_mode="api_key",
        )

    for path, configured in _candidate_claude_paths(config):
        if not _credential_file_is_readable(path):
            continue
        data = _read_json_object(path)
        if data is None:
            continue
        # Claude Code's canonical shape is claudeAiOauth. The generic token
        # scanner also accepts the small set of current alternate shapes.
        oauth = data.get("claudeAiOauth")
        if isinstance(oauth, dict):
            normalized = {
                "tokens": {
                    "accessToken": oauth.get("accessToken") or oauth.get("access_token"),
                    "refreshToken": oauth.get("refreshToken") or oauth.get("refresh_token"),
                    "expiresAt": oauth.get("expiresAt") or oauth.get("expires_at"),
                    "scopes": oauth.get("scopes"),
                }
            }
            state = _state_from_file(
                "claude",
                path,
                normalized,
                source_label=_file_source_label("claude", path, configured),
            )
            if state:
                state.container_path = ("claudeAiOauth",)
                state.access_key = "accessToken"
                state.refresh_key = "refreshToken" if oauth.get("refreshToken") else None
                state.expiry_key = "expiresAt" if "expiresAt" in oauth else (
                    "expires_at" if "expires_at" in oauth else None
                )
                state.scopes = list(oauth.get("scopes")) if isinstance(oauth.get("scopes"), list) else None
                return state
        state = _state_from_file(
            "claude", path, data, source_label=_file_source_label("claude", path, configured)
        )
        if state:
            return state

    return _read_claude_keychain()


def _urlopen(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener: Callable[..., Any] | None = None,
) -> Any:
    if opener is not None:
        return opener(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def _http_status(error: BaseException) -> int | None:
    value = getattr(error, "code", None)
    return int(value) if isinstance(value, int) else None


def _json_response(response: Any) -> dict[str, Any]:
    try:
        value = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError) as exc:
        raise ProviderError("OAuth endpoint returned invalid JSON.", status=502, code="oauth_invalid_json") from exc
    if not isinstance(value, dict):
        raise ProviderError("OAuth endpoint returned an invalid response.", status=502, code="oauth_invalid_response")
    return value


def refresh_codex_oauth(
    refresh_token: str,
    *,
    token_url: str = DEFAULT_CODEX_TOKEN_URL,
    timeout: float = 20.0,
    opener: Callable[..., Any] | None = None,
    allow_host_override: bool = False,
) -> dict[str, Any]:
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex credentials have no refresh token. Run `codex login` and try again.",
            code="codex_missing_refresh_token",
        )
    token_url = validate_codex_token_url(
        token_url,
        allow_host_override=allow_host_override,
    )
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }
    ).encode("ascii")
    request = urllib.request.Request(
        token_url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": CODEX_USER_AGENT,
        },
    )
    try:
        with _urlopen(request, timeout=timeout, opener=opener) as response:
            payload = _json_response(response)
    except urllib.error.HTTPError as exc:
        status = _http_status(exc) or 502
        if status == 429:
            raise AuthError(
                "Codex OAuth refresh is rate-limited; retry later. Credentials were not changed.",
                code="codex_refresh_rate_limited",
            ) from exc
        raise AuthError(
            f"Codex OAuth refresh failed (HTTP {status}). Run `codex login` and try again.",
            code="codex_refresh_failed",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError("Codex OAuth refresh could not reach the token service.", status=502, code="codex_refresh_unreachable") from exc

    access = payload.get("access_token") or payload.get("accessToken")
    if not isinstance(access, str) or not access.strip():
        raise AuthError(
            "Codex OAuth refresh returned no access token. Run `codex login` and try again.",
            code="codex_refresh_missing_access_token",
        )
    next_refresh = payload.get("refresh_token") or payload.get("refreshToken")
    if not isinstance(next_refresh, str) or not next_refresh.strip():
        next_refresh = refresh_token.strip()
    expires_at = _parse_expiry(payload.get("expires_at") or payload.get("expiresAt"))
    if expires_at is None:
        expires_in = _parse_expiry(payload.get("expires_in"))
        if expires_in is not None and expires_in < 10_000_000:
            expires_at = _now() + expires_in
    return {
        "access_token": access.strip(),
        "refresh_token": next_refresh.strip(),
        "expires_at": expires_at,
    }


def refresh_claude_oauth(
    refresh_token: str,
    *,
    token_urls: Iterable[str] = CLAUDE_TOKEN_URLS,
    timeout: float = 20.0,
    opener: Callable[..., Any] | None = None,
    allow_host_override: bool = False,
) -> dict[str, Any]:
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Claude OAuth credentials have no refresh token. Run `claude setup-token` or `claude login`.",
            code="claude_missing_refresh_token",
        )
    token_urls = tuple(
        validate_claude_token_url(
            endpoint,
            allow_host_override=allow_host_override,
        )
        for endpoint in token_urls
    )
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLAUDE_OAUTH_CLIENT_ID,
        }
    ).encode("ascii")
    last_error: BaseException | None = None
    for endpoint in token_urls:
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": CLAUDE_TOKEN_USER_AGENT,
            },
        )
        try:
            with _urlopen(request, timeout=timeout, opener=opener) as response:
                payload = _json_response(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            continue
        access = payload.get("access_token") or payload.get("accessToken")
        if not isinstance(access, str) or not access.strip():
            raise AuthError(
                "Claude OAuth refresh returned no access token. Run `claude setup-token` or `claude login`.",
                code="claude_refresh_missing_access_token",
            )
        next_refresh = payload.get("refresh_token") or payload.get("refreshToken")
        if not isinstance(next_refresh, str) or not next_refresh.strip():
            next_refresh = refresh_token.strip()
        expires_at = _parse_expiry(payload.get("expires_at") or payload.get("expiresAt"))
        if expires_at is None:
            expires_in = _parse_expiry(payload.get("expires_in"))
            if expires_in is not None and expires_in < 10_000_000:
                expires_at = _now() + expires_in
        return {
            "access_token": access.strip(),
            "refresh_token": next_refresh.strip(),
            "expires_at": expires_at,
            "scopes": payload.get("scope") if isinstance(payload.get("scope"), list) else None,
        }
    status = _http_status(last_error) if last_error else None
    suffix = f" (HTTP {status})" if status else ""
    raise AuthError(
        f"Claude OAuth refresh failed{suffix}. Run `claude setup-token` or `claude login`.",
        code="claude_refresh_failed",
    ) from last_error


@contextmanager
def _credential_refresh_lock(path: Path) -> Iterator[None]:
    """Serialize file-backed refreshes when POSIX locking is available.

    The lock is advisory and only coordinates processes that use this gateway's
    lock file. Provider CLIs that ignore it remain outside that coordination.
    """
    if fcntl is None:
        yield
        return
    lock_path = path.with_name(path.name + ".lock")
    fd: int | None = None
    try:
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, stat.S_IRUSR | stat.S_IWUSR)
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    except OSError as exc:
        raise ProviderError(
            "Could not lock the provider credential file safely.",
            status=500,
            code="credential_lock_failed",
        ) from exc
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass


def _ensure_safe_credential_target(path: Path) -> None:
    try:
        if path.is_symlink():
            raise ProviderError(
                "Refusing to update a symlinked provider credential file.",
                status=500,
                code="credential_symlink_refused",
            )
        if path.exists() and not path.is_file():
            raise ProviderError(
                "Provider credential target is not a regular file.",
                status=500,
                code="credential_target_invalid",
            )
    except OSError as exc:
        raise ProviderError(
            "Could not inspect the provider credential file safely.",
            status=500,
            code="credential_target_invalid",
        ) from exc


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a credential JSON file with mode 0600."""
    parent = path.parent
    if not parent.exists():
        raise ProviderError("Credential directory does not exist; re-run the provider login.", status=500, code="credential_directory_missing")
    _ensure_safe_credential_target(path)
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
        temporary_path = Path(temporary)
        try:
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        except AttributeError:  # pragma: no cover - Windows fallback
            os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Check again immediately before replacement. os.replace does not
            # follow the destination symlink, but refusing a changed target is
            # clearer and avoids overwriting a newly introduced link.
            _ensure_safe_credential_target(path)
            os.replace(temporary_path, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise ProviderError("Could not safely update the provider credential file.", status=500, code="credential_write_failed") from exc


def _update_credential_file(state: CredentialState, refreshed: Mapping[str, Any]) -> None:
    if state.path is None:
        return
    _ensure_safe_credential_target(state.path)
    current = _read_json_object(state.path)
    if current is None:
        raise ProviderError("Credential file is not valid JSON; run the provider login again.", status=500, code="credential_file_invalid")

    if state.provider == "claude" and state.container_path == ("claudeAiOauth",):
        container = current.get("claudeAiOauth")
        if not isinstance(container, dict):
            raise ProviderError("Claude credential file has an invalid OAuth section; run `claude login`.", status=500, code="claude_credentials_invalid")
    else:
        container = _walk_path(current, state.container_path)
    if container is None:
        raise ProviderError("Credential file changed shape; run the provider login again.", status=500, code="credential_shape_changed")

    access = str(refreshed.get("access_token") or "").strip()
    refresh = str(refreshed.get("refresh_token") or "").strip()
    if not access or not refresh:
        raise ProviderError("Provider refresh returned incomplete credentials.", status=502, code="refresh_incomplete")
    container[state.access_key] = access
    if state.refresh_key:
        container[state.refresh_key] = refresh
    else:
        # A canonical Claude file always uses this spelling. For a supported
        # alternate shape, add the conventional key only inside its OAuth
        # container so future refreshes can rotate it too.
        container["refreshToken" if state.provider == "claude" else "refresh_token"] = refresh
    if state.expiry_key and refreshed.get("expires_at") is not None:
        old_value = container.get(state.expiry_key)
        expiry = float(refreshed["expires_at"])
        container[state.expiry_key] = int(expiry * 1000) if isinstance(old_value, (int, float)) and old_value > 1_000_000_000_000 else int(expiry)
    elif state.provider == "claude" and refreshed.get("expires_at") is not None:
        container["expiresAt"] = int(float(refreshed["expires_at"]) * 1000)
    elif state.provider == "codex" and refreshed.get("expires_at") is not None:
        container["expires_at"] = int(float(refreshed["expires_at"]))
    if state.provider == "claude" and refreshed.get("scopes") and isinstance(refreshed["scopes"], list):
        container["scopes"] = refreshed["scopes"]
    _atomic_write_json(state.path, current)


def _reload_file_state(state: CredentialState) -> CredentialState | None:
    """Re-read a file-backed state immediately before a refresh.

    This intentionally uses the same narrow shape recognition as discovery so
    a rotated credential is adopted without copying a stale snapshot back over
    unrelated fields.
    """
    if state.path is None:
        return state
    _ensure_safe_credential_target(state.path)
    current = _read_json_object(state.path)
    if current is None:
        return None
    if state.provider == "claude" and state.container_path == ("claudeAiOauth",):
        oauth = current.get("claudeAiOauth")
        if not isinstance(oauth, dict):
            return None
        normalized = {
            "tokens": {
                "accessToken": oauth.get("accessToken") or oauth.get("access_token"),
                "refreshToken": oauth.get("refreshToken") or oauth.get("refresh_token"),
                "expiresAt": oauth.get("expiresAt") or oauth.get("expires_at"),
                "scopes": oauth.get("scopes"),
            }
        }
        latest = _state_from_file(
            "claude",
            state.path,
            normalized,
            source_label=state.source,
        )
        if latest is None:
            return None
        latest.container_path = ("claudeAiOauth",)
        latest.access_key = "accessToken"
        latest.refresh_key = "refreshToken" if oauth.get("refreshToken") else None
        latest.expiry_key = "expiresAt" if "expiresAt" in oauth else (
            "expires_at" if "expires_at" in oauth else None
        )
        latest.scopes = list(oauth.get("scopes")) if isinstance(oauth.get("scopes"), list) else None
        return latest
    return _state_from_file("codex", state.path, current, source_label=state.source) if state.provider == "codex" else _state_from_file(
        "claude", state.path, current, source_label=state.source
    )


def _set_refreshed_state(state: CredentialState, refreshed: Mapping[str, Any]) -> CredentialState:
    _update_credential_file(state, refreshed)
    state.access_token = str(refreshed["access_token"])
    state.refresh_token = str(refreshed["refresh_token"])
    state.expires_at = refreshed.get("expires_at")
    return state


def resolve_codex_credentials(
    config: Any,
    *,
    force_refresh: bool = False,
    opener: Callable[..., Any] | None = None,
) -> CredentialState:
    state = discover_codex_credentials(config)
    if state is None or not state.access_token:
        raise AuthError(
            "Codex OAuth credentials were not found. Run `codex login` and try again.",
            code="codex_credentials_missing",
        )
    if any(character in state.access_token for character in "\r\n"):
        raise AuthError(
            "Codex credential data is malformed. Run `codex login` and try again.",
            code="codex_credentials_invalid",
        )
    skew = int(getattr(config, "refresh_skew_seconds", 120) or 120)
    should_refresh = force_refresh or token_is_expiring(state.access_token, state.expires_at, skew_seconds=skew)
    if should_refresh and state.refresh_token:
        allow_override = bool(getattr(config, "allow_oauth_host_override", False))
        token_url = str(
            getattr(config, "codex_token_url", "")
            or os.environ.get("MUSE_GATEWAY_CODEX_TOKEN_URL", "")
            or os.environ.get("MUSE_SHIM_CODEX_TOKEN_URL", "")
            or DEFAULT_CODEX_TOKEN_URL
        )
        if state.path is not None:
            with _credential_refresh_lock(state.path):
                latest = _reload_file_state(state)
                if latest is None or not latest.access_token:
                    raise AuthError(
                        "Codex credential file changed or is invalid. Run `codex login` and try again.",
                        code="codex_credentials_invalid",
                    )
                if (
                    latest.access_token != state.access_token
                    and not token_is_expiring(latest.access_token, latest.expires_at, skew_seconds=skew)
                ):
                    return latest
                state = latest
                if not state.refresh_token:
                    raise AuthError(
                        "Codex credentials have no refresh token. Run `codex login` and try again.",
                        code="codex_missing_refresh_token",
                    )
                refreshed = refresh_codex_oauth(
                    state.refresh_token,
                    token_url=token_url,
                    timeout=float(getattr(config, "timeout", 20.0) or 20.0),
                    opener=opener,
                    allow_host_override=allow_override,
                )
                return _set_refreshed_state(state, refreshed)
        refreshed = refresh_codex_oauth(
            state.refresh_token,
            token_url=token_url,
            timeout=float(getattr(config, "timeout", 20.0) or 20.0),
            opener=opener,
            allow_host_override=allow_override,
        )
        return _set_refreshed_state(state, refreshed)
    elif should_refresh and (
        force_refresh or (state.expires_at is not None and state.expires_at <= _now())
    ):
        raise AuthError(
            "Codex access token has expired and cannot be refreshed. Run `codex login` and try again.",
            code="codex_credentials_expired",
        )
    return state


def resolve_claude_credentials(
    config: Any,
    *,
    force_refresh: bool = False,
    opener: Callable[..., Any] | None = None,
) -> CredentialState:
    state = discover_claude_credentials(config)
    if state is None or not state.access_token:
        raise AuthError(
            "Claude credentials were not found. Run `claude setup-token` for OAuth or `claude login` for a normal login.",
            code="claude_credentials_missing",
        )
    if any(character in state.access_token for character in "\r\n"):
        raise AuthError(
            "Claude credential data is malformed. Run `claude setup-token` or `claude login`.",
            code="claude_credentials_invalid",
        )
    if state.auth_mode == "api_key":
        return state
    skew = int(getattr(config, "refresh_skew_seconds", 120) or 120)
    should_refresh = force_refresh or token_is_expiring(state.access_token, state.expires_at, skew_seconds=skew)
    if should_refresh and state.refresh_token:
        allow_override = bool(getattr(config, "allow_oauth_host_override", False))
        urls = str(
            getattr(config, "claude_token_url", "")
            or os.environ.get("MUSE_GATEWAY_CLAUDE_TOKEN_URL", "")
            or os.environ.get("MUSE_SHIM_CLAUDE_TOKEN_URL", "")
        ).strip()
        token_urls = (urls,) if urls else CLAUDE_TOKEN_URLS
        if state.path is not None:
            with _credential_refresh_lock(state.path):
                latest = _reload_file_state(state)
                if latest is None or not latest.access_token:
                    raise AuthError(
                        "Claude credential file changed or is invalid. Run `claude setup-token` or `claude login`.",
                        code="claude_credentials_invalid",
                    )
                if (
                    latest.access_token != state.access_token
                    and not token_is_expiring(latest.access_token, latest.expires_at, skew_seconds=skew)
                ):
                    return latest
                state = latest
                if not state.refresh_token:
                    raise AuthError(
                        "Claude credentials have no refresh token. Run `claude setup-token` or `claude login`.",
                        code="claude_missing_refresh_token",
                    )
                refreshed = refresh_claude_oauth(
                    state.refresh_token,
                    token_urls=token_urls,
                    timeout=float(getattr(config, "timeout", 20.0) or 20.0),
                    opener=opener,
                    allow_host_override=allow_override,
                )
                return _set_refreshed_state(state, refreshed)
        refreshed = refresh_claude_oauth(
            state.refresh_token,
            token_urls=token_urls,
            timeout=float(getattr(config, "timeout", 20.0) or 20.0),
            opener=opener,
            allow_host_override=allow_override,
        )
        return _set_refreshed_state(state, refreshed)
    elif should_refresh and (
        force_refresh or (state.expires_at is not None and state.expires_at <= _now())
    ):
        raise AuthError(
            "Claude access token has expired and cannot be refreshed. Run `claude setup-token` or `claude login`.",
            code="claude_credentials_expired",
        )
    return state


def codex_headers(access_token: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "text/event-stream, application/json",
        "Content-Type": "application/json",
        "User-Agent": CODEX_USER_AGENT,
        "originator": "codex_cli_rs",
    }
    account_id = token_account_id(access_token)
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    return headers


def claude_headers(state: CredentialState, *, stream: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream, application/json" if stream else "application/json",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if state.auth_mode == "api_key":
        headers["x-api-key"] = state.access_token
        headers["anthropic-beta"] = ",".join(CLAUDE_COMMON_BETAS)
    else:
        headers.update(
            {
                "Authorization": f"Bearer {state.access_token}",
                "anthropic-beta": ",".join((*CLAUDE_COMMON_BETAS, *CLAUDE_OAUTH_BETAS)),
                "user-agent": f"claude-code/{CLAUDE_CODE_VERSION} (external, cli)",
                "x-app": "cli",
            }
        )
    return headers


def _fold_system_messages(
    turn_input: Any,
) -> tuple[Any, str] | None:
    """Split system-role items out of a Responses input list.

    Returns (remaining_input, system_text), or None for non-list input.
    Unsupported system content is rejected rather than silently discarded.
    """
    if not isinstance(turn_input, list):
        return None
    remaining: list[Any] = []
    texts: list[str] = []
    for item in turn_input:
        if not isinstance(item, dict) or item.get("role") != "system":
            remaining.append(item)
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if (
                    not isinstance(part, dict)
                    or part.get("type") not in (None, "text", "input_text")
                    or not isinstance(part.get("text"), str)
                ):
                    raise ProviderError(
                        "Codex system messages must contain text only; no context was discarded.",
                        code="codex_system_content_unsupported",
                    )
                texts.append(part["text"])
        else:
            raise ProviderError(
                "Codex system messages must contain text only; no context was discarded.",
                code="codex_system_content_unsupported",
            )
    return remaining, "\n\n".join(texts)


# Client-side mislabels for Codex wire efforts, keyed by lowercase alias.
# Applied only when the target model rejects the sent value but accepts the
# correction (see correct_codex_effort). Known case: a stale Muse variant menu
# sends "minimal" to gpt-6-astra, whose wire efforts are low/medium/high/xhigh/max.
CODEX_EFFORT_CORRECTIONS = {"minimal": "low", "ultra": "max"}


def correct_codex_effort(
    effort: Any, supported_efforts: list[str] | tuple[str, ...]
) -> tuple[Any, bool]:
    """Reconcile one inbound Codex reasoning effort with model evidence.

    Returns (effort_to_send, was_corrected). With no evidence (empty supported
    list, e.g. bundled fallbacks) the value passes through untouched. A known
    mislabel is corrected only when the model supports the correction.
    Anything else unsupported raises ProviderError with the supported list so
    Muse surfaces an actionable error instead of an opaque upstream 400.
    """
    if not isinstance(effort, str) or not supported_efforts:
        return effort, False
    supported = [level for level in supported_efforts if isinstance(level, str)]
    if effort in supported:
        return effort, False
    lowered = effort.lower()
    if lowered in supported:
        return lowered, True
    corrected = CODEX_EFFORT_CORRECTIONS.get(lowered)
    if corrected is not None and corrected in supported:
        return corrected, True
    raise ProviderError(
        f"Reasoning effort {effort!r} is not supported by this Codex model "
        f"(supported: {', '.join(supported)}). Reselect the effort in Muse.",
        code="codex_unsupported_reasoning_effort",
    )


def prepare_codex_request(body: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the Codex Responses backend's required request preflight."""
    payload = copy.deepcopy(dict(body))
    payload["store"] = False
    payload.pop("max_output_tokens", None)
    if isinstance(payload.get("tools"), list) and not payload["tools"]:
        payload.pop("tools", None)
    if isinstance(payload.get("input"), str):
        # The Codex backend rejects string input ("Input must be a list";
        # observed on gpt-6.1-sol and gpt-6-astra alike), which Muse sends for
        # simple tool-free prompts. Wrap it in the canonical message form,
        # which the same models accept. Semantics are unchanged.
        payload["input"] = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": payload["input"]}],
            }
        ]
    # The Codex backend rejects system-role messages ("System messages are
    # not allowed"), which resumed sessions carry as context. Fold their text
    # into `instructions` (the Responses system-prompt field) so the context
    # survives instead of 400ing or being silently dropped. Muse's own
    # `instructions` system prompt is always preserved first.
    folded = _fold_system_messages(payload.get("input"))
    if folded is not None:
        remaining, system_text = folded
        payload["input"] = remaining
        if system_text:
            prior = payload.get("instructions")
            if prior is not None and not isinstance(prior, str):
                raise ProviderError("Codex instructions must be text.", code="codex_instructions_invalid")
            prefix = f"{prior}\n\n" if prior else ""
            payload["instructions"] = f"{prefix}{system_text}"
    if payload.get("previous_response_id"):
        # store=False keeps no server-side state, so a chained ID can never
        # resolve upstream and forwarding it only produces an opaque 400.
        # Observed Muse traffic carries the full conversation inline, so the
        # ID is redundant: drop it and proceed with the complete turn. The
        # drop is logged by the caller (never silent). Only a turn with no
        # input at all is rejected, since there is nothing to forward.
        if payload.get("input"):
            payload.pop("previous_response_id", None)
        else:
            raise ProviderError(
                "Codex response chaining is not supported by this stateless "
                "gateway, and this turn carries no input to send instead. "
                "Retry as a fresh turn with complete conversation input.",
                code="codex_previous_response_unsupported",
            )
    payload.pop("previous_response_id", None)
    history = payload.get("input")
    if isinstance(history, list):
        # Muse qualifies replayed item ids as response_id:item_id. Codex
        # accepts only alphanumerics, underscores and dashes. Keep the native
        # component when unambiguous; hash other invalid ids without changing
        # call_id (which pairs function calls with their results).
        occupied = {
            item["id"] for item in history if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", item["id"])
        }
        remapped = {}
        for item in history:
            if not isinstance(item, dict) or item.get("type") == "item_reference":
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", item_id):
                continue
            if item_id not in remapped:
                prefix = {"reasoning": "rs_", "function_call": "fc_"}.get(item.get("type"), "msg_")
                tail = item_id.rsplit(":", 1)[-1]
                candidate = tail if tail.startswith(prefix) and re.fullmatch(r"[A-Za-z0-9_-]+", tail) else ""
                if not candidate or candidate in occupied:
                    candidate = prefix + hashlib.sha256(item_id.encode()).hexdigest()[:32]
                    while candidate in occupied:
                        candidate = prefix + hashlib.sha256(candidate.encode()).hexdigest()[:32]
                occupied.add(candidate)
                remapped[item_id] = candidate
            item["id"] = remapped[item_id]
        for item in history:
            if (isinstance(item, dict) and item.get("type") == "item_reference"
                    and isinstance(item.get("id"), str) and item["id"] in remapped):
                item["id"] = remapped[item["id"]]
    return payload


def demote_codex_encrypted_reasoning(body: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Portable fallback only after Codex explicitly rejects encrypted state."""
    payload = copy.deepcopy(dict(body))
    history = payload.get("input")
    if not isinstance(history, list):
        return payload, 0
    portable = []
    demoted = 0
    demoted_ids = set()
    for item in history:
        if not (isinstance(item, dict) and item.get("type") == "reasoning" and item.get("encrypted_content")):
            portable.append(item)
            continue
        summary = item.get("summary", [])
        text = summary if isinstance(summary, str) else "\n".join(
            part["text"] for part in summary
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ) if isinstance(summary, list) else ""
        if text:
            portable.append({"type": "message", "role": "assistant",
                             "content": [{"type": "output_text", "text": text}]})
        if isinstance(item.get("id"), str):
            demoted_ids.add(item["id"])
        demoted += 1
    payload["input"] = [item for item in portable if not (
        isinstance(item, dict) and item.get("type") == "item_reference"
        and isinstance(item.get("id"), str) and item["id"] in demoted_ids
    )]
    return payload, demoted


def _flatten_tools(tools: Any) -> list[Any]:
    if not isinstance(tools, list):
        return []
    result: list[Any] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "namespace":
            result.append(tool)
            name = tool.get("name") if isinstance(tool, dict) else None
            if isinstance(name, str):
                if name in names:
                    raise ProviderError(f"duplicate tool name {name!r} after namespace flattening.", code="duplicate_tool")
                names.add(name)
            continue
        namespace = tool.get("name", "<unnamed>")
        children = tool.get("tools")
        if not isinstance(children, list):
            raise ProviderError(f"Muse namespace {namespace!r} has no tools list.", code="invalid_tools")
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("name"), str) or not child.get("name"):
                raise ProviderError(f"Muse namespace {namespace!r} contains an unnamed tool.", code="invalid_tools")
            name = child["name"]
            if name in names:
                raise ProviderError(f"duplicate tool name {name!r} after namespace flattening.", code="duplicate_tool")
            names.add(name)
            result.append(child)
    return result


def _wire_tool_name(name: str, oauth: bool) -> str:
    if not oauth:
        return name
    # Anthropic subscription routing rejects a single-underscore mcp_ name.
    # The two escape cases keep this mapping injective, including for tools
    # that already begin with mcp__.
    if name.startswith("mcp__"):
        return "mcp___" + name[len("mcp__") :]
    if name.startswith("mcp_"):
        return "mcp__mcp_" + name[len("mcp_") :]
    return "mcp__" + name


def _decode_wire_tool_name(name: str, oauth: bool) -> str:
    if not oauth or not isinstance(name, str):
        return name
    if name.startswith("mcp___"):
        return "mcp__" + name[len("mcp___") :]
    if name.startswith("mcp__mcp_"):
        return "mcp_" + name[len("mcp__mcp_") :]
    if name.startswith("mcp__"):
        return name[len("mcp__") :]
    return name


@dataclass
class ToolNameMap:
    oauth: bool = False
    original_to_wire: dict[str, str] = field(default_factory=dict)
    wire_to_original: dict[str, str] = field(default_factory=dict)

    def add(self, name: str) -> str:
        if name in self.original_to_wire:
            return self.original_to_wire[name]
        wire = _wire_tool_name(name, self.oauth)
        self.original_to_wire[name] = wire
        self.wire_to_original[wire] = name
        return wire

    def decode(self, name: str) -> str:
        return self.wire_to_original.get(name, _decode_wire_tool_name(name, self.oauth))


def _responses_tool_parts(tool: Mapping[str, Any]) -> tuple[str, str, Any] | None:
    if tool.get("type") != "function":
        return None
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = fn.get("description", "")
    parameters = fn.get("parameters") or {"type": "object", "properties": {}}
    return name.strip(), str(description or ""), parameters


def convert_tools_to_anthropic(tools: Any, *, oauth: bool = False) -> tuple[list[dict[str, Any]], ToolNameMap]:
    mapping = ToolNameMap(oauth=oauth)
    converted: list[dict[str, Any]] = []
    for raw in _flatten_tools(tools):
        if not isinstance(raw, dict):
            continue
        parts = _responses_tool_parts(raw)
        if parts is None:
            continue
        name, description, parameters = parts
        wire_name = mapping.add(name)
        converted.append(
            {
                "name": wire_name,
                "description": description,
                "input_schema": copy.deepcopy(parameters),
            }
        )
    return converted, mapping


def _content_parts(content: Any, *, role: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content or "")}]
    result: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                result.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str):
                result.append({"type": "text", "text": text})
        elif ptype in {"input_image", "image_url"}:
            image_url = part.get("image_url")
            detail = part.get("detail")
            if isinstance(image_url, dict):
                detail = image_url.get("detail", detail)
                image_url = image_url.get("url")
            if not isinstance(image_url, str) or not image_url:
                continue
            if image_url.startswith("data:") and ";base64," in image_url:
                header, encoded = image_url.split(";base64,", 1)
                media_type = header[5:] or "image/png"
                result.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": encoded},
                    }
                )
            else:
                result.append(
                    {"type": "image", "source": {"type": "url", "url": image_url}}
                )
        elif ptype == "image":
            source = part.get("source")
            if isinstance(source, dict):
                result.append({"type": "image", "source": copy.deepcopy(source)})
    return result


def _text_from_content(content: Any) -> str:
    parts = _content_parts(content, role="user")
    return "".join(part.get("text", "") for part in parts if part.get("type") == "text")


def _tool_result_content(output: Any) -> Any:
    if isinstance(output, list):
        parts = _content_parts(output, role="user")
        return parts or ""
    if output is None:
        return ""
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


def _append_message(messages: list[dict[str, Any]], role: str, content: Any) -> None:
    if role not in {"user", "assistant"}:
        return
    if not content:
        content = [{"type": "text", "text": "(empty)"}]
    if messages and messages[-1].get("role") == role:
        previous = messages[-1].get("content")
        if not isinstance(previous, list):
            previous = [{"type": "text", "text": str(previous)}]
        current = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        messages[-1]["content"] = previous + current
    else:
        messages.append({"role": role, "content": content})


def _normalize_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if normalized and normalized[-1]["role"] == role:
            prev = normalized[-1].get("content")
            prev_list = prev if isinstance(prev, list) else [{"type": "text", "text": str(prev)}]
            current_list = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            normalized[-1]["content"] = prev_list + current_list
        else:
            normalized.append(message)
    if not normalized:
        return [{"role": "user", "content": "(empty request)"}]
    if normalized[0]["role"] != "user":
        normalized.insert(0, {"role": "user", "content": "(conversation start)"})
    return normalized


def _input_items(body: Mapping[str, Any]) -> list[Any]:
    value = body.get("input")
    if value is None:
        value = body.get("messages", [])
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _reasoning_config(body: Mapping[str, Any]) -> tuple[bool, str]:
    value = body.get("reasoning")
    if isinstance(value, dict):
        enabled = value.get("enabled") is not False
        effort = str(value.get("effort") or "medium").lower()
        return enabled, effort
    effort = body.get("reasoning_effort")
    return (True, str(effort).lower()) if effort else (False, "medium")


_THINKING_BUDGETS = {
    "minimal": 4096,
    "low": 4096,
    "medium": 8000,
    "high": 16000,
    "xhigh": 32000,
    "ultra": 32000,
    "max": 32000,
}


def _supports_adaptive_thinking(model: str) -> bool:
    lower = model.lower()
    if "claude" not in lower:
        return False
    legacy = (
        "claude-3",
        "claude-opus-4-0",
        "claude-opus-4.0",
        "claude-opus-4-1",
        "claude-opus-4.1",
        "claude-sonnet-4-0",
        "claude-sonnet-4.0",
        "claude-opus-4-5",
        "claude-opus-4.5",
        "claude-opus-4-2025",
        "claude-sonnet-4-5",
        "claude-sonnet-4.5",
        "claude-sonnet-4-2025",
        "claude-haiku-4-5",
        "claude-haiku-4.5",
    )
    if lower in {"claude-opus-4", "claude-sonnet-4"}:
        return False
    return not any(marker in lower for marker in legacy)


def _anthropic_effort(effort: str, model: str) -> str:
    value = {"minimal": "low", "ultra": "max"}.get(effort, effort)
    if value not in {"low", "medium", "high", "xhigh", "max"}:
        value = "medium"
    if value == "xhigh" and ("4-6" in model or "4.6" in model):
        return "max"
    return value


def responses_to_anthropic(
    body: Mapping[str, Any],
    *,
    oauth: bool = False,
    model_override: str | None = None,
    max_output_tokens: int | None = None,
) -> tuple[dict[str, Any], ToolNameMap]:
    if "previous_response_id" in body:
        raise ProviderError(
            "Claude mode does not support previous_response_id because the gateway has no durable response-state store.",
            code="claude_previous_response_unsupported",
        )
    model = str(model_override or body.get("model") or "").strip()
    if not model:
        raise ProviderError("request is missing model; set MUSE_GATEWAY_MODEL or send model.", code="model_missing")
    anthropic_tools, tool_names = convert_tools_to_anthropic(body.get("tools"), oauth=oauth)
    system_parts: list[dict[str, Any]] = []
    instructions = body.get("instructions", body.get("system"))
    if instructions is not None:
        system_parts.extend(_content_parts(instructions, role="system"))
    messages: list[dict[str, Any]] = []
    call_ids: dict[str, str] = {}

    for raw in _input_items(body):
        if not isinstance(raw, dict):
            continue
        item_type = str(raw.get("type") or "").strip()
        if item_type == "function_call":
            original_call_id = str(raw.get("call_id") or raw.get("id") or "").strip()
            if not original_call_id:
                original_call_id = "call_" + hashlib.sha256(
                    str(raw.get("name") or "tool").encode("utf-8")
                ).hexdigest()[:16]
            wire_call_id = original_call_id
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", wire_call_id):
                wire_call_id = "muse_call_" + base64.urlsafe_b64encode(
                    original_call_id.encode("utf-8")
                ).decode("ascii").rstrip("=")
                if len(wire_call_id) > 64:
                    wire_call_id = "muse_call_" + hashlib.sha256(
                        original_call_id.encode("utf-8")
                    ).hexdigest()[:48]
            call_ids[wire_call_id] = original_call_id
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            tool_names.add(name)
            arguments = raw.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    parsed_arguments = {}
            elif isinstance(arguments, dict):
                parsed_arguments = copy.deepcopy(arguments)
            else:
                parsed_arguments = {}
            if not isinstance(parsed_arguments, dict):
                parsed_arguments = {"value": parsed_arguments}
            block = {
                "type": "tool_use",
                "id": wire_call_id,
                "name": tool_names.original_to_wire.get(name, _wire_tool_name(name, oauth)),
                "input": parsed_arguments,
            }
            if messages and messages[-1].get("role") == "assistant":
                content = messages[-1].setdefault("content", [])
                if not isinstance(content, list):
                    content = [{"type": "text", "text": str(content)}]
                    messages[-1]["content"] = content
                content.append(block)
            else:
                _append_message(messages, "assistant", [block])
            continue

        if item_type == "function_call_output":
            original_call_id = str(raw.get("call_id") or "").strip()
            wire_call_id = next(
                (wire for wire, original in call_ids.items() if original == original_call_id),
                original_call_id,
            )
            block = {
                "type": "tool_result",
                "tool_use_id": wire_call_id,
                "content": _tool_result_content(raw.get("output", "")),
            }
            if messages and messages[-1].get("role") == "user":
                content = messages[-1].setdefault("content", [])
                if isinstance(content, list):
                    content.append(block)
                else:
                    messages[-1]["content"] = [{"type": "text", "text": str(content)}, block]
            else:
                _append_message(messages, "user", [block])
            continue

        if item_type == "reasoning":
            # OpenAI encrypted reasoning is endpoint-bound and cannot be
            # replayed to Anthropic. A visible summary, when supplied, is safe.
            summary = raw.get("summary")
            if isinstance(summary, str) and summary.strip():
                _append_message(messages, "assistant", [{"type": "text", "text": summary}])
            continue

        role = raw.get("role")
        if item_type == "message" and role is None:
            role = "assistant"
        if role in {"system", "developer"}:
            system_parts.extend(_content_parts(raw.get("content", ""), role="system"))
            continue
        if role not in {"user", "assistant"}:
            if item_type in {"input_text", "output_text", "text"}:
                role = "user"
            else:
                continue
        parts = _content_parts(raw.get("content", raw.get("text", "")), role=str(role))
        _append_message(messages, str(role), parts)

    messages = _normalize_anthropic_messages(messages)
    if oauth:
        prefix = {"type": "text", "text": CLAUDE_SYSTEM_PREFIX}
        system_parts.insert(0, prefix)
        for part in system_parts:
            if part.get("type") == "text":
                text = str(part.get("text") or "")
                part["text"] = (
                    text.replace("Hermes Agent", "Claude Code")
                    .replace("Hermes agent", "Claude Code")
                    .replace("hermes-agent", "claude-code")
                    .replace("Nous Research", "Anthropic")
                )
        # History tool_use blocks are already wire-encoded where they are
        # built (original_to_wire lookup with a single-encode fallback), so
        # there is intentionally no second pass here: re-encoding a wire name
        # such as mcp__read_file would produce mcp___read_file, which matches
        # neither the schema nor the decode map.

    requested_max = body.get("max_output_tokens", body.get("max_tokens"))
    try:
        max_tokens = int(requested_max) if requested_max is not None else 32768
    except (TypeError, ValueError):
        max_tokens = 32768
    if max_tokens <= 0:
        raise ProviderError("max_output_tokens must be a positive integer.", code="invalid_output_limit")
    if max_output_tokens is not None:
        max_tokens = min(max_tokens, int(max_output_tokens))
    if max_tokens <= 0:
        raise ProviderError("max_output_tokens must be a positive integer.", code="invalid_output_limit")
    # Muse reasoning is intentionally not mapped to Anthropic thinking yet.
    # Signed thinking/redacted-thinking blocks must be replayed exactly across
    # tool turns; silently enabling thinking here would make those turns
    # lossy. Keep the request on the ordinary Messages contract until a
    # lossless round-trip is implemented.
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": bool(body.get("stream")),
    }
    if system_parts:
        payload["system"] = system_parts
    if anthropic_tools:
        payload["tools"] = anthropic_tools
        choice = body.get("tool_choice", "auto")
        if choice in (None, "auto"):
            payload["tool_choice"] = {"type": "auto"}
        elif choice == "required":
            payload["tool_choice"] = {"type": "any"}
        elif choice == "none":
            payload.pop("tools", None)
        elif isinstance(choice, dict):
            requested_name = choice.get("name")
            if not isinstance(requested_name, str) and isinstance(choice.get("function"), dict):
                requested_name = choice["function"].get("name")
            if choice.get("type") in {"function", "tool"} and isinstance(requested_name, str):
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": tool_names.original_to_wire.get(requested_name, _wire_tool_name(requested_name, oauth)),
                }
        elif isinstance(choice, str):
            payload["tool_choice"] = {
                "type": "tool",
                "name": tool_names.original_to_wire.get(choice, _wire_tool_name(choice, oauth)),
            }
    for source, destination in (("temperature", "temperature"), ("top_p", "top_p")):
        value = body.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[destination] = value
    stop = body.get("stop") or body.get("stop_sequences")
    if isinstance(stop, str):
        payload["stop_sequences"] = [stop]
    elif isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        payload["stop_sequences"] = stop
    if isinstance(body.get("metadata"), dict):
        payload["metadata"] = copy.deepcopy(body["metadata"])
    return payload, tool_names


def _responses_function_id(raw_id: str, call_id: str) -> str:
    candidate = str(raw_id or "")
    if candidate.startswith("fc_") and re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        return candidate
    if call_id.startswith("call_"):
        return "fc_" + call_id[len("call_") :]
    digest = hashlib.sha256((candidate or call_id).encode("utf-8")).hexdigest()[:24]
    return "fc_" + digest


def anthropic_json_to_responses(
    payload: Mapping[str, Any],
    *,
    model: str | None = None,
    tool_names: ToolNameMap | None = None,
) -> dict[str, Any]:
    tool_names = tool_names or ToolNameMap(oauth=False)
    if payload.get("type") == "error":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        return {
            "id": "resp_" + secrets.token_hex(8),
            "object": "response",
            "status": "failed",
            "model": model or payload.get("model"),
            "output": [],
            "error": {
                "message": str(error.get("message") or "Claude returned an error."),
                "type": str(error.get("type") or "upstream_error"),
            },
        }
    raw_response_id = str(payload.get("id") or "")
    response_id = (
        raw_response_id
        if raw_response_id.startswith("resp_")
        else "resp_" + hashlib.sha256(raw_response_id.encode("utf-8")).hexdigest()[:24]
    )
    if not raw_response_id:
        response_id = "resp_" + secrets.token_hex(8)
    output: list[dict[str, Any]] = []
    text_message: dict[str, Any] | None = None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    for index, block in enumerate(payload.get("content", []) if isinstance(payload.get("content"), list) else []):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            if text_message is None or not output or output[-1] is not text_message:
                text_message = {
                    "type": "message",
                    "id": "msg_" + hashlib.sha256(response_id.encode("utf-8")).hexdigest()[:20],
                    "role": "assistant",
                    "status": "completed",
                    "content": [],
                }
                output.append(text_message)
            text_message["content"].append(
                {
                    "type": "output_text",
                    "text": str(block.get("text") or ""),
                    "annotations": [],
                }
            )
        elif block_type == "tool_use":
            raw_id = str(block.get("id") or "tool_" + str(index))
            call_id = tool_names.wire_to_original.get(raw_id, raw_id)
            name = tool_names.decode(str(block.get("name") or ""))
            arguments = block.get("input", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            output.append(
                {
                    "type": "function_call",
                    "id": _responses_function_id(raw_id, call_id),
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }
            )
    stop_reason = payload.get("stop_reason")
    status = "incomplete" if stop_reason == "max_tokens" else "completed"
    result: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model or payload.get("model"),
        "output": output,
    }
    if usage:
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        result["usage"] = {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }
    if status == "incomplete":
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    return result


@dataclass
class StreamState:
    model: str | None
    tool_names: ToolNameMap
    response_id: str = ""
    output: list[dict[str, Any]] = field(default_factory=list)
    blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    started: bool = False
    finished: bool = False

    def ensure_id(self, value: Any = None) -> None:
        if self.response_id:
            return
        candidate = str(value or "")
        self.response_id = (
            candidate
            if candidate.startswith("resp_")
            else "resp_" + hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:24]
        ) if candidate else "resp_" + secrets.token_hex(8)

    def response(self, *, status: str = "in_progress") -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.response_id or "resp_" + secrets.token_hex(8),
            "object": "response",
            "status": status,
            "model": self.model,
            "output": copy.deepcopy(self.output),
        }
        if self.usage:
            result["usage"] = {
                "input_tokens": self.usage.get("input_tokens", 0),
                "output_tokens": self.usage.get("output_tokens", 0),
                "total_tokens": self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0),
            }
        if status == "incomplete":
            result["incomplete_details"] = {"reason": "max_output_tokens"}
        if status == "failed" and self.error_message:
            result["error"] = {
                "message": self.error_message,
                "type": self.error_type or "upstream_error",
            }
        return result


def _event_payload(event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    payload["type"] = event_type
    return payload


def convert_anthropic_event(
    event_type: str,
    data: Mapping[str, Any],
    state: StreamState,
) -> list[tuple[str, dict[str, Any]]]:
    """Convert one Anthropic SSE event to OpenAI Responses event payloads."""
    event_type = str(event_type or data.get("type") or "")
    events: list[tuple[str, dict[str, Any]]] = []
    if state.finished:
        return events
    if event_type == "message_start":
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        state.ensure_id(message.get("id"))
        if not state.model:
            state.model = message.get("model")
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        if isinstance(usage.get("input_tokens"), int):
            state.usage["input_tokens"] = usage["input_tokens"]
        if not state.started:
            state.started = True
            events.append(
                (
                    "response.created",
                    _event_payload(
                        "response.created", {"response": state.response()}
                    ),
                )
            )
            events.append(
                (
                    "response.in_progress",
                    _event_payload(
                        "response.in_progress", {"response": state.response()}
                    ),
                )
            )
        return events
    if not state.started:
        state.ensure_id()
        state.started = True
        events.append(
            (
                "response.created",
                _event_payload("response.created", {"response": state.response()}),
            )
        )
        events.append(
            (
                "response.in_progress",
                _event_payload("response.in_progress", {"response": state.response()}),
            )
        )

    if event_type == "content_block_start":
        index = int(data.get("index", len(state.blocks)))
        block = data.get("content_block") if isinstance(data.get("content_block"), dict) else {}
        block_type = block.get("type")
        if block_type == "text":
            item_id = "msg_" + hashlib.sha256(f"{state.response_id}:{index}".encode()).hexdigest()[:20]
            item = {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
            state.blocks[index] = {"kind": "text", "item": item, "text": "", "content_index": 0}
            state.output.append(item)
            events.append(("response.output_item.added", _event_payload("response.output_item.added", {"output_index": len(state.output) - 1, "item": copy.deepcopy(item)})))
            events.append(("response.content_part.added", _event_payload("response.content_part.added", {"item_id": item_id, "output_index": len(state.output) - 1, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})))
        elif block_type == "tool_use":
            raw_id = str(block.get("id") or f"tool_{index}")
            call_id = state.tool_names.wire_to_original.get(raw_id, raw_id)
            name = state.tool_names.decode(str(block.get("name") or ""))
            item_id = _responses_function_id(raw_id, call_id)
            item = {
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress",
            }
            state.blocks[index] = {"kind": "tool", "item": item, "arguments": ""}
            state.output.append(item)
            events.append(("response.output_item.added", _event_payload("response.output_item.added", {"output_index": len(state.output) - 1, "item": copy.deepcopy(item)})))
        return events

    if event_type == "content_block_delta":
        index = int(data.get("index", 0))
        block = state.blocks.get(index)
        delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
        if block and block.get("kind") == "text" and delta.get("type") == "text_delta":
            text = str(delta.get("text") or "")
            block["text"] += text
            item = block["item"]
            content = item.setdefault("content", [])
            if not content:
                content.append({"type": "output_text", "text": "", "annotations": []})
            content[0]["text"] = block["text"]
            output_index = state.output.index(item)
            events.append(("response.output_text.delta", _event_payload("response.output_text.delta", {"item_id": item["id"], "output_index": output_index, "content_index": 0, "delta": text})))
        elif block and block.get("kind") == "tool" and delta.get("type") == "input_json_delta":
            fragment = str(delta.get("partial_json") or "")
            block["arguments"] += fragment
            block["item"]["arguments"] = block["arguments"]
            output_index = state.output.index(block["item"])
            events.append(("response.function_call_arguments.delta", _event_payload("response.function_call_arguments.delta", {"item_id": block["item"]["id"], "output_index": output_index, "delta": fragment})))
        return events

    if event_type == "content_block_stop":
        index = int(data.get("index", 0))
        block = state.blocks.get(index)
        if not block:
            return events
        item = block["item"]
        output_index = state.output.index(item)
        if block.get("kind") == "text":
            events.append(("response.output_text.done", _event_payload("response.output_text.done", {"item_id": item["id"], "output_index": output_index, "content_index": 0, "text": block["text"]})))
            events.append(("response.content_part.done", _event_payload("response.content_part.done", {"item_id": item["id"], "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": block["text"], "annotations": []}})))
        else:
            arguments = block.get("arguments") or "{}"
            item["arguments"] = arguments
            events.append(("response.function_call_arguments.done", _event_payload("response.function_call_arguments.done", {"item_id": item["id"], "output_index": output_index, "arguments": arguments})))
        item["status"] = "completed"
        events.append(("response.output_item.done", _event_payload("response.output_item.done", {"output_index": output_index, "item": copy.deepcopy(item)})))
        return events

    if event_type == "message_delta":
        delta = data.get("delta") if isinstance(data.get("delta"), dict) else {}
        if delta.get("stop_reason") is not None:
            state.stop_reason = str(delta.get("stop_reason"))
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        for key in ("input_tokens", "output_tokens"):
            if isinstance(usage.get(key), int):
                state.usage[key] = usage[key]
        return events

    if event_type == "error":
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        message = str(error.get("message") or "Claude upstream returned an error.")
        state.error_message = message
        state.error_type = str(error.get("type") or "upstream_error")
        events.append(("error", _event_payload("error", {"error": {"message": message, "type": str(error.get("type") or "upstream_error")}})))
        state.stop_reason = "error"
        return events

    if event_type == "message_stop":
        status = (
            "incomplete"
            if state.stop_reason == "max_tokens"
            else "failed"
            if state.stop_reason == "error"
            else "completed"
        )
        final = state.response(status=status)
        final_event = (
            "response.incomplete"
            if status == "incomplete"
            else "response.failed"
            if status == "failed"
            else "response.completed"
        )
        events.append((final_event, _event_payload(final_event, {"response": final})))
        state.finished = True
        return events
    return events


def _iter_sse_text(stream: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    event_name = ""
    data_lines: list[str] = []
    while True:
        line = stream.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"type": "error", "error": {"type": "invalid_sse", "message": "Claude returned invalid SSE data."}}
                if isinstance(data, dict):
                    yield event_name or str(data.get("type") or ""), data
            event_name = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            data = {"type": "error", "error": {"type": "invalid_sse", "message": "Claude returned invalid SSE data."}}
        if isinstance(data, dict):
            yield event_name or str(data.get("type") or ""), data


def response_sse_line(event_name: str, payload: Mapping[str, Any]) -> bytes:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(dict(payload), ensure_ascii=False, separators=(',', ':'))}\n\n"
    ).encode("utf-8")


def convert_anthropic_stream(
    stream: Any,
    *,
    model: str | None,
    tool_names: ToolNameMap,
) -> Iterator[bytes]:
    state = StreamState(model=model, tool_names=tool_names)
    sequence = 0

    def encode_event(event_name, payload):
        nonlocal sequence
        # Muse's Responses decoder requires numbering on every event,
        # including synthetic lifecycle/failure events on truncated streams.
        numbered = {**payload, "sequence_number": sequence}
        sequence += 1
        return response_sse_line(event_name, numbered)

    for event_name, payload in _iter_sse_text(stream):
        for output_event, output_payload in convert_anthropic_event(event_name, payload, state):
            yield encode_event(output_event, output_payload)
    if not state.finished:
        if not state.started:
            for output_event, output_payload in convert_anthropic_event(
                "message_start", {"type": "message_start", "message": {}}, state
            ):
                yield encode_event(output_event, output_payload)
        state.stop_reason = "error"
        state.error_message = state.error_message or (
            "Claude stream ended before message_stop."
        )
        state.error_type = state.error_type or "truncated_stream"
        status = "failed"
        event_name = "response.failed"
        yield encode_event(event_name, _event_payload(event_name, {"response": state.response(status=status)}))
        state.finished = True


def translate_anthropic_sse(
    raw_sse: str,
    *,
    model: str | None = None,
    tool_names: ToolNameMap | None = None,
) -> str:
    """Convenience conversion for deterministic tests and small callers."""
    import io

    return b"".join(
        convert_anthropic_stream(
            io.BytesIO(raw_sse.encode("utf-8")),
            model=model,
            tool_names=tool_names or ToolNameMap(oauth=False),
        )
    ).decode("utf-8")


def endpoint_for(base: str, suffix: str) -> str:
    value = str(base or "").rstrip("/")
    if value.endswith("/" + suffix.lstrip("/")) or value.endswith(suffix.lstrip("/")):
        return value
    return value + "/" + suffix.lstrip("/")


# Descriptive aliases kept public for small integrations and tests that want
# to use the adapters without constructing an HTTP server.
convert_responses_to_anthropic = responses_to_anthropic
convert_anthropic_response = anthropic_json_to_responses
convert_anthropic_sse = translate_anthropic_sse
