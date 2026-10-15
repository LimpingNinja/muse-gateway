"""Devin (Codeium Cascade) provider for muse-gateway.

Native provider speaking the Connect/protobuf wire protocol directly
(``muse_devin_wire``), following OMP at pinned rev 818dd42 action by action.
Rule 2 applies throughout: match OMP where mechanically sound; our own
mechanisms are additive only. Deviations and junie-audit adoptions are
marked inline. Native signed-reasoning replay is not implemented: no
caller-supplied signature is treated as trusted provider state.

Layout:
  credentials ......... OMP SQLite store (read-only) + env token + selection
  discovery ........... GetCliModelConfigs normalization, lanes, fusion filter
  chat ................ history/tools translation, assign-before-chat, stream
"""

import base64
import copy
import hashlib
from http.client import HTTPException
import json
import os
import re
import sqlite3
import struct
import time
import uuid

from muse_devin_wire import (
    ASSIGN_MODEL_PATH,
    AUTH_PATH,
    CONNECT_COMPRESSED_FLAG,
    CONNECT_END_STREAM_FLAG,
    DEFAULT_BASE_URL,
    GET_CLI_MODEL_CONFIGS_PATH,
    DevinAuthError,
    DevinHttpError,
    DevinTrailerError,
    DevinWireError,
    decode_unary,
    discovery_metadata,
    iter_chat_frames,
    decompress_frame,
    legacy_metadata,
    normalize_base_url,
    normalize_session_token,
    open_chat_stream,
    parse_trailer_error,
    unary_rpc,
    wire_metadata,
)

import devin_pb2
from muse_capture import current_capture


class DevinCredentialError(DevinWireError):
    """No usable Devin credential (unconfigured/ambiguous/expired)."""


class DevinDiscoveryError(DevinWireError):
    """Model discovery failed or returned nothing usable."""


class DevinChatError(DevinWireError):
    """Chat request construction or streaming failed."""

    def __init__(self, message, code="devin_chat_error"):
        self.code = code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Credentials (OMP auth/sqlite-credential-store.ts, registry/engine/common.ts)
# ---------------------------------------------------------------------------

OMP_PROVIDER_KEY = "devin"
OAUTH_CREDENTIAL_TYPE = "oauth"
# KDL fallback when the token carries no usable exp claim.
OAUTH_EXPIRY_FALLBACK_MS = 365 * 24 * 60 * 60 * 1000
# B1: clamp absurd exp claims (float overflow would never expire).
MAX_SANE_EXP_S = 4102444800  # 2100-01-01


def _default_auth_db_candidates():
    """Candidate agent.db paths, best-effort mirror of OMP dirs.ts.

    ``~/.omp/agent/agent.db`` by default, honoring PI_CONFIG_DIR for the
    config root name plus the XDG data-home flattening on non-Windows.
    """
    home = os.path.expanduser("~")
    root_name = os.environ.get("PI_CONFIG_DIR", ".omp") or ".omp"
    candidates = [os.path.join(home, root_name, "agent", "agent.db")]
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg and os.name != "nt":
        candidates.append(os.path.join(xdg, "omp", "agent.db"))
    return candidates


def resolve_auth_db(explicit_path=None):
    """Resolve the credential database: explicit path or first candidate."""
    if explicit_path:
        return explicit_path
    for candidate in _default_auth_db_candidates():
        if os.path.isfile(candidate):
            return candidate
    return None


def _cli_credentials_candidates():
    """Default Devin CLI credentials.toml locations (keys only, never logged)."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming"
        )
        return [os.path.join(appdata, "devin", "credentials.toml")]
    home = os.path.expanduser("~")
    candidates = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates.append(os.path.join(xdg, "devin", "credentials.toml"))
    candidates.append(os.path.join(home, ".config", "devin", "credentials.toml"))
    candidates.append(os.path.join(home, ".devin", "credentials.toml"))
    return candidates


def _parse_toml_string_table(text):
    """Minimal fallback parser for flat string tables (no tomllib)."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in "\"'"
            and name.replace("_", "").isalnum()
        ):
            values[name] = (
                value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            )
    return values


def read_cli_token(explicit_path=None):
    """Read the Devin CLI's own login (windsurf_api_key). Read-only.

    Returns ``(token, path)`` or ``(None, None)``. Values are never logged.
    """
    candidates = (
        [explicit_path] if explicit_path else _cli_credentials_candidates()
    )
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        try:
            import tomllib

            parsed = tomllib.loads(text)
        except ImportError:
            parsed = _parse_toml_string_table(text)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        token = parsed.get("windsurf_api_key")
        if isinstance(token, str) and token.strip():
            return token.strip(), candidate
    return None, None


def _jwt_expiry_ms(token):
    """Decode a JWT exp claim without verifying (OMP decodeJwtPayload).

    Returns None when the token is not a JWT or carries no numeric exp.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
    if not isinstance(claims, dict):
        return None
    exp = claims.get("exp")
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    if exp > MAX_SANE_EXP_S:  # B1: overflow guard, fall through to fallback
        return None
    return int(exp * 1000)


def _credential_expiry_ms(access_token):
    expiry = _jwt_expiry_ms(access_token or "")
    if expiry is None:
        return int(time.time() * 1000) + OAUTH_EXPIRY_FALLBACK_MS
    return expiry


def read_store_credentials(auth_db_path):
    """Read active devin rows from OMP's store. Read-only; never migrates.

    Returns ``(active, disabled)`` row dicts: id/identity_key/access/refresh/
    expires_at/email/disabled_cause. Malformed rows are silently skipped,
    matching OMP. Missing file/table means no credentials, not an error.
    """
    active, disabled = [], []
    if not auth_db_path or not os.path.isfile(auth_db_path):
        return active, disabled
    try:
        connection = sqlite3.connect(
            f"file:{auth_db_path}?mode=ro", uri=True, timeout=2.0
        )
    except sqlite3.Error:
        return active, disabled
    try:
        try:
            rows = connection.execute(
                "SELECT id, credential_type, data, disabled_cause,"
                " identity_key FROM auth_credentials"
                " WHERE provider = ? ORDER BY id ASC",
                (OMP_PROVIDER_KEY,),
            ).fetchall()
        except sqlite3.Error:
            return active, disabled
        for row_id, cred_type, data, cause, identity in rows:
            try:
                payload = json.loads(data)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            record = {
                "id": row_id,
                "credential_type": cred_type,
                "identity_key": identity,
                "access": payload.get("access") or "",
                "refresh": payload.get("refresh") or "",
                "email": payload.get("email"),
                "disabled_cause": cause,
            }
            if cause is not None:
                disabled.append(record)
                continue
            if cred_type != OAUTH_CREDENTIAL_TYPE or not record["access"]:
                continue
            record["expires_at"] = payload.get("expires")
            if not isinstance(record["expires_at"], (int, float)):
                record["expires_at"] = _credential_expiry_ms(record["access"])
            else:
                record["expires_at"] = int(record["expires_at"])
            active.append(record)
    finally:
        connection.close()
    return active, disabled


def select_credential(active_rows, credential_id=None):
    """Pick one credential row. Ambiguity without an explicit selector is an
    error (ratified: stricter than OMP's silent first-row pick)."""
    if not active_rows:
        raise DevinCredentialError(
            "no active Devin credential: log in via the Devin CLI or set a token"
        )
    if credential_id is not None:
        wanted = str(credential_id)
        for row in active_rows:
            if str(row["id"]) == wanted or (row["identity_key"] or "") == wanted:
                return row
        raise DevinCredentialError(
            f"Devin credential {credential_id!r} not found among"
            f" {len(active_rows)} active row(s)"
        )
    if len(active_rows) > 1:
        raise DevinCredentialError(
            f"{len(active_rows)} active Devin credentials: pass an explicit"
            " credential id (row id or identity key)"
        )
    return active_rows[0]


# Both reference clients refuse to *use* a token within its last 60 s
# (OMP OAUTH_REFRESH_SKEW_MS; junie lease_expiry_is_usable). Do not invent a
# larger cushion: 60 s is the ratified cross-client bound.
EXPIRY_USE_SKEW_MS = 60_000

# In-memory auth-failure marks, keyed by token fingerprint: junie's
# ReauthRequired quarantine without touching OMP's store (read-only by
# design). A re-login mints new token bytes, which unmarks naturally.
_AUTH_FAILURE_MARKS = {}


def _token_fingerprint(access_token):
    return hashlib.sha256((access_token or "").encode("utf-8")).hexdigest()[:16]


def mark_auth_failure(access_token, detail=""):
    """Remember a persistent GetUserJwt 401 for this token (process lifetime)."""
    _AUTH_FAILURE_MARKS[_token_fingerprint(access_token)] = {
        "at": int(time.time() * 1000),
        "detail": detail,
    }


def auth_failure_mark(access_token):
    """Return the mark dict for this token, or None."""
    return _AUTH_FAILURE_MARKS.get(_token_fingerprint(access_token))


def credential_state(auth_db_path=None, credential_id=None, api_key=None):
    """Resolve credentials into a status dict for diagnostics.

    States: ``ready`` | ``unconfigured`` | ``ambiguous`` | ``expired`` |
    ``not-found`` | ``auth-failed``. Never raises for credential problems.
    """
    if api_key:
        mark = auth_failure_mark(api_key)
        if mark is not None:
            return {
                "state": "auth-failed",
                "source": "explicit",
                "credential_id": None,
                "identity": None,
                "note": (
                    "Devin account requires sign-in again (auth failed;"
                    " explicit token quarantined)"
                ),
            }
        return {
            "state": "ready",
            "source": "explicit",
            "credential_id": None,
            "identity": None,
            "expires_at": None,
            "note": "explicit token; expiry unchecked",
        }
    if credential_id is None:
        # The Devin CLI's own login wins over the OMP-compatible store: it is
        # the user's primary login. An explicit credential id skips it and
        # targets store rows below.
        cli_token, cli_path = read_cli_token()
        if cli_token is not None:
            if auth_failure_mark(cli_token) is not None:
                return {
                    "state": "auth-failed",
                    "source": "devin-cli",
                    "path": cli_path,
                    "note": (
                        "Devin account requires sign-in again (run"
                        " `devin auth login`)"
                    ),
                }
            return {
                "state": "ready",
                "source": "devin-cli",
                "path": cli_path,
                "access_token": cli_token,
                "expires_at": None,
                "note": "Devin CLI login; expiry unchecked",
            }
    resolved = resolve_auth_db(auth_db_path)
    if resolved is None:
        return {"state": "unconfigured", "source": None}
    active, disabled = read_store_credentials(resolved)
    try:
        row = select_credential(active, credential_id)
    except DevinCredentialError as error:
        text = str(error)
        if "explicit" in text:
            return {
                "state": "ambiguous",
                "source": "store",
                "path": resolved,
                "count": len(active),
                "note": text,
            }
        if "not found" in text:
            return {
                "state": "not-found",
                "source": "store",
                "path": resolved,
                "note": text,
            }
        return {
            "state": "unconfigured",
            "source": "store",
            "path": resolved,
            "disabled": len(disabled),
        }
    now_ms = int(time.time() * 1000)
    if row["expires_at"] <= now_ms + EXPIRY_USE_SKEW_MS:
        return {
            "state": "expired",
            "source": "store",
            "path": resolved,
            "credential_id": row["id"],
            "identity": row["identity_key"] or row["email"],
            "note": "log in again via the Devin CLI (no refresh grant exists)",
        }
    mark = auth_failure_mark(row["access"])
    if mark is not None:
        return {
            "state": "auth-failed",
            "source": "store",
            "path": resolved,
            "credential_id": row["id"],
            "identity": row["identity_key"] or row["email"],
            "note": (
                "Devin account requires sign-in again (GetUserJwt keeps"
                " returning 401 for this token)"
            ),
        }
    return {
        "state": "ready",
        "source": "store",
        "path": resolved,
        "credential_id": row["id"],
        "identity": row["identity_key"] or row["email"],
        "access_token": row["access"],
        "expires_at": row["expires_at"],
    }


def resolve_access_token(auth_db_path=None, credential_id=None, api_key=None):
    """Return ``(access_token, credential_identity)`` or raise."""
    if api_key:
        state = credential_state(api_key=api_key)
        if state["state"] != "ready":
            raise DevinCredentialError(state["note"])
        return api_key, "explicit"
    state = credential_state(auth_db_path, credential_id)
    if state["state"] != "ready" or not state.get("access_token"):
        raise DevinCredentialError(
            state.get("note") or f"Devin credential {state['state']}"
        )
    # B15: bind downstream work (discovery cache, diagnostics) to the account.
    identity = (
        state.get("identity")
        or (f"devin-cli:{state['path']}" if state.get("source") == "devin-cli" else None)
        or f"id:{state.get('credential_id')}"
    )
    return state["access_token"], identity


# ---------------------------------------------------------------------------
# Discovery (OMP catalog/src/discovery/devin.ts)
# ---------------------------------------------------------------------------

THINKING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
IMAGE_BLIND_UIDS = frozenset({"swe-1-6", "swe-1-6-fast"})
SEED_IDS = frozenset({"swe-1-6", "swe-1-6-fast"})

# Family-lane axes (devin.ts:150-168).
_FAMILY_FAST_KEY = "fast mode"
_FAMILY_FAST_ORDER = 1
_FAMILY_THINKING_KEY = "thinking"
_FAMILY_THINKING_ORDER = 1
_FAMILY_CONTEXT_1M_KEY = "1m context"
_FAMILY_CONTEXT_1M_ORDER = 1
_FAMILY_EFFORT_BY_NAME = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
    "off": "off",
    "none": "off",
    "nothinking": "off",
}

# Curated seeds, the entire bundled surface (providers/devin.kdl). Discovery
# replaces them; the account default wins over the seed default.
SEEDS = (
    {
        "uid": "swe-1-6-fast",
        "label": "SWE-1.6 Fast",
        "reasoning": True,
        "supports_tools": True,
        "supports_images": False,
        "supports_parallel": True,
        "context_limit": 200000,
        "output_limit": 128000,
        "is_default": False,
    },
    {
        "uid": "swe-1-6",
        "label": "SWE-1.6",
        "reasoning": True,
        "supports_tools": True,
        "supports_images": False,
        "supports_parallel": True,
        "context_limit": 200000,
        "output_limit": 128000,
        "is_default": True,
    },
)


class DevinModel:
    """One gateway-addressable Devin model (standalone uid or family lane)."""

    def __init__(
        self,
        uid,
        logical_id,
        label,
        context_limit,
        output_limit,
        supports_tools=True,
        supports_images=False,
        supports_parallel=False,
        reasoning=True,
        is_router=False,
        efforts=(),
        default_effort=None,
        effort_routing=None,
        is_default=False,
        base_url=None,
        unverified=False,
    ):
        self.uid = uid
        self.logical_id = logical_id
        self.label = label
        self.context_limit = context_limit
        self.output_limit = output_limit
        self.supports_tools = supports_tools
        self.supports_images = supports_images
        self.supports_parallel = supports_parallel
        self.reasoning = reasoning
        self.is_router = is_router
        self.efforts = list(efforts)
        self.default_effort = default_effort
        self.effort_routing = dict(effort_routing or {})
        self.is_default = is_default
        self.base_url = base_url
        self.unverified = unverified

    def wire_uid(self, effort=None):
        """Resolve the chat uid for an effort tier (lane routing or self)."""
        if effort is not None and effort in self.effort_routing:
            return self.effort_routing[effort]
        if self.effort_routing and self.default_effort in self.effort_routing:
            return self.effort_routing[self.default_effort]
        return self.uid


def _model_info(config):
    if not config.HasField("model_info"):
        return None
    return config.model_info


def _model_features(info):
    if info is not None and info.HasField("model_features"):
        return info.model_features
    return None


_REASONING_LABEL = re.compile(
    r"think|thinking|minimal|high|medium|low|xhigh|max|reasoning", re.IGNORECASE
)
_NO_REASONING_LABEL = re.compile(r"\bno thinking\b", re.IGNORECASE)


def supports_thinking(config):
    """Server features authoritative; label heuristic only when absent."""
    features = _model_features(_model_info(config))
    if features is not None:
        return bool(features.supports_thinking)
    if _NO_REASONING_LABEL.search(config.label or ""):
        return False
    return bool(_REASONING_LABEL.search(config.label or ""))


def _normalize_key(text):
    return "".join(
        ch if ch.isalnum() else " " for ch in (text or "").lower()
    ).strip()


def _normalize_effort_name(text):
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def fusion_pairing_lead(uid, live_uids):
    """Lead chat uid of a ``fusion-<lead>[-fast]-sidekick-<sidekick>`` pairing.

    Returns None for non-pairings, False for pairings with no live lead (the
    composite uid itself is never servable), else the lead uid. Verbatim port
    of OMP devinFusionLeadUid — including the exact-vs-priority lane order.
    """
    if not uid.startswith("fusion-"):
        return None
    cut = uid.find("-sidekick-")
    if cut <= len("fusion-"):
        return None
    lead = uid[len("fusion-") : cut]
    if lead in live_uids:
        return lead
    if lead.endswith("-fast"):
        base = lead[: -len("-fast")]
        if f"{base}-priority" in live_uids:
            return f"{base}-priority"
        if base in live_uids:
            return base
    return False


def _slug(text):
    slug = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return slug.strip("-")


def _collect_lane(lanes, config, uid):
    metadata = config.model_family_metadata
    label = (metadata.model_family_label or "").strip()
    if not label:
        return
    effort = None
    thinking = None
    fast = False
    one_m = False
    for entry in metadata.entries:
        value = entry.value
        if value is None:
            continue
        key = _normalize_key(entry.key)
        if key == _FAMILY_FAST_KEY:
            fast = value.order == _FAMILY_FAST_ORDER
        elif key == _FAMILY_THINKING_KEY:
            thinking = value.order == _FAMILY_THINKING_ORDER
        elif key == _FAMILY_CONTEXT_1M_KEY:
            one_m = value.order == _FAMILY_CONTEXT_1M_ORDER
        elif key == "effort" or key == "reasoning effort":
            effort = _FAMILY_EFFORT_BY_NAME.get(_normalize_effort_name(value.name))
    # Paired non-thinking/thinking configs share one effort label; the
    # explicit Thinking axis decides whether the route is off.
    if thinking is False:
        effort = "off"
    base_id = _slug(label)
    if not base_id:
        return
    lane_id = base_id + ("-1m" if one_m else "") + ("-fast" if fast else "")
    lane = lanes.get(lane_id)
    if lane is None:
        lane = {
            "id": lane_id,
            "name": label + (" 1M" if one_m else "") + (" Fast" if fast else ""),
            "members": [],
            "routing": {},
            "default_member": None,
        }
        lanes[lane_id] = lane
    lane["members"].append(uid)
    if lane["default_member"] is None and (
        config.is_default_model_in_family or metadata.is_default_model_in_family
    ):
        lane["default_member"] = uid
    if effort is not None and effort not in lane["routing"]:
        lane["routing"][effort] = uid


def _spec_from_config(config, uid, base_url, is_router):
    info = _model_info(config)
    features = _model_features(info)
    if features is not None:
        images = bool(features.supports_images)
    else:
        images = bool(config.supports_images)
    if uid in IMAGE_BLIND_UIDS:
        images = False
    tools = bool(features.supports_tool_calls) if features is not None else True
    parallel = (
        bool(features.supports_parallel_tool_calls)
        if features is not None
        else False
    )
    max_tokens = int(config.max_tokens or 0)
    return {
        "uid": uid,
        "label": config.label or uid,
        "description": config.description or "",
        "is_new": bool(config.is_new),
        "is_beta": bool(config.is_beta),
        "is_recommended": bool(config.is_recommended),
        "context_limit": max_tokens if max_tokens > 0 else 200000,
        "output_limit": int(info.max_output_tokens)
        if info is not None and info.max_output_tokens > 0
        else 64000,
        "supports_tools": tools,
        "supports_images": images,
        "supports_parallel": parallel,
        "reasoning": supports_thinking(config),
        "is_router": is_router,
        "harness_uids": list(info.harness_uids) if info is not None else [],
        "base_url": base_url,
    }


def normalize_configs(response, base_url):
    """Normalize a discovery response into gateway DevinModels.

    Returns ``(models, dropped_fusion_pairings)``. Fusion pairings are
    dropped entirely (option B: the sidekick half cannot run gateway-side);
    harness-backed composites stay. Routers stay standalone and never join
    lanes; lane members collapse into one family row.
    """
    configs = list(response.client_model_configs)
    live_uids = set()
    for config in configs:
        uid = (config.model_uid or "").strip()
        if not config.disabled and uid and uid not in live_uids:
            live_uids.add(uid)
    seen = set()
    lanes = {}
    specs = {}
    dropped_fusion = 0
    for config in configs:
        if config.disabled:
            continue
        info = _model_info(config)
        display = info.display_option if info is not None else 0
        if display in (4, 6):  # QUICK_REVIEW + INTERNAL_DEFAULT
            continue
        uid = (config.model_uid or "").strip()
        if not uid or uid in seen:
            continue
        seen.add(uid)
        lead = fusion_pairing_lead(uid, live_uids)
        if lead is not None:
            # Option B: pairings never list (dead-lead AND live-lead).
            dropped_fusion += 1
            continue
        is_router = display == 3 or (info is not None and info.is_model_router)
        is_assign_router = is_router and not (
            info is not None and list(info.harness_uids)
        )
        specs[uid] = _spec_from_config(config, uid, base_url, is_assign_router)
        if not is_router:
            _collect_lane(lanes, config, uid)
    models = []
    consumed = set()
    for lane in lanes.values():
        named_efforts = [e for e in THINKING_EFFORTS if e in lane["routing"]]
        if not named_efforts:
            continue
        off_uid = lane["routing"].get("off")
        efforts = (["none"] if off_uid is not None else []) + named_efforts
        default_member = lane["default_member"]
        members = lane["members"]
        if default_member is not None:
            members = [default_member] + [
                uid for uid in members if uid != default_member
            ]
        default_level = None
        if default_member is not None:
            if default_member == off_uid:
                default_level = "none"
            for effort in named_efforts:
                if lane["routing"][effort] == default_member:
                    default_level = effort
                    break
        first = specs[members[0]]
        models.append(
            DevinModel(
                uid=members[0],
                logical_id=lane["id"],
                label=lane["name"],
                context_limit=first["context_limit"],
                output_limit=first["output_limit"],
                supports_tools=first["supports_tools"],
                supports_images=first["supports_images"],
                supports_parallel=first["supports_parallel"],
                reasoning=True,
                efforts=efforts,
                default_effort=default_level if default_member is not None else named_efforts[0],
                effort_routing={
                    **{e: lane["routing"][e] for e in named_efforts},
                    **({"off": off_uid, "none": off_uid} if off_uid is not None else {}),
                },
                base_url=base_url,
            )
        )
        consumed.update(members)
    for uid, spec in specs.items():
        if uid in consumed:
            continue
        models.append(
            DevinModel(
                uid=uid,
                logical_id=uid,
                label=spec["label"],
                context_limit=spec["context_limit"],
                output_limit=spec["output_limit"],
                supports_tools=spec["supports_tools"],
                supports_images=spec["supports_images"],
                supports_parallel=spec["supports_parallel"],
                reasoning=spec["reasoning"],
                is_router=spec["is_router"],
                base_url=base_url,
            )
        )
    account_default = ""
    if response.HasField("default_override_model_config"):
        account_default = (
            response.default_override_model_config.model_uid or ""
        ).strip()
    if account_default:
        for model in models:
            if model.uid == account_default or model.logical_id == account_default:
                model.is_default = True
    return models, dropped_fusion


def seed_models(base_url=None):
    """Bundled fallback seeds (KDL). Discovery replaces these when possible."""
    return [
        DevinModel(
            uid=seed["uid"],
            logical_id=seed["uid"],
            label=seed["label"],
            context_limit=seed["context_limit"],
            output_limit=seed["output_limit"],
            supports_tools=seed["supports_tools"],
            supports_images=seed["supports_images"],
            supports_parallel=seed["supports_parallel"],
            reasoning=seed["reasoning"],
            is_default=seed["is_default"],
            base_url=base_url or DEFAULT_BASE_URL,
            unverified=True,
        )
        for seed in SEEDS
    ]


def _fetch_catalog(api_key, base_url, metadata, timeout=5.0):
    request = devin_pb2.GetCliModelConfigsRequest()
    request.metadata.CopyFrom(metadata)
    try:
        _, _, body = unary_rpc(
            base_url, GET_CLI_MODEL_CONFIGS_PATH, request.SerializeToString(),
            timeout=timeout,
        )
    except DevinWireError:
        return None
    try:
        return decode_unary(devin_pb2.GetCliModelConfigsResponse, body)
    except Exception:
        return None


def discover_models(api_key, base_url=None, timeout=5.0):
    """Fetch + normalize the credential-scoped roster.

    Native dev-channel identity first; legacy Windsurf identity when native
    yields nothing, is empty, or is seed-only; the longer roster wins.
    Returns ``(models, dropped_fusion, warnings)``. Empty-but-200 is the
    stale-pin failure signature (warn, return no models).
    """
    resolved_base = normalize_base_url(base_url or DEFAULT_BASE_URL)
    warnings = []
    native_models = None
    response = _fetch_catalog(
        api_key, resolved_base, discovery_metadata(api_key), timeout
    )
    if response is not None:
        native_models, _ = normalize_configs(response, resolved_base)
    seed_only = (
        native_models is not None
        and len(native_models) > 0
        and all(model.uid in SEED_IDS for model in native_models)
    )
    if native_models and not seed_only:
        models, dropped = normalize_configs(response, resolved_base)
        return models, dropped, warnings
    legacy_response = _fetch_catalog(
        api_key, resolved_base, legacy_metadata(api_key), timeout
    )
    legacy_models = None
    if legacy_response is not None:
        legacy_models, _ = normalize_configs(legacy_response, resolved_base)
    if legacy_models and (
        native_models is None or len(legacy_models) > len(native_models)
    ):
        models, dropped = normalize_configs(legacy_response, resolved_base)
        return models, dropped, warnings
    if native_models:
        models, dropped = normalize_configs(response, resolved_base)
        return models, dropped, warnings
    warnings.append(
        "Devin returned an empty model catalog; the pinned client identities"
        " may be stale"
    )
    return [], 0, warnings


# ---------------------------------------------------------------------------
# Chat translation (OMP providers/devin.ts, utils/schema/*)
# ---------------------------------------------------------------------------

TOOL_CALL_ID_MAX_LENGTH = 40
MAX_SCHEMA_DEPTH = 128

# OMP UNSUPPORTED_SCHEMA_FIELDS + const (schema/fields.ts, compatibility.ts).
_GOOGLE_STRIP_KEYS = frozenset({
    "$schema", "$id", "$anchor", "$ref", "$defs", "$dynamicRef",
    "$dynamicAnchor", "examples", "prefixItems", "unevaluatedProperties",
    "unevaluatedItems", "patternProperties", "additionalProperties",
    "propertyNames", "minItems", "maxItems", "uniqueItems", "contains",
    "minContains", "maxContains", "additionalItems", "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "pattern", "format", "contentEncoding", "contentMediaType",
    "contentSchema", "dependencies", "dependentSchemas", "dependentRequired",
    "x-mcp-header", "deprecated", "readOnly", "writeOnly", "$comment",
    "const",
})
# OMP LIFTABLE_TO_DESCRIPTION_FIELDS: stripped keys preserved as model context.
_GOOGLE_SPILL_KEYS = frozenset({
    "pattern", "format", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems",
    "maxItems", "uniqueItems", "minProperties", "maxProperties", "default",
    "examples",
})
_GOOGLE_SCALAR_TYPES = frozenset({
    "string", "number", "integer", "boolean", "object", "array", "null",
})
_ID_SANITIZE = re.compile(r"[^A-Za-z0-9_-]")


def deterministic_uuid(seed):
    """First 128 bits of SHA-256 formatted 8-4-4-4-12 (NOT RFC 4122)."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (
        f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}"
        f"-{digest[16:20]}-{digest[20:32]}"
    )


def normalize_tool_call_id(raw_id, seen):
    """Sanitize + 40-cap + dedupe a tool call id (OMP transformMessages).

    Responses composites (``call_id|item_id``) keep the wire ``call_id``
    segment; opaque ids keep every segment. ``seen`` maps wire ids already
    claimed; collisions gain a ``-2``/``-3`` suffix.
    """
    base = str(raw_id or "").split("|")[0]
    base = _ID_SANITIZE.sub("_", base)[:TOOL_CALL_ID_MAX_LENGTH]
    if not base:
        base = "call"
    candidate, counter = base, 2
    while candidate in seen:
        suffix = f"-{counter}"
        candidate = base[: TOOL_CALL_ID_MAX_LENGTH - len(suffix)] + suffix
        counter += 1
    seen.add(candidate)
    return candidate


def _resolve_pointer(root, pointer):
    """Resolve a local JSON pointer (``#/...``); None when unresolvable."""
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        return None
    node = root
    for segment in pointer[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and segment in node:
            node = node[segment]
        else:
            return None
    return node


def deref_schema_refs(schema):
    """Inline local ``$ref``s (E1). B3: conflicting $defs/definitions names
    skip dereferencing entirely (fail-open); cycles resolve once then stop."""
    if not isinstance(schema, dict):
        return schema
    definitions = schema.get("definitions")
    dollar_defs = schema.get("$defs")
    if isinstance(definitions, dict) and isinstance(dollar_defs, dict):
        for name, entry in definitions.items():
            if name in dollar_defs and dollar_defs[name] != entry:
                return schema

    def _walk(node, depth, stack):
        if depth > MAX_SCHEMA_DEPTH or not isinstance(node, dict):
            return node
        if isinstance(node.get("$ref"), str):
            target = _resolve_pointer(schema, node["$ref"])
            if target is None or id(target) in stack:
                return node
            merged = _walk(target, depth + 1, stack | {id(target)})
            if not isinstance(merged, dict):
                return node
            rest = {k: v for k, v in node.items() if k != "$ref"}
            combined = dict(merged)
            combined.update(_walk(rest, depth + 1, stack))
            return combined
        out = {}
        for key, value in node.items():
            if isinstance(value, dict):
                out[key] = _walk(value, depth + 1, stack)
            elif isinstance(value, list):
                out[key] = [
                    _walk(item, depth + 1, stack)
                    if isinstance(item, (dict, list))
                    else item
                    for item in value
                ]
            else:
                out[key] = value
        return out

    return _walk(schema, 0, set())


def _spill_to_description(node, stripped):
    kept = [(k, v) for k, v in stripped if v is not None]
    if not kept:
        return
    formatted = (
        "{" + ", ".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in kept) + "}"
    )
    existing = node.get("description")
    existing = existing if isinstance(existing, str) else ""
    node["description"] = f"{existing}\n\n{formatted}" if existing else formatted


def normalize_schema_for_google(schema):
    """Focused port of OMP normalizeSchemaForGoogle for gemini backends.

    Strip+spill unsupported keys, scalarize type arrays, infer string type
    for bare enums, keep ``nullable``. Hard errors only where OMP would 400
    downstream (B12 multi-type, B13 unknown scalar). Budget exhaustion fails
    open to the source schema (junie behavior).
    """
    try:
        source = json.dumps(schema, ensure_ascii=False)
    except (TypeError, ValueError):
        raise DevinChatError("tool schema could not be serialized")
    budget = [max(1, len(source) * 4)]

    def spend():
        budget[0] -= 1
        if budget[0] < 0:
            raise _BudgetExhausted

    class _BudgetExhausted(Exception):
        pass

    def walk(node, depth):
        spend()
        if depth > MAX_SCHEMA_DEPTH:
            raise _BudgetExhausted
        if isinstance(node, list):
            return [walk(item, depth + 1) for item in node]
        if not isinstance(node, dict):
            return node
        out = {}
        stripped = []
        for key, value in node.items():
            if key in _GOOGLE_STRIP_KEYS:
                if key in _GOOGLE_SPILL_KEYS:
                    stripped.append((key, value))
                continue
            out[key] = value
        _spill_to_description(out, stripped)
        node_type = out.get("type")
        if isinstance(node_type, list):
            members = [t for t in node_type if t != "null"]
            had_null = len(members) != len(node_type)
            if not members:
                out.pop("type", None)
                out["nullable"] = True
            elif len(members) == 1:
                out["type"] = members[0]
            else:
                raise DevinChatError(
                    f"tool schema type array {node_type!r} is not representable"
                    " on the Devin gemini backend"
                )
            node_type = out.get("type")
        if node_type == "null":
            out.pop("type", None)
            out["nullable"] = True
            node_type = None
        if node_type is not None and node_type not in _GOOGLE_SCALAR_TYPES:
            raise DevinChatError(
                f"tool schema type {node_type!r} is not representable"
                " on the Devin gemini backend"
            )
        if "type" not in out and isinstance(out.get("enum"), list):
            out["type"] = "string"
        skip = {"enum", "required", "examples", "type", "const", "default", "example"}
        for key, value in list(out.items()):
            if key in skip:
                continue
            if key in ("properties", "$defs", "definitions", "dependentSchemas"):
                if isinstance(value, dict):
                    out[key] = {
                        name: walk(child, depth + 1)
                        if isinstance(child, (dict, list))
                        else child
                        for name, child in value.items()
                    }
            elif isinstance(value, (dict, list)):
                out[key] = walk(value, depth + 1)
        return out

    try:
        return walk(schema, 0)
    except _BudgetExhausted:
        return schema


def is_gemini_uid(uid):
    """Gemini-backend detection: MODEL_GOOGLE prefix (B6) or gemini naming."""
    text = str(uid or "")
    return text.startswith("MODEL_GOOGLE") or "gemini" in text.lower()


def _input_items(body):
    value = body.get("input")
    if value is None:
        value = body.get("messages", [])
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def _text_and_images(content):
    """Split message content into joined text + base64 image list."""
    texts, images = [], []
    parts = content if isinstance(content, list) else [content]
    for part in parts:
        if isinstance(part, str):
            if part:
                texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
        elif ptype in {"input_image", "image_url", "image"}:
            url = part.get("image_url", part.get("url", ""))
            if isinstance(url, dict):
                url = url.get("url", "")
            source = part.get("source")
            if isinstance(source, dict) and source.get("type") == "base64":
                images.append((source.get("data", ""), source.get("media_type", "")))
                continue
            if not isinstance(url, str) or not url.startswith("data:"):
                raise DevinChatError(
                    "Devin image parts must be base64 data URIs; URL images"
                    " cannot be represented on the Cascade wire format"
                )
            meta, _, data = url[5:].partition(";base64,")
            images.append((data, meta or "image/png"))
    return "".join(texts), images


def _output_text(output):
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text, _ = _text_and_images(output)
        return text
    return json.dumps(output, ensure_ascii=False)


def history_to_prompts(body, cascade_id):
    """Translate Responses input into Cascade prompts + system text.

    Returns ``(system_text, prompts, call_id_map)``. Reasoning summaries
    become assistant text (no signature store exists to replay thinking).
    """
    system_parts = []
    instructions = body.get("instructions", body.get("system"))
    if isinstance(instructions, str) and instructions.strip():
        system_parts.append(instructions.strip())
    prompts = []
    call_ids = {}
    seen_ids = set()
    open_calls = None

    def close_calls():
        nonlocal open_calls
        if open_calls is not None and open_calls["tool_calls"]:
            prompts.append(open_calls)
        open_calls = None

    for raw in _input_items(body):
        if not isinstance(raw, dict):
            continue
        item_type = str(raw.get("type") or "").strip()
        if item_type == "function_call":
            original = str(raw.get("call_id") or raw.get("id") or "").strip()
            if not original:
                continue
            wire_id = call_ids.get(original)
            if wire_id is None:
                wire_id = normalize_tool_call_id(original, seen_ids)
                call_ids[original] = wire_id
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            arguments = raw.get("arguments", {})
            if isinstance(arguments, str):
                arguments_text = arguments
            else:
                try:
                    arguments_text = json.dumps(arguments, ensure_ascii=False)
                except (TypeError, ValueError):
                    arguments_text = "{}"
            if open_calls is None:
                open_calls = {
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [],
                }
            open_calls["tool_calls"].append(
                {"id": wire_id, "name": name, "arguments": arguments_text}
            )
            continue
        if item_type == "function_call_output":
            close_calls()
            original = str(raw.get("call_id") or "").strip()
            wire_id = call_ids.get(original)
            if wire_id is None:
                wire_id = normalize_tool_call_id(original, seen_ids)
                call_ids[original] = wire_id
            prompts.append(
                {
                    "role": "tool",
                    "text": _output_text(raw.get("output", "")),
                    "tool_call_id": wire_id,
                    "is_error": bool(
                        raw.get("is_error") is True
                        or str(raw.get("status") or "") == "error"
                    ),
                    "images": [],
                }
            )
            continue
        if item_type == "reasoning":
            close_calls()
            summary = raw.get("summary", "")
            if isinstance(summary, list):
                summary = "".join(
                    part.get("text", "")
                    if isinstance(part, dict)
                    else str(part)
                    for part in summary
                )
            if isinstance(summary, str) and summary.strip():
                prompts.append({"role": "assistant", "text": summary})
            continue
        role = raw.get("role")
        if item_type == "message" and role is None:
            role = "assistant"
        if role == "system":
            text, _ = _text_and_images(raw.get("content", ""))
            if text.strip():
                system_parts.append(text.strip())
            continue
        if role not in {"user", "developer", "assistant"}:
            if item_type in {"input_text", "output_text", "text"}:
                role = "user"
            else:
                continue
        close_calls()
        text, images = _text_and_images(
            raw.get("content", raw.get("text", ""))
        )
        if role == "assistant":
            if text.strip():
                prompts.append({"role": "assistant", "text": text})
        else:
            prompts.append(
                {"role": role, "text": text, "images": images}
            )
    close_calls()
    system_text = "\n\n".join(part for part in system_parts if part)
    staged = []
    for prompt in prompts:
        if prompt["role"] == "assistant" and not prompt.get(
            "text", ""
        ).strip() and not prompt.get("tool_calls"):
            continue
        staged.append(prompt)
    return system_text, staged, call_ids


def _message_id(cascade_id, index, role, tool_call_id=""):
    seed = f"{cascade_id}\0{index}\0{role}"
    if tool_call_id:
        seed += f"\0{tool_call_id}"
    return deterministic_uuid(seed)


def build_history_prompts(staged, cascade_id):
    """Build ChatMessagePrompt protos with deterministic ids."""
    protos = []
    for index, prompt in enumerate(staged):
        message = devin_pb2.ChatMessagePrompt()
        role = prompt["role"]
        if role == "tool":
            message.message_id = _message_id(
                cascade_id, index, "tool", prompt.get("tool_call_id", "")
            )
            message.source = devin_pb2.ChatMessageSource.Value("CHAT_MESSAGE_SOURCE_TOOL")
            message.prompt = prompt.get("text", "")
            message.tool_call_id = prompt.get("tool_call_id", "")
            message.tool_result_is_error = bool(prompt.get("is_error"))
        else:
            message.message_id = _message_id(cascade_id, index, role)
            if role == "assistant":
                message.source = devin_pb2.ChatMessageSource.Value(
                    "CHAT_MESSAGE_SOURCE_SYSTEM"
                )
            else:
                message.source = devin_pb2.ChatMessageSource.Value("CHAT_MESSAGE_SOURCE_USER")
            message.prompt = prompt.get("text", "")
            for call in prompt.get("tool_calls", []):
                entry = message.tool_calls.add()
                entry.id = call["id"]
                entry.name = call["name"]
                entry.arguments_json = call["arguments"]
            for data, mime in prompt.get("images", []):
                image = message.images.add()
                image.base64_data = data
                image.mime_type = mime or "image/png"
        protos.append(message)
    return protos


def build_tools(tools, gemini):
    """Build ChatToolDefinitions; Google-normalize for gemini backends."""
    definitions = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        parameters = tool.get("parameters", tool.get("input_schema", {}))
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise DevinChatError(
                f"tool {name!r} parameters must be a JSON object"
            )
        schema = json.loads(json.dumps(parameters))
        schema = deref_schema_refs(schema)
        if gemini:
            schema = normalize_schema_for_google(schema)
        definition = devin_pb2.ChatToolDefinition()
        definition.name = name
        definition.description = str(tool.get("description") or "")
        definition.json_schema_string = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        definition.strict = bool(tool.get("strict", False))
        definitions.append(definition)
    return definitions


# ---------------------------------------------------------------------------
# Chat requests (OMP providers/devin.ts fetchDevinAuthMetadata +
# assignDevinModel + buildDevinChatRequest)
# ---------------------------------------------------------------------------


def fround(value):
    """Round a float the way Devin's CLI widens float32 to double."""
    return struct.unpack("d", struct.pack("d", struct.unpack("f", struct.pack("f", float(value)))[0]))[0]


def fetch_auth(api_key, base_url, timeout=30.0):
    """Per-turn JWT: prefixed credential first, unprefixed on 401 only.

    Returns ``(wire_api_key, user_jwt, effective_base_url)``. The custom
    server URL rides Assign+Chat, never GetUserJwt. (ADDITIVE: HTTP(S)
    validation on the custom URL; 30s bound instead of a bare signal.)
    """
    resolved_base = normalize_base_url(base_url or DEFAULT_BASE_URL)
    prefixed = normalize_session_token(api_key)
    attempts = [prefixed]
    if api_key and api_key != prefixed:
        attempts.append(api_key)
    last_error = None
    for attempt in attempts:
        request = devin_pb2.GetUserJwtRequest()
        request.metadata.CopyFrom(wire_metadata(attempt))
        try:
            _, _, body = unary_rpc(
                resolved_base, AUTH_PATH, request.SerializeToString(),
                timeout=timeout,
            )
        except DevinHttpError as error:
            if error.status == 401 and attempt is attempts[0] and len(attempts) > 1:
                last_error = error
                continue
            if error.status == 401:
                # Persistent 401 (alternate form exhausted or N/A): quarantine
                # this token like junie's ReauthRequired (in-memory only).
                mark_auth_failure(api_key, "GetUserJwt 401")
            raise DevinAuthError(str(error)) from error
        decoded = decode_unary(devin_pb2.GetUserJwtResponse, body)
        if decoded is None or not decoded.user_jwt:
            raise DevinAuthError("Devin auth error: GetUserJwt returned an empty user JWT")
        effective = resolved_base
        custom = (decoded.custom_api_server_url or "").strip()
        if custom:
            effective = normalize_base_url(custom)
        return attempt, decoded.user_jwt, effective
    raise DevinAuthError(str(last_error) if last_error else "Devin auth error: GetUserJwt failed")


def build_router_prompt(staged):
    """The prompt the router scores: the last user/developer turn alone.

    messageId stays empty (minted by the chat request that follows); no
    cache options (verified against OMP buildRouterPrompt).
    """
    for prompt in reversed(staged):
        if prompt["role"] not in {"user", "developer"}:
            continue
        message = devin_pb2.ChatMessagePrompt()
        message.message_id = ""
        message.source = devin_pb2.ChatMessageSource.Value("CHAT_MESSAGE_SOURCE_USER")
        message.prompt = prompt.get("text", "")
        for data, mime in prompt.get("images", []):
            image = message.images.add()
            image.base64_data = data
            image.mime_type = mime or "image/png"
        return message
    return None


def assign_if_router(model, wire_api_key, cascade_id, staged, base_url, timeout=30.0):
    """Resolve a router into (model_uid, assignment_jwt). Both-or-throw: a
    failed assignment fails the turn rather than sending the router id."""
    router_prompt = build_router_prompt(staged)
    request = devin_pb2.AssignModelRequest()
    request.metadata.CopyFrom(wire_metadata(wire_api_key))
    request.model_router_uid = model.uid
    request.cascade_id = cascade_id
    if router_prompt is not None:
        request.chat_message_prompt.CopyFrom(router_prompt)
    try:
        _, _, body = unary_rpc(
            base_url, ASSIGN_MODEL_PATH, request.SerializeToString(), timeout=timeout
        )
    except DevinHttpError as error:
        raise DevinChatError(f"Devin AssignModel error: {error}") from error
    decoded = decode_unary(devin_pb2.AssignModelResponse, body)
    assignment = decoded.assignment if decoded is not None else None
    if assignment is None or not assignment.assignment_jwt or not assignment.model_uid:
        raise DevinChatError("Devin AssignModel error: response carried no assignment JWT and model uid")
    return assignment.model_uid, assignment.assignment_jwt


def _body_number(body, name, default):
    value = body.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _body_int(body, name, default):
    value = body.get(name, default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def build_chat_request(
    system_text,
    body,
    model,
    wire_uid,
    wire_api_key,
    user_jwt,
    cascade_id,
    history_protos,
    tools,
    assignment_jwt=None,
):
    """Assemble GetChatMessageRequest exactly as OMP buildDevinChatRequest."""
    request = devin_pb2.GetChatMessageRequest()
    request.metadata.CopyFrom(wire_metadata(wire_api_key, user_jwt))
    request.prompt = system_text
    request.chat_message_prompts.extend(history_protos)
    request.chat_model_uid = wire_uid
    if assignment_jwt:
        request.model_assignment_jwt = assignment_jwt
    request.request_type = devin_pb2.ChatMessageRequestType.Value(
        "CHAT_MESSAGE_REQUEST_TYPE_CASCADE"
    )
    request.planner_mode = devin_pb2.ConversationalPlannerMode.Value(
        "CONVERSATIONAL_PLANNER_MODE_DEFAULT"
    )
    request.tool_choice.option_name = "auto"
    request.system_prompt_cache_options.type = devin_pb2.CacheControlType.Value(
        "CACHE_CONTROL_TYPE_EPHEMERAL"
    )
    request.disable_parallel_tool_calls = not model.supports_parallel
    request.cascade_id = cascade_id
    request.execution_id = str(uuid.uuid4())
    configuration = request.configuration
    configuration.num_completions = 1
    configuration.max_tokens = _body_int(body, "max_output_tokens", model.output_limit or 64000)
    configuration.max_newlines = 400
    configuration.temperature = _body_number(body, "temperature", 1.0)
    configuration.top_k = 40
    configuration.top_p = fround(_body_number(body, "top_p", 0.95))
    stops = body.get("stop", [])
    if isinstance(stops, str):
        stops = [stops]
    for stop in stops if isinstance(stops, list) else []:
        if isinstance(stop, str) and stop:
            configuration.stop_patterns.append(stop)
    request.tools.extend(tools)
    return request


# ---------------------------------------------------------------------------
# Stream consumption -> Responses events (OMP devin.ts stream loop)
# ---------------------------------------------------------------------------


def _responses_function_id(raw_id, call_id):
    candidate = str(raw_id or "")
    if candidate.startswith("fc_") and re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        return candidate
    if call_id.startswith("call_"):
        return "fc_" + call_id[len("call_") :]
    digest = hashlib.sha256((candidate or call_id).encode("utf-8")).hexdigest()[:24]
    return "fc_" + digest


def _finalize_tool_arguments(accumulated, invalid_fallback=""):
    """Authoritative final parse with repair-or-mark (E8, ratified OMP)."""
    for candidate in (accumulated, invalid_fallback):
        if not candidate:
            continue
        try:
            json.loads(candidate)
        except ValueError:
            continue
        return candidate
    return json.dumps({"__parseError": accumulated}, ensure_ascii=False)


class DevinStreamState:
    """Fold Cascade stream messages into Responses (event, payload) tuples.

    Shapes mirror the gateway's Claude translator exactly (plain dicts; no
    gateway import, keeping this module dependency-light). Thinking deltas
    accumulate for diagnostics only: Responses has no thinking stream event
    and no signature store exists to replay them (documented limitation).
    """

    def __init__(self, model_id):
        self.model_id = model_id
        self.response_id = ""
        self.upstream_model = None
        self.output = []
        self.text_block = None
        self.tool_blocks = {}
        self.tool_order = []
        self.active_tool_id = None
        self.thinking_text = ""
        self.usage = {}
        self.stop_reason = None
        self.error_message = None
        self.error_type = None
        self.had_output = False
        self.finished = False
        self.started = False
        self.sequence_number = 0

    def _number_events(self, events):
        for event_name, payload in events:
            payload["sequence_number"] = self.sequence_number
            self.sequence_number += 1
        return events

    def _ensure_started(self):
        """Emit response.created + response.in_progress first.

        The Responses SSE contract (and the gateway's Claude translator)
        opens every turn with both lifecycle events carrying the response
        object; clients reject streams that start at output_item.added.
        The id is decided once here so created/completed always agree, and
        forced to resp_ form (Devin upstream ids look like bot-uuid).
        """
        if self.started:
            return []
        self.started = True
        if not self.response_id:
            self.response_id = "resp_" + uuid.uuid4().hex[:16]
        snapshot = self.final_response(status="in_progress")
        return [
            ("response.created", {
                "type": "response.created",
                "response": copy.deepcopy(snapshot),
            }),
            ("response.in_progress", {
                "type": "response.in_progress",
                "response": copy.deepcopy(snapshot),
            }),
        ]

    def _message_item_id(self):
        seed = f"{self.response_id}:{len(self.output)}"
        return "msg_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]

    def _ensure_text_block(self):
        if self.text_block is not None:
            return []
        item_id = self._message_item_id()
        item = {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        self.text_block = {"item": item, "text": ""}
        self.output.append(item)
        index = len(self.output) - 1
        return [
            ("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": index,
                "item": copy.deepcopy(item),
            }),
            ("response.content_part.added", {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        ]

    def _ensure_tool_block(self, wire_id, name):
        block = self.tool_blocks.get(wire_id)
        if block is not None:
            if name:
                block["name"] = name
                block["item"]["name"] = name
            return []
        item_id = _responses_function_id(wire_id, wire_id)
        item = {
            "type": "function_call",
            "id": item_id,
            "call_id": wire_id,
            "name": name or "",
            "arguments": "",
            "status": "in_progress",
        }
        self.tool_blocks[wire_id] = {
            "item": item, "name": name or "", "arguments": "",
            "invalid": "",
        }
        self.tool_order.append(wire_id)
        self.output.append(item)
        # HTTP deltas cannot be replaced after argument repair or late naming.
        # Release finalized tool events together; text still streams normally.
        return []

    def feed(self, message):
        """Fold one GetChatMessageResponse; return event tuples."""
        events = []
        if self.finished:
            return events
        if message.message_id and not self.response_id:
            candidate = str(message.message_id)
            if candidate.startswith("resp_"):
                self.response_id = candidate
            else:
                digest = hashlib.sha256(
                    candidate.encode("utf-8")
                ).hexdigest()[:24]
                self.response_id = "resp_" + digest
        if message.actual_model_uid:
            self.upstream_model = message.actual_model_uid
        events.extend(self._ensure_started())
        if message.delta_thinking:
            self.had_output = True
            self.thinking_text += message.delta_thinking
        if message.delta_signature:
            self.had_output = True
        if message.delta_text:
            self.had_output = True
            events.extend(self._ensure_text_block())
            block = self.text_block
            block["text"] += message.delta_text
            item = block["item"]
            content = item.setdefault("content", [])
            if not content:
                content.append({"type": "output_text", "text": "", "annotations": []})
            content[0]["text"] = block["text"]
            events.append(("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": self.output.index(item),
                "content_index": 0,
                "delta": message.delta_text,
            }))
        for call in message.delta_tool_calls:
            wire_id = call.id or self.active_tool_id
            if not wire_id:
                continue
            self.had_output = True
            events.extend(self._ensure_tool_block(wire_id, call.name))
            self.active_tool_id = wire_id
            block = self.tool_blocks[wire_id]
            if call.arguments_json:
                previous = block["arguments"]
                block["arguments"] = (
                    call.arguments_json if call.arguments_json.startswith(previous)
                    else previous + call.arguments_json
                )
                block["item"]["arguments"] = block["arguments"]
            if call.invalid_json_str:
                block["invalid"] = call.invalid_json_str
        if message.stop_reason != 0:
            self.stop_reason = int(message.stop_reason)
        usage = message.usage
        if message.HasField("usage"):
            cached = int(usage.cache_read_tokens)
            cache_write = int(usage.cache_write_tokens)
            self.usage = {
                "input_tokens": int(usage.input_tokens) + cached + cache_write,
                "output_tokens": int(usage.output_tokens),
                "cached_tokens": cached,
                "cache_write_tokens": cache_write,
            }
        return self._number_events(events)

    def fail(self, message, error_type="upstream_error"):
        """Emit one failed terminal outcome, retaining partial output."""
        self.error_message = message
        self.error_type = error_type
        self.stop_reason = "error"
        events = self._ensure_started()
        final = self.final_response(status="failed")
        events.append(("response.failed", {
            "type": "response.failed", "response": final,
        }))
        self.finished = True
        return self._number_events(events)

    def _finish_items(self):
        events = []
        if self.text_block is not None:
            item = self.text_block["item"]
            index = self.output.index(item)
            events.append(("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": item["id"], "output_index": index,
                "content_index": 0, "text": self.text_block["text"],
            }))
            events.append(("response.content_part.done", {
                "type": "response.content_part.done",
                "item_id": item["id"], "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.text_block["text"],
                         "annotations": []},
            }))
            item["status"] = "completed"
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": index, "item": copy.deepcopy(item),
            }))
        for wire_id in self.tool_order:
            block = self.tool_blocks[wire_id]
            item = block["item"]
            arguments = _finalize_tool_arguments(
                block["arguments"], block["invalid"]
            )
            index = self.output.index(item)
            added = copy.deepcopy(item)
            added["arguments"] = ""
            events.append(("response.output_item.added", {
                "type": "response.output_item.added", "output_index": index,
                "item": added,
            }))
            item["arguments"] = arguments
            events.append(("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"], "output_index": index, "delta": arguments,
            }))
            events.append(("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"], "output_index": index,
                "arguments": arguments,
            }))
            item["status"] = "completed"
            events.append(("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": index, "item": copy.deepcopy(item),
            }))
        return events

    def finish(self):
        """Emit done + terminal events. Done mapping: toolUse > length > stop."""
        if self.finished:
            return []
        if self.stop_reason in (1, 9):
            return self.fail("Devin stream ended before a completion signal", "upstream_error")
        if any(not block["name"] for block in self.tool_blocks.values()):
            return self.fail("Devin returned a tool call without a name", "invalid_tool_call")
        events = self._ensure_started()
        events.extend(self._finish_items())
        if self.tool_blocks:
            status = "completed"
        elif self.stop_reason == 3:  # STOP_REASON_MAX_TOKENS
            status = "incomplete"
        else:
            status = "completed"
        final_event = (
            "response.incomplete" if status == "incomplete" else "response.completed"
        )
        events.append((final_event, {
            "type": final_event, "response": self.final_response(status=status),
        }))
        self.finished = True
        return self._number_events(events)

    def final_response(self, status="in_progress"):
        result = {
            "id": self.response_id or "resp_" + uuid.uuid4().hex[:16],
            "object": "response",
            "status": status,
            "model": self.model_id,
            "output": copy.deepcopy(self.output),
        }
        if self.usage:
            result["usage"] = {
                "input_tokens": self.usage.get("input_tokens", 0),
                "output_tokens": self.usage.get("output_tokens", 0),
                "total_tokens": self.usage.get("input_tokens", 0)
                + self.usage.get("output_tokens", 0),
            }
            if self.usage.get("cached_tokens") or self.usage.get("cache_write_tokens"):
                result["usage"]["input_tokens_details"] = {
                    "cached_tokens": self.usage.get("cached_tokens", 0),
                }
        if status == "incomplete":
            result["incomplete_details"] = {"reason": "max_output_tokens"}
        if status == "failed" and self.error_message:
            result["error"] = {
                "message": self.error_message,
                "type": self.error_type or "upstream_error",
                "code": self.error_type or "upstream_error",
            }
        return result


def _shrinkable_history_bytes(history_protos, staged):
    """Encoded bytes of the shrinkable history (OMP recovery heuristic)."""
    tail = 1 if staged and staged[-1]["role"] in {"user", "developer"} else 0
    if staged and staged[-1]["role"] == "developer":
        for prompt in reversed(staged[:-1]):
            if prompt["role"] not in {"user", "developer"}:
                break
            tail += 1
    shrinkable = history_protos[: len(history_protos) - tail] if tail else history_protos
    probe = devin_pb2.GetChatMessageRequest()
    probe.chat_message_prompts.extend(shrinkable)
    return len(probe.SerializeToString())


def _classify_trailer(trailer_error, had_output, history_bytes):
    """Map a Connect trailer to an action. Returns "overflow" only for the
    large-history invalid_argument signature; otherwise None (plain error)."""
    if (
        not had_output
        and trailer_error.code.lower() == "invalid_argument"
        and re.search(r"\binternal error\b", trailer_error.message, re.IGNORECASE)
        and history_bytes >= 512 * 1024
    ):
        return "overflow"
    return None


def run_turn_events(
    body,
    flat_tools,
    model,
    access_token,
    base_url=None,
    effort=None,
    timeout=300.0,
    conversation_id=None,
):
    """Bound upstream connection failures before the first event as well."""
    try:
        yield from _run_turn_events(
            body, flat_tools, model, access_token, base_url, effort,
            timeout, conversation_id,
        )
    except (HTTPException, ConnectionError) as error:
        raise DevinChatError("Devin upstream connection failed before a response") from error


def _run_turn_events(
    body,
    flat_tools,
    model,
    access_token,
    base_url=None,
    effort=None,
    timeout=300.0,
    conversation_id=None,
):
    """Run one Devin turn, yielding (event_type, payload) tuples.

    Auth (per-turn JWT) -> assign-if-router -> gzipped Connect chat stream.
    Raises DevinChatError/DevinAuthError for pre-stream failures; in-stream
    failures surface as one response.failed event. (ADDITIVE: bounded
    socket timeout instead of OMP's bare abort signal.)
    """
    resolved_base = normalize_base_url(base_url or model.base_url or DEFAULT_BASE_URL)
    identity = conversation_id or body.get("conversation_id") or body.get("session_id")
    cascade_id = deterministic_uuid(str(identity)) if identity else str(uuid.uuid4())
    system_text, staged, _call_ids = history_to_prompts(body, cascade_id)
    if not model.supports_images and any(prompt.get("images") for prompt in staged):
        raise DevinChatError("The selected Devin model does not support images")
    if not model.supports_tools and flat_tools:
        raise DevinChatError("The selected Devin model does not support tools")
    wire_uid = model.wire_uid(effort)
    wire_api_key, user_jwt, chat_base = fetch_auth(access_token, resolved_base, timeout=min(timeout, 30.0))
    assignment_jwt = None
    request_uid = wire_uid
    if model.is_router:
        assigned_uid, assignment_jwt = assign_if_router(
            model, wire_api_key, cascade_id, staged, chat_base, timeout=min(timeout, 30.0)
        )
        request_uid = assigned_uid
    gemini = is_gemini_uid(model.uid) or is_gemini_uid(request_uid)
    tools = build_tools(flat_tools, gemini)
    history_protos = build_history_prompts(staged, cascade_id)
    request = build_chat_request(
        system_text, body, model, request_uid, wire_api_key, user_jwt,
        cascade_id, history_protos, tools, assignment_jwt,
    )
    state = DevinStreamState(f"devin/{model.logical_id}")
    history_bytes = _shrinkable_history_bytes(history_protos, staged)
    response = open_chat_stream(chat_base, request.SerializeToString(), timeout=timeout)
    saw_end_stream = False
    try:
        for flags, payload in iter_chat_frames(response):
            if flags & CONNECT_END_STREAM_FLAG:
                saw_end_stream = True
                raw = payload
                if flags & CONNECT_COMPRESSED_FLAG:
                    raw = decompress_frame(payload)
                text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else ""
                trailer = parse_trailer_error(text)
                if trailer is None:
                    continue
                overflow = _classify_trailer(trailer, state.had_output, history_bytes)
                if overflow:
                    raise DevinChatError(
                        "Devin rejected the turn as context overflow (large-history"
                        " invalid_argument): start a fresh turn with less history.",
                        code="devin_context_overflow",
                    )
                for event in state.fail(trailer.formatted, "upstream_error"):
                    yield event
                return
            raw = payload
            if flags & CONNECT_COMPRESSED_FLAG:
                raw = decompress_frame(payload)
            message = devin_pb2.GetChatMessageResponse()
            try:
                message.ParseFromString(bytes(raw))
            except Exception as error:
                raise DevinChatError(f"Devin stream message does not decode: {error}")
            for event in state.feed(message):
                yield event
        if not state.had_output:
            raise DevinChatError("Devin stream ended without model output")
        if state.stop_reason in (1, 9) or (not saw_end_stream and state.stop_reason is None):
            raise DevinChatError("Devin stream ended before a completion signal")
    except (DevinWireError, OSError, EOFError, HTTPException) as error:
        if not state.started:
            if isinstance(error, HTTPException):
                raise DevinChatError("Devin HTTP stream framing failed") from error
            raise
        for event in state.fail(str(error), "upstream_error"):
            yield event
        return
    finally:
        response.close()
        if current_capture() is not None:
            current_capture().event("devin.chat.closed")
    for event in state.finish():
        yield event


def run_turn_final(
    body, flat_tools, model, access_token, base_url=None, effort=None, timeout=300.0,
    conversation_id=None,
):
    """Non-streaming turn: consume the same stream, return the final object."""
    final = None
    for event_type, payload in run_turn_events(
        body, flat_tools, model, access_token, base_url, effort, timeout, conversation_id
    ):
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            final = payload.get("response")
    if final is None:
        raise DevinChatError("Devin turn ended without a terminal event")
    return final
