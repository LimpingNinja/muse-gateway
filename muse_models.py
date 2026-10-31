"""Load bundled model data and fill missing metadata, never account entitlement."""

import json
from importlib.resources import files
from typing import Any

PROVIDERS = ("meta", "claude", "codex", "kiro", "devin", "local")
_SOURCES = {"documented", "user-approved", "unverified-fallback", "bundled-seed"}
_SEED_FLAGS = {"reasoning", "supports_tools", "supports_images", "supports_parallel", "is_default"}


def _bounded_label(value: Any) -> str:
    text = "".join(
        char if ord(char) >= 32 and not 127 <= ord(char) <= 159 else " "
        for char in str(value or "")
    )
    if len(text) > 128:
        text = text[:125] + "..."
    return repr(text)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {_bounded_label(key)}")
        result[key] = value
    return result


def _parse_catalog(provider: str, text: str) -> dict[str, Any]:
    """Reject malformed bundled data rather than silently omitting it."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
        allowed = {"fallbacks", "models", "reasoning_efforts", "default_reasoning_effort"}
        if not isinstance(data, dict) or set(data) - allowed:
            raise ValueError("expected a catalog object with recognized fields")
        models = data.get("models")
        if not isinstance(models, dict):
            raise ValueError("models must be an object")
        for name, record in models.items():
            if not name or name.strip() != name or "/" in name:
                raise ValueError("model IDs must be nonempty exact upstream IDs")
            record_keys = {"context", "output", "sources"}
            if provider == "devin":
                record_keys |= {"label"} | _SEED_FLAGS
            if not isinstance(record, dict) or set(record) - record_keys:
                raise ValueError(f"invalid model record {_bounded_label(name)}")
            for field in ("context", "output"):
                value = record.get(field)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{_bounded_label(name)}: {field} must be a positive integer")
            sources = record.get("sources")
            if (not isinstance(sources, dict) or set(sources) != {"context", "output"}
                    or any(not isinstance(source, str) or source not in _SOURCES
                           for source in sources.values())):
                raise ValueError(f"{_bounded_label(name)}: invalid capacity provenance")
            for flag in _SEED_FLAGS & record.keys():
                if type(record[flag]) is not bool:
                    raise ValueError(f"{_bounded_label(name)}: {flag} must be boolean")
            if "label" in record and (not isinstance(record["label"], str) or not record["label"].strip()):
                raise ValueError(f"{_bounded_label(name)}: label must be nonempty text")
        fallbacks = data.get("fallbacks")
        if (not isinstance(fallbacks, list)
                or any(not isinstance(name, str) or name not in models for name in fallbacks)
                or len(set(fallbacks)) != len(fallbacks)):
            raise ValueError("fallbacks must be unique IDs present in models")
        if provider == "devin":
            for name in fallbacks:
                if not ({"label"} | _SEED_FLAGS) <= models[name].keys():
                    raise ValueError(f"{_bounded_label(name)}: incomplete Devin fallback capabilities")
        efforts = data.get("reasoning_efforts", [])
        if (not isinstance(efforts, list)
                or any(not isinstance(effort, str) or not effort for effort in efforts)
                or len(set(efforts)) != len(efforts)):
            raise ValueError("reasoning_efforts must contain unique nonempty strings")
        if "default_reasoning_effort" in data and data["default_reasoning_effort"] not in efforts:
            raise ValueError("default_reasoning_effort must belong to reasoning_efforts")
        return data
    except ValueError as error:
        raise ValueError(f"Invalid provider catalog providers/{provider}.json: {error}") from error


def _load_catalog(provider: str) -> dict[str, Any]:
    try:
        text = files("providers").joinpath(f"{provider}.json").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read provider catalog providers/{provider}.json") from error
    return _parse_catalog(provider, text)


_CATALOGS = {provider: _load_catalog(provider) for provider in PROVIDERS}
FALLBACK_REGISTRY = {
    f"{provider}/{name}": {
        f"{field}_limit": (record[field], record["sources"][field])
        for field in ("context", "output")
    }
    for provider, catalog in _CATALOGS.items()
    for name, record in catalog["models"].items()
}


def provider_catalog(provider: str) -> dict[str, Any]:
    """Bundled data, loaded and validated once per process; callers must not mutate."""
    return _CATALOGS[provider]


def positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def resolve_limits(model_id: str, context: Any = None, output: Any = None) -> dict[str, Any]:
    fallback = FALLBACK_REGISTRY.get(model_id, {})
    result: dict[str, Any] = {}
    for field, supplied in (("context_limit", context), ("output_limit", output)):
        value = positive_int(supplied)
        value, source = (value, "live") if value is not None else fallback.get(field, (None, "missing"))
        result[field] = value
        result[field.removesuffix("_limit") + "_source"] = source
    result["limits_verified"] = result["context_source"] == result["output_source"] == "live"
    result["limit_source"] = (result["context_source"] if result["context_source"] == result["output_source"] else "mixed")
    return result
