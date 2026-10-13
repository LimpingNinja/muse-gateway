#!/usr/bin/env python3
"""Backward-compatible alias for :mod:`muse_gateway`.

The project was renamed from muse-shim to muse-gateway. This module keeps
``python -m muse_shim``, ``muse_shim.py``-based launchers, and the
``muse-shim`` console script working: every attribute (including ``main``)
is served from :mod:`muse_gateway`, so ``from muse_shim import Config`` and
``patch("muse_shim.X")`` keep resolving to the live implementation.

New code should ``import muse_gateway`` and prefer the ``MUSE_GATEWAY_*``
environment-variable spelling (``MUSE_SHIM_*`` still works as a fallback).
"""

from muse_gateway import main
import muse_gateway as _gateway_module

__all__ = [name for name in dir(_gateway_module) if not name.startswith("_")]


def __getattr__(name: str):
    return getattr(_gateway_module, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_gateway_module)))


if __name__ == "__main__":
    raise SystemExit(main())
