#!/usr/bin/env python3
"""Backward-compatible alias for :mod:`muse_gateway`.

The project was renamed from muse-shim to muse-gateway. Importing this
module returns :mod:`muse_gateway` itself (one module object), so
``from muse_shim import Config`` and ``patch("muse_shim.X")`` read and patch
the live implementation. ``python -m muse_shim`` and ``muse_shim.py``-based
launchers keep working.

New code should ``import muse_gateway`` and use supported ``MUSE_GATEWAY_*``
settings. Retired ``MUSE_SHIM_*`` names produce migration guidance, not aliases.
"""

import sys

import muse_gateway

if __name__ == "__main__":
    raise SystemExit(muse_gateway.main())

sys.modules[__name__] = muse_gateway
