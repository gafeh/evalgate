"""Vercel entrypoint. The platform serves the ASGI app exported as `app`."""

from __future__ import annotations

import sys
from pathlib import Path

# The package lives under src/, which Vercel does not add to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalgate.api.app import app

__all__ = ["app"]