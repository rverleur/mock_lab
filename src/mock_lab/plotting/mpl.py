"""Centralized headless Matplotlib configuration for pipeline exports."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "mock_lab_matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


__all__ = ["matplotlib", "plt"]
