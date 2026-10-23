"""Lightweight checks for the Dask scheduler and worker preload modules."""

from __future__ import annotations

import os
import sys
from typing import Any


def _preload_status(module_name: str) -> dict[str, Any]:
    """Return preload state without importing the target module."""
    module = sys.modules.get(module_name)
    if module is None:
        return {
            "completed": False,
            "pid": os.getpid(),
            "seconds": None,
            "peak_rss_mib": None,
        }
    return {
        "completed": bool(getattr(module, "PRELOAD_COMPLETED", False)),
        "pid": int(getattr(module, "PRELOAD_PID", os.getpid())),
        "seconds": float(getattr(module, "PRELOAD_SECONDS", -1.0)),
        "peak_rss_mib": float(getattr(module, "PRELOAD_PEAK_RSS_MIB", -1.0)),
    }


def worker_preload_status() -> dict[str, Any]:
    """Return the homogenization worker-preload state."""
    return _preload_status("homogenization_study.dask_worker_preload")


def scheduler_preload_status() -> dict[str, Any]:
    """Return the homogenization scheduler-preload state."""
    return _preload_status("homogenization_study.dask_scheduler_preload")
