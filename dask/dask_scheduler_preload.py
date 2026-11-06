"""
Preload the MLMC task module before the Dask scheduler accepts clients.
"""

import importlib
import logging
import os
import resource
import sys
import time


logger = logging.getLogger(__name__)
PRELOAD_STARTED_AT = time.monotonic()
PRELOAD_COMPLETED = False

APP_MODULE = os.environ["ENDORSE_DASK_APP_MODULE"]
SCHEDULER_MODULE = os.environ["ENDORSE_DASK_SCHEDULER_MODULE"]
importlib.import_module(SCHEDULER_MODULE)
sys.modules[f"{APP_MODULE}.dask_scheduler_preload"] = sys.modules[__name__]

PRELOAD_SECONDS = time.monotonic() - PRELOAD_STARTED_AT
PRELOAD_PEAK_RSS_MIB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
PRELOAD_PID = os.getpid()
PRELOAD_COMPLETED = True

logger.info(
    "Preloaded MLMC scheduler module %s in %.2fs: pid=%s, peak_rss=%.1f MiB.",
    SCHEDULER_MODULE,
    PRELOAD_SECONDS,
    PRELOAD_PID,
    PRELOAD_PEAK_RSS_MIB,
)


def dask_setup(scheduler) -> None:
    """
    Log completion against the final Dask scheduler address.
    """
    logger.info(
        "MLMC scheduler preload ready on %s: pid=%s, elapsed=%.2fs, peak_rss=%.1f MiB.",
        scheduler.address,
        PRELOAD_PID,
        PRELOAD_SECONDS,
        PRELOAD_PEAK_RSS_MIB,
    )
