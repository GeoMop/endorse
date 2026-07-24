"""
Preload the heavy transport stack before a Dask worker accepts MLMC tasks.
"""

import importlib
import logging
import os
import resource
import time


logger = logging.getLogger(__name__)
PRELOAD_STARTED_AT = time.monotonic()
PRELOAD_COMPLETED = False

importlib.import_module("numpy")
importlib.import_module("chodby_trans.transport_simulation")

PRELOAD_SECONDS = time.monotonic() - PRELOAD_STARTED_AT
PRELOAD_PEAK_RSS_MIB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
PRELOAD_PID = os.getpid()
PRELOAD_COMPLETED = True

logger.info(
    "Preloaded MLMC transport modules in %.2fs: pid=%s, peak_rss=%.1f MiB.",
    PRELOAD_SECONDS,
    PRELOAD_PID,
    PRELOAD_PEAK_RSS_MIB,
)


def dask_setup(worker) -> None:
    """
    Log completion against the final Dask worker address.
    """
    logger.info(
        "MLMC transport preload ready on worker %s: pid=%s, elapsed=%.2fs, peak_rss=%.1f MiB.",
        worker.address,
        PRELOAD_PID,
        PRELOAD_SECONDS,
        PRELOAD_PEAK_RSS_MIB,
    )
