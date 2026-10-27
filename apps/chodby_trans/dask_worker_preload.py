"""
Preload the heavy transport stack before a Dask worker accepts MLMC tasks.
"""

import importlib
import logging
import os
from pathlib import Path
import resource
import time

from chodby_trans import job


logger = logging.getLogger(__name__)
PRELOAD_STARTED_AT = time.monotonic()
PRELOAD_COMPLETED = False

importlib.import_module("numpy")
importlib.import_module("chodby_trans.transport_simulation")

PRELOAD_SECONDS = time.monotonic() - PRELOAD_STARTED_AT
PRELOAD_PEAK_RSS_MIB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
PRELOAD_PID = os.getpid()

logger.info(
    "Preloaded MLMC transport modules in %.2fs: pid=%s, peak_rss=%.1f MiB.",
    PRELOAD_SECONDS,
    PRELOAD_PID,
    PRELOAD_PEAK_RSS_MIB,
)


def dask_setup(worker) -> None:
    """
    Initialize job paths before the worker registers and accepts MLMC tasks.
    """
    output_dir = os.environ.get(job.OUTPUT_DIR_ENV)
    input_dir = os.environ.get(job.INPUT_DIR_ENV)
    missing = [
        env_name
        for env_name, value in (
            (job.OUTPUT_DIR_ENV, output_dir),
            (job.INPUT_DIR_ENV, input_dir),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing MLMC worker preload environment variables: "
            + ", ".join(missing)
        )

    job.set_workdir(Path(output_dir), Path(input_dir))

    global PRELOAD_COMPLETED
    PRELOAD_COMPLETED = True
    logger.info(
        "MLMC transport preload ready on worker %s: pid=%s, elapsed=%.2fs, "
        "peak_rss=%.1f MiB, input=%s, scratch=%s, output=%s.",
        worker.address,
        PRELOAD_PID,
        PRELOAD_SECONDS,
        PRELOAD_PEAK_RSS_MIB,
        job.input.dir_path,
        job.scratch.dir_path,
        job.output.dir_path,
    )
