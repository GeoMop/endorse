"""
Preload application modules before a Dask worker accepts MLMC tasks.
"""

import importlib
import logging
import os
from pathlib import Path
import resource
import sys
import time

logger = logging.getLogger(__name__)
PRELOAD_STARTED_AT = time.monotonic()
PRELOAD_COMPLETED = False

APP_MODULE = os.environ["ENDORSE_DASK_APP_MODULE"]
WORKER_MODULE = os.environ["ENDORSE_DASK_WORKER_MODULE"]
job = importlib.import_module(f"{APP_MODULE}.job")
importlib.import_module(WORKER_MODULE)
sys.modules[f"{APP_MODULE}.dask_worker_preload"] = sys.modules[__name__]

PRELOAD_SECONDS = time.monotonic() - PRELOAD_STARTED_AT
PRELOAD_PEAK_RSS_MIB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
PRELOAD_PID = os.getpid()

logger.info(
    "Preloaded MLMC worker module %s in %.2fs: pid=%s, peak_rss=%.1f MiB.",
    WORKER_MODULE,
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

    assert output_dir is not None and input_dir is not None
    job.set_workdir(Path(output_dir), Path(input_dir))

    global PRELOAD_COMPLETED
    PRELOAD_COMPLETED = True
    logger.info(
        "MLMC preload ready on worker %s: pid=%s, elapsed=%.2fs, "
        "peak_rss=%.1f MiB, input=%s, scratch=%s, output=%s.",
        worker.address,
        PRELOAD_PID,
        PRELOAD_SECONDS,
        PRELOAD_PEAK_RSS_MIB,
        job.input.dir_path,
        job.scratch.dir_path,
        job.output.dir_path,
    )
