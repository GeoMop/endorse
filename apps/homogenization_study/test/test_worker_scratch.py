"""Exercise the real MLMC workspace lifecycle with distinct PBS scratch roots."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from mlmc.level_simulation import LevelSimulation

from homogenization_study.mlmc_worker import WorkerScratchSamplingPool


def write_sample(config, sample_input):
    """Leave a file that identifies where MLMC ran the task."""
    Path("sample.txt").write_text(str(Path.cwd()))
    if config["fail"]:
        raise ValueError("sample failed")
    return 1, 0


class InlineClient:
    """Execute the submitted callable locally without opening Dask sockets."""

    def submit(self, function, *args, **kwargs):
        """Return a finished future using the actual MLMC worker callable."""
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        future.release = lambda: None
        return future


@pytest.mark.parametrize(
    ("debug", "fail", "sample_id", "retained_dir"),
    [
        (False, False, "L00_S0000000", "several_successful"),
        (False, False, "L00_S0000005", None),
        (True, False, "L00_S0000005", ""),
        (True, True, "L00_S0000005", "failed"),
    ],
)
def test_pbs_task_uses_worker_scratch(tmp_path, monkeypatch, debug, fail, sample_id, retained_dir):
    """Create and retain sample files under the worker root, with no driver-side copies."""
    driver = tmp_path / "driver"
    driver.mkdir()
    worker = tmp_path / "worker"
    worker.mkdir()
    monkeypatch.setenv("SCRATCHDIR", str(worker))
    level = LevelSimulation({"fail": fail}, need_sample_workspace=True, calculate=write_sample, level_id=0)
    pool = WorkerScratchSamplingPool(InlineClient(), work_dir=str(driver), debug=debug)
    pool._distributed = SimpleNamespace(fire_and_forget=lambda future: None)
    pool.schedule_sample((sample_id, 1), level)
    successful, failed, running, _ = pool.get_finished()
    assert running == 0
    assert bool(successful.get(0)) == (not fail)
    assert bool(failed.get(0)) == fail
    assert not list(driver.rglob(sample_id))
    worker_output = worker / "workdir" / "output"
    if retained_dir is None:
        assert not list(worker_output.rglob(sample_id))
    else:
        marker = worker_output / retained_dir / sample_id / "sample.txt"
        assert marker.read_text() == str(worker_output / sample_id)
