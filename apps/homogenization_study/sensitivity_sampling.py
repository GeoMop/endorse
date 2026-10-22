"""Single-level MLMC driver for homogenized conductivity sampling."""

from __future__ import annotations

import logging
import shutil
import subprocess
import socket
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import dask
from dask.distributed import Client, LocalCluster

from endorse import common
from endorse.common import dotdict
from endorse.mesh_class import File, load_mesh
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampler import Sampler
from mlmc.sampling_pool_dask import SamplingPoolDask

from homogenization_study import job
from homogenization_study.create_mesh import create_homogenization_mesh
from homogenization_study.model import StudyMeshFiles
from homogenization_study.problem_simulation import (
    HomogenizationSimulation,
    RandomSimulation,
    ensure_field_storage,
    full_parameter_names,
)


def setup_logging(name: str = "homogenization") -> None:
    """Configure process-safe logging for the driver and workers."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{name}] {socket.gethostname()}:%(process)d %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def mlmc_level_parameters(cfg: dotdict) -> list[list[float]]:
    """Return exactly one MLMC level for this single-resolution study."""
    if len(cfg.mlmc.levels) != 1:
        raise ValueError("Stage B homogenization uses exactly one MLMC level.")
    return [[1.0]]


def simulation_class_name(cfg: dotdict) -> str:
    """Return the configured simulation adapter name persisted with restart data."""
    return str(cfg.mlmc.get("sim_class", "HomogenizationSimulation"))


def _homogenization_geometry(meshes: StudyMeshFiles) -> tuple[np.ndarray, np.ndarray]:
    """Return bulk-element barycentres and stable Gmsh ids from the fixed mesh."""
    mesh = load_mesh(File(str(meshes.homogenization_mesh)), heal_tol=None)
    bulk = mesh.el_dim_slice(dim=3)
    indices = np.arange(bulk.start, bulk.stop, dtype=int)
    if not indices.size:
        raise ValueError("The homogenization mesh contains no bulk elements.")
    return mesh.el_barycenters()[indices], np.asarray(mesh.el_ids, dtype=np.int64)[indices]


def prepare_sampling(cfg: dotdict, meshes: StudyMeshFiles, workdir: Path):
    """Create fixed geometry metadata and the one-level simulation adapter."""
    barycenter, gmsh_id = _homogenization_geometry(meshes)
    parameter_names = full_parameter_names(cfg)
    n_samples = int(cfg.ot_sensitivity.n_samples)
    ensure_field_storage(
        job.output.zarr_store_path,
        n_samples=n_samples,
        n_elements=len(gmsh_id),
        parameter_names=parameter_names,
        el_barycenter=barycenter,
        gmsh_id=gmsh_id,
    )
    simulation_name = simulation_class_name(cfg)
    simulation_class = {
        "HomogenizationSimulation": HomogenizationSimulation,
        "RandomSimulation": RandomSimulation,
    }.get(simulation_name)
    if simulation_class is None:
        raise ValueError(
            f"Unsupported cfg.mlmc.sim_class={simulation_name!r}; "
            "expected HomogenizationSimulation or RandomSimulation."
        )
    return simulation_class(
        cfg=cfg,
        workdir=workdir,
        meshes=meshes,
        n_elements=len(gmsh_id),
        store_path=job.output.zarr_store_path,
    )


def resubmit_unfinished_samples(sampler: Sampler) -> int:
    """Resubmit unfinished HDF samples after a driver or worker restart."""
    unfinished = sampler.sample_storage.unfinished_ids()
    for sample in unfinished:
        sample_id, _sample_input = sample
        level_id = int(sample_id.split("_", 1)[0][1:])
        sampler._sampling_pool.schedule_sample(sample, sampler._level_sim_objects[level_id])
    return len(unfinished)


def sample_counts(sampler: Sampler) -> tuple[int, int]:
    """Return successful and failed sample counts for the single level."""
    finished = int(sampler.n_finished_samples[0])
    failed = len(sampler.sample_storage.failed_samples().get("0", []))
    return finished - failed, failed


def _move_stale_hdf(path: Path) -> Path:
    """Move a broken HDF file aside and return its backup path."""
    stale_path = path.with_name(path.stem + ".stale" + path.suffix)
    index = 1
    while stale_path.exists():
        stale_path = path.with_name(path.stem + f".stale{index}" + path.suffix)
        index += 1
    path.replace(stale_path)
    return stale_path


def initialize_hdf_storage(cfg: dotdict) -> SampleStorageHDF:
    """Open HDF storage, leaving level creation to ``Sampler``.

    This follows the Chodby lifecycle exactly: ``Sampler`` calls
    ``save_global_data`` on the newly opened storage. The helper only moves a
    structurally incomplete file left by an interrupted earlier run.
    """
    path = Path(job.output.mlmc_hdf_path)

    def structurally_valid(candidate: Path) -> bool:
        if not candidate.exists():
            return True
        try:
            with h5py.File(candidate, "r") as hdf_file:
                if hdf_file.attrs.get("simulation_class") != simulation_class_name(cfg):
                    return False
                levels = hdf_file.get("Levels")
                if levels is None:
                    return False
                for level_id in range(len(mlmc_level_parameters(cfg))):
                    group = levels.get(str(level_id))
                    if group is None or not {"scheduled", "collected_ids", "failed"}.issubset(group):
                        return False
            return True
        except (OSError, KeyError):
            return False

    if not structurally_valid(path):
        stale_path = _move_stale_hdf(path)
        logging.warning("Moved structurally invalid MLMC HDF store %s to %s.", path, stale_path)
    return SampleStorageHDF(str(path))


def save_hdf_simulation_class(cfg: dotdict) -> None:
    """Record which simulation produced the restart data."""
    with h5py.File(job.output.mlmc_hdf_path, "r+") as hdf_file:
        hdf_file.attrs["simulation_class"] = simulation_class_name(cfg)


def run_mlmc_sampling(
    cfg: dotdict,
    client: Client,
    seed: int = 1234,
    *,
    meshes: StudyMeshFiles | None = None,
    workdir: Path | None = None,
) -> Sampler:
    """Run or resume the configured single-level sampling calculation."""
    if job.output is None:
        raise RuntimeError("Call job.set_workdir before starting sampling.")
    workdir = Path(workdir or job.scratch.dir_path).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if meshes is None:
        homogenization_mesh = create_homogenization_mesh(cfg, job.output.homogenization_mesh_dir)
        meshes = StudyMeshFiles(micro_mesh=Path(""), homogenization_mesh=homogenization_mesh)

    simulation = prepare_sampling(cfg, meshes, workdir)
    storage = initialize_hdf_storage(cfg)
    pool = SamplingPoolDask(
        client,
        work_dir=str(workdir),
        debug=not bool(cfg.ot_sensitivity.clean_sample_dir),
        clean=bool(cfg.ot_sensitivity.clean_sample_dir),
    )
    try:
        sampler = Sampler(storage, pool, simulation, mlmc_level_parameters(cfg), seed=seed)
    except (ValueError, OSError, KeyError) as error:
        stale_path = _move_stale_hdf(Path(job.output.mlmc_hdf_path))
        logging.warning("Moved incompatible MLMC HDF store to %s: %s", stale_path, error)
        storage = SampleStorageHDF(str(job.output.mlmc_hdf_path))
        sampler = Sampler(storage, pool, simulation, mlmc_level_parameters(cfg), seed=seed)
    save_hdf_simulation_class(cfg)
    if bool(cfg.ot_sensitivity.recompute_failed):
        sampler.renew_failed_samples()
    resubmitted = resubmit_unfinished_samples(sampler)
    target = int(cfg.ot_sensitivity.n_samples)
    scheduled = int(sampler.l_scheduled_samples()[0])
    if scheduled < target:
        sampler.schedule_samples(level_id=0, n_samples=target - scheduled)
    logging.info("Sampling target=%s, resumed=%s, scheduled=%s.", target, resubmitted, scheduled)
    successful, failed = sample_counts(sampler)
    while successful < target:
        sampler.ask_sampling_pool_for_samples(sleep=1.0, timeout=5.0)
        successful, failed = sample_counts(sampler)
        logging.info("Successful samples: %s/%s; failed: %s", successful, target, failed)
        if failed:
            raise RuntimeError(
                f"MLMC sampling stopped with {failed} failed samples and {successful}/{target} successful samples."
            )
    return sampler


PBS_SCRIPT_TEMPLATE = """#!/bin/bash
#PBS -S /bin/bash
#PBS -l select={n_nodes}:ncpus={n_cpus}:mem={mem}:scratch_local={scratch}
#PBS -l place=scatter
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -N {job_name}
#PBS -W umask=0002
#PBS -o {out_log}
#PBS -j oe
#PBS -m e

set -euo pipefail
set -x
export TMPDIR="$SCRATCHDIR"
export ENDORSE_DISABLE_MEMOIZE=1

OUTPUT_DIR={output_dir}
PROJECT_DIR={project_dir}
PYEXEC={python_executable}
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"
"$PYEXEC" -u "$PROJECT_DIR/sensitivity_sampling.py" "$OUTPUT_DIR" {driver_command}
"""


def submit_pbs(cfg_machine: dotdict, cmd: str = "local", *, submit: bool = True) -> Path:
    """Create the PBS script and optionally submit it with ``qsub``.

    ``cmd`` is retained for compatibility with the Chodby launcher.  The
    homogenization driver always executes the single-level sampling command;
    ``local`` and ``meta`` are accepted aliases.
    """
    if cmd not in {"local", "meta", "mlmc", "continue", "random"}:
        raise ValueError(f"Unsupported PBS command {cmd!r}.")
    machine = cfg_machine.get("__resolved__", cfg_machine)
    pbs = machine.get("pbs")
    if not pbs and machine is not cfg_machine:
        machine = cfg_machine.get("__default__", machine)
        pbs = machine.get("pbs")
    if not pbs:
        raise ValueError("machine_config does not define PBS resources.")
    if job.output is None:
        raise RuntimeError("Call job.set_workdir before creating a PBS script.")
    script_path = Path(job.output.pbs_script)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "n_nodes": int(pbs.n_nodes),
        "n_cpus": int(pbs.n_cores),
        "mem": pbs.mem,
        "scratch": pbs.scratch,
        "walltime": pbs.walltime,
        "queue": pbs.queue,
        "job_name": pbs.pbs_name,
        "out_log": job.output.dir_path / f"{pbs.pbs_name}.out",
        "output_dir": job.output.dir_path,
        "project_dir": Path(__file__).resolve().parent,
        "python_executable": sys.executable,
        "driver_command": cmd,
    }
    script_path.write_text(PBS_SCRIPT_TEMPLATE.format(**values), encoding="utf-8")
    logging.info("Wrote PBS script: %s", script_path)
    if submit:
        subprocess.run(["qsub", str(script_path)], check=True)
    return script_path


def main() -> None:
    """Run the local driver using the configured study workdir."""
    setup_logging()
    workdir = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("workdir_sampling")).resolve()
    job.set_workdir(workdir)
    if not job.input.study_cfg_path.exists():
        source_input = Path(__file__).resolve().parent / "input_data"
        if not source_input.is_dir():
            raise FileNotFoundError(f"Study input directory does not exist: {source_input}")
        shutil.copytree(source_input, job.input.dir_path, dirs_exist_ok=True)
        logging.info("Copied study inputs from %s to %s.", source_input, job.input.dir_path)
    cfg = common.config.load_config(str(job.input.study_cfg_path))
    command = sys.argv[2] if len(sys.argv) > 2 else "local"
    if command == "submit":
        submit_pbs(cfg.machine_config, cmd=sys.argv[3] if len(sys.argv) > 3 else "meta")
        return
    if command == "random":
        cfg.mlmc.sim_class = "RandomSimulation"
    elif command not in {"local", "meta", "mlmc"}:
        raise ValueError(f"Unknown command {command!r}; use submit, local, or random.")
    scheduler = sys.argv[3] if len(sys.argv) > 3 else None
    if scheduler:
        with Client(scheduler) as client:
            run_mlmc_sampling(cfg, client)
        return

    with dask.config.set({"distributed.worker.daemon": False}):
        cluster = LocalCluster(processes=True, n_workers=1, threads_per_worker=1, dashboard_address=None)
        try:
            with Client(cluster) as client:
                run_mlmc_sampling(cfg, client)
        finally:
            cluster.close()


if __name__ == "__main__":
    main()
