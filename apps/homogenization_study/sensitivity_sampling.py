"""Single-level MLMC driver for homogenized conductivity sampling."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import socket
import sys
from pathlib import Path

import dask
import h5py
import numpy as np
from dask.distributed import Client

from endorse import common
from endorse.common import dotdict
from endorse.mesh_class import File, load_mesh
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampler import Sampler
from mlmc.sampling_pool_dask import SamplingPoolDask

from homogenization_study import job
from homogenization_study.create_mesh import create_homogenization_mesh
from homogenization_study.dask_status import scheduler_preload_status, worker_preload_status
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
        schema_path=job.input.data_schema_yaml,
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


def wait_for_expected_workers(client: Client) -> int | None:
    """Wait for the worker count published by the PBS Dask runner."""
    expected_raw = os.environ.get("DASK_EXPECTED_WORKERS")
    if expected_raw is None:
        logging.warning("DASK_EXPECTED_WORKERS is not set; skipping the worker registration barrier.")
        return None
    expected = int(expected_raw)
    timeout = float(os.environ.get("DASK_WORKER_STARTUP_TIMEOUT", "600"))
    if expected < 1:
        raise ValueError(f"DASK_EXPECTED_WORKERS must be positive, got {expected}.")
    logging.info("Waiting up to %.1fs for %s Dask workers.", timeout, expected)
    client.wait_for_workers(expected, timeout=timeout)
    connected = len(client.scheduler_info(n_workers=-1)["workers"])
    logging.info("Dask worker registration complete: expected=%s connected=%s.", expected, connected)
    return connected


def validate_cluster_preloads(client: Client) -> None:
    """Require the scheduler and every connected worker to finish preloading."""
    scheduler_state = client.run_on_scheduler(scheduler_preload_status)
    if not scheduler_state.get("completed", False):
        raise RuntimeError("Homogenization preload did not complete on the Dask scheduler.")
    worker_states = client.run(worker_preload_status)
    if not worker_states:
        raise RuntimeError("No Dask workers are connected for homogenization sampling.")
    incomplete = [address for address, state in worker_states.items() if not state.get("completed", False)]
    if incomplete:
        raise RuntimeError("Homogenization preload did not complete on workers: " + ", ".join(sorted(incomplete)))
    logging.info(
        "Verified Dask preloads: scheduler pid=%s, workers=%s.",
        scheduler_state["pid"],
        len(worker_states),
    )


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
    expected_workers = wait_for_expected_workers(client)
    if expected_workers is not None:
        validate_cluster_preloads(client)
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


PBS_SCRIPT_TEMPLATE = """
#!/bin/bash
#PBS -S /bin/bash
#PBS -l select={n_chunks}:ncpus={n_cpus}:mem={mem}:scratch_local={scratch}
#PBS -l place=scatter
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -N {job_name}
#PBS -W umask=0002
#PBS -o {out_log}
#PBS -j oe
#PBS -m e

set -x
env | grep PBS_
export TMPDIR=$SCRATCHDIR
export ENDORSE_DISABLE_MEMOIZE=1

output_dir={outputdir}

SCRIPT_PATH={script_path}
PROJECT_DIR=$(dirname -- $SCRIPT_PATH)
WORK_DIRNAME="workdir"

# A node list with each host once
UNIQ_HOSTS=$(sort -u "$PBS_NODEFILE")
NCPUS=$(wc -l < "$PBS_NODEFILE")

echo "Staging data to scratch ..."
for node in $UNIQ_HOSTS; do
    pbsdsh -vh "$node" -- rsync -a --delete "$output_dir/" "$SCRATCHDIR/" &
done
wait

bash $PROJECT_DIR/dask_cluster_env.sh $output_dir start

cd "$SCRATCHDIR"
echo "START SAMPLING"
bash $PROJECT_DIR/dask_cluster_env.sh $output_dir run {app_cmd}
echo "FINISHED SAMPLING"

echo "Copying results back ..."
uname -n
for node in $UNIQ_HOSTS; do
    pbsdsh -vh "$node" -- rsync -a "$SCRATCHDIR/$WORK_DIRNAME/" "$output_dir/workdir_$node" &
done
wait

bash $PROJECT_DIR/dask_cluster_env.sh $output_dir stop

clean_scratch

echo "FINISHED"
"""


def submit_pbs(cfg_machine: dotdict, cmd: str = "meta", *, submit: bool = True) -> Path:
    """Create the PBS script and optionally submit it with ``qsub``."""
    cfg_machine = cfg_machine.get("__resolved__", cfg_machine)
    cfg_pbs = cfg_machine.pbs
    pbs_path = Path(job.output.pbs_script)
    n_workers = int(cfg_pbs.n_nodes * (cfg_pbs.n_cores - 1) - 2)
    parameters = {
        "n_chunks": cfg_pbs.n_nodes,
        "n_cpus": cfg_pbs.n_cores,
        "queue": cfg_pbs.queue,
        "mem": cfg_pbs.mem,
        "scratch": cfg_pbs.scratch,
        "walltime": cfg_pbs.walltime,
        "job_name": cfg_pbs.pbs_name,
        "python": sys.executable,
        "n_workers": n_workers,
        "script_path": Path(__file__).resolve(),
        "workdir": job.scratch.dir_path,
        "outputdir": job.output.dir_path,
        "out_log": job.output.dir_path / f"{cfg_pbs.pbs_name}.out",
        "app_cmd": cmd,
    }
    pbs_path.write_text(PBS_SCRIPT_TEMPLATE.format(**parameters), encoding="utf-8")

    qsub_cmd = ["qsub", str(pbs_path)]
    logging.info("submit pbs: '%s'", qsub_cmd)
    if submit:
        subprocess.run(qsub_cmd, check=True)
    return pbs_path


def main() -> None:
    """Resolve the workdir, command, and optional Dask scheduler address."""
    setup_logging()
    scheduler = None
    if len(sys.argv) == 3:
        work_dir = Path(sys.argv[1]).absolute()
        cmd = sys.argv[2]
    elif len(sys.argv) == 4:
        work_dir = Path(sys.argv[1]).absolute()
        cmd = sys.argv[2]
        scheduler = sys.argv[3]
    else:
        sys.exit("Provide <workdir> <command: (submit|local|meta|mlmc|continue|random)> <command_args>.")

    job.set_workdir(work_dir)
    resolve_subcmd(cmd, work_dir, scheduler)


def resolve_subcmd(
    cmd: str,
    work_dir: Path,
    scheduler: str | None,
    copy_flag: bool = True,
) -> None:
    """Resolve one sampling command using the optional external scheduler."""
    if copy_flag:
        shutil.copytree(Path(__file__).resolve().parent / job.input.dir_path.name,
                        job.input.dir_path, dirs_exist_ok=True)

    logging.info(job.to_str())
    if not job.input.dir_path.exists():
        raise FileNotFoundError(f"Input data '{job.input.dir_path}' not found in workdir '{work_dir}'")

    cfg = common.config.load_config(str(job.input.study_cfg_path))
    if cmd == "submit":
        if len(sys.argv) == 4:
            app_cmd = sys.argv[3]
            assert app_cmd in ["continue", "meta", "mlmc", "random"]
            submit_pbs(cfg.machine_config, cmd=app_cmd)
        else:
            submit_pbs(cfg.machine_config)
        return

    if cmd == "random":
        cfg.mlmc.sim_class = "RandomSimulation"
    elif cmd not in {"local", "meta", "mlmc", "continue"}:
        sys.exit(f"Unknown command provided: '{cmd}'!")

    with dask.config.set({"distributed.worker.daemon": False}):
        client = Client(scheduler)
        logging.info("Connected to: %s", client)
        run_mlmc_sampling(cfg, client)


if __name__ == "__main__":
    main()
