from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/endorse_matplotlib")

import numpy as np
import openturns as ot

from endorse import common

import chodby_trans.job as job
import chodby_trans.transport_simulation as transport_simulation
from chodby_trans import ot_sa
from chodby_trans.fullscale_transport import prepare_common_homogenization_mesh


SCRIPT_PATH = Path(__file__).absolute()
DEFAULT_INPUT_DIR = SCRIPT_PATH.parent / "input_data"


def setup_logging() -> None:
    fmt = "%(asctime)s [seq-saltelli] %(hostname)s:%(process)d %(levelname)s: %(message)s"

    class HostnameFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.hostname = socket.gethostname()
            return True

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(HostnameFilter())
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[handler], force=True)


def set_threadsafe_environ() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def copy_input_data(workdir: Path, source_input_dir: Path, overwrite: bool) -> Path:
    input_dir = workdir / "input_data"
    if input_dir.exists() and overwrite:
        shutil.rmtree(input_dir)
    if not input_dir.exists():
        shutil.copytree(source_input_dir, input_dir)
    return input_dir


def make_group_matrix(sa_obj: ot_sa.SensitivityAnalysis, n_rows: int) -> np.ndarray:
    group_distr = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * len(sa_obj.groups))
    experiment = sa_obj._experiment_design(group_distr, int(n_rows))
    return np.asarray(experiment.generate(), dtype=float)


def saltelli_terms(a_row: np.ndarray, b_row: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    """
    Yield Saltelli rows in the MLMC Sobol wrapper order.
    """
    yield "A", a_row
    for i_param in range(len(a_row)):
        row = np.array(a_row, copy=True)
        row[i_param] = b_row[i_param]
        yield f"AB_{i_param:02d}", row
    for i_param in range(len(a_row)):
        row = np.array(b_row, copy=True)
        row[i_param] = a_row[i_param]
        yield f"BA_{i_param:02d}", row
    yield "B", b_row


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def run_one_term(
    sim: transport_simulation.TransportSimulation,
    root_cfg,
    output_dir: Path,
    level_id: int,
    saltelli_index: int,
    term_name: str,
    group_row: np.ndarray,
    finer_sample_count: int,
) -> dict:
    sample_dir = output_dir / f"L{level_id:02d}_S{saltelli_index:07d}_{term_name}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "level_id": level_id,
        "root_cfg": copy.deepcopy(root_cfg),
    }
    sample_input = np.concatenate([group_row, np.array([float(finer_sample_count)])])
    full_parameters = transport_simulation.expand_sample_parameters(root_cfg, group_row)
    parameter_dict = ot_sa.SensitivityAnalysis.from_cfg(root_cfg.ot_sensitivity).param_vec_to_dict(full_parameters)

    payload = {
        "level_id": level_id,
        "saltelli_index": saltelli_index,
        "term": term_name,
        "group_row": group_row.tolist(),
        "sample_input": sample_input.tolist(),
        "parameters": {name: float(value) for name, value in parameter_dict.items()},
        "sample_dir": str(sample_dir),
    }
    atomic_write_json(sample_dir / "input.json", payload)

    old_cwd = Path.cwd()
    start = time.time()
    try:
        os.chdir(sample_dir)
        fine, coarse = sim.calculate(config_dict, sample_input)
        elapsed = time.time() - start
        np.savez(
            sample_dir / "result.npz",
            fine=np.asarray(fine),
            coarse=np.asarray(coarse),
            sample_input=sample_input,
            group_row=group_row,
        )
        status = {
            **payload,
            "status": "ok",
            "elapsed_sec": elapsed,
            "fine_shape": list(np.asarray(fine).shape),
            "coarse_shape": list(np.asarray(coarse).shape),
        }
    except Exception as exc:
        elapsed = time.time() - start
        status = {
            **payload,
            "status": "failed",
            "elapsed_sec": elapsed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (sample_dir / "error.txt").write_text(status["traceback"], encoding="utf-8")
    finally:
        os.chdir(old_cwd)

    atomic_write_json(sample_dir / "status.json", status)
    return status


def run_sequential_saltelli(args: argparse.Namespace) -> int:
    set_threadsafe_environ()
    if args.disable_memoize:
        os.environ["ENDORSE_DISABLE_MEMOIZE"] = "1"

    workdir = args.workdir.absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    input_dir = copy_input_data(workdir, args.input_dir.absolute(), args.overwrite_input)
    job.set_workdir(workdir, input_dir)

    cfg_path = input_dir / args.config_name
    cfg = common.config.load_config(str(cfg_path))
    logging.info("Job dirs:\n%s", job.to_str())
    logging.info("Config: %s", cfg_path)

    output_dir = workdir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with common.workdir(str(job.scratch.dir_path), clean=False):
        prepare_common_homogenization_mesh(cfg)

    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    if args.level_id < 0 or args.level_id >= len(cfg.mlmc.levels):
        raise ValueError(f"level-id {args.level_id} is outside cfg.mlmc.levels")

    ot.RandomGenerator.SetSeed(args.seed)
    a_matrix = make_group_matrix(sa_obj, args.n_saltelli)
    b_matrix = make_group_matrix(sa_obj, args.n_saltelli)

    sim = transport_simulation.TransportSimulation(cfg, workdir)
    summary = {
        "config": str(cfg_path),
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "level_id": args.level_id,
        "seed": args.seed,
        "n_saltelli": args.n_saltelli,
        "n_groups": len(sa_obj.groups),
        "groups": list(sa_obj.groups),
        "disable_memoize": bool(args.disable_memoize),
        "statuses": [],
    }

    for i_sample in range(args.n_saltelli):
        for term_name, group_row in saltelli_terms(a_matrix[i_sample], b_matrix[i_sample]):
            logging.info("Running sample %s term %s", i_sample, term_name)
            status = run_one_term(
                sim=sim,
                root_cfg=cfg,
                output_dir=output_dir,
                level_id=args.level_id,
                saltelli_index=i_sample,
                term_name=term_name,
                group_row=group_row,
                finer_sample_count=args.finer_sample_count,
            )
            logging.info(
                "Finished sample %s term %s: %s in %.1fs",
                i_sample,
                term_name,
                status["status"],
                status["elapsed_sec"],
            )
            summary["statuses"].append(status)
            atomic_write_json(output_dir / "summary.json", summary)
            if status["status"] != "ok" and args.stop_on_error:
                return 1

    n_failed = sum(1 for status in summary["statuses"] if status["status"] != "ok")
    logging.info("Sequential Saltelli run finished, failed=%s/%s", n_failed, len(summary["statuses"]))
    return 1 if n_failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real transport Saltelli sample inputs sequentially without MLMC/Dask machinery.",
    )
    parser.add_argument("workdir", type=Path, help="Working directory for copied input_data and sequential outputs.")
    parser.add_argument(
        "-n",
        "--n-saltelli",
        type=int,
        required=True,
        help="Number of Saltelli base samples. Each sample runs 2 * (n_groups + 1) model evaluations.",
    )
    parser.add_argument("--seed", type=int, default=101, help="OpenTURNS random seed.")
    parser.add_argument("--level-id", type=int, default=0, help="Transport MLMC level id from cfg.mlmc.levels.")
    parser.add_argument(
        "--finer-sample-count",
        type=int,
        default=0,
        help="Value appended to sample_input for the Goal 3 finer-level count field.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Source real input_data directory.")
    parser.add_argument("--config-name", default="transport_mlmc.yaml", help="Config file name inside input_data.")
    parser.add_argument("--output-dir", default="sequential_saltelli", help="Output subdirectory under workdir.")
    parser.add_argument("--overwrite-input", action="store_true", help="Replace workdir/input_data before running.")
    parser.add_argument(
        "--enable-memoize",
        action="store_false",
        dest="disable_memoize",
        help="Leave @memoize enabled. By default this script disables memoize.",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed term.")
    parser.set_defaults(disable_memoize=True)
    args = parser.parse_args()
    if args.n_saltelli <= 0:
        parser.error("--n-saltelli must be positive")
    return args


def main() -> int:
    setup_logging()
    return run_sequential_saltelli(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
