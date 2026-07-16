"""
Main driver: buffered micro mesh -> six KUBC Flow123d runs -> inner-cube averaging -> effective
elastic tensor report.

Run (Docker Desktop must be running for the Flow123d wrapper):

    cd apps/upscale_m
    ../../venv/Scripts/python.exe run_upscale.py RUN_NAME [CONFIG]

Layout convention (source vs data, like Python_scripts' src/Raw_data split):
    configs/  - study definitions (CONFIG is looked up there; bare names work: config.yaml)
    runs/     - ALL run products, git-ignored; RUN_NAME becomes runs/RUN_NAME/ automatically
Each run dir contains the healed mesh, one directory per load case with all Flow123d
in/outputs, and the tensor report (name from config key output.report_name; all matrices are
the MEASURED inner-cube averages; prescribed load values are only boundary conditions, see
upscale_tensor.py).
"""
import logging
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_app_dir, "src"))

from endorse import common

from micro_mesh import make_micro_mesh, make_averaging_grid
from kubc import run_kubc
from upscale_tensor import write_all_reports


def main(work_dir="default", config_file="config.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    common.CallCache.instance()
    # configs live in configs/; accept both bare names and explicit paths
    cfg_path = os.path.join(_app_dir, "configs", config_file)
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(_app_dir, config_file)
    cfg = common.load_config(cfg_path)
    # all run products go under runs/ (git-ignored), unless an absolute path is given
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(_app_dir, "runs", work_dir)
    print(f"[upscale_m] config: {cfg_path}")
    print(f"[upscale_m] run dir: {work_dir}")

    bc_type = str(cfg.loads.bc_type).strip().lower()
    assert bc_type == "kubc", \
        f"Only 'kubc' is implemented so far (got {bc_type!r}); SUBC pending PLAN.md decisions."

    os.makedirs(work_dir, exist_ok=True)
    with common.workdir(work_dir, inputs=[]):
        print("=== upscale_m: micro mesh ===", flush=True)
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures,
                                cfg.materials.aperture_per_r, work_dir=".")

        print("=== upscale_m: KUBC load cases ===", flush=True)
        grid = make_averaging_grid(cfg.geometry, micro)
        results = run_kubc(cfg, micro, grid)

        print("=== upscale_m: tensor assembly ===", flush=True)
        write_all_reports(cfg, micro, grid, results)

    print("\n=== upscale_m: DONE ===")


if __name__ == "__main__":
    main(*sys.argv[1:])
