"""
The micro problem: buffered fractured mesh -> six Flow123d load cases per BC type.
"""
import logging
import os
import sys
from pathlib import Path

from endorse import common

from micro_mesh import make_micro_mesh
from bc import run_bc


def main(work_dir="default", config_file="config_master.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    common.CallCache.instance()
    app_dir = Path(__file__).parents[1]
    cfg_path = app_dir / "configs" / config_file
    if not cfg_path.is_file():
        cfg_path = app_dir / config_file
    cfg = common.load_config(cfg_path)
    if not os.path.isabs(work_dir):
        work_dir = app_dir / "runs" / work_dir

    print(f"[upscale_m] config: {cfg_path}")
    print(f"[upscale_m] run dir: {work_dir}")

    bc_type = str(cfg.loads.bc_type).strip().lower()
    assert bc_type in ("kubc", "subc", "both"), \
        f"loads.bc_type must be 'kubc', 'subc' or 'both' (got {bc_type!r})"

    bc_types = [t for t in ("kubc", "subc") if bc_type in (t, "both")]

    os.makedirs(work_dir, exist_ok=True)
    with common.workdir(work_dir, inputs=[]):
        print("=== upscale_m: micro mesh ===", flush=True)
        # SUBC needs the rigid-body support tetrahedra built INTO the mesh, so a 'both' run
        # solves both BC types on identical geometry; a pure KUBC run gets none
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures,
                                cfg.materials.aperture_per_r, work_dir=".",
                                subc_support="subc" in bc_types)

        for t in bc_types:
            print(f"=== upscale_m: {t.upper()} load cases ===", flush=True)
            run_bc(cfg, micro, t)

    print("\n=== upscale_m: DONE ===")


if __name__ == "__main__":
    main(*sys.argv[1:])
