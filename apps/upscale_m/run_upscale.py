"""
Main driver: buffered micro mesh -> six KUBC Flow123d runs -> inner-cube averaging -> effective
elastic tensor report.

Run (Docker Desktop must be running for the Flow123d wrapper):

    cd apps/upscale_m
    ../../venv/Scripts/python.exe run_upscale.py RUN_NAME [CONFIG]

Layout convention (source vs data, like Python_scripts' src/Raw_data split):
    configs/  - study definitions (CONFIG is looked up there; bare names work: config_dfn.yaml)
    runs/     - ALL run products, git-ignored; RUN_NAME becomes runs/RUN_NAME/ automatically
Each run dir contains the healed mesh, one directory per load case with all Flow123d
in/outputs, and the effective_tensor_C_k_kinematic_bc.txt report (all matrices are the MEASURED
inner-cube averages; prescribed load values are only boundary conditions, see upscale_tensor.py).
"""
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_app_dir, "src"))

import numpy as np

from endorse import common

from micro_mesh import make_micro_mesh
from kubc import run_kubc
from upscale_tensor import assemble_matrices, equivalent_tensor, write_report


def main(work_dir="default", config_file="config.yaml"):
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
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures, work_dir=".")

        print("=== upscale_m: KUBC load cases ===", flush=True)
        results = run_kubc(cfg, micro)

        print("=== upscale_m: tensor assembly ===", flush=True)
        meta = dict(
            L_inner=cfg.geometry.L_inner,
            L_ext_factor=cfg.geometry.get("L_ext_factor", 2.0),
            subdomains=list(micro.subdivision),
            beta=cfg.loads.deformation_parameter_beta,
            E_rock=cfg.materials.young_modulus_rock,
            nu_rock=cfg.materials.poisson_ratio_rock,
            E_fracture=cfg.materials.young_modulus_fracture,
            nu_fracture=cfg.materials.poisson_ratio_fracture,
            fractures=[(fr.r, list(fr.center), list(fr.normal)) for fr in micro.fractures],
        )
        # one tensor per subdomain cell, all from the same six solutions
        # (mirrors macro_conductivity's per-subdomain loop; single cell = the classic run)
        n_sub = len(micro.subdomain_boxes)
        for i_sub, (lo3, hi3) in enumerate(micro.subdomain_boxes):
            E, Sigma = assemble_matrices(results, i_sub)
            C_k = equivalent_tensor(E, Sigma, method="exact")
            meta_i = dict(meta)
            meta_i["subdomain_box"] = [list(np.round(lo3, 6)), list(np.round(hi3, 6))]
            if n_sub == 1:
                report_path = "effective_tensor_C_k_kinematic_bc.txt"
            else:
                ix, iy, iz = micro.subdomain_index(i_sub)
                report_path = os.path.join(f"subdomain_{ix}_{iy}_{iz}",
                                           "effective_tensor_C_k_kinematic_bc.txt")
            write_report(report_path, results, E, Sigma, C_k, meta_i)

            if n_sub == 1:
                with np.printoptions(precision=3, suppress=False, linewidth=140):
                    print("\nC_k:\n", C_k)
            else:
                ix, iy, iz = micro.subdomain_index(i_sub)
                print(f"  subdomain ({ix},{iy},{iz}): "
                      f"C11 = {C_k[0, 0]:.4e}  C33 = {C_k[2, 2]:.4e}  C66 = {C_k[5, 5]:.4e}")
    print("\n=== upscale_m: DONE ===")


if __name__ == "__main__":
    main(*sys.argv[1:])
