import shutil
from pathlib import Path
import logging

from endorse import common

from homogenization_study.create_mesh import (
    make_fractures,
    make_geometry_homo,
    make_geometry_micro,
    make_mesh,
    update_dfn_params,
)
from homogenization_study import job
from homogenization_study.model import StudyMeshFiles
from homogenization_study.problem import prepare_conductivity_homogenization, run_homogenization_problem

def update_mesh_cfg(cfg_mesh, cfg_params, new_mesh_name):
    mcfg = common.apply_variant(cfg_mesh, cfg_params)
    mcfg.mesh_name = new_mesh_name
    return mcfg


def test_mesh() -> None:
    """Build both meshes and prepare the averaging domains for conductivity homogenization."""
    app_dir = Path(__file__).absolute().parents[1]
    work_dir = app_dir / "workdir_msh"
    job.set_workdir(work_dir)
    shutil.copytree(app_dir / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)
    logging.info(job.to_str())

    if not job.input.dir_path.exists():
        raise Exception(f"Input data '{job.input.dir_path}' not found in workdir '{work_dir}'")
    cfg_path = job.input.study_cfg_path
    cfg = common.config.load_config(str(cfg_path))

    param_dict = {
        'dfn_pop_1_alpha': 2.42941414948701,
        'dfn_pop_1_p30': 0.1151259392307745,
        'dfn_pop_1_tr_a': 8.85896252732927e-09,
        'dfn_pop_2_alpha': 3.024403009869272,
        'dfn_pop_2_p30': 0.1600760419324986,
        'dfn_pop_2_tr_a': 3.990062310689759e-09,
        'dfn_pop_3_alpha': 3.028455929427835,
        'dfn_pop_3_p30': 0.1547317897009867,
        'dfn_pop_3_tr_a': 3.0339712706664374e-08,
        'dfn_pop_4_alpha': 2.507600614787692,
        'dfn_pop_4_p30': 0.1304209040611591,
        'dfn_pop_4_tr_a': 5.929355247766668e-09,
        'dfn_pop_5_alpha': 2.434281368148621,
        'dfn_pop_5_p30': 0.1594357853078999,
        'dfn_pop_5_tr_a': 1.4162405896638502e-07,
        'dfn_seed': 1, 'meshing_seed': 1}
    update_dfn_params(cfg.mesh.fractures, param_dict)

    with common.workdir(work_dir, clean=False):
        fr_pop, fracture_set, n_large = make_fractures(cfg.mesh, param_dict["dfn_seed"])

        cfg_mesh = update_mesh_cfg(cfg.mesh, cfg.mlmc.levels[0].homo_params, "homogenization_mesh")
        homogenization_mesh = make_mesh(
            cfg_mesh,
            fracture_set,
            param_dict["meshing_seed"],
            geometry_func=make_geometry_homo,
        )

        cfg_mesh = update_mesh_cfg(cfg.mesh, cfg.mlmc.levels[0].params, "micro_mesh")
        micro_mesh = make_mesh(
            cfg_mesh,
            fracture_set,
            param_dict["meshing_seed"],
            geometry_func=make_geometry_micro,
        )

        meshes = StudyMeshFiles(
                micro_mesh=micro_mesh.path,
                homogenization_mesh=homogenization_mesh.path)
        prepared = prepare_conductivity_homogenization(cfg, meshes, fractures=fracture_set)

        result = run_homogenization_problem(cfg, work_dir, meshes, prepared)

    assert prepared.homogenized_elements.size > 0
    assert prepared.subproblems.n_subdomains == prepared.homogenized_elements.size
    assert prepared.empty_macro_elements.size == 0
